import copy
import random
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import gym
import numpy as np
import pygame
import torch
from gym.spaces import Box, Discrete
from matplotlib import pyplot as plt
from numba import jit
from torch import nn

ARENA_WIDTH = 31
ARENA_HEIGHT = 31

assert ARENA_WIDTH % 2 != 0, "Need odd sized grid for egocentric view"
assert ARENA_HEIGHT == ARENA_WIDTH, "current only support square arenas"

BLOCK_SIZE = 10

SCREEN_WIDTH = ARENA_WIDTH * BLOCK_SIZE
SCREEN_HEIGHT = ARENA_HEIGHT * BLOCK_SIZE

# Game terminates after this number of steps
MAX_STEPS = 2000

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 102)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 100, 0)

HERE = Path(__file__).parent.resolve()


def load_network(team_name: str, network_folder: Path = HERE) -> nn.Module:
    net_path = network_folder / f"{team_name}_network.pt"
    assert (
        net_path.exists()
    ), f"Network saved using TEAM_NAME='{team_name}' doesn't exist! ({net_path})"
    model = torch.load(str(HERE / net_path))
    model.eval()
    return model


def save_network(network: nn.Module, team_name: str) -> None:
    assert isinstance(
        network, nn.Module
    ), f"train() function outputs an network type: {type(network)}"
    assert "/" not in team_name, "Invalid TEAM_NAME. '/' are illegal in TEAM_NAME"
    net_path = HERE / f"{team_name}_network.pt"
    n_retries = 5
    for attempt in range(n_retries):
        try:
            torch.save(network, net_path)
            load_network(str(HERE / team_name))
            return
        except Exception:
            if attempt == n_retries - 1:
                raise


def choose_move_randomly(state):
    move = int(random.random() * 3) + 1
    return move


def play_snake(
    your_choose_move: Callable,
    opponent_choose_moves: List[Callable],
    game_speed_multiplier=1,
    render=True,
    verbose=False,
) -> None:
    env = SnakeEnv(opponent_choose_moves=opponent_choose_moves, verbose=False, render=True)
    state = env.reset()
    done = False
    while not done:
        action = your_choose_move(state)

        state, reward, done, _ = env.step(action)
        time.sleep(1 / game_speed_multiplier)


def wrap_position(pos: Tuple[int, int]) -> Tuple[int, int]:
    # Have removed the wrapping for now
    return pos
    # x, y = pos
    # # This could easily be wrong
    # x = ARENA_WIDTH + x if x < 0 else x - ARENA_WIDTH if x >= ARENA_WIDTH else x
    # y = ARENA_HEIGHT + y if y < 0 else y - ARENA_HEIGHT if y >= ARENA_HEIGHT else y
    # return (x, y)


def in_arena(pos: Tuple[int, int]):
    y_out = pos[1] <= 0 or pos[1] >= ARENA_HEIGHT - 1
    x_out = pos[0] <= 0 or pos[0] >= ARENA_WIDTH - 1
    return not x_out and not y_out


class Action:
    """The action taken by the snake.

    The snake has 3 options:
        1. Go forward
        2. Turn left (and go forward 1 step)
        3. Turn right (and go forward 1 step)
    """

    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


class Orientation:
    """Direction the snake is pointing."""

    SOUTH = 0  # negative y-direction
    EAST = 1  # positive x-direction
    NORTH = 2  # positive y-direction
    WEST = 3  # negative x-direction


REMAP_ORIENTATION = {0: [0, -1], 1: [1, 0], 2: [0, 1], 3: [-1, 0]}

# how many orientations for grid to be north from your perspective?
ORIENTATION_2_ROT = {0: 2, 2: 0, 1: 1, 3: 3}


class Snake:
    def __init__(
        self, name: str = "snek", starting_position: Optional[Tuple[int, int]] = None
    ) -> None:

        self.snake_direction = random.choice(
            [Orientation.EAST, Orientation.WEST, Orientation.NORTH, Orientation.SOUTH]
        )

        if starting_position is None:
            snake_head_x = random.randint(ARENA_WIDTH // 4, 3 * ARENA_WIDTH // 4)
            snake_head_y = random.randint(ARENA_HEIGHT // 4, 3 * ARENA_HEIGHT // 4)
        else:
            snake_head_x, snake_head_y = starting_position

        snake_tail_x = (
            snake_head_x - 1
            if self.snake_direction == Orientation.EAST
            else snake_head_x + 1
            if self.snake_direction == Orientation.WEST
            else snake_head_x
        )
        snake_tail_y = (
            snake_head_y - 1
            if self.snake_direction == Orientation.NORTH
            else snake_head_y + 1
            if self.snake_direction == Orientation.SOUTH
            else snake_head_y
        )
        self.snake_positions = [(snake_head_x, snake_head_y), (snake_tail_x, snake_tail_y)]
        self.alive = True
        self.name = name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Snake) and self.name == other.name

    def has_hit_boundaries(self) -> bool:
        # y_boundary_hit = self.snake_head[1] <= 0 or self.snake_head[1] >= ARENA_HEIGHT - 1
        # x_boundary_hit = self.snake_head[0] <= 0 or self.snake_head[0] >= ARENA_WIDTH - 1
        return not in_arena(self.snake_head)
        # return y_boundary_hit or x_boundary_hit

    def kill_snake(self) -> None:
        self.alive = False

    def has_hit_self(self) -> bool:
        return self.snake_head in self.snake_body

    @property
    def snake_length(self) -> int:
        return len(self.snake_positions)

    @property
    def snake_head(self) -> Tuple[int, int]:
        return self.snake_positions[0]

    @property
    def snake_body(self) -> List[Tuple[int, int]]:
        return self.snake_positions[1:]

    def take_action(self, action: int) -> None:

        if action == 2:
            new_orientation = (self.snake_direction + 1) % 4
        elif action == 3:
            new_orientation = (self.snake_direction - 1) % 4
        else:
            new_orientation = self.snake_direction

        x, y = self.snake_head
        if new_orientation % 2 == 0:
            # South is 0 (y -= 1), North is 2 (y += 1)
            y += new_orientation - 1
        else:
            # East is 1 (x += 1), West is 3 (x -= 1)
            x += 2 - new_orientation

        # Update position and orientation
        if action is not None:
            self.snake_positions.insert(0, (x, y))
            self.snake_direction = new_orientation

        self.snake_positions = [wrap_position(pos) for pos in self.snake_positions]

    def remove_tail_end(self) -> None:
        del self.snake_positions[-1]


def get_starting_positions() -> List[Tuple[int, int]]:
    """Get a list of starting positions that are not too close together."""
    min_x = ARENA_WIDTH // 4
    max_x = 3 * ARENA_WIDTH // 4
    min_y = ARENA_HEIGHT // 4
    max_y = 3 * ARENA_HEIGHT // 4
    positions = []

    # Return n**2 points
    n = 3

    offset_x = (max_x - min_x) // (n - 1)
    offset_y = (max_y - min_y) // (n - 1)
    for i in range(n):
        for j in range(n):
            positions.append((min_x + offset_x * i, min_y + offset_y * j))

    return positions


class SnakeEnv(gym.Env):
    def __init__(
        self,
        opponent_choose_moves: List[Callable],
        verbose: bool = False,
        render: bool = False,
        game_speed_multiplier: int = 1,
    ):
        self.choose_move_store = copy.deepcopy(opponent_choose_moves)

        self.opponent_choose_moves = opponent_choose_moves
        self._render = render
        self.verbose = verbose
        self.game_speed_multiplier = game_speed_multiplier

        self.action_space = Discrete(3)

        self.observation_space = Box(
            low=0, high=255, shape=(ARENA_WIDTH, ARENA_HEIGHT, 1), dtype=np.uint8
        )

        self.metadata = ""
        self.arena = np.zeros((ARENA_WIDTH, ARENA_HEIGHT, 1))
        self.starting_positions = get_starting_positions()
        self.score = 0
        if self._render:
            self.init_visuals()

    def reset(self) -> np.ndarray:
        self.opponent_choose_moves = self.choose_move_store
        self.player_dead = False
        self.food_position = (
            random.randint(2, ARENA_WIDTH - 2),
            random.randint(2, ARENA_HEIGHT - 2),
        )
        self.num_steps_taken = 0
        # Currently player snake is just stored as the first element of the
        # overall snakes list. Maybe just separate variable?
        random.shuffle(self.starting_positions)

        self.player_snake = Snake(name="player", starting_position=self.starting_positions[0])
        self.snakes = [self.player_snake]
        self.snakes += [
            Snake(name=f"opponent_{idx}", starting_position=self.starting_positions[idx + 1])
            for idx in range(len(self.opponent_choose_moves))
        ]
        self.dead_snakes: List[Snake] = []
        assert len(self.snakes) == len(self.opponent_choose_moves) + 1

        return self.get_snake_state(self.snakes[0])

    @property
    def done(self) -> bool:
        # done if player is not in list of snakes anymore
        return not any([snake.name == "player" for snake in self.snakes]) or len(self.snakes) < 2

    def generate_food(self) -> None:
        possible_food_positions = [
            (x, y)
            for x in range(ARENA_WIDTH)
            for y in range(ARENA_HEIGHT)
            if (x, y) not in [snake.snake_positions for snake in self.snakes]
        ]
        self.food_position = random.choice(possible_food_positions)

    def _step(self, action: int, snake: Snake) -> int:

        if action is None:
            return 0

        # AHHHHHHHHHHHHHHH BADDD
        # action += 1
        snake.take_action(action)

        if action not in [Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT]:
            raise ValueError(f"Invalid action: {action}")

        # Remove end of tail unless food eaten
        if snake.snake_head != self.food_position:
            reward = 0
            snake.remove_tail_end()
        else:
            reward = 1
            self.generate_food()
            if snake.name == "player":
                self.score += 1

        if self.has_hit_tails(snake.snake_head) or snake.has_hit_boundaries():
            snake.kill_snake()
            reward = 0

        if self.head_to_head_collision(snake):
            reward = 0

        self.num_steps_taken += 1
        if self.verbose and self.num_steps_taken % 1000 == 0:
            print(f"{self.num_steps_taken} steps taken")

        if self.num_steps_taken >= MAX_STEPS:
            if self.verbose:
                print("RUN OUT OF TIME!")
            snake.kill_snake()

        return reward

    def head_to_head_collision(self, snake: Snake) -> bool:
        for other_snake in self.snakes:
            if other_snake == snake:
                continue
            if other_snake.snake_head == snake.snake_head:
                other_snake.kill_snake()
                snake.kill_snake()
                return True
        return False

    def has_hit_tails(self, snake_head: Tuple[int, int]) -> bool:
        return any([snake_head in snake.snake_body for snake in self.snakes])

    @staticmethod
    def idx_to_flat(pos: Tuple[int, int], dim: int) -> int:
        """idx of a matrix to its position in the flattened vector."""
        return dim * pos[0] + pos[1]

    @staticmethod
    def boundary_elements_mask(matrix: np.ndarray) -> np.ndarray:
        mask = np.ones(matrix.shape, dtype=bool)
        mask[matrix.ndim * (slice(1, -1),)] = False
        return mask

    def in_arena(self, pos: Tuple[int, int]):
        return pos[0]

    def get_snake_state(self, ego_snake: Snake) -> np.ndarray:
        """Get egocentric positioning for a single snake 'ego_snake'."""
        return np.array([1, 1, 1])

        # self.arena[:] = 0
        # # Will break stable baselines
        # self.arena = self.arena.squeeze()
        # boundary_pos = np.where(self.boundary_elements_mask(self.arena))
        # # self.arena[self.boundary_elements_mask(self.arena)] = 88

        # #  Subtract to normalise to snake head position
        # norm_x = ego_snake.snake_head[0] - ARENA_WIDTH // 2
        # norm_y = ego_snake.snake_head[1] - ARENA_HEIGHT // 2

        # norm_boundary = (boundary_pos[0] - norm_x, boundary_pos[1] - norm_y)
        # keep_idx = np.logical_and(
        #     np.logical_and(0 <= norm_boundary[0], norm_boundary[0] < ARENA_WIDTH),
        #     np.logical_and(0 <= norm_boundary[1], norm_boundary[1] < ARENA_HEIGHT),
        # )
        # norm_boundary = (norm_boundary[0][keep_idx], norm_boundary[1][keep_idx])
        # self.arena[norm_boundary] = 88

        # norm_head = wrap_position(
        #     (ego_snake.snake_head[0] - norm_x, ego_snake.snake_head[1] - norm_y)
        # )

        # relative_food_position = wrap_position(
        #     (self.food_position[0] - norm_x, self.food_position[1] - norm_y)
        # )

        # if in_arena(relative_food_position):
        #     self.arena[relative_food_position] = 10

        # for snake in self.snakes:
        #     # Currently can't tell apart opponent head from tail
        #     for pos in snake.snake_positions:
        #         norm_pos = wrap_position((pos[0] - norm_x, pos[1] - norm_y))
        #         if in_arena(norm_pos):
        #             self.arena[norm_pos] = 255
        # self.arena[norm_head] = 255

        # self.arena = np.rot90(self.arena, k=ORIENTATION_2_ROT[ego_snake.snake_direction])

        # # print(self.arena)
        # # print("\n")

        # return self.arena

    def step(self, action: int) -> Tuple:

        # Step player's snake
        reward = self._step(action, self.snakes[0])

        assert len(self.snakes) == len(self.opponent_choose_moves) + 1
        for snake, choose_move in zip(self.snakes[1:], self.opponent_choose_moves):
            if not self.done:
                snake_state = self.get_snake_state(snake)
                action = choose_move(snake_state)
                reward = self._step(action, snake)
                reward = 0

        # Remove me
        assert self.snakes[0] == self.player_snake
        assert self.player_snake.name == "player"

        idx_alive = []
        for idx, snake in enumerate(self.snakes):
            if not snake.alive:
                if snake.name == "player":
                    self.player_dead = True
                self.dead_snakes.append(snake)
            else:
                idx_alive.append(idx)

        self.snakes = [self.snakes[idx] for idx in idx_alive]

        self.opponent_choose_moves = [
            self.opponent_choose_moves[idx - 1] for idx in idx_alive if idx != 0
        ]

        # self.snakes = [snake for snake in self.snakes if snake.alive]

        # self.snakes, self.opponent_choose_moves = list(
        #     zip(
        #         *[
        #             (snake, choose_move)
        #             for snake, choose_move in zip(self.snakes, self.opponent_choose_moves)
        #             if snake.alive
        #         ]
        #     )
        # )

        # keep_snakes = []
        # keep_moves = []

        # for snake, choose_move in zip(self.snakes, self.opponent_choose_moves):
        #     if snake.alive:
        #         keep_snakes.append(snake)
        #         keep_moves.append(choose_move)

        # self.snakes = keep_snakes
        # self.opponent_choose_moves = keep_moves

        if self._render:
            self.render_game()

        # if self.done:
        #     result = "won" if reward > 0 else "lost"
        #     msg = f"You {result} {abs(reward*2)} chips"

        # if self.verbose:
        #     print(msg)

        return self.get_snake_state(self.player_snake), reward, self.done, {}

    def init_visuals(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((ARENA_WIDTH * BLOCK_SIZE, ARENA_HEIGHT * BLOCK_SIZE))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.screen.fill(WHITE)
        self.score_font = pygame.font.SysFont("comicsansms", 35)

    def render_game(self) -> None:

        # Maybe necessary maybe not
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        self.screen.fill(WHITE)
        # Draw apple
        food_screen_x, food_screen_y = self.food_position

        food_screen_y = (
            ARENA_HEIGHT - food_screen_y - 1
        )  # Flip y axis because pygame counts 0,0 as top left
        pygame.draw.rect(
            self.screen,
            GREEN,
            [food_screen_x * BLOCK_SIZE, food_screen_y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE],
        )

        # Draw boundaries
        pygame.draw.rect(
            self.screen, BLACK, [1, 1, SCREEN_WIDTH - 1, SCREEN_HEIGHT - 1], width=BLOCK_SIZE
        )

        # Draw snake
        for snake in self.snakes:
            color = BLUE if snake.name == "player" else BLACK

            for snake_pos in snake.snake_body:
                snake_y = (
                    ARENA_HEIGHT - snake_pos[1] - 1
                )  # Flip y axis because pygame counts 0,0 as top left
                pygame.draw.rect(
                    self.screen,
                    color,
                    [snake_pos[0] * BLOCK_SIZE, snake_y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE],
                )
            # Flip y axis because pygame counts 0,0 as top left
            snake_y = ARENA_HEIGHT - snake.snake_head[1] - 1
            pygame.draw.rect(
                self.screen,
                DARK_GREEN,
                [
                    snake.snake_head[0] * BLOCK_SIZE,
                    snake_y * BLOCK_SIZE,
                    BLOCK_SIZE,
                    BLOCK_SIZE,
                ],
            )

        # draw score
        value = self.score_font.render(f"Your score: {self.score}", True, BLACK)
        self.screen.blit(value, [0, 0])
        pygame.display.update()


def human_player(state) -> Optional[int]:
    """Controls quite janky."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            quit()
        # elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
        #     return 3
    is_key_pressed = pygame.key.get_pressed()
    if is_key_pressed[pygame.K_RIGHT]:
        return 3
    elif is_key_pressed[pygame.K_LEFT]:
        return 2
    if is_key_pressed[pygame.K_UP]:
        return 1

    return None
