import random
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

import gym
import pygame
import torch
from gym.spaces import Box, Discrete
from numba import jit
from torch import nn

ARENA_WIDTH = 41
ARENA_HEIGHT = 41

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
    return random.choice([Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT])


def play_snake(
    your_choose_move: Callable,
    opponent_choose_move: Callable,
    game_speed_multiplier=1,
    render=True,
    verbose=False,
) -> None:
    pass


def wrap_position(pos: Tuple[int, int]) -> Tuple[int, int]:
    x, y = pos
    # This could easily be wrong
    x = ARENA_WIDTH + x if x < 0 else x - ARENA_WIDTH if x >= ARENA_WIDTH else x
    y = ARENA_HEIGHT + y if y < 0 else y - ARENA_HEIGHT if y >= ARENA_HEIGHT else y
    return (x, y)


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
    def __init__(self) -> None:
        self.snake_direction = random.choice(
            [Orientation.EAST, Orientation.WEST, Orientation.NORTH, Orientation.SOUTH]
        )
        snake_head_x = random.randint(ARENA_WIDTH // 4, 3 * ARENA_WIDTH // 4)
        snake_head_y = random.randint(ARENA_HEIGHT // 4, 3 * ARENA_HEIGHT // 4)
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

    def take_action(self, action: int):
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


class SnakeEnv(gym.Env):
    def __init__(
        self,
        opponent_choose_move: Callable,
        verbose: bool = False,
        render: bool = False,
        game_speed_multiplier: int = 1,
    ):
        self.opponent_choose_move = opponent_choose_move
        self.render = render
        self.verbose = verbose
        self.game_speed_multiplier = game_speed_multiplier

        self.action_space = Discrete(3)

        self.observation_space = Box(
            low=0, high=255, shape=(ARENA_WIDTH, ARENA_HEIGHT, 1), dtype=np.uint8
        )

        self.metadata = ""
        self.arena = np.zeros((ARENA_WIDTH, ARENA_HEIGHT, 1))
        if render:
            self.init_visuals()

    def reset(self) -> List[int]:
        self.food_position = (
            random.randint(0, ARENA_WIDTH - 1),
            random.randint(0, ARENA_HEIGHT - 1),
        )
        self.num_steps_taken = 0
        self.snake = Snake()

        # return self.state, 0, False, {}
        return self.state

    @property
    def done(self) -> bool:

        # Make better
        if not self.snake.alive:
            return True
        return False

    def generate_food(self) -> None:
        possible_food_positions = [
            (x, y)
            for x in range(ARENA_WIDTH)
            for y in range(ARENA_HEIGHT)
            if (x, y) not in self.snake.snake_positions
        ]
        self.food_position = random.choice(possible_food_positions)

    # @jit
    # @jit(nopython=True)
    def _step(self, action: int) -> int:

        if action is None:
            return 0

        # AHHHHHHHHHHHHHHH BADDD
        action += 1
        self.snake.take_action(action)

        if action not in [Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT]:
            raise ValueError(f"Invalid action: {action}")

        # Remove end of tail unless food eaten
        if self.snake.snake_head != self.food_position:
            reward = 0
            self.snake.remove_tail_end()
        else:
            reward = 1
            self.generate_food()

        # If you hit more snake, game over
        if self.snake.has_hit_self():
            self.snake.kill_snake()
            reward = 0

        self.num_steps_taken += 1
        if self.verbose and self.num_steps_taken % 1000 == 0:
            print(f"{self.num_steps_taken} steps taken")

        if self.num_steps_taken >= MAX_STEPS:
            if self.verbose:
                print("RUN OUT OF TIME!")
            self.snake.kill_snake()

        if self.render:
            self.render_game()

        return reward

    @staticmethod
    def idx_to_flat(pos: Tuple[int, int], dim: int) -> int:
        """idx of a matrix to its position in the flattened vector."""
        return dim * pos[0] + pos[1]

    @property
    def state(self) -> np.ndarray:

        self.arena[:] = 0

        #  Subtract to normalise to snake head position
        norm_x = self.snake.snake_head[0] - ARENA_WIDTH // 2
        norm_y = self.snake.snake_head[1] - ARENA_HEIGHT // 2
        norm_head = wrap_position(
            (self.snake.snake_head[0] - norm_x, self.snake.snake_head[1] - norm_y)
        )

        relative_food_position = wrap_position(
            (self.food_position[0] - norm_x, self.food_position[1] - norm_y)
        )

        self.arena[relative_food_position] = 10
        for pos in self.snake.snake_body:
            norm_pos = wrap_position((pos[0] - norm_x, pos[1] - norm_y))
            self.arena[norm_pos] = 255
        self.arena[norm_head] = 255

        rotated = np.rot90(self.arena, k=ORIENTATION_2_ROT[self.snake.snake_direction])

        return rotated

    def step(self, action: int) -> Tuple:

        reward = self._step(action)

        if not self.done:
            # reward = self._step(self.opponent_choose_move(state=self.observation))
            # reward = 0
            pass
        # if self.done:
        #     result = "won" if reward > 0 else "lost"
        #     msg = f"You {result} {abs(reward*2)} chips"

        # if self.verbose:
        #     print(msg)
        # if self.render:
        #     self.env.render(
        #         most_recent_move=self.most_recent_move,
        #         win_message=msg,
        #         render_opponent_cards=True,
        #     )

        return self.state, reward, self.done, {}

    def has_hit_boundaries(self) -> bool:
        y_boundary_hit = self.snake_head[1] < 0 or self.snake_head[1] >= ARENA_HEIGHT
        x_boundary_hit = self.snake_head[0] < 0 or self.snake_head[0] >= ARENA_WIDTH
        return y_boundary_hit or x_boundary_hit

    def init_visuals(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((ARENA_WIDTH * BLOCK_SIZE, ARENA_HEIGHT * BLOCK_SIZE))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.screen.fill(WHITE)
        self.score_font = pygame.font.SysFont("comicsansms", 35)

    def print_state(self) -> None:
        # Maybe useful
        arena = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))
        arena[self.food_position] = 1
        for new in self.snake_positions:
            try:
                arena[new] = 2
            except:
                pass
        print(arena)
        print("\n")
        # time.sleep(0.1)

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

        # Draw snake
        for snake_pos in self.snake.snake_body:
            snake_y = (
                ARENA_HEIGHT - snake_pos[1] - 1
            )  # Flip y axis because pygame counts 0,0 as top left
            pygame.draw.rect(
                self.screen,
                BLACK,
                [snake_pos[0] * BLOCK_SIZE, snake_y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE],
            )
        # Flip y axis because pygame counts 0,0 as top left
        snake_y = ARENA_HEIGHT - self.snake.snake_head[1] - 1
        pygame.draw.rect(
            self.screen,
            DARK_GREEN,
            [self.snake.snake_head[0] * BLOCK_SIZE, snake_y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE],
        )

        # draw score
        value = self.score_font.render(f"Your score: {self.snake.snake_length - 2}", True, BLACK)
        self.screen.blit(value, [0, 0])
        pygame.display.update()


def human_player(state) -> Optional[int]:
    """Controls quite janky."""
    time.sleep(0.1)
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            quit()
        # elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
        #     return 3
    is_key_pressed = pygame.key.get_pressed()
    if is_key_pressed[pygame.K_RIGHT]:
        return 2
    elif is_key_pressed[pygame.K_LEFT]:
        return 1
    if is_key_pressed[pygame.K_UP]:
        return 0

    return None
