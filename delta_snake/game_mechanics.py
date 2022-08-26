import random
import time
from copy import copy, deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gym
import numpy as np
import pygame
import torch
from torch import nn

ARENA_WIDTH = 51
ARENA_HEIGHT = 51
BLOCK_SIZE = 10

assert ARENA_HEIGHT == ARENA_WIDTH, "current only support square arenas"


SCREEN_WIDTH = ARENA_WIDTH * BLOCK_SIZE
SCREEN_HEIGHT = ARENA_HEIGHT * BLOCK_SIZE

TAIL_STARTING_LENGTH = 1

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# will break with >6 teams
BIKE_COLORS = [
    (237, 0, 3),
    (53, 0, 255),
    (1, 254, 1),
    (255, 134, 0),
    (255, 254, 55),
    (140, 0, 252),
]

HERE = Path(__file__).parent.resolve()


def choose_move_randomly(state: Dict) -> int:
    """This works but the bots die very fast."""
    return int(random.random() * 3) + 1


def choose_move_square(state: Dict) -> int:
    """This bot happily goes round the edge in a square."""

    orientation = state["player"].direction
    head = state["player"].head

    if orientation == 0 and head[1] <= 3:
        return 3
    if orientation == 3 and head[0] <= 3:
        return 3
    if orientation == 2 and head[1] >= ARENA_HEIGHT - 3:
        return 3
    if orientation == 1 and head[0] >= ARENA_WIDTH - 3:
        return 3
    return 1


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


def play_tron(
    your_choose_move: Callable,
    opponent_choose_moves: List[Callable],
    game_speed_multiplier: float = 1.0,
    render=True,
    verbose=False,
) -> float:
    env = TronEnv(opponent_choose_moves=opponent_choose_moves, verbose=verbose, render=render)

    state, reward, done, _ = env.reset()
    done = False

    return_ = 0
    while not done:
        action = your_choose_move(state)
        state, reward, done, _ = env.step(action)
        time.sleep(1 / game_speed_multiplier)
        return_ += reward

    return return_


def wrap_position(pos: Tuple[int, int]) -> Tuple[int, int]:
    # wrapping taken out
    return pos


def in_arena(pos: Tuple[int, int]) -> bool:
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


class Bike:
    def __init__(
        self, name: str = "snek", starting_position: Optional[Tuple[int, int]] = None
    ) -> None:

        self.direction = random.choice(
            [Orientation.EAST, Orientation.WEST, Orientation.NORTH, Orientation.SOUTH]
        )

        if starting_position is None:
            snake_head_x = random.randint(ARENA_WIDTH // 4, 3 * ARENA_WIDTH // 4)
            snake_head_y = random.randint(ARENA_HEIGHT // 4, 3 * ARENA_HEIGHT // 4)
        else:
            snake_head_x, snake_head_y = starting_position

        self.positions = [(snake_head_x, snake_head_y)]

        for offset in range(1, TAIL_STARTING_LENGTH + 1):
            snake_tail_x = (
                snake_head_x - offset
                if self.direction == Orientation.EAST
                else snake_head_x + offset
                if self.direction == Orientation.WEST
                else snake_head_x
            )
            snake_tail_y = (
                snake_head_y - offset
                if self.direction == Orientation.NORTH
                else snake_head_y + offset
                if self.direction == Orientation.SOUTH
                else snake_head_y
            )
            self.positions.append((snake_tail_x, snake_tail_y))

        self.alive = True
        self.name = name
        self.is_murderer = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bike):
            raise NotImplementedError
        return self.name == other.name

    def __copy__(self) -> "Bike":
        print("OOOH i be copying")
        positions = deepcopy(self.positions)
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.set_positions(positions)
        return result

    def set_positions(self, positions: List[Tuple[int, int]]) -> None:
        self.positions = positions

    def has_hit_boundaries(self) -> bool:
        return not in_arena(self.head)

    def kill_bike(self) -> None:
        self.alive = False

    def has_hit_self(self) -> bool:
        return self.head in self.body

    @property
    def length(self) -> int:
        return len(self.positions)

    @property
    def head(self) -> Tuple[int, int]:
        return self.positions[0]

    @property
    def body(self) -> List[Tuple[int, int]]:
        return self.positions[1:]

    def take_action(self, action: int) -> None:

        if action == 2:
            new_orientation = (self.direction + 1) % 4
        elif action == 3:
            new_orientation = (self.direction - 1) % 4
        else:
            new_orientation = self.direction

        x, y = self.head
        if new_orientation % 2 == 0:
            # South is 0 (y -= 1), North is 2 (y += 1)
            y += new_orientation - 1
        else:
            # East is 1 (x += 1), West is 3 (x -= 1)
            x += 2 - new_orientation

        # Update position and orientation
        if action is not None:
            self.positions.insert(0, (x, y))
            self.direction = new_orientation

        self.positions = [wrap_position(pos) for pos in self.positions]

    def remove_tail_end(self) -> None:
        del self.positions[-1]

    def make_a_murderer(self) -> None:
        self.is_murderer = True

    def make_innocent(self) -> None:
        self.is_murderer = False


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


class TronEnv(gym.Env):
    def __init__(
        self,
        opponent_choose_moves: List[Callable],
        verbose: bool = False,
        render: bool = False,
        game_speed_multiplier: int = 1,
    ):
        """Number of opponents set by the length of opponent_choose_moves."""

        self.choose_move_store = deepcopy(opponent_choose_moves)

        self.opponent_choose_moves = opponent_choose_moves
        self.n_foods = self.n_opponents = len(self.opponent_choose_moves)
        self._render = render
        self.verbose = verbose
        self.game_speed_multiplier = game_speed_multiplier

        self.starting_positions = get_starting_positions()
        self.score = 0
        if self._render:
            self.init_visuals()

    def reset(self) -> Tuple[Dict, int, bool, Dict]:
        self.opponent_choose_moves = self.choose_move_store
        self.player_dead = False
        self.num_steps_taken = 0

        random.shuffle(self.starting_positions)

        self.player_bike = Bike(name="player", starting_position=self.starting_positions[0])
        self.bikes = [self.player_bike]
        self.bikes += [
            Bike(name=f"opponent_{idx}", starting_position=self.starting_positions[idx + 1])
            for idx in range(len(self.opponent_choose_moves))
        ]
        self.dead_snakes: List[Bike] = []
        assert len(self.bikes) == len(self.opponent_choose_moves) + 1

        # Aint no food no more
        self.generate_food()

        self.color_lookup = {
            name: color for name, color in zip([snake.name for snake in self.bikes], BIKE_COLORS)
        }

        return self.get_snake_state(self.bikes[0]), 0, False, {}

    @property
    def done(self) -> bool:
        return not any([snake.name == "player" for snake in self.bikes]) or len(self.bikes) < 2

    def generate_food(self, eaten_pos: Optional[Tuple[int, int]] = None) -> None:
        """pass eaten_pos if a food has just been eaten."""
        # No food no more
        return

    def _step(self, action: int, bike: Bike) -> None:

        bike.take_action(action)

        if action not in [Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT]:
            raise ValueError(f"Invalid action: {action}")

        if self.has_hit_tails(bike.head) or bike.has_hit_boundaries():
            bike.kill_bike()
        self.head_to_head_collision(bike)

        if self.verbose and self.num_steps_taken % 100 == 0:
            print(f"{self.num_steps_taken} steps taken")

        return

    def head_to_head_collision(self, snake: Bike) -> bool:
        for other_snake in self.bikes:
            if other_snake == snake:
                continue
            if other_snake.head == snake.head:
                other_snake.kill_bike()
                snake.kill_bike()
                return True
        return False

    def has_hit_tails(self, snake_head: Tuple[int, int]) -> bool:
        for other_snake in self.bikes:
            if snake_head in other_snake.body:
                # Did other_snake kill with body, not suicide?
                if snake_head != other_snake.head:
                    other_snake.make_a_murderer()
                return True
        return False

    @staticmethod
    def boundary_elements_mask(matrix: np.ndarray) -> np.ndarray:
        mask = np.ones(matrix.shape, dtype=bool)
        mask[matrix.ndim * (slice(1, -1),)] = False
        return mask

    def get_snake_state(self, snake: Bike) -> Dict:
        state: Dict[str, Any] = {}
        state["player"] = snake
        state["opponents"] = [other_snake for other_snake in self.bikes if other_snake != snake]

        return state

    def step(self, action: int) -> Tuple:

        # Step player's snake
        self._step(action, self.bikes[0])

        assert len(self.bikes) == len(self.opponent_choose_moves) + 1
        for snake, choose_move in zip(self.bikes[1:], self.opponent_choose_moves):
            if not self.done:
                snake_state = self.get_snake_state(snake)
                action = choose_move(snake_state)
                self._step(action, snake)

        # Remove me
        assert self.bikes[0] == self.player_bike
        assert self.player_bike.name == "player"

        idx_alive = []
        for idx, snake in enumerate(self.bikes):
            if not snake.alive:
                if snake.name == "player":
                    self.player_dead = True
                self.dead_snakes.append(snake)
            else:
                idx_alive.append(idx)

        self.bikes = [self.bikes[idx] for idx in idx_alive]

        self.opponent_choose_moves = [
            self.opponent_choose_moves[idx - 1] for idx in idx_alive if idx != 0
        ]

        if self._render:
            self.render_game()

        self.num_steps_taken += 1

        reward = 0
        if self.done:
            winner = self.find_winner()
            if winner is not None and winner == self.player_bike:
                reward += 1

        return self.get_snake_state(self.player_bike), reward, self.done, {}

    def find_winner(self) -> Optional[Bike]:
        assert self.done
        if len(self.bikes) == 0:
            return None
        return self.bikes[np.argmax([snake.length for snake in self.bikes])]

    def init_visuals(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (ARENA_WIDTH * BLOCK_SIZE, ARENA_HEIGHT * BLOCK_SIZE), pygame.FULLSCREEN
        )
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()
        self.screen.fill(WHITE)
        self.score_font = pygame.font.SysFont("comicsansms", 35)

    def render_game(self) -> None:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit()

        self.screen.fill(WHITE)
        # Draw boundaries
        pygame.draw.rect(
            self.screen, BLACK, [1, 1, SCREEN_WIDTH - 1, SCREEN_HEIGHT - 1], width=BLOCK_SIZE
        )

        # Draw snake
        for snake in self.bikes:
            color = self.color_lookup[snake.name]

            for snake_pos in snake.body:
                snake_y = (
                    ARENA_HEIGHT - snake_pos[1] - 1
                )  # Flip y axis because pygame counts 0,0 as top left
                pygame.draw.rect(
                    self.screen,
                    color,
                    [snake_pos[0] * BLOCK_SIZE, snake_y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE],
                )
            # Flip y axis because pygame counts 0,0 as top left
            snake_y = ARENA_HEIGHT - snake.head[1] - 1
            pygame.draw.rect(
                self.screen,
                BLACK,
                [
                    snake.head[0] * BLOCK_SIZE,
                    snake_y * BLOCK_SIZE,
                    BLOCK_SIZE,
                    BLOCK_SIZE,
                ],
            )

        pygame.display.update()


def human_player(state) -> Optional[int]:
    """Controls quite janky."""
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            quit()
    is_key_pressed = pygame.key.get_pressed()
    if is_key_pressed[pygame.K_RIGHT]:
        return 3
    elif is_key_pressed[pygame.K_LEFT]:
        return 2
    if is_key_pressed[pygame.K_UP]:
        return 1
    return 1


# Functional reimplementation of some above logic
def transition_function(state: Dict, action: int, bike_move: Bike) -> Dict:

    state = state.copy()
    bike_move = copy(bike_move)
    state["player"] = copy(state["player"])
    state["opponents"] = [copy(bike) for bike in state["opponents"]]

    bike_move.take_action(action)

    if has_hit_tails(bike_move.head, state) or bike_move.has_hit_boundaries():
        bike_move.kill_bike()

    state = head_to_head_collision(bike_move, state)

    # If have the same name put the newly moved bike back in the state
    if state["player"] == bike_move:
        state["player"] = bike_move
    else:
        for idx, bike in enumerate(state["opponents"]):
            if bike == bike_move:
                state["opponents"][idx] = bike_move

    return state


def has_hit_tails(snake_head: Tuple[int, int], state: Dict) -> bool:
    bikes = [state["player"]] + state["opponents"]
    return any([snake_head in bike.body for bike in bikes])


def head_to_head_collision(bike_move: Bike, state: Dict) -> Dict:
    """Kill bikes involved in head to head collisions."""

    bikes = [state["player"]] + state["opponents"]

    for other_bike in bikes:
        if other_bike == bike_move:
            continue
        if other_bike.head == bike_move.head:
            other_bike.kill_bike()
            bike_move.kill_bike()

    return state
