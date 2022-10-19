import random
import time
from copy import copy, deepcopy
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gym
import numpy as np
import pygame

ARENA_WIDTH = 15
ARENA_HEIGHT = 15
BLOCK_SIZE = 50

assert ARENA_HEIGHT == ARENA_WIDTH, "current only support square arenas"


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


@dataclass
class State:
    player: "Bike"
    opponents: List["Bike"]

    @property
    def state_id(self) -> str:
        """Unique identifier of state."""
        # Ensure opponents always in the same order by position
        self.opponents.sort(key=lambda x: sum(chain(*x.positions)))
        s = f"player:{self.player.bike_state}"
        for opponent in self.opponents:
            s += f"opponent{opponent.bike_state}"
        return s


def choose_move_randomly(state: State) -> int:
    """This works but the bots die very fast."""
    return int(random.random() * 3) + 1


def choose_move_square(state: State) -> int:
    """This bot happily goes round the edge in a square."""

    orientation = state.player.direction
    head = state.player.head

    if orientation == 0 and head[1] <= 3:
        return 3
    if orientation == 3 and head[0] <= 3:
        return 3
    if orientation == 2 and head[1] >= ARENA_HEIGHT - 3:
        return 3
    if orientation == 1 and head[0] >= ARENA_WIDTH - 3:
        return 3
    return 1


def play_tron(
    your_choose_move: Callable,
    opponent_choose_moves: List[Callable],
    game_speed_multiplier: float = 1.0,
    render: bool = True,
    verbose: bool = False,
) -> float:
    env = TronEnv(
        opponent_choose_moves=opponent_choose_moves,
        verbose=verbose,
        render=render,
        game_speed_multiplier=game_speed_multiplier,
    )

    state, reward, done, _ = env.reset()
    done = False

    return_ = 0
    while not done:
        action = your_choose_move(state)
        state, reward, done, _ = env.step(action)
        return_ += reward

    return return_


def in_arena(pos: Tuple[int, int]) -> bool:
    y_out = pos[1] <= 0 or pos[1] >= ARENA_HEIGHT - 1
    x_out = pos[0] <= 0 or pos[0] >= ARENA_WIDTH - 1
    return not x_out and not y_out


class Action:
    """The action taken by the bike.

    The bike has 3 options:
        1. Go forward
        2. Turn left (and go forward 1 step)
        3. Turn right (and go forward 1 step)
    """

    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


class Orientation:
    """Direction the bike is pointing."""

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
            bike_head_x = random.randint(ARENA_WIDTH // 4, 3 * ARENA_WIDTH // 4)
            bike_head_y = random.randint(ARENA_HEIGHT // 4, 3 * ARENA_HEIGHT // 4)
        else:
            bike_head_x, bike_head_y = starting_position

        self.positions = [(bike_head_x, bike_head_y)]

        for offset in range(1, TAIL_STARTING_LENGTH + 1):
            bike_tail_x = (
                bike_head_x - offset
                if self.direction == Orientation.EAST
                else bike_head_x + offset
                if self.direction == Orientation.WEST
                else bike_head_x
            )
            bike_tail_y = (
                bike_head_y - offset
                if self.direction == Orientation.NORTH
                else bike_head_y + offset
                if self.direction == Orientation.SOUTH
                else bike_head_y
            )
            self.positions.append((bike_tail_x, bike_tail_y))

        self.alive = True
        self.name = name
        self.is_murderer = False

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Bike):
            raise NotImplementedError
        return self.name == other.name

    def __copy__(self) -> "Bike":
        positions = deepcopy(self.positions)
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result.set_positions(positions)
        return result

    def __repr__(self) -> str:
        return f"Bike {self.name}"

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
    def bike_state(self) -> str:
        """Describes fully the state of a bike.

        Can be used as a dictionary key
        """
        return f"position{self.positions}alive{self.alive}direction{self.direction}"

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

        self.positions = list(self.positions)

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

    SCREEN_WIDTH = ARENA_WIDTH * BLOCK_SIZE
    SCREEN_HEIGHT = ARENA_HEIGHT * BLOCK_SIZE

    def __init__(
        self,
        opponent_choose_moves: List[Callable],
        verbose: bool = False,
        render: bool = False,
        game_speed_multiplier: float = 1.0,
        single_player_mode: bool = True,
    ):
        """Number of opponents set by the length of opponent_choose_moves.

        If single_player_mode = True, the env will be done if player1 (the
        player not controlled by opponent_choose_moves) dies. Otherwise
        env continues until a single bike remains.
        """

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

        self.single_player_mode = single_player_mode

    def reset(self) -> Tuple[State, int, bool, Dict]:

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
        self.dead_bikes: List[Bike] = []
        assert len(self.bikes) == len(self.opponent_choose_moves) + 1

        self.color_lookup = dict(zip([bike.name for bike in self.bikes], BIKE_COLORS))
        return self.get_bike_state(self.bikes[0]), 0, False, {}

    @property
    def done(self) -> bool:
        return (
            self.player_dead or len(self.bikes) < 2
            if self.single_player_mode
            else sum(bike.alive for bike in self.bikes) < 2
        )

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

    def head_to_head_collision(self, bike: Bike) -> bool:
        for other_bike in self.bikes:
            if other_bike == bike:
                continue
            if other_bike.head == bike.head:
                other_bike.kill_bike()
                bike.kill_bike()
                return True
        return False

    def has_hit_tails(self, bike_head: Tuple[int, int]) -> bool:
        for other_bike in self.bikes:
            if bike_head in other_bike.body:
                # Did other_bike kill with body, not suicide?
                if bike_head != other_bike.head:
                    other_bike.make_a_murderer()
                return True
        return False

    @staticmethod
    def boundary_elements_mask(matrix: np.ndarray) -> np.ndarray:
        mask = np.ones(matrix.shape, dtype=bool)
        mask[matrix.ndim * (slice(1, -1),)] = False
        return mask

    def get_bike_state(self, bike: Bike) -> State:
        return State(
            player=bike,
            opponents=[other_bike for other_bike in self.bikes if other_bike != bike],
        )

    def step(self, action: int) -> Tuple[State, int, bool, Dict]:

        # Step the player's bike if its not dead (tournament)
        if not self.player_dead:
            self._step(action, self.bikes[0])

        assert len(self.bikes) == len(self.opponent_choose_moves) + 1
        for bike, choose_move in zip(self.bikes[1:], self.opponent_choose_moves):
            if not self.done:
                bike_state = self.get_bike_state(bike)
                action = choose_move(state=bike_state)
                self._step(action, bike)

        idx_alive = []
        for idx, bike in enumerate(self.bikes):
            if not bike.alive:
                if bike.name == "player":
                    self.player_dead = True
                self.dead_bikes.append(bike)
            else:
                idx_alive.append(idx)

        if self.player_dead:
            idx_alive.insert(0, 0)
            # Make sure you don't crash into dead bikes
            self.bikes[0].set_positions([(-100, -100)])

        self.bikes = [self.bikes[idx] for idx in idx_alive]

        self.opponent_choose_moves = [
            self.opponent_choose_moves[idx - 1] for idx in idx_alive if idx != 0
        ]

        if self._render:
            self.render_game()
            time.sleep(1 / self.game_speed_multiplier)

        self.num_steps_taken += 1

        reward = 0
        if self.done:
            winner = self.find_winner()
            if winner is not None and winner == self.player_bike:
                reward += 1

        return self.get_bike_state(self.player_bike), reward, self.done, {}

    def find_winner(self) -> Optional[Bike]:
        assert self.done
        if len(self.bikes) == 0:
            return None
        return self.bikes[np.argmax([bike.length for bike in self.bikes])]

    def init_visuals(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (ARENA_WIDTH * BLOCK_SIZE, ARENA_HEIGHT * BLOCK_SIZE)  # , pygame.FULLSCREEN
        )
        pygame.display.set_caption("bike Game")
        self.clock = pygame.time.Clock()
        self.screen.fill(WHITE)
        self.score_font = pygame.font.SysFont("comicsansms", 35)

    def render_game(self, screen: Optional[pygame.Surface] = None) -> None:

        if screen is None:
            screen = self.screen

        screen.fill(WHITE)

        # Draw boundaries
        pygame.draw.rect(
            screen, BLACK, [1, 1, self.SCREEN_WIDTH - 1, self.SCREEN_HEIGHT - 1], width=BLOCK_SIZE
        )

        for bike in self.bikes:

            color = self.color_lookup[bike.name]

            for bike_pos in bike.body:
                bike_y = (
                    ARENA_HEIGHT - bike_pos[1] - 1
                )  # Flip y axis because pygame counts 0,0 as top left
                pygame.draw.rect(
                    screen,
                    color,
                    [bike_pos[0] * BLOCK_SIZE, bike_y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE],
                )
            # Flip y axis because pygame counts 0,0 as top left
            bike_y = ARENA_HEIGHT - bike.head[1] - 1
            pygame.draw.rect(
                screen,
                BLACK,
                [
                    bike.head[0] * BLOCK_SIZE,
                    bike_y * BLOCK_SIZE,
                    BLOCK_SIZE,
                    BLOCK_SIZE,
                ],
            )

        # This may cause flashing in the tournament
        pygame.display.update()


def human_player(*args: Any, **kwargs: Any) -> int:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                return 3
            if event.key == pygame.K_LEFT:
                return 2
    return 1


# Functional reimplementation of some above logic
def transition_function(state: State, action: int, bike_move: Bike) -> State:

    state = copy(state)
    bike_move = copy(bike_move)
    state.player = copy(state.player)
    state.opponents = [copy(bike) for bike in state.opponents]

    bike_move.take_action(action)

    if has_hit_tails(bike_move.head, state) or bike_move.has_hit_boundaries():
        bike_move.kill_bike()

    state = head_to_head_collision(bike_move, state)

    # Put the newly moved bike back in the state copy
    if state.player == bike_move:
        state.player = bike_move
    else:
        for idx, bike in enumerate(state.opponents):
            if bike == bike_move:
                state.opponents[idx] = bike_move

    return state


def reward_function(successor_state: State, bike_move: Bike) -> int:
    bikes = [successor_state.player] + successor_state.opponents
    return int(all(not bike.alive for bike in bikes if bike != bike_move) and bike_move.alive)


def has_hit_tails(bike_head: Tuple[int, int], state: State) -> bool:
    bikes = [state.player] + state.opponents
    return any(bike_head in bike.body for bike in bikes)


def head_to_head_collision(bike_move: Bike, state: State) -> State:
    """Kill bikes involved in head to head collisions."""

    bikes = [state.player] + state.opponents

    for other_bike in bikes:
        if other_bike == bike_move:
            continue
        if other_bike.head == bike_move.head:
            other_bike.kill_bike()
            bike_move.kill_bike()

    return state


def is_terminal(successor_state: State) -> bool:
    bikes = [successor_state.player] + successor_state.opponents
    return not successor_state.player.alive or sum(bike.alive for bike in bikes) < 2


def get_possible_actions():
    return [1, 2, 3]
