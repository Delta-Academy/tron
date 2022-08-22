import enum
from pathlib import Path
import random
from typing import Callable, List, Tuple

import pygame
import torch
from torch import nn

ARENA_WIDTH = 80
ARENA_HEIGHT = 60
BLOCK_SIZE = 10

SCREEN_WIDTH = ARENA_WIDTH * BLOCK_SIZE
SCREEN_HEIGHT = ARENA_HEIGHT * BLOCK_SIZE

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
    model = torch.load(net_path)
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
            load_network(team_name)
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


class Action(enum.Enum):
    """The action taken by the snake.

    The snake has 3 options:
        1. Go forward
        2. Turn left (and go forward 1 step)
        3. Turn right (and go forward 1 step)
    """

    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


class Orientation(enum.Enum):
    """Direction the snake is pointing."""

    SOUTH = 0  # negative y-direction
    EAST = 1  # positive x-direction
    NORTH = 2  # positive y-direction
    WEST = 3  # negative x-direction


class Snake:
    def __init__(self):
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
        self.food_position = (
            random.randint(0, ARENA_WIDTH - 1),
            random.randint(0, ARENA_HEIGHT - 1),
        )
        self.snake_alive = True
        self.num_steps_taken = 0

    def generate_food(self):
        possible_food_positions = [
            (x, y)
            for x in range(ARENA_WIDTH)
            for y in range(ARENA_HEIGHT)
            if (x, y) not in self.snake_positions
        ]
        self.food_position = random.choice(possible_food_positions)

    def update(
        self,
        choose_action: Callable[[List[Tuple[int, int]], Orientation, Tuple[int, int]], Action],
    ) -> None:
        try:
            action = choose_action(self.snake_positions, self.snake_direction, self.food_position)
            if action not in [Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT]:
                raise ValueError(f"Invalid action: {action}")
        except Exception as e:
            print(e)
            action = random.choice([Action.MOVE_FORWARD, Action.TURN_LEFT, Action.TURN_RIGHT])
        if action.value == Action.MOVE_FORWARD.value:
            new_orientation = self.snake_direction.value
        elif action.value == Action.TURN_LEFT.value:
            new_orientation = (self.snake_direction.value + 1) % 4
        else:
            new_orientation = (self.snake_direction.value - 1) % 4

        x, y = self.snake_head
        if new_orientation % 2 == 0:
            # South is 0 (y -= 1), North is 2 (y += 1)
            y += new_orientation - 1
        else:
            # East is 1 (x += 1), West is 3 (x -= 1)
            x += 2 - new_orientation

        # Update snake position and orientation
        self.snake_positions.insert(0, (x, y))
        self.snake_direction = Orientation(new_orientation)

        # If snake eats apple, don't remove the end of the tail
        if self.snake_head != self.food_position:
            del self.snake_positions[-1]
        else:
            # Generate new apple
            self.generate_food()

        # If you hit more snake or boundary, game over
        if self.has_hit_boundaries() or self.has_hit_self():
            self.snake_alive = False

        self.num_steps_taken += 1
        if self.num_steps_taken % 1000 == 0:
            print(f"{self.num_steps_taken} steps taken")

        if self.num_steps_taken >= 10000:
            print("RUN OUT OF TIME!")
            self.snake_alive = False

    def has_hit_boundaries(self) -> bool:
        y_boundary_hit = self.snake_head[1] < 0 or self.snake_head[1] >= ARENA_HEIGHT
        x_boundary_hit = self.snake_head[0] < 0 or self.snake_head[0] >= ARENA_WIDTH
        return y_boundary_hit or x_boundary_hit

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


# def play_snake(
#     choose_move: Callable[[List[Tuple[int, int]], Orientation, Tuple[int, int]], Action],
#     game_speed_multiplier: float = 1.0,
# ):
#     """Play a game of Pong where both paddles are controlled by `move_paddle()`."""
#     game = Snake()
#     pygame.init()
#     screen = pygame.display.set_mode((ARENA_WIDTH * BLOCK_SIZE, ARENA_HEIGHT * BLOCK_SIZE))
#     pygame.display.set_caption("Snake Game")
#     clock = pygame.time.Clock()
#     score_font = pygame.font.SysFont("comicsansms", 35)

#     game_quit = False
#     while game.snake_alive and not game_quit:
#         game.update(choose_move)
#         draw_game(screen, game, score_font)
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 game_quit = True
#                 print("Game quit")
#         clock.tick(round(30 * game_speed_multiplier))
#     pygame.quit()
#     print(game.snake_length - 2)


def draw_game(screen, game: Snake, score_font):
    # White background
    screen.fill(WHITE)

    # Draw apple
    food_screen_x, food_screen_y = game.food_position
    food_screen_y = (
        ARENA_HEIGHT - food_screen_y - 1
    )  # Flip y axis because pygame counts 0,0 as top left
    pygame.draw.rect(
        screen,
        GREEN,
        [food_screen_x * BLOCK_SIZE, food_screen_y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE],
    )

    # Draw snake
    for snake_pos in game.snake_body:
        snake_y = (
            ARENA_HEIGHT - snake_pos[1] - 1
        )  # Flip y axis because pygame counts 0,0 as top left
        pygame.draw.rect(
            screen, BLACK, [snake_pos[0] * BLOCK_SIZE, snake_y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE]
        )
    # Flip y axis because pygame counts 0,0 as top left
    snake_y = ARENA_HEIGHT - game.snake_head[1] - 1
    pygame.draw.rect(
        screen,
        DARK_GREEN,
        [game.snake_head[0] * BLOCK_SIZE, snake_y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE],
    )

    # draw score
    value = score_font.render(f"Your score: {game.snake_length - 2}", True, BLACK)
    screen.blit(value, [0, 0])

    pygame.display.update()
