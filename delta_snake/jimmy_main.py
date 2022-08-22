import time
from typing import Any, Dict

import numpy as np
from torch import nn
from tqdm import tqdm

from check_submission import check_submission
from game_mechanics import SnakeEnv, choose_move_randomly, load_network, play_snake, save_network

TEAM_NAME = "Team jimmy"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


def train() -> nn.Module:
    """
    TODO: Write this function to train your algorithm.

    Returns:
    """

    raise NotImplementedError("You need to implement this function!")


def choose_move(state: np.ndarray, network: Any) -> int:
    """Called during competitive play. It acts greedily given current state of the board and value
    function dictionary. It returns a single move to play.

    Args:
        state:

    Returns:
    """
    raise NotImplementedError("You need to implement this function!")


def test():

    env = SnakeEnv(lambda x: x, verbose=False, render=False)
    t1 = time.time()

    for _ in tqdm(range(1000)):
        state, reward, done, _ = env.reset()
        while not done:
            state, reward, done, _ = env.step(np.random.randint(1, 4))
    t2 = time.time()
    print(t2 - t1)


if __name__ == "__main__":
    test()

    # ## Example workflow, feel free to edit this! ###
    # network = train()
    # save_network(network, TEAM_NAME)

    # check_submission(
    #     TEAM_NAME
    # )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    # my_value_fn = load_network(TEAM_NAME)

    # # Code below plays a single game against a random
    # #  opponent, think about how you might want to adapt this to
    # #  test the performance of your algorithm.
    # def choose_move_no_network(state: Any) -> int:
    #     """The arguments in play_game() require functions that only take the state as input.

    #     This converts choose_move() to that format.
    #     """
    #     return choose_move(state, my_value_fn)

    # play_snake(
    #     your_choose_move=choose_move_no_network,
    #     opponent_choose_move=choose_move_randomly,
    #     game_speed_multiplier=1,
    #     render=True,
    #     verbose=False,
    # )
