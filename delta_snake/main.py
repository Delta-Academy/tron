import time
from typing import Any, Dict, Optional

import numpy as np

from check_submission import check_submission
from game_mechanics import (
    State,
    TronEnv,
    choose_move_randomly,
    choose_move_square,
    human_player,
    load_network,
    play_tron,
    save_network,
)
from torch import nn

TEAM_NAME = (
    "Team Nameeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"  # <---- Enter your team name here!
)
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


def train() -> nn.Module:
    """
    TODO: Write this function to train your algorithm.

    Returns:
    """
    raise NotImplementedError("You need to implement this function!")


def choose_move(state: State, neural_network: Optional[nn.Module] = None) -> int:
    """Called during competitive play. It acts greedily given current state of the board and value
    function dictionary. It returns a single move to play.

    Args:
        state:

    Returns:
    """
    raise NotImplementedError("You need to implement this function!")


if __name__ == "__main__":

    # ## Example workflow, feel free to edit this! ###
    # network = train()
    # save_network(network, TEAM_NAME)

    # my_network = load_network(TEAM_NAME)

    # # Code below plays a single game against a random
    # #  opponent, think about how you might want to adapt this to
    # #  test the performance of your algorithm.
    # def choose_move_no_network(state: Any) -> int:
    #     """The arguments in play_game() require functions that only take the state as input.

    #     This converts choose_move() to that format.
    #     """
    #     return choose_move(state, my_network)

    # check_submission(
    #     TEAM_NAME, choose_move_no_network
    # )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    # Play against your bot!
    play_tron(
        your_choose_move=human_player,
        opponent_choose_moves=[choose_move_square],
        game_speed_multiplier=1,
        render=True,
        verbose=True,
    )
