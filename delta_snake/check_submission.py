from pathlib import Path
from typing import Dict

import delta_utils.check_submission as checker
from torch import nn

from game_mechanics import SnakeEnv


def check_submission(team_name: str) -> None:

    example_state, _, _, _ = SnakeEnv(opponent_choose_moves=[lambda x: x]).reset()
    expected_choose_move_return_type = int
    game_mechanics_expected_hash = (
        "6367cbb701bd6fa04635a1ea85cd9eb5d173ab653df3559f5cacc083b8d42cf2",
    )
    expected_pkl_output_type = (None,)
    pkl_file = None

    return checker.check_submission(
        example_state=example_state,
        expected_choose_move_return_type=expected_choose_move_return_type,
        expected_pkl_type=expected_pkl_output_type,
        pkl_file=None,
        pkl_checker_function=None,
        game_mechanics_hash=game_mechanics_expected_hash,
        current_folder=Path(__file__).parent.resolve(),
    )
