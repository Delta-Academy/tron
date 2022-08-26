import time
from pathlib import Path
from typing import Dict

import delta_utils.check_submission as checker
from torch import nn

from game_mechanics import TronEnv


def check_submission(team_name: str) -> None:

    example_state, _, _, _ = TronEnv(opponent_choose_moves=[lambda x: x]).reset()
    expected_choose_move_return_type = int
    game_mechanics_expected_hash = (
        "0ad1277d2c92e85b1e5d2db98595765d032c350e417a3b10ef8755cbb6dd5404"
    )
    expected_pkl_output_type = (None,)
    pkl_file = None

    t1 = time.time()

    return checker.check_submission(
        example_state=example_state,
        expected_choose_move_return_type=expected_choose_move_return_type,
        expected_pkl_type=expected_pkl_output_type,
        pkl_file=None,
        pkl_checker_function=None,
        game_mechanics_hash=game_mechanics_expected_hash,
        current_folder=Path(__file__).parent.resolve(),
    )
