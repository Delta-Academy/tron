from copy import deepcopy

import numpy as np

from delta_snake.game_mechanics import TronEnv, transition_function


def test_transition_function_one_step() -> None:

    starting_state, _, _, _ = TronEnv(opponent_choose_move=lambda x: x).reset()
    not_moved_state = transition_function(starting_state, 1, make_copies=True)

    prev_opponent = starting_state.opponent
    new_opponent = not_moved_state.opponent

    prev_player = starting_state.player
    new_player = not_moved_state.player

    assert prev_opponent.positions == new_opponent.positions
    assert prev_opponent.direction == new_opponent.direction
    assert prev_player.positions == new_player.positions
    assert new_player.direction == new_player.direction
    assert prev_opponent.alive
    assert new_opponent.alive
    assert prev_player.alive
    assert prev_opponent.alive
    assert starting_state.player.positions == not_moved_state.player.positions

    moved_state = transition_function(not_moved_state, 1, make_copies=True)

    assert moved_state.player.positions != not_moved_state.player.positions
    assert moved_state.opponent.positions != starting_state.opponent.positions

    prev_pos = not_moved_state.player.head
    new_pos = moved_state.player.head

    assert abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1]) == 1

    prev_pos = not_moved_state.opponent.head
    new_pos = moved_state.opponent.head

    assert abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1]) == 1


def test_transition_function_death() -> None:

    starting_state, _, _, _ = TronEnv(opponent_choose_move=lambda x: x).reset()
    assert starting_state.player.alive

    new_state = deepcopy(starting_state)

    # Drive the bikes to oblivion
    for _ in range(100):
        new_state = transition_function(new_state, 1, make_copies=True)

    assert starting_state.player.alive
    assert not new_state.player.alive
    assert not new_state.opponent.alive
