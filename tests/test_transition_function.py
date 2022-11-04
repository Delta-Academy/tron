from copy import deepcopy

import numpy as np

from delta_snake.game_mechanics import TronEnv, transition_function


def test_transition_function_one_step() -> None:

    prev_state, _, _, _ = TronEnv(opponent_choose_moves=[lambda x: x]).reset()
    new_state = transition_function(prev_state, 1, prev_state.player)

    for prev_opponent, new_opponent in zip(prev_state.opponents, new_state.opponents):
        assert prev_opponent.positions == new_opponent.positions
        assert prev_opponent.direction == new_opponent.direction
        assert prev_opponent.alive
        assert new_opponent.alive

    assert prev_state.player.positions != new_state.player.positions

    prev_pos = prev_state.player.head
    new_pos = new_state.player.head

    assert abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1]) == 1


def test_transition_function_death() -> None:

    prev_state, _, _, _ = TronEnv(opponent_choose_moves=[lambda x: x]).reset()
    assert prev_state.player.alive
    new_state = deepcopy(prev_state)

    # Drive the bike to oblivion
    for _ in range(100):
        new_state = transition_function(new_state, 1, new_state.player)
        print(new_state.player.positions)

    assert prev_state.player.alive
    assert not new_state.player.alive


def transition_function_one_step_opponent() -> None:

    prev_state, _, _, _ = TronEnv(opponent_choose_moves=[lambda x: x] * 3).reset()
    opponent_idx = np.random.randint(3)
    new_state = transition_function(prev_state, 1, prev_state.opponents[opponent_idx])

    for idx, (prev_opponent, new_opponent) in enumerate(
        zip(prev_state.opponents, new_state.opponents)
    ):
        if idx == opponent_idx:
            continue
        assert prev_opponent.positions == new_opponent.positions
        assert prev_opponent.direction == new_opponent.direction
        assert prev_opponent.alive
        assert new_opponent.alive

    assert prev_state.player.positions == new_state.player.positions
    assert prev_state.player.alive == new_state.player.alive

    prev_pos = prev_state.opponents[opponent_idx].head
    new_pos = new_state.opponents[opponent_idx].head

    assert abs(new_pos[0] - prev_pos[0]) + abs(new_pos[1] - prev_pos[1]) == 1


def test_transition_function_one_step_opponent() -> None:
    for _ in range(100):
        transition_function_one_step_opponent()


def transition_function_opponent_death() -> None:

    prev_state, _, _, _ = TronEnv(opponent_choose_moves=[lambda x: x] * 3).reset()
    opponent_idx = np.random.randint(3)
    assert prev_state.opponents[opponent_idx].alive

    new_state = transition_function(prev_state, 1, prev_state.opponents[opponent_idx])
    # Drive the bike to oblivion
    for _ in range(100):
        new_state = transition_function(new_state, 1, new_state.opponents[opponent_idx])

    assert prev_state.opponents[opponent_idx].alive
    assert not new_state.opponents[opponent_idx].alive


def test_transition_function_opponent_death() -> None:
    for _ in range(100):
        transition_function_opponent_death()
