from check_submission import check_submission
from game_mechanics import (
    State,
    TronEnv,
    choose_move_randomly,
    choose_move_square,
    human_player,
    play_tron,
)

TEAM_NAME = "Team Name"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


def choose_move(state: State) -> int:
    """Called during competitive play. It acts greedily given current state of the game. It returns
    a single action to take.

    Args:
        state: a State object containing the positions of yours and your opponents snakes

    Returns:
        The action to take
    """
    return choose_move_randomly(state)


if __name__ == "__main__":

    # Example workflow, feel free to edit this!

    # Make sure this passes, or your solution will not work in the tournament!!
    check_submission(choose_move)

    # Play against your bot!
    play_tron(
        your_choose_move=human_player,
        opponent_choose_move=choose_move,
        game_speed_multiplier=5,
        render=True,
        verbose=True,
    )
