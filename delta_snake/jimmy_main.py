import cProfile
import random
import time
from typing import Any, Dict

import numpy as np
import pygame
from matplotlib import pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
from torch import nn
from tqdm import tqdm

from check_submission import check_submission
from game_mechanics import (
    ARENA_HEIGHT,
    ARENA_WIDTH,
    HERE,
    TronEnv,
    choose_move_randomly,
    choose_move_square,
    human_player,
    load_network,
    save_network,
    transition_function,
)

TEAM_NAME = "Team jimmy"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> None:
        """Called every step()"""
        self.rewards.append(safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]))


def smooth_trace(trace: np.ndarray, one_sided_window_size: int = 3) -> np.ndarray:
    """Smooths a trace by averaging over a window of size one_sided_window_size."""
    window_size = int(2 * one_sided_window_size + 1)
    trace[one_sided_window_size:-one_sided_window_size] = (
        np.convolve(trace, np.ones(window_size), mode="valid") / window_size
    )
    return trace


def train() -> nn.Module:
    """
    TODO: Write this function to train your algorithm.

    Returns:
    """
    env = TronEnv([choose_move_square] * 2, verbose=False, render=False)
    env.reset()

    # model = PPO("CnnPolicy", env, verbose=2, ent_coef=0.01)
    # model = PPO("MlpPolicy", env, verbose=2, ent_coef=0.01)

    model = PPO.load(str(HERE / "howdy_model"))
    model.set_env(env)

    callback = CustomCallback()
    model.learn(
        total_timesteps=1_000_000,
        callback=callback,
    )
    model.save(str(HERE / "howdy_model"))
    plt.plot(smooth_trace(callback.rewards, 1_000))
    plt.title(f"mean: {np.nanmean(callback.rewards)}, std: {np.nanstd(callback.rewards)}")
    plt.show()


def choose_move(state: np.ndarray, network: Any) -> int:
    """Called during competitive play. It acts greedily given current state of the board and value
    function dictionary. It returns a single move to play.

    Args:
        state:

    Returns:
    """
    raise NotImplementedError("You need to implement this function!")


def test():

    env = TronEnv([choose_move_square] * 2, verbose=False, render=True)

    state, reward, done, _ = env.reset()
    done = False
    model = PPO.load(str(HERE / "howdy_model"))
    rewards = 0

    while not done:

        state, reward, done, _ = env.step(choose_move_square(state))
        rewards += reward
        time.sleep(0.1)

    print(f"score = {rewards}")


def n_games():

    env = TronEnv([choose_move_square] * 2, verbose=False, render=False)
    n = 3000
    n_steps_list = []

    for _ in tqdm(range(n)):

        # state = env.reset()
        state, reward, done, _ = env.reset()
        done = False
        n_steps = 0

        while not done:

            action = choose_move_square(state)
            state, reward, done, _ = env.step(action)
            n_steps += 1
        n_steps_list.append(n_steps)

    plt.show()
    print(f"n steps = {np.mean(n_steps_list)}")


if __name__ == "__main__":

    # cProfile.run("n_games()", "profile.prof")
    # n_games()

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
    #

    # model = PPO.load(str(HERE / "howdy_model"))
    # def choose_move_model(state) -> int:
    #     return model.predict(state.copy())[0] + 1

    # play_snake(
    #     your_choose_move=choose_move_model,
    #     opponent_choose_moves=[choose_move_model] * 3,
    #     game_speed_multiplier=0.1,
    #     render=True,
    #     verbose=False,
    # )

    # play_snake(
    #     your_choose_move=human_player,
    #     # opponent_choose_moves=[choose_move_model] * 1,
    #     opponent_choose_moves=[lambda x: None] * 1,
    #     game_speed_multiplier=5,
    #     render=True,
    #     verbose=False,
    # )
    # train()
    test()

    # state, _, _, _ = TronEnv(opponent_choose_moves=[lambda x: x]).reset()
    # transition_function(state, 0)
