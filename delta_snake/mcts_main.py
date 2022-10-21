import cProfile
import math
import random
import time
from typing import Callable, Dict, List

import numpy as np
from tqdm import tqdm

from check_submission import check_submission
from game_mechanics import (
    State,
    TronEnv,
    choose_move_randomly,
    choose_move_square,
    human_player,
    is_terminal,
    play_tron,
    reward_function,
    rules_rollout,
    transition_function,
)
from node import Node, NodeID

TEAM_NAME = "Team mcts"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


class MCTS:
    def __init__(
        self,
        initial_state: State,
        rollout_policy: Callable[[State], int],
        explore_coeff: float,
        verbose: int = 0,
    ):
        self.root_node = Node(initial_state, None)
        self.total_return: Dict[NodeID, float] = {self.root_node.key: 0.0}
        self.N: Dict[NodeID, int] = {self.root_node.key: 0}
        self.tree: Dict[NodeID, Node] = {self.root_node.key: self.root_node}

        self.rollout_policy = rollout_policy
        self.explore_coeff = explore_coeff

        self.verbose = verbose
        self.rollout_len = []

    def do_rollout(self) -> None:
        if self.verbose:
            print("\nNew rollout started from", self.root_node.state)
        path_taken = self._select()
        simulation_node = self._expand(path_taken[-1])
        total_return = self._simulate(simulation_node)
        self._backup(path_taken, total_return)

    def _select(self) -> List[Node]:
        """Selects a node to simulate from, given the current state and tree.

         Returns a list of nodes of the path taken from the root
          to the selected node.

        Write this as exercise 4
        """
        node = self.root_node
        path_taken = [node]
        # If not fully expanded children, select this node
        while not node.is_terminal and all(
            (state.state_id, action) in self.tree for action, state in node.child_states.items()
        ):
            child_nodes = {
                a: self.tree[(state.state_id, a)] for a, state in node.child_states.items()
            }
            node = self._uct_select(self.N[node.key], child_nodes)
            path_taken.append(node)
            if self.verbose:
                print("UCT selected:", node.state)
        return path_taken

    def _expand(self, node: Node) -> Node:
        """Unless the selected node is a terminal state, expand the selected node by adding its
        children nodes to the tree.

        Write this as exercise 5
        """
        assert node.key in self.tree
        if node.is_terminal:
            return node

        child_nodes = []
        for action, state in node.child_states.items():
            child_node = Node(state, action)
            self.tree[child_node.key] = child_node
            self.total_return[child_node.key] = 0
            self.N[child_node.key] = 0
            child_nodes.append(child_node)

        return random.choice(child_nodes) if child_nodes else node

    def _simulate(self, node: Node) -> float:
        """Simulates a full episode to completion from `node`, outputting the total return from the
        episode."""
        n_steps = 0
        if not node.is_terminal:
            action = self.rollout_policy(node.state, node.state.player)
            # Arbitrarily step player first
            state = node.state
            state = transition_function(node.state, action, node.state.player)

            while not is_terminal(state):
                n_steps += 1
                bikes = [state.player] + state.opponents
                for bike in bikes:
                    action = self.rollout_policy(state, bike)
                    if self.verbose:
                        print(f"Simulation take move: {action} on {bike}")
                    state = transition_function(state, action, bike, make_copies=False)
        else:
            state = node.state

        self.rollout_len.append(n_steps)
        # Not convinced the second argument is correct
        total_return = reward_function(state, state.player)

        if self.verbose:
            print("Simulation return:", total_return, state)

        return total_return

    def _backup(self, path_taken: List[Node], total_ep_return: float) -> None:
        """Update the Monte Carlo action-value estimates of all parent nodes in the tree with the
        return from the simulated trajectory.

        Write this as exercise 3
        """
        for node in path_taken:
            self.total_return[node.key] += total_ep_return
            self.N[node.key] += 1
            if self.verbose >= 2:
                print(
                    "Backing up node:",
                    node.state,
                    self.N[node.key],
                    self.total_return[node.key],
                )

    def choose_action(self) -> int:
        """Once we've simulated all the trajectories, we want to select the action at the current
        timestep which maximises the action-value estimate."""
        if self.verbose:
            print("HI")
            # print(
            #     "Q estimates & N:",
            #     {
            #         a: (round(self.Q(state.state_id), 2), self.N[state.state_id])
            #         for a, state in self.root_node.child_states.items()
            #     },
            # )
        return max(
            self.root_node.child_states.keys(),
            key=lambda a: self.N[(self.root_node.child_states[a].state_id, a)],
        )

    def Q(self, node_id: NodeID) -> float:
        return self.total_return[node_id] / (self.N[node_id] + 1e-15)

    def _uct_select(self, N: int, children_nodes: Dict[int, Node]) -> Node:
        max_uct_value = -math.inf
        max_uct_nodes = []
        for child_node in children_nodes.values():
            # q = -child_node.state.player_to_move * self.Q(child_node.key)
            # Could be a sign flip here
            q = self.Q(child_node.key)
            uct_value = q + self.explore_coeff * math.sqrt(
                math.log(N + 1) / (self.N[child_node.key] + 1e-15)
            )
            if self.verbose >= 2:
                print(
                    child_node.state,
                    "UCT value",
                    round(uct_value, 2),
                    "Q",
                    round(q, 2),
                )

            if uct_value > max_uct_value:
                max_uct_value = uct_value
                max_uct_nodes = [child_node]
            elif uct_value == max_uct_value:
                max_uct_nodes.append(child_node)

        len_nodes = len(max_uct_nodes)
        chosen_node = max_uct_nodes[int(random.random() * len_nodes)]
        if self.verbose and self.N[chosen_node.key] == 0:
            print("Exploring!")
        return chosen_node

    def prune_tree(self, action_taken, successor_state: State) -> None:
        """Between steps in the real environment, clear out the old tree."""
        # If it's the terminal state we don't care about pruning the tree
        if is_terminal(successor_state):
            return

        self.root_node = self.tree.get(
            (successor_state.state_id, action_taken), Node(successor_state, action_taken)
        )

        self.N[self.root_node.key] = 0
        self.total_return[self.root_node.key] = 0

        # Build a new tree dictionary
        new_tree: Dict[NodeID, Node] = {self.root_node.key: self.root_node}

        prev_added_nodes: Dict[NodeID, Node] = {self.root_node.key: self.root_node}
        while prev_added_nodes:
            newly_added_nodes: Dict[NodeID, Node] = {}

            for node in prev_added_nodes.values():
                child_nodes = {
                    (state.state_id, action): self.tree[(state.state_id, action)]
                    for action, state in node.child_states.items()
                    if (state.state_id, action) in self.tree
                }
                new_tree |= child_nodes  # type: ignore
                newly_added_nodes |= child_nodes  # type: ignore

            prev_added_nodes = newly_added_nodes

        self.tree = new_tree
        self.total_return = {key: self.total_return[key] for key in self.tree}
        self.N = {key: self.N[key] for key in self.tree}


def choose_move(state: State) -> int:
    """Called during competitive play. It acts greedily given current state of the game. It returns
    a single action to take.

    Args:
        state: a State object containing the positions of yours and your opponents snakes

    Returns:
        The action to take
    """
    mcts = MCTS(state, rollout_policy=rules_rollout, explore_coeff=0.5)
    start_time = time.time()
    time_taken = 0.0
    n_rollout = 0
    while time_taken < 0.1:
        mcts.do_rollout()
        n_rollout += 1
        time_taken = time.time() - start_time

    # print(f"n_rollouts = {n_rollout}")
    # print(f"rollout_len = {(mcts.rollout_len)}")
    # print("\n")

    return mcts.choose_action()


# if __name__ == "__main__":
#     from validation_tests import *

#     simulate_from_terminal_state(MCTS)
#     simulation_from_base(MCTS)
#     backup_win_base(MCTS)
#     backup_lose_state_and_parent(MCTS)
#     select_empty(MCTS)
#     select_exploit(MCTS)
#     expand_terminal(MCTS)
#     expand_root(MCTS)


def profile_me():

    for _ in tqdm(range(100)):
        play_tron(
            your_choose_move=choose_move,
            opponent_choose_move=choose_move,
            game_speed_multiplier=5,
            render=False,
            verbose=False,
        )


if __name__ == "__main__":

    #     # # ## Example workflow, feel free to edit this! ###

    #     check_submission(
    #         TEAM_NAME, choose_move
    #     )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    #     # Play against your bot!

    cProfile.run("profile_me()", "profile.prof")

    # play_tron(
    #     your_choose_move=human_player,
    #     opponent_choose_move=choose_move_square,
    #     game_speed_multiplier=5,
    #     render=True,
    #     verbose=False,
    # )
