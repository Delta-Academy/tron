import cProfile
import math
import random
import time
from typing import Callable, Dict, List, Tuple

from game_mechanics import (
    Bike,
    State,
    get_possible_actions,
    human_player,
    in_arena,
    is_terminal,
    play_tron,
    reward_function,
    transition_function,
)
from node import Node, NodeID
from tqdm import tqdm

TEAM_NAME = "Henry"  # <---- Enter your team name here!
assert TEAM_NAME != "Team Name", "Please change your TEAM_NAME!"


def player_sign(state: State):
    return 2 * (int(state.bike_to_move == state.player) - 0.5)


class MCTS:
    def __init__(
        self,
        initial_state: State,
        rollout_policy: Callable[[State], int],
        explore_coeff: float,
        verbose: int = 0,
    ):
        self.root_node = Node(initial_state)
        self.total_return: Dict[NodeID, float] = {self.root_node.key: 0.0}
        self.N: Dict[NodeID, int] = {self.root_node.key: 0}
        self.tree: Dict[NodeID, Node] = {self.root_node.key: self.root_node}

        self.rollout_policy = rollout_policy
        self.explore_coeff = explore_coeff

        self.verbose = verbose
        self.rollout_len: List[int] = []
        self.select_len: List[int] = []

    def do_rollout(self) -> None:
        if self.verbose:
            print("\nNew rollout started from", self.root_node.state.state_id)
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
            state.state_id in self.tree for state in node.child_states.values()
        ):
            child_nodes = {a: self.tree[state.state_id] for a, state in node.child_states.items()}
            node = self._uct_select(self.N[node.key], child_nodes)
            path_taken.append(node)
            if self.verbose:
                print("UCT selected:", node.state.state_id)
        self.select_len.append(len(path_taken))
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
            child_node = Node(state)
            self.tree[child_node.key] = child_node
            self.total_return[child_node.key] = 0
            self.N[child_node.key] = 0
            child_nodes.append(child_node)

        return random.choice(child_nodes) if child_nodes else node

    def _simulate(self, node: Node) -> float:
        """Simulates a full episode to completion from `node`, outputting the total return from the
        episode."""
        n_steps = 0
        state = node.state.copy()
        while not is_terminal(state):
            n_steps += 1
            for bike in state.bikes:
                action = self.rollout_policy(state)
                if self.verbose:
                    print(f"Simulation take move: {action} on {bike} with state: {state.state_id}")
                state = transition_function(state, action, make_copies=False)

        self.rollout_len.append(n_steps)
        total_return = reward_function(state)

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
            print(
                "Q estimates & N:",
                {
                    a: (round(self.Q(state.state_id), 2), self.N[state.state_id])
                    for a, state in self.root_node.child_states.items()
                },
            )
        return max(
            self.root_node.child_states.keys(),
            key=lambda a: self.N[self.root_node.child_states[a].state_id],
        )

    def Q(self, node_id: NodeID) -> float:
        return self.total_return[node_id] / (self.N[node_id] + 1e-15)

    def _uct_select(self, N: int, children_nodes: Dict[int, Node]) -> Node:
        max_uct_value = -math.inf
        max_uct_nodes = []
        for child_node in children_nodes.values():
            q = -player_sign(child_node.state) * self.Q(child_node.key)
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

    # def prune_tree(self, action_taken, successor_state: State) -> None:
    #     """Between steps in the real environment, clear out the old tree."""
    #     # If it's the terminal state we don't care about pruning the tree
    #     if is_terminal(successor_state):
    #         return
    #
    #     self.root_node = self.tree.get(
    #         (successor_state.state_id, action_taken), Node(successor_state, action_taken)
    #     )
    #
    #     self.N[self.root_node.key] = 0
    #     self.total_return[self.root_node.key] = 0
    #
    #     # Build a new tree dictionary
    #     new_tree: Dict[NodeID, Node] = {self.root_node.key: self.root_node}
    #
    #     prev_added_nodes: Dict[NodeID, Node] = {self.root_node.key: self.root_node}
    #     while prev_added_nodes:
    #         newly_added_nodes: Dict[NodeID, Node] = {}
    #
    #         for node in prev_added_nodes.values():
    #             child_nodes = {
    #                 (state.state_id, action): self.tree[(state.state_id, action)]
    #                 for action, state in node.child_states.items()
    #                 if (state.state_id, action) in self.tree
    #             }
    #             new_tree |= child_nodes  # type: ignore
    #             newly_added_nodes |= child_nodes  # type: ignore
    #
    #         prev_added_nodes = newly_added_nodes
    #
    #     self.tree = new_tree
    #     self.total_return = {key: self.total_return[key] for key in self.tree}
    #     self.N = {key: self.N[key] for key in self.tree}


def choose_move(state: State) -> int:
    """Called during competitive play. It acts greedily given current state of the game. It returns
    a single action to take.

    Args:
        state: a State object containing the positions of yours and your opponents snakes

    Returns:
        The action to take
    """
    mcts = MCTS(state, rollout_policy=henry_rules_rollout, explore_coeff=0.5, verbose=0)
    start_time = time.time()

    n_rollout = 0
    while time.time() - start_time < 0.05 and n_rollout < 10_000:
        mcts.do_rollout()
        n_rollout += 1

    print(
        f"n_rollouts = {n_rollout}, "
        f"rollout_lengths = {sum(mcts.rollout_len) / len(mcts.rollout_len)},"
        f" select_lengths = {sum(mcts.select_len) / len(mcts.select_len)}\n"
    )

    return mcts.choose_action()


def get_new_head_location(bike: Bike, action: int) -> Tuple[int, int]:
    if action == 2:
        new_orientation = (bike.direction + 1) % 4
    elif action == 3:
        new_orientation = (bike.direction - 1) % 4
    else:
        new_orientation = bike.direction

    x, y = bike.head
    if new_orientation % 2 == 0:
        # South is 0 (y -= 1), North is 2 (y += 1)
        y += new_orientation - 1
    else:
        # East is 1 (x += 1), West is 3 (x -= 1)
        x += 2 - new_orientation
    return x, y


def not_in_operator(a, b):
    return a in b


def henry_rules_rollout(state: State) -> int:
    """Rollout policy that tries not to hit anything."""
    obstacles = state.player.positions + state.opponent.positions

    poss_actions = get_possible_actions()
    while poss_actions:
        action = poss_actions[math.floor(random.random() * len(poss_actions))]
        new_head = get_new_head_location(state.bike_to_move, action)
        if in_arena(new_head) and new_head not in obstacles:
            return action
        else:
            poss_actions.remove(action)
    return 1


def profile_me():
    for _ in tqdm(range(1)):
        play_tron(
            your_choose_move=human_player,
            opponent_choose_move=choose_move,
            game_speed_multiplier=5,
            render=True,
            verbose=False,
        )


if __name__ == "__main__":

    #     # # ## Example workflow, feel free to edit this! ###

    #     check_submission(
    #         TEAM_NAME, choose_move
    #     )  # <---- Make sure I pass! Or your solution will not work in the tournament!!

    #     # Play against your bot!

    cProfile.run("profile_me()", "profile.prof")
    # profile_me()
