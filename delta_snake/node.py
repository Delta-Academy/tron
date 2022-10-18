from typing import Dict, Optional, Tuple

from game_mechanics import Action, State, get_possible_actions, is_terminal, transition_function

NodeID = Tuple[Tuple[Tuple[int], int], int]


class Node:
    def __init__(self, state: State):
        self.state = state
        self.is_terminal = is_terminal(state)
        # No guarantee that these NODES exist in the MCTS TREE!
        self.child_states = self._get_possible_children()
        self.key = self.state.state_id

    def _get_possible_children(self) -> Dict[str, State]:
        """Gets the possible children of this node."""
        if self.is_terminal:
            return {}
        children = {}

        # I'm not sure these loops cover every possible action for opponets and player
        for action in get_possible_actions():
            player_moved_state = transition_function(self.state, action, self.state.player)
            for opponent in self.state.opponents:
                for opponent_action in get_possible_actions():
                    opponent_moved_state = transition_function(
                        player_moved_state, opponent_action, opponent
                    )
                    children[opponent_moved_state.state_id] = opponent_moved_state
        return children
