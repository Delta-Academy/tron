from typing import Dict, Optional, Tuple

from game_mechanics import State, get_possible_actions, is_terminal, transition_function

NodeID = Tuple[Tuple, Tuple, Optional[int]]


class Node:
    def __init__(self, state: State):
        self.state = state
        self.is_terminal = is_terminal(state)
        # No guarantee that these NODES exist in the MCTS TREE!
        self.child_states = self._get_possible_children()
        self.key: NodeID = self.state.state_id

    def _get_possible_children(self) -> Dict[int, State]:
        """Gets the possible children of this node."""
        if self.is_terminal:
            return {}
        return {
            action: transition_function(self.state, action) for action in get_possible_actions()
        }
