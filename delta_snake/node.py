from typing import Dict, Optional, Tuple

from game_mechanics import Action, State, get_possible_actions, is_terminal, transition_function

NodeID = Tuple[str, Optional[int]]


class Node:
    def __init__(self, state: State, last_action: Optional[int]):
        self.state = state
        self.last_action = last_action
        self.is_terminal = is_terminal(state) if last_action is not None else False
        # No guarantee that these NODES exist in the MCTS TREE!
        self.child_states = self._get_possible_children()
        self.key: NodeID = self.state.state_id, last_action

    def _get_possible_children(self) -> Dict[int, State]:
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
                    children[action] = opponent_moved_state
        return children

    # def _get_possible_children(self) -> Dict[Action, State]:
    #     """Gets the possible children of this node."""
    #     if self.is_terminal:
    #         return {}
    #     children = {}
    #     for action in get_possible_actions(self.state):
    #         state = self.state.board.copy()
    #         state[action] = self.state.player_to_move
    #         children[action] = State(state, self.state.other_player)
    #     return children
