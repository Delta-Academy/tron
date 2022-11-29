import copy
import random

from game_mechanics import (
    State,
    TronEnv,
    choose_move_randomly,
    get_possible_actions,
    transition_function,
)
from node import Node, NodeID


def simulate_from_terminal_state(MCTS):
    initial_state, _, _, _ = TronEnv(opponent_choose_moves=[choose_move_randomly]).reset()
    mcts = MCTS(
        initial_state=initial_state,
        rollout_policy=lambda x: get_possible_actions()[
            int(random.random() * len(get_possible_actions()))
        ],
        explore_coeff=0.5,
        verbose=True,
    )
    terminal_state = copy.deepcopy(initial_state)
    terminal_state.player.alive = False
    node = Node(terminal_state, 0)

    total_return = mcts._simulate(node)
    assert total_return in {
        -1,
        0,
        1,
    }, f"total_return returned from ._simulate() = {total_return}, must be in [-1, 0, 1]"


def simulation_from_base(MCTS) -> None:
    initial_state, _, _, _ = TronEnv(opponent_choose_moves=[choose_move_randomly]).reset()
    mcts = MCTS(
        initial_state=initial_state,
        rollout_policy=lambda x: choose_move_randomly(x),
        explore_coeff=0.5,
        verbose=True,
    )
    node = Node(initial_state, None)
    total_return = mcts._simulate(node)
    assert total_return in {
        -1,
        0,
        1,
    }, f"total_return returned from ._simulate() = {total_return}, must be in [-1, 0, 1]"


def backup_win_base(MCTS):
    initial_state, _, _, _ = TronEnv(opponent_choose_moves=[choose_move_randomly]).reset()
    mcts = MCTS(
        initial_state=initial_state,
        rollout_policy=lambda x: choose_move_randomly(x),
        explore_coeff=0.5,
        verbose=True,
    )

    state = initial_state
    node = Node(state, None)
    total_return = 1
    mcts.tree[node.key] = node
    mcts.N[node.key] = 3
    mcts.total_return[node.key] = 0

    mcts._backup([node], total_return)

    assert (
        mcts.total_return[node.key] == 1
    ), f"total_return dictionary not updated correctly! total_return[state.id] = {mcts.total_return[node.key]}, when it should be 1!"
    assert (
        mcts.N[node.key] == 4
    ), f"N dictionary not updated correctly! N[state.id] = {mcts.N[node.key]}, when it should be 4!"


def backup_lose_state_and_parent(MCTS):
    # Enter code here
    initial_state, _, _, _ = TronEnv(opponent_choose_moves=[choose_move_randomly]).reset()
    mcts = MCTS(
        initial_state=initial_state,
        rollout_policy=lambda x: choose_move_randomly(x),
        explore_coeff=0.5,
        verbose=True,
    )
    root_state = initial_state
    # Move both bikes forward 1
    state_1 = transition_function(root_state, 0, root_state.player)
    state_2 = transition_function(root_state, 0, root_state.opponents[0])

    root_node = Node(root_state, None)
    node_1 = Node(state_1, 0)
    node_2 = Node(state_2, 0)

    mcts.tree[node_2.key] = node_2
    mcts.tree[node_1.key] = node_1
    mcts.tree[root_node.key] = root_node

    mcts.N[node_2.key] = 3
    mcts.N[node_1.key] = 4
    mcts.N[root_node.key] = 5

    mcts.total_return[node_2.key] = 0
    mcts.total_return[node_1.key] = 1
    mcts.total_return[root_node.key] = 1

    total_return = 1

    mcts._backup([root_node, node_1, node_2], total_return)

    assert (
        mcts.total_return[node_2.key] == 1
    ), f"total_return dictionary not updated correctly! total_return[state.key] = {mcts.total_return[node_2.key]}, when it should be 1!"
    assert (
        mcts.total_return[node_1.key] == 2
    ), f"total_return dictionary not updated correctly for parent nodes! total_return[state.key] = {mcts.total_return[node_1.key]}, when it should be 2!"
    assert (
        mcts.total_return[root_node.key] == 2
    ), f"total_return dictionary not updated correctly for parents! total_return[state.key] = {mcts.total_return[root_node.key]}, when it should be 2!"

    assert (
        mcts.N[node_2.key] == 4
    ), f"N dictionary not updated correctly! N[state.key] = {mcts.N[node_2.key]}, when it should be 4!"
    assert (
        mcts.N[node_1.key] == 5
    ), f"N dictionary not updated correctly for parents! N[state.key] = {mcts.N[node_1.key]}, when it should be 5!"
    assert (
        mcts.N[root_node.key] == 6
    ), f"N dictionary not updated correctly for parents! N[state.key] = {mcts.N[root_node.key]}, when it should be 6!"


def select_empty(MCTS):
    # Enter code here
    root, _, _, _ = TronEnv(opponent_choose_moves=[choose_move_randomly]).reset()
    mcts = MCTS(
        initial_state=root,
        rollout_policy=lambda x: choose_move_randomly(x),
        explore_coeff=0.5,
        verbose=True,
    )

    path = mcts._select()
    assert isinstance(
        path, list
    ), f"Path returned from ._select() is not a list, instead it's of type: {type(path)}"
    assert isinstance(
        path[0], Node
    ), f"Elements of list returned from ._select() are not of type Node, instead they're of type: {type(path[0])}"
    assert not path[
        0
    ].is_terminal, "Your ._select() function selected a terminal node from the root. It should select a node where 1 move has been played."
    assert (
        path[0].state == root
    ), f"Your ._select() function returns a node which isn't the root node given at initialization, with state: {root}. Instead, it returned the node with this state: {path[0].state}"


def select_exploit(MCTS):
    # Enter code here
    root, _, _, _ = TronEnv(opponent_choose_moves=[choose_move_randomly]).reset()
    mcts = MCTS(
        initial_state=root,
        rollout_policy=lambda x: choose_move_randomly(x),
        explore_coeff=0.5,
        verbose=True,
    )

    big_boy = 4
    for i in range(6):
        state = copy.deepcopy(root)
        player_to_move = state.player if i > 2 else state.opponents[0]
        action = i % 3
        state = transition_function(state, action, player_to_move)
        node = Node(state, 0)
        mcts.tree[node.key] = node
        mcts.N[node.key] = 1 if i != big_boy else 10
        mcts.total_return[node.key] = -2 if i != big_boy else 10

    path = mcts._select()

    assert isinstance(
        path, list
    ), f"Path returned from ._select() is not of type List, instead it's of type: {type(path)}"
    assert isinstance(
        path[0], Node
    ), f"Elements of the list returned from ._select() are not of type Node, instead they're of type: {type(path[0])}"
    assert not path[
        -1
    ].is_terminal, f"Your ._select() function selected a terminal node with state: {path[0].state} from the root with no pieces played. It should select a node where 1 move has been played."
    assert (
        path[-1].state == node.state
    ), f"Your ._select() function selected a node with state: {path[0].state}, when it should have selected state: {node.state}"


def expand_terminal(MCTS):
    # Enter code here
    root, _, _, _ = TronEnv(opponent_choose_moves=[choose_move_randomly]).reset()
    mcts = MCTS(
        initial_state=root,
        rollout_policy=lambda x: choose_move_randomly(x),
        explore_coeff=0,
        verbose=True,
    )

    # Turn the player's bike into itself, terminating game
    terminal_state = transition_function(root, 2, root.player)
    terminal_state = transition_function(terminal_state, 2, terminal_state.player)
    terminal_state = transition_function(terminal_state, 2, terminal_state.player)

    terminal_node = Node(terminal_state, 2)
    mcts.tree[terminal_node.key] = terminal_node
    mcts.N[terminal_node.key] = 0
    mcts.total_return[terminal_node.key] = 0

    node = mcts._expand(terminal_node)

    assert isinstance(
        node, Node
    ), f"Node returned from ._expand() is not of type Node, instead it's of type: {type(node)}"
    assert (
        node == terminal_node
    ), f"Node returned from ._expand() should be the terminal node, with state: {terminal_node.state}. Instead returned node with state: {node.state}"
    assert (
        node.is_terminal
    ), f"Your ._expand() function selected a terminal node with state: {node.state} from the root with no pieces played. It should select a node where 1 move has been played."


def expand_root(MCTS):
    # Enter code here
    root, _, _, _ = TronEnv(opponent_choose_moves=[choose_move_randomly]).reset()
    mcts = MCTS(
        initial_state=root,
        rollout_policy=lambda x: choose_move_randomly(x),
        explore_coeff=0,
        verbose=True,
    )

    node = mcts._expand(mcts.tree[mcts.root_node.key])

    assert isinstance(
        node, Node
    ), f"Node returned from ._expand() is not of type Node, instead it's of type: {type(node)}"
    assert (
        not node.is_terminal
    ), f"Your ._expand() function selected a terminal node with state: {node.state} from the root with no pieces played. It should select a node where 1 move has been played."
