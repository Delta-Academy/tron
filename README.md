# Tron

![Drive your light cycle to victory](images/Tron.jpeg)

## Rules of Tron :bullettrain_side:

You control a [LightCycle](https://en.wikipedia.org/wiki/Light_Cycle) and must defeat your enemies.

You are navigating a grid. At each timestep, you can take 1 of 3 possible actions to navigate.

1. Move forward
2. Turn left
3. Turn right.

**Moving forward** will continue the bike moving in the same direction 1 square.

**Turning left** or **right** will change the direction of the bike in that direction, and move the bike forward 1 square in that direction.

You should use Monte Carlo Tree Search to play.

:rotating_light: :rotating_light: :rotating_light: **You will have a maximum of 0.5 seconds to compute each move!** :rotating_light: :rotating_light: :rotating_light:

### The game

Each step you drive forward, your light tail extends behind you

If you **hit any walls** or you **collide with another bike's tail** (including youself) then you die, and the **game ends**

You win and are rewarded if you are vanquish your opponent and remain the last bike standing.

### Opponents

You will share the arena with another **light bike**

Your **goal** is to defeat your opponent.

# Competition Rules :scroll:

1. You must build an agent to play tron using Monte Carlo Tree Search :deciduous_tree:
2. You can only write code in `main.py`
   - Any code not in `main.py` will not be used.
3. You can store the state of your tree in your MCTS class. This will persist between calls to choose_move.
4. Submission deadline: **2:30pm UTC, Sunday**.
   - You can update your code after submitting, but **not after the deadline**.
   - **Check your submission is valid with `check_submission()`**

## Competition Format :crossed_swords:

Each matchup will be played against a single opponent.

The competition will consist of multiple rounds against the opponent

The competition & discussion will be in [Gather Town](https://app.gather.town/app/nJwquzJjD4TLKcTy/Delta%20Academy) at **3:30pm UTC on Sunday** (60 mins after submission deadline)!

## Technical Details :hammer:

### Rewards :moneybag:

Your reward :moneybag: is:

`+1` for winning the game. `-1` for losing the game. `0` for drawing the game.

On all other steps, your reward is `0`.

### Observations each timestep :mag:

The **tuple** returned from each `env.step()` has:

- A **`state` object** describing the positions of objects on the board and the state of the game.

  - `player`: A Bike object containing information about your bike
  - `opponent`: A Bike object containing information about your opponent
  - `player_move` (int | None): None if the player has not selected a move yet, otherwise the move the player took last timestep. (See `transition_function` for more details)

- The `reward` for each timestep
- Whether the round is `done` (boolean)
- Extra information

### The Bike object

**Useful attributes**

- `positions` a list of coordinate tuples, describing the position in space of each segment of the bike (head first)
- `direction` (int) - which way you're facing (0: south, 1: east, 2: north, 3: west)
- `alive` are you a deceased bike?

## Arena Layout

The arena is `ARENA_WIDTH` wide and`ARENA_HEIGHT`tall.

The width of the arena lies along the x-axis and the height along the y-axis.

**Positions:** top left corner is at `(0, 0)`, the bottom right corner is at `(ARENA_WIDTH, ARENA_HEIGHT)` and the top right corner is at `(ARENA_WIDTH, 0)`.

**Bikes (your bike is red)** moves East, West, North, and South. The bike cannot move diagonally.

## Functions you write :point_left:

<details>
<summary><code style="white-space:nowrap;">  choose_move()</code></summary>
Choose an action given the state

In the competition, the choose_move() function is called to make your next move.

It takes the State of the game and your initialised MCTS object as input.

</details>

## Existing Code :pray:

<details>
<summary><code style="white-space:nowrap;">  TronEnv</code></summary>
The environment class controls the game and runs the opponents. It should be used for training your agent.
<br />
<br />
See example usage in <code style="white-space:nowrap;">play_tron()</code>.
<br />
<br />
The opponent's <code style="white-space:nowrap;">choose_move</code> functions are input at initialisation (when <code style="white-space:nowrap;">Env(opponent_choose_move)</code> is called). Every time you call <code style="white-space:nowrap;">Env.step()</code>, all bikes make a move according to their choose_move function. All player's perspecitves on the arena is the same but they will recieve their own position as player_bike in the state object.
    <br />
    <br />

<code style="white-space:nowrap;">TronEnv</code> has a <code style="white-space:nowrap;"> verbose</code> argument which prints the information about the game to the console when set to <code style="white-space:nowrap;">True</code>. <code style="white-space:nowrap;"> TronEnv</code> also has a render argument which visualises the game in pygame when set to <code style="white-space:nowrap;">True</code>. This allows you to visualise your AI's skills. You can play against your agent using the <code style="white-space:nowrap;">human_choose_move()</code> function!

</details>

<details>
<summary><code style="white-space:nowrap;">  play_tron()</code></summary>
Plays a game of tron, which can be rendered through pygame (if <code style="white-space:nowrap;">render=True</code>).

You can play against your own bot if you set <code style="white-space:nowrap;">your_choose_move</code> to <code style="white-space:nowrap;">human_player</code>!
<br />
<br />
Inputs:

<code style="white-space:nowrap;">your_choose_move</code>: Function that takes the state and outputs the action for your agent.

<code style="white-space:nowrap;">opponent_choose_move</code>: Function that takes the state and outputs the action for the opponent.

<code style="white-space:nowrap;">game_speed_multiplier</code>: controls the gameplay speed. High numbers mean fast games, low numbers mean slow games.

<code style="white-space:nowrap;">verbose</code>: whether to print info to the console.

<code styae="white-space:nowrap;">render</code>: whether to render the match through pygame

</details>

<details>
<summary><code style="white-space:nowrap;"> transition_function()</code></summary>
The transition function return a new <code style="white-space:nowrap;">State</code>  after an <code style="white-space:nowrap;">action</code> has been taken in the current <code style="white-space:nowrap;">State</code>.
Tron requires both players to move simultaneously. The first time this function is called it should be called with the action of the player. This move is stored but the player is not moved. The second time this function is called it should be called with the action of the opponent. This time both players are moved.

<br />
<br />

Whether the players are moved or not is determined by the <code style="white-space:nowrap;">player_move</code> attribute of the <code style="white-space:nowrap;">State</code>. If this is <code style="white-space:nowrap;">None</code> then the player has not moved yet and the player's action is stored. If this is not <code style="white-space:nowrap;">None</code> then both players are moved.

If make_copies is set to True the state is copied before the action is taken to avoid mutating the original state.

</details>

<details>
<summary><code style="white-space:nowrap;"> reward_function()</code></summary>
The reward function returns the reward that would be given to the player in a given <code style="white-space:nowrap;">State</code>.
<br />
If the <code style="white-space:nowrap;">State</code> is terminal then the reward is +1 if the player won and -1 if the opponent won and 0 if the game was drawn. If the <code style="white-space:nowrap;">State</code> is not terminal then the reward is 0.
</details>

<details>
<summary><code style="white-space:nowrap;">  is_terminal()</code></summary>
Returns whether the <code style="white-space:nowrap;">State</code> is terminal or not.
<br />
<br />
A <code style="white-space:nowrap;">State</code> is terminal if either player has died or if the game has reached the maximum number of timesteps.
</details>

### Hard-coded polices

<details>
<summary><code style="white-space:nowrap;"> choose_move_randomly()</code></summary>
A policy that chooses a move randomly.
Takes the state as input and outputs an action.
<br />
<br />
</details>

Random players in Tron are very poor and die after a few steps. So we provide two further hardcoded policies to use in MCTS rollout and as opponents

<details>
<summary><code style="white-space:nowrap;"> choose_move_square()</code></summary>
A basic tron bot that won't die immediately, a useful first opponent!
Takes the state as input and outputs an action.
<br />
<br />
</details>

<details>
<summary><code style="white-space:nowrap;"> rules_rollout()</code></summary>
A policy that will move to avoid obstacles and the opponent if possible. An excellent rollout policy.
Takes the state as input and outputs an action.
<br />
<br />
</details>

## Suggested Approach :+1:

1. Use Monte Carlo Tree Search to simlulate possible future trajectories.
2. Play around with: exploration coefficient, tree policy & rollout policy to see which combination plays the best
3. Validate your performance first against simple rules-based bots, then against each other.
4. Use `verbose` arguments to **print out important values** - otherwise bugs in your code may slip through the cracks :astonished:
