# Week three: Multiplayer battle snake

![Watch out for the sting in his tail](images/battlesnake.webp)

## Rules of Snake :snake:

You control a Snake searching for delicious apples. :apple::apple::snake::apple::apple:

You are navigating a grid. At each timestep, you can take 1 of 3 possible actions to navigate towards the food:

1. Move forward
2. Turn left
3. Turn right.

**Moving forward** will continue the Snake moving in the same direction 1 square.

**Turning left** or **right** will change the direction of the snake in that direction, and move the snake forward 1 square in that direction.

### Points

For eating every apple **you score 1 point.** The snake also grows 1 block longer for each :apple: eaten. The block is added to the tail of the :snake:.

If you **hit any walls** or you **collide with another snake** (including youself) then you die, and the **game ends**

The game :joystick: will also end after `1000` steps

### Opponents

Unlike regular snake :snake:, you will share the arena with multiple **opponents**!

These **opponents** are subject to the same rules and logic as you.

Your **goal** is to defeat your opponents.

The winner :boom: :boom: :boom: of the game is the either:

- The last :snake: still alive
  or
- The snake with the highest number of points at the end of the time :hourglass:

# Competition Rules :scroll:

1. You must build an agent to play snake using either a **reinforcement learning** algorithm or a **heuristic search algorithm** (such as monte carlo tree search :deciduous_tree:)
2. You can only write code in `main.py`
   - You can only store data to be used in a competition in a `.pt` file by `save_network()` (it is not mandatory to use this however).
   - In the competition, your agent will call the `choose_move()` function in `main.py` to select a move (`choose_move()` may call other functions in `main.py`)
   - Any code not in `main.py` will not be used.
3. Submission deadline: **2pm UTC, Sunday**.
   - You can update your code after submitting, but **not after the deadline**.
   - **Check your submission is valid with `check_submission()`**

## Competition Format :crossed_swords:

Each matchup will be played on a single board with all submitted agents. So your should make sure your agent works against `N` opponents!

The competition will consist of multiple rounds with the ranking of bots in each round decided through the same method as the **training environment**

The competition & discussion will be in [Gather Town](https://app.gather.town/app/nJwquzJjD4TLKcTy/Delta%20Academy) at **3pm UTC on Sunday** (60 mins after submission deadline)!

## Technical Details :hammer:

### Rewards :moneybag:

The following **rewards** will be recieved in the **training environment**, however they may not map perfectly onto good performance in the tournament, so think carefully about how you use them.

Your reward :moneybag: is:

`+1` for eating an :apple:
`+2` for murdering :knife: your opponent \*
`+5` on the final step if you win the game

Otherwise your reward is `0`

\*They crash into your tail

### Observations each timestep :mag:

The **tuple** returned from each `env.step()` has:

- A **dictionary** :book: describing the positions of objects on the board

  - `player_snake`: **list** of **tuples** describing your snake's coordinates on the grid location
  - `player_orientation`: **tuple** the orientation of your snake
  - `opponent_snakes`: a **list** of **lists** of **tuples** decribing the opponent snake orientations
  - `opponent_orientations`: a **list** of **lists** of **tuples** decribing the opponent snake orientations
  - `food_location`: a **list** of **tuples**. Location of apples in the grid

- The reward for each timestep
- Whether the point is done (boolean)
- Extra information

## Arena Layout

The court is`ARENA_WIDTH` wide and`ARENA_HEIGHT`tall.

The width of the arena lies along the x-axis and the height along the y-axis.

**Positions:** top left corner is at `(0, 0)`, the bottom right corner is at `(ARENA_WIDTH, ARENA_HEIGHT)` and the top right corner is at `(ARENA_WIDTH, 0)`.

**Snake (in black**) moves East, West, North, and South. The snake cannot move diagonally.

**Apple (in green**) yum!

## Functions you write :point_left:

<details>
<summary><code style="white-space:nowrap;">  train()</code></summary>
Write this to train your algorithm from experience in the environment.
<br />
<br />
(Optional) Return a trained network so it can be saved.
</details>
<details>
<summary><code style="white-space:nowrap;">  choose_move()</code></summary>
This acts greedily given the state and network.

In the competition, the choose_move() function is called to make your next move. Takes the state as input and outputs an action.

</details>

## Existing Code :pray:

<details>
<summary><code style="white-space:nowrap;">  SnakeEnv</code></summary>
The environment class controls the game and runs the opponents. It should be used for training your agent.
<br />
<br />
See example usage in <code style="white-space:nowrap;">play_snake()</code>.
<br />
<br />
The opponents' <code style="white-space:nowrap;">choose_move</code> functions are input at initialisation (when <code style="white-space:nowrap;">Env(opponent_choose_moves)</code> is called). Every time you call <code style="white-space:nowrap;">Env.step()</code>, all snakes make a move according to their choose_move function. All player's perspecitves on the arena is the same but they will recieve their own position as player_snake in the state dictionary.
    <br />
    <br />

<code style="white-space:nowrap;">SnakeEnv</code> has a <code style="white-space:nowrap;"> verbose</code> argument which prints the information about the game to the console when set to <code style="white-space:nowrap;">True</code>. <code style="white-space:nowrap;"> SnakeEnv</code> also has a render argument which visualises the game in pygame when set to <code style="white-space:nowrap;">True</code>. This allows you to visualise your AI's skills. You can play against your agent using the <code style="white-space:nowrap;">human_choose_move()</code> function!

</details>

<details>
<summary><code style="white-space:nowrap;"> choose_move_square()</code></summary>
A basic snake bot that won't die immediately, a useful first opponent!
<br />
<br />
Takes the state as input and outputs an action.
</details>

<details>
<summary><code style="white-space:nowrap;">  play_snake()</code></summary>
Plays a game of snake, which can be rendered through pygame (if <code style="white-space:nowrap;">render=True</code>).

You can play against your own bot if you set <code style="white-space:nowrap;">your_choose_move</code> to <code style="white-space:nowrap;">human_player</code>!
<br />
<br />
Inputs:

<code style="white-space:nowrap;">your_choose_move</code>: Function that takes the state and outputs the action for your agent.

<code style="white-space:nowrap;">opponent_choose_move</code>: Function that takes the state and outputs the action for the opponent.

<code style="white-space:nowrap;">game_speed_multiplier</code>: controls the gameplay speed. High numbers mean fast games, low numbers mean slow games.

<code style="white-space:nowrap;">verbose</code>: whether to print info to the console.

<code style="white-space:nowrap;">render</code>: whether to render the match through pygame

</details>

## Suggested Approach :+1:

1. Use monte carlo tree search to simlulate possible future trajectories>
2. **Write `train()`**, borrowing from past exercises
3. **Iterate, iterate, iterate** on that `train()` function
4. **Print out important values** - otherwise bugs in your code may slip through the cracks :astonished:
