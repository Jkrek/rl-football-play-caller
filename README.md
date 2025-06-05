🏈 RL Football Play-Calling Bot

RL Football Play-Calling Bot is a machine learning simulation that trains an AI agent to call offensive plays in a football game using real NFL data. It combines reinforcement learning, custom Gym environments, and a statistical play outcome model trained on play-by-play data from the NFL (2020–2023). This is an intersection between my technical interests and my hobby(football)!

🧠 Project Overview
The AI agent learns to choose between:
Run, Short Pass, Deep Pass, Field Goal, or Punt.

The game state includes:
Down, Yards to First, Yards to Goal, Field Position, and Score Differential.

Instead of using random outcomes, the simulation pulls realistic yardage results from a model trained on real-world NFL data. This gives the simulation a much more authentic feel.

The model is trained using Stable-Baselines3 with a Deep Q-Network (DQN), and episodes are visualized using an animated GIF that captures each play the agent calls.

🔬 What This Project Demonstrates
How reinforcement learning can be used to replicate decision-making in sports.

How real-world data (via nfl_data_py) can enhance the quality of RL environments.

The use of Gym environments to simulate realistic decision trees for agents.

Basic reward engineering to teach the agent strategy (score, convert, don’t turn over the ball).

🛠 Technologies Used

Python

Gymnasium (custom environment)

Stable-Baselines3 (DQN)

nfl_data_py + NFLFastR dataset

Matplotlib (for visualization)

Pandas / NumPy (data processing)

📈 Room for Expansion

This simulation can be made more advanced by:

Adding game clock and timeout logic

Introducing defensive modeling

Including formations or player-level detail

Comparing agent decisions to real NFL team playcalling

