# -*- coding: utf-8 -*-
"""
Created on Wed May 28 15:14:00 2025

@author: Jared.Krekeler
"""

from env.football_env import FootballPlayCallerEnv

# Create the environment
env = FootballPlayCallerEnv()

# Reset the environment
state, _ = env.reset()
print("Initial State:", state)

# Take 10 random actions
for step in range(10):
    action = env.action_space.sample()
    next_state, reward, done, _, _ = env.step(action)
    print(f"Step {step+1}: Action={action}, Reward={reward}, Done={done}")
    if done:
        break

env.close()
