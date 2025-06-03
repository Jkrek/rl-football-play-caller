# -*- coding: utf-8 -*-
"""
Created on Wed May 28 15:20:23 2025

@author: Jared.Krekeler
"""
from stable_baselines3 import DQN
from env.football_env import FootballPlayCallerEnv

# Load the trained model
model = DQN.load("dqn_football_playcaller")

# Initialize environment
env = FootballPlayCallerEnv()

# Run evaluation episodes
num_episodes = 5

for episode in range(num_episodes):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    print(f"\nEpisode {episode + 1}")
    print("-" * 30)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        step_count += 1

        print(f"Step {step_count}: Action={action}, Reward={reward:.2f}, State={obs.tolist()}")

    print(f"Total Reward: {total_reward:.2f}")

env.close()

