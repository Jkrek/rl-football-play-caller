# -*- coding: utf-8 -*-
"""
Created on Wed May 28 14:57:15 2025

@author: Jared.Krekeler
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FootballPlayCallerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self):
        super(FootballPlayCallerEnv, self).__init__()

        # Observation space:
        # [down, distance_to_first, yards_to_goal, field_position, score_diff]
        self.observation_space = spaces.Box(
            low=np.array([1, 0, 0, 0, -100]),
            high=np.array([4, 10, 100, 100, 100]),
            dtype=np.float32
        )

        # Action space:
        # 0 = Run, 1 = Short Pass, 2 = Deep Pass, 3 = Field Goal, 4 = Punt
        self.action_space = spaces.Discrete(5)

        self.state = None
        self.done = False
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.down = 1
        self.distance_to_first = 10
        self.yards_to_goal = 80
        self.field_position = 20
        self.score_diff = 0
        self.play_count = 0
        self.max_plays = 30

        self._update_state()  # ✅ ensure state is consistent

        self.done = False
        return self.state, {}


    def step(self, action):
        reward = 0
        self.play_count += 1

        # Simulate play outcome
        yards_gained = 0
        turnover = False

        if action == 0:  # Run
            yards_gained = np.random.randint(0, 6)
        elif action == 1:  # Short Pass
            success = np.random.rand() < 0.7
            yards_gained = np.random.randint(3, 10) if success else 0
        elif action == 2:  # Deep Pass
            success = np.random.rand() < 0.4
            yards_gained = np.random.randint(10, 30) if success else 0
        elif action == 3:  # Field Goal
            if self.yards_to_goal <= 40:
                reward += 3
            else:
                turnover = True
        elif action == 4:  # Punt
            yards_gained = np.random.randint(30, 50)
            self.yards_to_goal += yards_gained
            reward -= 0.5  # cost of giving up the ball
            self._update_state()
            return self.state, reward, True, False, {}

        # Update position
        self.distance_to_first -= yards_gained
        self.yards_to_goal -= yards_gained
        self.field_position += yards_gained

        # Turnover on downs
        if self.distance_to_first <= 0:
            reward += 1  # first down
            self.down = 1
            self.distance_to_first = min(10, self.yards_to_goal)
        else:
            self.down += 1

        # If 4th down and not scoring, turnover
        if self.down > 4 or turnover:
            reward -= 3
            self.done = True

        # Touchdown
        if self.yards_to_goal <= 0:
            reward += 6
            self.done = True

        # Game over by play limit
        if self.play_count >= self.max_plays:
            self.done = True

        self._update_state()
        return self.state, reward, self.done, False, {}

    def _update_state(self):
        self.state = np.array([
        min(4, self.down),
        min(10, self.distance_to_first),
        min(100, max(0, self.yards_to_goal)),
        min(100, max(0, self.field_position)),
        max(-100, min(100, self.score_diff))
    ], dtype=np.float32)


    def render(self):
        print(f"Down: {self.down}, Distance: {self.distance_to_first}, "
              f"Yards to Goal: {self.yards_to_goal}, Position: {self.field_position}")

    def close(self):
        pass
