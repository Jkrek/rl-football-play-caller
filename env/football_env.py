import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Play_Outcome_model import PlayOutcomeModel

class FootballPlayCallerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self):
        super(FootballPlayCallerEnv, self).__init__()

        self.observation_space = spaces.Box(
            low=np.array([1, 0, 0, 0, -100]),
            high=np.array([4, 10, 100, 100, 100]),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(5)

        self.outcome_model = PlayOutcomeModel("processed_offensive_plays.csv")

        self.state = None
        self.done = False
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.down = 1
        self.distance_to_first = 10
        self.yards_to_goal = 80
        self.field_position = 20
        self.team_score = 0
        self.opponent_score = 0
        self.team_has_ball = True
        self.play_count = 0
        self.max_plays = 30
        self.done = False
        self._update_state()
        return self.state, {}

    def step(self, action):
        reward = 0
        self.play_count += 1
        turnover = False
        yards_gained = 0

        if action == 0:
            play_type = "run"
        elif action in [1, 2]:
            play_type = "pass"
        elif action == 3:
            if self.yards_to_goal <= 40:
                self.team_score += 3
                reward += 3
            else:
                turnover = True
            self._update_state()
            return self.state, reward, True, False, {}
        elif action == 4:
            yards_gained = np.random.randint(30, 50)
            self.yards_to_goal += yards_gained
            reward -= 0.5
            self._update_state()
            return self.state, reward, True, False, {}

        yards_gained = self.outcome_model.sample_yards(
            down=self.down,
            ydstogo=self.distance_to_first,
            yardline_100=self.yards_to_goal,
            play_type=play_type
        )

        self.distance_to_first -= yards_gained
        self.yards_to_goal -= yards_gained
        self.field_position += yards_gained

        if self.distance_to_first <= 0:
            reward += 1
            self.down = 1
            self.distance_to_first = min(10, self.yards_to_goal)
        else:
            self.down += 1

        if self.down > 4 or turnover:
            reward -= 3
            self.opponent_score += 3
            self.done = True
        elif self.yards_to_goal <= 0:
            self.team_score += 6
            reward += 6
            self.done = True

        if self.play_count >= self.max_plays:
            self.done = True

        self._update_state()
        return self.state, reward, self.done, False, {}

    def _update_state(self):
        score_diff = self.team_score - self.opponent_score
        self.state = np.array([
            min(4, self.down),
            min(10, self.distance_to_first),
            min(100, max(0, self.yards_to_goal)),
            min(100, max(0, self.field_position)),
            max(-100, min(100, score_diff))
        ], dtype=np.float32)

    def render(self):
        print(f"Down: {self.down}, Distance: {self.distance_to_first}, "
              f"Yards to Goal: {self.yards_to_goal}, "
              f"Position: {self.field_position}, "
              f"Score: {self.team_score}–{self.opponent_score}")

    def close(self):
        pass
