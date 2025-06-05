import os
import shutil
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from visualize import draw_play, make_gif


class VisualizationCallback(BaseCallback):
    def __init__(self, verbose=0, gif_output="episode_play.gif"):
        super().__init__(verbose)
        self.step_count = 0
        self.gif_output = gif_output
        self.frame_dir = "frames"
        self.last_possession = None

        # Clean up frame folder
        if os.path.exists(self.frame_dir):
            shutil.rmtree(self.frame_dir)
        os.makedirs(self.frame_dir, exist_ok=True)

    def _on_step(self) -> bool:
        try:
            if hasattr(self.training_env, 'envs'):
                wrapped_env = self.training_env.envs[0]
                env = getattr(wrapped_env, 'unwrapped', wrapped_env)
            else:
                return True

            state = getattr(env, 'state', None)
            if state is None or not isinstance(state, (list, tuple, np.ndarray)) or len(state) < 5:
                if self.verbose:
                    print(f"[VisualizationCallback Warning] Invalid or empty state: {state}")
                return True

            down = int(state[0])
            distance = int(state[1])
            yardline_start = float(state[3])
            score_diff = int(state[4])
            score_team = max(score_diff, 0)
            score_opp = max(-score_diff, 0)

            raw_action = self.locals.get("actions")
            raw_reward = self.locals.get("rewards")
            if raw_action is None or raw_reward is None:
                return True

            action = raw_action[0] if isinstance(raw_action, (list, tuple, np.ndarray)) else raw_action
            reward = raw_reward[0] if isinstance(raw_reward, (list, tuple, np.ndarray)) else raw_reward

            action_labels = ["Run", "Short Pass", "Deep Pass", "Field Goal", "Punt"]
            action_name = action_labels[action] if isinstance(action, int) and 0 <= action < len(action_labels) else f"Action {action}"

            possession = "Team A"
            reset_flag = self.last_possession != possession
            self.last_possession = possession

            draw_play(
                yardline_start=yardline_start,
                action=action_name,
                yards_gained=reward,
                step=self.step_count,
                down=down,
                distance=distance,
                score_team=score_team,
                score_opp=score_opp,
                possession=possession,
                save=True,
                folder=self.frame_dir,
                reset=reset_flag
            )

            self.step_count += 1

        except Exception as e:
            if self.verbose:
                print(f"[VisualizationCallback Error] {e}")

        return True

    def _on_training_end(self) -> None:
        try:
            make_gif(folder=self.frame_dir, filename=self.gif_output)
            if self.verbose:
                print(f"[GIF Saved] {self.gif_output}")
        except Exception as e:
            print(f"[GIF Creation Error] {e}")
