import os
import shutil
from stable_baselines3.common.callbacks import BaseCallback
from visualize import draw_play, make_gif
import numpy as np


class VisualizationCallback(BaseCallback):
    def __init__(self, verbose=0, gif_output="episode_play.gif"):
        super().__init__(verbose)
        self.step_count = 0
        self.gif_output = gif_output
        self.frame_dir = "frames"

        # Clean up frames folder on init
        if os.path.exists(self.frame_dir):
            shutil.rmtree(self.frame_dir)
        os.makedirs(self.frame_dir, exist_ok=True)

    def _on_step(self) -> bool:
        try:
            # Unwrap the monitored/stable-baselines3 environment
            if hasattr(self.training_env, 'envs'):
                wrapped_env = self.training_env.envs[0]
                env = getattr(wrapped_env, 'unwrapped', wrapped_env)
            else:
                return True  # No env found, skip

            # Safely get the internal state
            state = getattr(env, 'state', None)
            if state is None or not isinstance(state, (list, tuple, np.ndarray)) or len(state) == 0:
                if self.verbose:
                    print(f"[VisualizationCallback Warning] Invalid or empty state: {state}")
                return True

            yardline = state[0] if len(state) > 0 else 0  # Defensive default

            # Extract actions and rewards safely
            raw_action = self.locals.get("actions")
            raw_reward = self.locals.get("rewards")

            if raw_action is None or raw_reward is None:
                return True

            action = raw_action[0] if isinstance(raw_action, (list, tuple, np.ndarray)) else raw_action
            reward = raw_reward[0] if isinstance(raw_reward, (list, tuple, np.ndarray)) else raw_reward

            # Label mapping for actions
            action_labels = ["Run", "Short Pass", "Deep Pass"]
            action_name = action_labels[action] if isinstance(action, int) and 0 <= action < len(action_labels) else f"Action {action}"

            # Draw the play visualization
            draw_play(
                yardline_start=yardline,
                action=action_name,
                yards_gained=reward,
                step=self.step_count,
                save=True,
                folder=self.frame_dir
            )

            self.step_count += 1

        except Exception as e:
            if self.verbose:
                print(f"[VisualizationCallback Error] {e}")

        return True

    def _on_training_end(self) -> None:
        # Generate GIF from saved frames
        try:
            make_gif(folder=self.frame_dir, filename=self.gif_output)
            if self.verbose:
                print(f"[GIF Saved] {self.gif_output}")
        except Exception as e:
            print(f"[GIF Creation Error] {e}")
