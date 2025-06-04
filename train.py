from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from env.football_env import FootballPlayCallerEnv
from visual_callback import VisualizationCallback

# Create and check the environment
env = FootballPlayCallerEnv()
check_env(env, warn=True)

# Set up the agent
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    verbose=1,
)

# Add the callback for drawing and saving a GIF
callback = VisualizationCallback(verbose=1, gif_output="episode_1_play.gif")

# Train the agent with visualization
model.learn(total_timesteps=10000, callback=callback)

# Save the model
model.save("dqn_football_playcaller")
env.close()
