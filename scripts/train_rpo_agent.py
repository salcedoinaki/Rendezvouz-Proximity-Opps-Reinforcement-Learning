import sys
import os
# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from stable_baselines3 import PPO
from environment.rpo_env import RPOEnvironment

def train_agent():
    """
    Trains the RL agent using the RPO environment and the PPO algorithm.
    """
    # Create the RPO environment
    env = RPOEnvironment()

    # Instantiate the PPO model
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_rpo_tensorboard/")

    # Train the agent
    model.learn(total_timesteps=100000)

    # Save the model
    model.save("ppo_rpo_agent")
    print("Model saved as ppo_rpo_agent.zip")

    # Optionally, close the environment
    env.close()

if __name__ == "__main__":
    train_agent()
