import sys
import os
# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from sb3_contrib import RecurrentPPO
#from stable_baselines3.common.policies import MlpPolicy
from environment.rpo_env import RPOEnvironment

def train_agent():
    """
    Trains the RL agent using the RPO environment and the Recurrent PPO algorithm with LSTM.
    """
    # Create the RPO environment
    env = RPOEnvironment()

    # Instantiate the Recurrent PPO model with LSTM policy
    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log="./ppo_rpo_tensorboard/")

    # Train the agent
    model.learn(total_timesteps=100000)

    # Save the model
    model.save("ppo_rpo_lstm_agent")
    print("Model saved as ppo_rpo_lstm_agent.zip")

    # Optionally, close the environment
    env.close()

if __name__ == "__main__":
    train_agent()
