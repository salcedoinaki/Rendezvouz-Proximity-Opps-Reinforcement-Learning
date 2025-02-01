import sys
import os
# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.logger import configure
from environment.rpo_env import RPOEnvironment



def make_env():
    """ Utility function to create an environment instance for parallelization. """
    return RPOEnvironment()

def train_agent():
    """
    Fast RL Training for refinement before deep model training.
    """
    # Create and normalize vectorized environments (parallelization for speed-up)
    num_envs = 2  # Reduce to 2 environments for faster training
    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Quick Refinement - Train MLP First
    ppo_params = {
            "learning_rate": 1.5e-4,  # Reduce LR for smoother training
            "clip_range": 0.2,       # Reduce clipping to avoid excessive limitations
            "batch_size": 256,        # Larger batch for stable updates
            "n_epochs": 20,           # More epochs to refine policy
            "normalize_advantage": True,  # Improve gradient stability
            "n_steps": 1024,          # Shorter rollouts for frequent updates
        }
    model_mlp = PPO(
        "MlpPolicy", env,
        verbose=1,
        tensorboard_log="./ppo_rpo_tensorboard/",
        **ppo_params,
        device="cuda"  # Enable GPU acceleration
    )

    model_mlp.learn(total_timesteps=20000)  # Reduce timesteps for quick refinement
    model_mlp.save("ppo_rpo_refined")  # Save refined model
    print("Refined model saved as ppo_rpo_refined.zip")

    # Train LSTM Model Separately from Scratch
    model_lstm = RecurrentPPO(
        "MlpLstmPolicy", env,
        verbose=1,
        tensorboard_log="./ppo_rpo_tensorboard/",
        learning_rate=0.0005,
        clip_range=0.1,
        vf_coef=0.5,
        n_steps=128,  # Shorter sequences for faster updates
        batch_size=128,
        device="cuda"
    )
    model_lstm.learn(total_timesteps=20000)  # Reduce timesteps for faster LSTM training
    model_lstm.save("ppo_rpo_lstm_agent")
    print("Deep model saved as ppo_rpo_lstm_agent.zip")
    new_logger = configure("./ppo_rpo_tensorboard/", ["stdout", "tensorboard"])
    model_lstm.set_logger(new_logger)
    # Close the environment
    env.close()

if __name__ == "__main__":
    train_agent()