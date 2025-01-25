import sys
import os
# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environment.orbital_rendezvous_env import OrbitalRendezvousEnv

def train_rl_model(total_timesteps=200000, save_path='models/orbital_rendezvous_model'):
    """Train a PPO model on the Orbital Rendezvous Environment."""
    vec_env = make_vec_env(lambda: OrbitalRendezvousEnv(), n_envs=4)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=0.0003,
        gamma=0.95,
        n_steps=1024,
        batch_size=256,
        clip_range=0.2,
        ent_coef=0.01,
    )
    model.learn(total_timesteps=total_timesteps)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    train_rl_model()
