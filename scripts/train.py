import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environment.orbital_rendezvous_env import OrbitalRendezvousEnv

# Instantiate the environment
env = OrbitalRendezvousEnv()

# Vectorize the environment for faster training
vec_env = make_vec_env(lambda: OrbitalRendezvousEnv(), n_envs=4)

# Train the PPO model
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=10000)

# Save the trained model to a specific path
save_path = "c:/Users/iniak/Documents/Rendezvouz-Proximity-Opps-Reinforcement-Learning/models/orbital_rendezvous_model"
print(f"Saving model to: {save_path}")
model.save(save_path)

