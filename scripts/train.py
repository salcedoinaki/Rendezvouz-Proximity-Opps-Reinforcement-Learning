import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environment.orbital_rendezvous_env import OrbitalRendezvousEnv

env = OrbitalRendezvousEnv()
vec_env = make_vec_env(lambda: env, n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=10000)
model.save("models/orbital_rendezvous_model")
