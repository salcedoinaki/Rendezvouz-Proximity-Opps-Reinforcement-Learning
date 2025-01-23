import sys
import os

# Add the project root to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from environment.orbital_rendezvous_env import OrbitalRendezvousEnv

# Load the trained model
model_path = "c:/Users/iniak/Documents/Rendezvouz-Proximity-Opps-Reinforcement-Learning/models/orbital_rendezvous_model"
model = PPO.load(model_path)

# Create an environment instance
env = OrbitalRendezvousEnv()

# Test the model
obs, _ = env.reset()
total_reward = 0
for step in range(100):  # Simulate 100 steps
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    env.render()

    # End the episode if terminated or truncated
    if terminated or truncated:
        print("Episode ended. Resetting environment.")
        obs, _ = env.reset()

print(f"Total Reward: {total_reward}")
