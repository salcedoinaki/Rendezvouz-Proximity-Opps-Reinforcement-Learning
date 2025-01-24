import sys
import os

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
from stable_baselines3 import PPO
from environment.orbital_rendezvous_env import OrbitalRendezvousEnv

def test_rl_model(model_path='models/orbital_rendezvous_model.zip', steps=100):
    """Test a trained PPO model on the Orbital Rendezvous Environment."""
    # Debug: Print the resolved path
    print(f"Looking for model at: {os.path.abspath(model_path)}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {os.path.abspath(model_path)}")

    # Load the model
    model = PPO.load(model_path)
    env = OrbitalRendezvousEnv()

    # Initialize environment and retrieve observation
    obs, _ = env.reset()  # Unpack obs from the reset tuple
    total_reward = 0

    for step in range(steps):
        # Use the observation for prediction
        action, _ = model.predict(obs)
        # Handle the extra values returned by gymnasium's step()
        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward
        env.render()

        # Reset environment if the episode ends
        if terminated or truncated:
            obs, _ = env.reset()

    print(f"Total Reward after {steps} steps: {total_reward}")

if __name__ == "__main__":
    test_rl_model()
