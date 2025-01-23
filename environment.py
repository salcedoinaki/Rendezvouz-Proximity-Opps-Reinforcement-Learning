# Orbital Rendezvous RL

# Initial Version for a Unique GitHub Repository

# Directory Structure

# Root Directory
# - README.md            # Project overview and installation instructions
# - LICENSE              # Open-source license file
# - CONTRIBUTING.md      # Guidelines for contributing
# - environment/         # Custom RL environment definition
# - models/              # Placeholder for trained models
# - scripts/             # Scripts for training and testing
# - visualizations/      # Tools for visualizing trajectories and results
# - tests/               # Unit and integration tests

# environment

import gym
from gym import spaces
import numpy as np

class OrbitalRendezvousEnv(gym.Env):
    """
    A custom Gym environment for simulating simplified orbital rendezvous dynamics.
    """

    def __init__(self):
        super(OrbitalRendezvousEnv, self).__init__()
        
        # Define action and observation space
        # Actions: thrust in X and Y directions
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observations: relative position (x, y) and velocity (vx, vy)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)
        
        self.state = None  # Initialize state

    def reset(self):
        """Reset the environment to an initial state."""
        self.state = np.array([np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0, 0], dtype=np.float32)
        return self.state

    def step(self, action):
        """Apply action to the environment and return the next state, reward, done, and info."""
        x, y, vx, vy = self.state
        thrust_x, thrust_y = action

        # Update velocities based on thrust
        vx += thrust_x * 0.1  # Assume a simple linear response
        vy += thrust_y * 0.1

        # Update positions
        x += vx
        y += vy

        # Update state
        self.state = np.array([x, y, vx, vy], dtype=np.float32)

        # Calculate reward: Encourage minimizing distance to (0, 0)
        distance = np.sqrt(x**2 + y**2)
        reward = -distance  # Negative distance as penalty

        # Termination condition: If the agent gets close enough
        done = distance < 0.1

        return self.state, reward, done, {}

    def render(self, mode='human'):
        """Visualize the environment (optional)."""
        print(f"Position: ({self.state[0]:.2f}, {self.state[1]:.2f})")

    def close(self):
        """Clean up resources."""
        pass

# Training Script Example (scripts/train.py)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Instantiate the environment
env = OrbitalRendezvousEnv()

# Vectorize environment for faster training
vec_env = make_vec_env(lambda: env, n_envs=4)

# Train PPO model
model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=10000)

# Save the trained model
model.save("models/orbital_rendezvous_model")

# Example Visualization Script (visualizations/plot_trajectory.py)

import matplotlib.pyplot as plt

# Placeholder trajectory data
trajectory = [
    [0, 0], [1, 1], [2, 1.5], [2.5, 0.8], [2.8, 0.3]
]

trajectory = np.array(trajectory)

plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o')
plt.title("Trajectory Visualization")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.show()
