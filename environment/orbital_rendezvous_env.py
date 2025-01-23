import gymnasium as gym
from gymnasium import spaces
import numpy as np
from dynamics.orbital_mechanics import update_state

class OrbitalRendezvousEnv(gym.Env):
    def __init__(self):
        super(OrbitalRendezvousEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)
        self.state = None
        self.np_random = None

    def reset(self, *, seed=None, options=None):
        """Reset the environment to its initial state."""
        # Set the seed for reproducibility
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        # Initialize state with seeded random values
        self.state = np.array([self.np_random.uniform(-5, 5), self.np_random.uniform(-5, 5), 0, 0], dtype=np.float32)
        # Return the state and an empty info dictionary
        return self.state, {}

    def step(self, action):
        """Apply the action and update the environment state."""
        self.state = update_state(self.state, action)
        x, y, _, _ = self.state

        # Calculate distance and reward
        distance = np.sqrt(x**2 + y**2)
        reward = -distance

        # Define termination criteria
        terminated = distance < 0.1  # Terminate if the agent is close enough
        truncated = False  # For now, there is no time limit or forced truncation

        # Return the updated state, reward, termination info, and metadata
        return self.state, reward, terminated, truncated, {}


    def render(self, mode='human'):
        print(f"Position: ({self.state[0]:.2f}, {self.state[1]:.2f})")

    def close(self):
        pass
