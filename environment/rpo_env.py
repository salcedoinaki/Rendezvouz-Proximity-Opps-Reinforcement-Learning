from .base_env import BaseEnvironment
from gymnasium import spaces
import numpy as np

class RPOEnvironment(BaseEnvironment):
    def __init__(self):
        # Define observation and action spaces
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        super(RPOEnvironment, self).__init__(observation_space, action_space)

        # Initialize specific environment parameters
        self.target_position = np.zeros(3)
        self.chaser_position = np.zeros(3)
        self.chaser_velocity = np.zeros(3)
        self.fuel = 100.0

    def step(self, action):
        # Update the chaser's state based on the action
        thrust = action
        self.chaser_velocity += thrust
        self.chaser_position += self.chaser_velocity

        # Calculate the reward
        distance_to_target = np.linalg.norm(self.target_position - self.chaser_position)
        reward = -distance_to_target

        # Check if the task is terminated (reached the target) or truncated (fuel depleted)
        terminated = distance_to_target < 0.1
        truncated = self.fuel <= 0
        self.fuel -= 1  # Decrement fuel

        # Return the new state, reward, termination flags, and additional info
        self.state = np.concatenate([self.chaser_position, self.chaser_velocity])
        return self.state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        # Reset the state of the environment
        if seed is not None:
            np.random.seed(seed)  # Set the random seed for reproducibility

        self.chaser_position = np.random.uniform(-1, 1, size=(3,))
        self.chaser_velocity = np.zeros(3)
        self.fuel = 100.0
        self.state = np.concatenate([self.chaser_position, self.chaser_velocity])

        # Gymnasium requires reset() to return (state, info)
        return self.state, {}
