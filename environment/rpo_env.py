from .base_env import BaseEnvironment
from gymnasium import spaces
from .reward_functions import RewardFormulation
from .dynamics import DynamicsDiscrete
import numpy as np


class RPOEnvironment(BaseEnvironment):
    def __init__(self):
        # Define observation and action spaces
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        super(RPOEnvironment, self).__init__(observation_space, action_space)

        # Initialize dynamics and specific environment parameters
        self.dynamics = DynamicsDiscrete()
        self.mass = 500.0  # Chaser mass in kilograms
        self.target_position = np.zeros(3)
        self.state = None  # Placeholder for the current state
        self.fuel = 100.0

        # Pass required arguments to RewardFormulation
        self.reward_function = RewardFormulation(chaser=self)
        

    def step(self, action):
        """
        Executes a single time step in the environment.

        Args:
            action (np.ndarray): The action taken by the agent.

        Returns:
            Tuple: (next_state, reward, terminated, truncated, info)
        """
        # Ensure the action is within bounds
        thrust = np.clip(action, self.action_space.low, self.action_space.high)

        # Update the state using the dynamics model
        next_state = self.dynamics.step(self.state, thrust, self.mass)

        # Update the internal state for the reward function
        self.reward_function.position = next_state[:3]
        self.reward_function.velocity = next_state[3:]
        self.reward_function.thrust = thrust

        # Compute the reward using the reward function
        reward, terminated = self.reward_function.compute_reward()

        # Check if fuel has run out
        truncated = self.fuel <= 0

        # Update fuel based on thrust usage
        self.fuel -= np.linalg.norm(thrust)

        # Update the environment's state
        self.state = next_state

        # Return the new state, reward, termination flags, and additional info
        return self.state, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.

        Args:
            seed (int, optional): Random seed for reproducibility.

        Returns:
            Tuple: (initial_state, info)
        """
        if seed is not None:
            np.random.seed(seed)  # Set the random seed

        # Randomly initialize the chaser's position and velocity
        chaser_position = np.random.uniform(-1, 1, size=(3,))
        chaser_velocity = np.zeros(3)
        self.state = np.concatenate([chaser_position, chaser_velocity])
        self.fuel = 100.0  # Reset fuel

        # Gymnasium requires reset() to return (state, info)
        return self.state, {}

    def render(self, mode="human"):
        """
        Renders the environment (optional).
        """
        print(f"Chaser Position: {self.state[:3]}, Chaser Velocity: {self.state[3:]}, Fuel: {self.fuel}")

    def close(self):
        """
        Cleans up resources (optional).
        """
        pass
