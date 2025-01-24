
import gym
from gym import spaces
import numpy as np

class OrbitalRendezvousEnv(gym.Env):
    """
    A custom Gym environment for simulating simplified orbital rendezvous dynamics.

    Observations:
        - Relative position: (x, y)
        - Relative velocity: (vx, vy)

    Actions:
        - Thrust: (thrust_x, thrust_y)

    Reward:
        - Negative of the distance to the target position (0, 0).

    Termination:
        - Episode ends when the distance to the target is below a threshold (0.1).
    """

    def __init__(self):
        super(OrbitalRendezvousEnv, self).__init__()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)
        self.state = None

    def reset(self):
        """Reset the environment to an initial state."""
        self.state = np.array([np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0, 0], dtype=np.float32)
        return self.state

    def step(self, action):
        """Apply action to the environment and return the next state, reward, done, and info."""
        x, y, vx, vy = self.state
        thrust_x, thrust_y = action

        vx += thrust_x * 0.1
        vy += thrust_y * 0.1

        x += vx
        y += vy

        self.state = np.array([x, y, vx, vy], dtype=np.float32)

        distance = np.sqrt(x**2 + y**2)
        reward = -distance

        done = distance < 0.1

        return self.state, reward, done, {}

    def render(self, mode='human'):
        """Visualize the environment (optional)."""
        print(f"Position: ({self.state[0]:.2f}, {self.state[1]:.2f})")

    def close(self):
        pass
