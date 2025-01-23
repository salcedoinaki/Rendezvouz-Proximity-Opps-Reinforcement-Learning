
import gym
from gym import spaces
import numpy as np
from dynamics.orbital_mechanics import update_state

class OrbitalRendezvousEnv(gym.Env):
    """A custom Gym environment for orbital rendezvous dynamics."""

    def __init__(self):
        super(OrbitalRendezvousEnv, self).__init__()
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(4,), dtype=np.float32)
        
        self.state = None

    def reset(self):
        self.state = np.array([np.random.uniform(-5, 5), np.random.uniform(-5, 5), 0, 0], dtype=np.float32)
        return self.state

    def step(self, action):
        self.state = update_state(self.state, action)
        x, y, _, _ = self.state
        distance = np.sqrt(x**2 + y**2)
        reward = -distance
        done = distance < 0.1
        return self.state, reward, done, {}

    def render(self, mode='human'):
        print(f"Position: ({self.state[0]:.2f}, {self.state[1]:.2f})")

    def close(self):
        pass
