import gymnasium as gym

class BaseEnvironment(gym.Env):
    def __init__(self, observation_space, action_space):
        super(BaseEnvironment, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.state = None

    def step(self, action):
        """
        Executes a single step in the environment.
        Must be overridden by the subclass.
        """
        raise NotImplementedError("The step() method must be implemented by the subclass.")

    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        Must be overridden by the subclass.
        """
        super().reset(seed=seed)
        raise NotImplementedError("The reset() method must be implemented by the subclass.")

    def render(self, mode="human"):
        """
        Renders the environment.
        Optional to override in subclasses.
        """
        pass

    def close(self):
        """
        Cleans up the environment.
        Optional to override in subclasses.
        """
        pass
