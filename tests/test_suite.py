# test_suite.py

# This script provides a comprehensive test suite for the entire project,
# including the environment, dynamics, reward functions, training, and visualization.
import sys
import os
# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import unittest
import numpy as np
import gym
import stable_baselines3 as sb3
from environment.rpo_env import RPOEnvironment
from environment.dynamics import DynamicsContinuous
from environment.orbital_rendezvous_env import OrbitalRendezvousEnv
from visualizations.visualize_trajectory import visualize_trajectory


class TestRPOEnv(unittest.TestCase):
    def setUp(self):
        self.env = RPOEnvironment()
        self.env.reset()
    
    def test_environment_reset(self):
        state = self.env.reset()
        self.assertEqual(state.shape, (6,))
    
    def test_environment_step(self):
        action = self.env.action_space.sample()
        new_state, reward, done, info = self.env.step(action)
        self.assertEqual(new_state.shape, (6,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)
    
    def test_action_limits(self):
        action = np.array([10.0, 10.0, 10.0])  # Out of bounds
        with self.assertRaises(Exception):
            self.env.step(action)
    
    def test_environment_stability(self):
        for _ in range(100):
            action = self.env.action_space.sample()
            self.env.step(action)


class TestVisualization(unittest.TestCase):

    def test_visualize_trajectory(self):
        positions = np.random.randn(100, 2) * 50
        try:
            visualize_trajectory(positions, target_position=[0, 0], animate=False)
            success = True
        except Exception as e:
            print(e)
            success = False
        self.assertTrue(success)


class TestTrainingAndModel(unittest.TestCase):
    def test_model_exists(self):
        import os
        self.assertTrue(os.path.exists(MODEL_PATH))
    
    def test_run_test_script(self):
        try:
            run_test(steps=5)
            success = True
        except Exception as e:
            print(e)
            success = False
        self.assertTrue(success)
    
    def test_model_training(self):
        env = RPOEnv()
        model = sb3.PPO("MlpPolicy", env, verbose=0)
        try:
            model.learn(total_timesteps=100)
            success = True
        except Exception as e:
            print(e)
            success = False
        self.assertTrue(success)
    

if __name__ == "__main__":
    unittest.main()
