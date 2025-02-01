import sys
import os
# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from environment.rpo_env import RPOEnvironment
from visualizations.visualization_utils import plot_all_states, plot_single, render_simulation


def test_agent():
    """
    Load and test the trained model in the RPO environment, then visualize the results.
    """
    env = RPOEnvironment()
    env = DummyVecEnv([lambda: RPOEnvironment()])  # Wrap in DummyVecEnv first
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, training=False)

    model_path = "ppo_rpo_lstm_agent.zip"
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return
    
    model = RecurrentPPO.load(model_path, env)
    obs = env.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    states = []  # Store trajectory for visualization
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step_count += 1
        states.append(obs)
        env.render()
    
    env.close()
    print(f"Test completed. Total Reward: {total_reward}, Steps Taken: {step_count}")
    
    # Generate plots for evaluation
    print("Generating visualizations...")
    plot_all_states(states)  # Plot all state variables
    plot_single(states)  # Plot a single trajectory
    render_simulation(states)  # Render the simulation visually
    print("Visualization complete.")
if __name__ == "__main__":
    test_agent()
