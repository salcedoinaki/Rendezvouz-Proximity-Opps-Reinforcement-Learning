import sys
import os
# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import numpy as np
from environment.rpo_env import RPOEnvironment
from stable_baselines3 import PPO

def visualize_trajectory():
    # Load the trained agent
    model = PPO.load("ppo_rpo_agent")

    # Create the environment
    env = RPOEnvironment()
    state, _ = env.reset()

    positions = [env.chaser_position.copy()]
    done = False

    # Simulate the agent
    while not done:
        action, _ = model.predict(state)
        state, _, terminated, truncated, _ = env.step(action)
        positions.append(env.chaser_position.copy())
        done = terminated or truncated

    # Plot the trajectory
    positions = np.array(positions)
    plt.figure(figsize=(8, 6))
    plt.plot(positions[:, 0], positions[:, 1], label="Chaser Trajectory")
    plt.scatter(env.target_position[0], env.target_position[1], c='red', label="Target")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.title("Chaser Trajectory Visualization")
    plt.show()

if __name__ == "__main__":
    visualize_trajectory()