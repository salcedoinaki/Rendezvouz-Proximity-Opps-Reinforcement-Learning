import sys
import os
# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
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

    # Convert trajectory to a numpy array
    positions = np.array(positions)

    # Create a figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d' if positions.shape[1] == 3 else None)

    # Plot trajectory with a color gradient
    if positions.shape[1] == 3:  # 3D Trajectory
        for i in range(len(positions) - 1):
            ax.plot(positions[i:i+2, 0], positions[i:i+2, 1], positions[i:i+2, 2],
                    color=plt.cm.viridis(i / len(positions)))
        ax.set_zlabel("Z Position")
    else:  # 2D Trajectory
        for i in range(len(positions) - 1):
            ax.plot(positions[i:i+2, 0], positions[i:i+2, 1],
                    color=plt.cm.viridis(i / len(positions)))

    # Highlight start, end, and target positions
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2] if positions.shape[1] == 3 else None,
               color="green", label="Start", s=100)
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2] if positions.shape[1] == 3 else None,
               color="red", label="End", s=100)
    ax.scatter(env.target_position[0], env.target_position[1],
               env.target_position[2] if positions.shape[1] == 3 else None,
               color="blue", label="Target", s=150, marker="X")

    # Add labels, grid, and legend
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Chaser Trajectory Visualization")
    ax.grid(True)
    ax.legend()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    visualize_trajectory()
