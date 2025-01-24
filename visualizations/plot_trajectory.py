
import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(trajectory, save_path=None):
    """Plot the trajectory of the orbital rendezvous simulation."""
    trajectory = np.array(trajectory)
    plt.figure(figsize=(8, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', label="Trajectory")
    plt.title("Trajectory Visualization")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
        print(f"Trajectory plot saved to: {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    example_trajectory = [[0, 0], [1, 1], [2, 1.5], [2.5, 0.8], [2.8, 0.3]]
    plot_trajectory(example_trajectory)
