
import matplotlib.pyplot as plt
import numpy as np

trajectory = [
    [0, 0], [1, 1], [2, 1.5], [2.5, 0.8], [2.8, 0.3]
]

trajectory = np.array(trajectory)

plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o')
plt.title("Trajectory Visualization")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.show()
