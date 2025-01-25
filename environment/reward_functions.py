import numpy as np

def proximity_reward(chaser_position, target_position):
    """Calculate the proximity reward based on the distance to the target."""
    return -np.linalg.norm(chaser_position - target_position)
