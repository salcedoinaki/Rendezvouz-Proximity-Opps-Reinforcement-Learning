import numpy as np

def proximity_reward(chaser_position, target_position):
    """
    Encourages the chaser to get closer to the target.
    
    Args:
        chaser_position (np.ndarray): Position of the chaser (3,).
        target_position (np.ndarray): Position of the target (3,).

    Returns:
        float: Negative distance to the target (closer is better).
    """
    distance = np.linalg.norm(chaser_position - target_position)
    return -distance  # Closer is better

def velocity_alignment_reward(chaser_velocity, relative_position):
    """
    Rewards the chaser for aligning velocity with the target direction.

    Args:
        chaser_velocity (np.ndarray): Velocity of the chaser (3,).
        relative_position (np.ndarray): Position of the target relative to the chaser (3,).

    Returns:
        float: Alignment score (higher is better).
    """
    norm_relative_position = np.linalg.norm(relative_position)
    if norm_relative_position > 0:
        unit_relative_position = relative_position / norm_relative_position
        alignment = np.dot(chaser_velocity, unit_relative_position)
    else:
        alignment = 0.0  # No alignment if already at the target
    return alignment

def fuel_penalty(thrust):
    """
    Penalizes the agent for using excessive thrust.

    Args:
        thrust (np.ndarray): Thrust applied by the chaser (3,).

    Returns:
        float: Negative fuel cost proportional to thrust magnitude.
    """
    fuel_cost = np.linalg.norm(thrust)
    return -fuel_cost

def safety_penalty(chaser_velocity, max_velocity=10.0):
    """
    Penalizes the agent for exceeding safe velocity limits.

    Args:
        chaser_velocity (np.ndarray): Velocity of the chaser (3,).
        max_velocity (float): Maximum allowed velocity magnitude.

    Returns:
        float: Fixed penalty if velocity exceeds the limit.
    """
    velocity_magnitude = np.linalg.norm(chaser_velocity)
    return -10.0 if velocity_magnitude > max_velocity else 0.0

def combined_reward(chaser_position, target_position, chaser_velocity, thrust):
    """
    Combines all reward components to calculate the total reward.

    Args:
        chaser_position (np.ndarray): Position of the chaser (3,).
        target_position (np.ndarray): Position of the target (3,).
        chaser_velocity (np.ndarray): Velocity of the chaser (3,).
        thrust (np.ndarray): Thrust applied by the chaser (3,).

    Returns:
        float: Total reward combining all components.
    """
    relative_position = target_position - chaser_position

    # Calculate individual rewards
    proximity = proximity_reward(chaser_position, target_position)
    alignment = velocity_alignment_reward(chaser_velocity, relative_position)
    fuel_cost = fuel_penalty(thrust)
    safety = safety_penalty(chaser_velocity)

    # Combine rewards with weights (tune these weights as needed)
    reward = (
        1.0 * proximity +    # Encourage getting closer to the target
        0.5 * alignment +    # Encourage velocity alignment
        0.1 * fuel_cost +    # Penalize excessive thrust
        1.0 * safety         # Penalize unsafe speeds
    )
    return reward
