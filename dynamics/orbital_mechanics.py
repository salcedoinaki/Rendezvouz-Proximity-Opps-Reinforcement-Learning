
import numpy as np

def update_state(state, action, thrust_scale=0.1):
    """
    Update the state of the orbital rendezvous system based on the applied action.

    Args:
        state (np.ndarray): Current state as [x, y, vx, vy].
        action (np.ndarray): Action as [thrust_x, thrust_y].
        thrust_scale (float): Scaling factor for thrust.

    Returns:
        np.ndarray: Updated state as [x, y, vx, vy].
    """
    x, y, vx, vy = state
    thrust_x, thrust_y = action

    vx += thrust_x * thrust_scale
    vy += thrust_y * thrust_scale

    x += vx
    y += vy

    return np.array([x, y, vx, vy], dtype=np.float32)
