
import numpy as np

def update_state(state, action):
    x, y, vx, vy = state
    thrust_x, thrust_y = action
    vx += thrust_x * 0.1
    vy += thrust_y * 0.1
    x += vx
    y += vy
    return np.array([x, y, vx, vy], dtype=np.float32)
