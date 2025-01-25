import numpy as np

def update_chaser_state(position, velocity, thrust, dt=1.0):
    """Updates the chaser's position and velocity based on the applied thrust."""
    velocity += thrust * dt
    position += velocity * dt
    return position, velocity
