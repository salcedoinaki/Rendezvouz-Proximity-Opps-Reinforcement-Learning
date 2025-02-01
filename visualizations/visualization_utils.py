import os
import matplotlib.pyplot as plt
import numpy as np

def plot_all_states(states):
    """ Generate plots for all state variables. """
    states = np.array(states)
    num_vars = states.shape[1]
    time_steps = range(states.shape[0])
    
    plt.figure(figsize=(12, 6))
    for i in range(num_vars):
        plt.plot(time_steps, states[:, i], label=f"State {i}")
    
    plt.xlabel("Time Step")
    plt.ylabel("State Values")
    plt.title("All State Variables Over Time")
    plt.legend()
    plt.grid()
    plt.show()

def plot_single(states, var_index=0):
    """ Generate a plot for a single state variable. """
    states = np.array(states)
    time_steps = range(states.shape[0])
    
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, states[:, var_index], label=f"State {var_index}", color='r')
    plt.xlabel("Time Step")
    plt.ylabel("State Value")
    plt.title(f"State {var_index} Over Time")
    plt.legend()
    plt.grid()
    plt.show()

def render_simulation(states):
    """ Render a simple visualization of the simulation over time. """
    states = np.array(states)
    
    plt.figure(figsize=(8, 8))
    plt.plot(states[:, 0], states[:, 1], marker='o', linestyle='-', label="Trajectory")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Chaser Trajectory")
    plt.legend()
    plt.grid()
    plt.show()
