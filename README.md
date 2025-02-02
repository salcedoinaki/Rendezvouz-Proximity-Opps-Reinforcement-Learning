# Rendezvous Proximity Operations Reinforcement Learning

## Overview
This project implements a reinforcement learning framework for autonomous rendezvous and proximity operations (RPO) in orbit. The framework includes environment simulation, reinforcement learning training, and visualization tools.

## Project Structure

### 1. `environment/`
This folder contains the core simulation environment for the reinforcement learning model.
- **`base_env.py`**: Defines the base class for the environment, including generic step and reset functions.
- **`dynamics.py`**: Implements orbital mechanics and dynamics equations for state propagation.
- **`orbital_rendezvous_env.py`**: The main environment class that simulates the orbital rendezvous scenario, handling state updates and rewards.
- **`reward_functions.py`**: Defines different reward functions used for reinforcement learning.
- **`rpo_env.py`**: Provides a high-level wrapper for the environment, integrating all components for reinforcement learning.

### 2. `dynamics/`
This folder contains physics-based models to simulate orbital maneuvers.
- **`orbital_mechanics.py`**: Implements functions such as orbital velocity calculations and Hohmann transfer maneuvers.

### 3. `scripts/`
This folder contains training and testing scripts for reinforcement learning.
- **`train.py`**: Trains a reinforcement learning agent using the Proximal Policy Optimization (PPO) algorithm.
- **`train_rpo_agent.py`**: Similar to `train.py`, but focused on training an agent for RPO specifically.
- **`test_model.py`**: Loads a trained model and runs it in the environment to evaluate performance.

### 4. `visualizations/`
This folder contains scripts for visualizing trajectories and results.
- **`plot_trajectory.py`**: Plots the trajectory of the spacecraft based on simulation data.
- **`visualization_utils.py`**: Provides utility functions for visualization, such as orbit plotting and animation.
- **`visualize_trajectory.py`**: Generates static and animated visualizations of the spacecraft's trajectory.

### 5. `models/`
This folder stores trained reinforcement learning models.
- **`orbital_rendezvous_model.zip`**: A trained PPO model for rendezvous operations.

### 6. `tests/`
- **`test_suite.py`**: A comprehensive test suite that verifies the functionality of environment dynamics, reward functions, training scripts, and visualizations.

## Installation
1. Install dependencies:
```sh
pip install -r requirements.txt
```

2. Run tests:
```sh
python tests/test_suite.py
```

3. Train the model:
```sh
python scripts/train.py
```

4. Test the model:
```sh
python scripts/test_model.py
```

## Usage
To train a new model, run the training script. To visualize the results, use the provided visualization scripts.


