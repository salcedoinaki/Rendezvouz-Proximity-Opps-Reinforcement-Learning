import sys
import os
# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from environment.rpo_env import RPOEnvironment

def test_rpo_environment():
    # Initialize the RPO environment
    env = RPOEnvironment()

    # Reset the environment
    state, _ = env.reset()
    print(f"Initial State: {state}")

    done = False
    total_reward = 0

    # Run a sample episode
    while not done:
        # Sample a random action
        action = env.action_space.sample()

        # Take a step in the environment
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Print the state and reward
        print(f"State: {state}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

        # Check if the episode is done
        done = terminated or truncated

    print(f"Total Reward: {total_reward}")

# Run the test
if __name__ == "__main__":
    test_rpo_environment()
