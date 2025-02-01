import numpy as np

class RewardConditions:
    def __init__(self, chaser):
        self.chaser = chaser
        self.time_limit = 2500
        self.chaser.docking_point = np.array([0.0, 800.0, 0.0])
        self.chaser.current_step = 0
        self.chaser.theta_cone = 60

    def inbounds(self):
        return self.get_current_distance() < 1500.0

    def get_current_distance(self):
        target_point = self.chaser.docking_point
        current_position = self.chaser.state[:3]
        return np.linalg.norm(target_point - current_position)

    def get_previous_distance(self):
        if len(self.chaser.state_trace) < 2:
            return self.get_current_distance()
        previous_position = self.chaser.state_trace[-2][:3]
        target_point = self.chaser.docking_point
        return np.linalg.norm(target_point - previous_position)

    def is_closer(self):
        return self.get_current_distance() < self.get_previous_distance()

    def in_los(self):
        dock_pos = np.array(self.chaser.docking_point)
        theta = self.chaser.theta_cone
        relative_position = self.chaser.state[:3] - dock_pos
        los_vector = np.array([0.0, 800, 0.0])
        cos_condition = np.cos(theta / 2.0)
        dot_product = np.dot(relative_position, los_vector) / (
            np.linalg.norm(relative_position) * np.linalg.norm(los_vector)
        )
        return 0.0 <= relative_position[1] <= 800.0 and dot_product >= cos_condition

    def is_docked(self):
        position_error = self.chaser.state[:3] - self.chaser.docking_point
        velocity_error = self.chaser.state[3:]
        return self.in_los() and np.linalg.norm(position_error) < 0.1 and np.linalg.norm(velocity_error) < 0.02

    def l2norm_state(self):
        position_error = self.chaser.state[:3] - self.chaser.docking_point
        velocity_error = self.chaser.state[3:]
        return np.linalg.norm(position_error) + 5.0 * np.linalg.norm(velocity_error)

class RewardFormulation(RewardConditions):
    def __init__(self, chaser):
        super().__init__(chaser)
        self.reset_rewards()

    def reset_rewards(self):
        self.time_in_los = 0
        self.total_penalty = 0
        self.total_reward = 0
        self.docked_reward = 0

    def terminal_conditions(self):
        if not self.inbounds():
            return -10, True  # Reduced from -100 to -10
        if self.chaser.current_step >= self.time_limit:
            return -10, True  # Reduced from -100 to -10
        return 0, False

    def calculate_penalty(self):
        penalty = -np.log(1 + self.l2norm_state())  # Log-scaling to prevent extreme penalties
        if not self.in_los():
            penalty -= 10.0  # Reduced from -50 to -10
        self.total_penalty += penalty
        return penalty

    def calculate_reward(self):
        reward = 0
        if self.in_los():
            self.time_in_los += 1
            distance = self.get_current_distance()
            reward += max(0, (800 - distance) ** 2) * 1e-4  # Increased from 1e-6 to 1e-4
        self.total_reward += reward
        return reward

    def docking_reward(self):
        if self.is_docked():
            reward = 500  # Reduced from 1000 to 500 for better balance
            self.docked_reward += reward
            return reward, True
        return 0, False

    def compute_reward(self):
        # Check terminal conditions
        terminal_reward, done = self.terminal_conditions()
        if done:
            return terminal_reward, done
        # Check docking condition
        docking_reward, docked = self.docking_reward()
        if docked:
            return docking_reward, True
        # Compute continuous reward and penalty
        return self.calculate_penalty() + self.calculate_reward(), False
