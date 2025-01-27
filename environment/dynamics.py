import numpy as np
from scipy.integrate import solve_ivp

class DynamicsDiscrete:
    def __init__(self):
        # Example: Gravitational constant and other parameters
        self.gravitational_constant = 4041.804  # Modify based on AADPOPPO requirements
        self.orbit_radius = 42164000  # Example: Orbit radius (modify if needed)
        self.n = np.sqrt(self.gravitational_constant / (self.orbit_radius ** 3.0))

        # State transition matrix (A) and control matrix (B)
        self.A = np.array([  # Modify based on your specific dynamics
            [4.0 - 3.0 * np.cos(self.n), 0, 0, (1.0 / self.n) * np.sin(self.n), (2.0 / self.n) * (1 - np.cos(self.n)), 0],
            [6 * (np.sin(self.n) - self.n), 1, 0, (2.0 / self.n) * (np.cos(self.n) - 1), (1.0 / self.n) * (4 * np.sin(self.n) - 3 * self.n), 0],
            [0, 0, np.cos(self.n), 0, 0, (1.0 / self.n) * np.sin(self.n)],
            [3 * self.n * np.sin(self.n), 0, 0, np.cos(self.n), 2.0 * np.sin(self.n), 0],
            [6 * self.n * (np.cos(self.n) - 1), 0, 0, -2 * np.sin(self.n), 4 * np.cos(self.n) - 3, 0],
            [0, 0, -self.n * np.sin(self.n), 0, 0, np.cos(self.n)],
        ], dtype=np.float64)

        self.B = np.array([
            [(1 / self.n**2.0) * (1 - np.cos(self.n)), (1 / self.n**2.0) * (2.0 * self.n - 2.0 * np.sin(self.n)), 0],
            [(1.0 / self.n**2.0) * (2.0 * (np.sin(self.n) - self.n)), -3 / 2 + (4 / (self.n**2.0)) * (1 - np.cos(self.n)), 0],
            [0, 0, (1.0 / self.n**2.0) * (1.0 - np.cos(self.n))],
            [np.sin(self.n) / self.n, (2.0 / self.n) * (1.0 - np.cos(self.n)), 0],
            [(2.0 / self.n) * (np.cos(self.n) - 1.0), -3 + (4.0 / self.n) * np.sin(self.n), 0],
            [0, 0, np.sin(self.n) / self.n],
        ], dtype=np.float64)

    def step(self, state, action, mass):
        """
        Perform one discrete time step.

        Args:
            state (np.ndarray): Current state (6,).
            action (np.ndarray): Control action (3,).
            mass (float): Mass of the agent.

        Returns:
            np.ndarray: Next state (6,).
        """
        x = np.reshape(np.array(state, dtype=np.float64), (6, 1))
        u = np.reshape(np.array(action, dtype=np.float64), (3, 1)) / mass
        x_next = np.matmul(self.A, x) + np.matmul(self.B, u)
        return x_next.flatten()


class DynamicsContinuous:
    def __init__(self):
        # Example: Gravitational constant and other parameters
        self.gravitational_constant = 3.986e+14  # Modify based on AADPOPPO
        self.orbit_radius = 42164000
        self.n = np.sqrt(self.gravitational_constant / (self.orbit_radius**3.0))

        self.A = np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [3 * self.n**2, 0, 0, 0, 2 * self.n, 0],
            [0, 0, 0, -2 * self.n, 0, 0],
            [0, 0, -self.n**2, 0, 0, 0],
        ], dtype=np.float64)

        self.B = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.float64)

    def step(self, state, action, mass):
        """
        Perform one continuous time step using ODE integration.

        Args:
            state (np.ndarray): Current state (6,).
            action (np.ndarray): Control action (3,).
            mass (float): Mass of the agent.

        Returns:
            np.ndarray: Next state (6,).
        """
        def ode(t, x):
            return self.A @ x + self.B @ (action / mass)

        sol = solve_ivp(ode, [0, 1], state, method='RK45')
        return sol.y[:, -1]
