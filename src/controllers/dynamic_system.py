# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 12:59:23 2025

@author: Mirhan Urkmez
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from functools import partial


# Base class for dynamic systems
class DynamicSystem:
    """
    Base class for dynamic systems — supports both continuous and discrete time.

    Continuous:  ẋ = f(x, u, t)   integrated via RK4
    Discrete:    x[k+1] = f(x[k], u[k], k)   stepped directly

    Args:
        n (int): State dimension.
        m (int): Control dimension.
        p (int): Output dimension.
        dt (float): Time step (sampling period for discrete systems).
        discrete (bool): If True, treat as a discrete-time system.
        state (np.ndarray, optional): Initial state.
        output (np.ndarray, optional): Initial output.
        t (float): Current time / step index.
    """

    def __init__(self, n=4, m=1, p=4, dt=0.01, discrete=False,
                 t=0.0, state=None, output=None):
        self.n = n
        self.m = m
        self.p = p
        self.dt = dt
        self.discrete = discrete
        self.state = state if state is not None else np.zeros(n)
        self.output = output if output is not None else np.zeros(p)
        self.t = t

    # ------------------------------------------------------------------
    # To be overridden by subclasses
    # ------------------------------------------------------------------

    def dynamics(self, state, u, t=0.0):
        """
        Continuous:  return ẋ = f(x, u, t)
        Discrete:    return x[k+1] = f(x[k], u[k], k)

        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "dynamics() must be implemented by subclasses.")

    def measure_output(self, state, u):
        """
        Returns system output. Override in subclasses.
        Must be implemented by subclasses.
        """
        raise NotImplementedError(
            "measure_output() must be implemented by subclasses.")

    def rk4_step(self, state, u, t=0.0):
        """Performs one Runge-Kutta step."""
        k1 = self.dynamics(state, u, t)
        k2 = self.dynamics(state + 0.5 * self.dt * k1, u, t + 0.5*self.dt)
        k3 = self.dynamics(state + 0.5 * self.dt * k2, u, t + 0.5*self.dt)
        k4 = self.dynamics(state + self.dt * k3, u, t + self.dt)
        return state + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def step(self, state, u, t=0.0):
        """
        Advance the system by one time step.
        Dispatches to RK4 (continuous) or direct map (discrete).
        """
        if self.discrete:
            return np.asarray(self.dynamics(state, u, t)).flatten()
        else:
            return self.rk4_step(state, u, t)

    def simulate(self, control_law=None, T=5.0, t0=None):
        """
        Simulate the system over a time horizon.

        Args:
            control_law (function, optional): Function computing control inputs, default is zero control.
            T (float): Total simulation time.
            add_noise (bool): Whether to add Gaussian noise.
            noise_std (float): Standard deviation of noise.

        Returns:
            tuple: (states, controls) as NumPy arrays.
        """
        # Define a default control law that returns a zero array of appropriate size
        if control_law is None:
            def control_law(self): return np.zeros(self.m)

        num_steps = int(T / self.dt)
        # Preallocate memory using self.n
        states = np.zeros((self.n, num_steps + 1))
        # Preallocate memory for control inputs
        controls = np.zeros((self.m, num_steps))
        states[:, 0] = self.state.flatten()  # Set initial state

        # y0 = self.measure_output(states[:, 0], control_law(states[:, 0], t0))
        outputs = np.zeros((self.p, num_steps))
        # outputs[:, 0] = y0
        if t0 is not None:
            self.t = t0

        for i in range(num_steps):
            u = control_law(self)  # Compute control input

            # Ensure control has correct dimensions
            if np.isscalar(u):
                u = np.array([u])

            controls[:, i] = u  # Store the control input
            # Compute next state
            states[:, i+1] = self.step(states[:, i], u, self.t)
            outputs[:, i] = self.measure_output(states[:, i], u)

            self.state = states[:, i+1]
            self.t = self.t + self.dt  # Current time
            # outputs[:, i + 1] = self.measure_output(next_state, u)

        self.state = states[:, -1]  # Update system state
        return outputs, states, controls
