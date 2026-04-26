# -*- coding: utf-8 -*-

import numpy as np
from src.controllers.dynamic_system import DynamicSystem
from src.util import *


class LTISystem(DynamicSystem):
    """Linear Time-Invariant (LTI) system.

    Implements the state-space model:
        ẋ = A x + B u + w      (continuous)
        x[k+1] = A x[k] + B u[k] + w[k]   (discrete)
        y = C x + D u + v

    where w is process noise and v is measurement noise. Supports three
    noise modes (mutually exclusive, checked at construction):
      - No noise: w = 0, v = 0.
      - Independent noise: w ~ N(0, Q), v ~ N(0, R), drawn independently.
      - Innovation form: a single innovation e ~ N(mu, R) is drawn once per
        step; w = K @ e enters the state equation and the same e is used as
        the measurement noise, matching the Kalman innovation structure.

    Integration is handled by the parent DynamicSystem.step(): continuous
    systems use RK4 (noise applied per sub-step, which is an approximation),
    discrete systems use the return value of dynamics() directly.

    Args:
        A: State matrix, shape (n, n).
        B: Input matrix, shape (n, m).
        C: Output matrix, shape (p, n). Defaults to identity (full-state).
        D: Feedthrough matrix, shape (p, m). Defaults to zeros.
        dt: Sampling / integration time step.
        state: Initial state vector, shape (n,). Defaults to parent class default.
        output: Initial output vector, shape (p,). Defaults to parent class default.
        Q: Process noise covariance, shape (n, n). Used in independent noise mode.
        R: Measurement (or innovation) noise covariance. Shape (p, p).
        K: Innovation gain matrix, shape (n, p). Activates innovation-form noise.
        discrete: If True, dynamics() is used as a difference equation (no RK4).
        innovation_mean: Mean of the innovation distribution, shape (p,).
            Defaults to zero. Only used in innovation-form mode.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray = None,
        D: np.ndarray = None,
        dt: float = 0.01,
        state: np.ndarray = None,
        output: np.ndarray = None,
        Q: np.ndarray = None,
        R: np.ndarray = None,
        K: np.ndarray = None,
        discrete: bool = False,
        innovation_mean: np.ndarray = None,
    ):
        n, n2 = A.shape
        assert n == n2,        "A must be square."
        assert B.shape[0] == n, "B must have the same row dimension as A."
        m = B.shape[1]

        C = C if C is not None else np.eye(n)
        p = C.shape[0]
        D = D if D is not None else np.zeros((p, m))

        super().__init__(n=n, m=m, p=p, dt=dt, discrete=discrete, state=state, output=output)

        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Q = np.atleast_2d(Q) if Q is not None else None
        self.R = np.atleast_2d(R) if R is not None else None

        # Innovation-form noise: w = K @ e, e ~ N(innovation_mean, R)
        if K is not None:
            self.K = np.atleast_2d(K)
            self.innovation_form = True
            # set in _process_noise, read in _measurement_noise
            self._innovation: np.ndarray = None
        else:
            self.K = None
            self.innovation_form = False

        self.innovation_mean = (
            np.zeros(self.p) if innovation_mean is None else np.atleast_1d(
                innovation_mean)
        )

    # ------------------------------------------------------------------
    # Noise sampling
    # ------------------------------------------------------------------

    def _process_noise(self) -> np.ndarray:
        """Sample and return the process noise vector w, shape (n,).

        In innovation form, also caches the drawn innovation for reuse in
        _measurement_noise() within the same time step.
        """
        if self.innovation_form:
            e = np.random.multivariate_normal(self.innovation_mean, self.R)
            self._innovation = e
            return self.K @ e
        elif self.Q is not None:
            return np.random.multivariate_normal(np.zeros(self.n), self.Q)
        return np.zeros(self.n)

    def _measurement_noise(self) -> np.ndarray:
        """Return the measurement noise vector v, shape (p,).

        In innovation form, returns the same draw cached by _process_noise()
        so that w and v share the same innovation realisation.
        """
        if self.innovation_form:
            return self._innovation
        elif self.R is not None:
            return np.random.multivariate_normal(np.zeros(self.p), self.R)
        return np.zeros(self.p)

    # ------------------------------------------------------------------
    # DynamicSystem interface
    # ------------------------------------------------------------------

    def dynamics(self, state: np.ndarray, u: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Evaluate the state derivative (continuous) or next state (discrete).

        Args:
            state: Current state vector, shape (n,).
            u: Control input vector, shape (m,).
            t: Current time (unused in LTI, included for interface compatibility).

        Returns:
            dx or x_next, shape (n,).
        """
        return (self.A @ state + self.B @ u + self._process_noise()).flatten()

    def measure_output(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute and store the noisy system output y = C x + D u + v.

        Args:
            state: Current state vector, shape (n,).
            u: Control input vector, shape (m,).

        Returns:
            y: Output vector, shape (p,). Also stored as self.output.
        """
        self.output = self.C @ state + self.D @ u + self._measurement_noise()
        return self.output
