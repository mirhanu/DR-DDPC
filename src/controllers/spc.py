# -*- coding: utf-8 -*-

import cvxpy as cp
import numpy as np
import copy
from typing import Callable, Optional
from src.util import *


class SPCController:
    """Subspace Predictive Controller (SPC) — nominal baseline.

    Standard (non-robust) SPC using the least-squares predictor
    K_hat = H_yf @ pinv([H_up; H_yp; H_uf]). No distributional robustness;
    the predicted output is deterministic given u_p, y_p, and u_f.

    Solves at each step:
        min_{u_f}  f1(u_f) + f2(y_hat)
        s.t.       h(y_hat) <= 0   (or soft penalty)
                   umin <= u_f <= umax

    where y_hat = K_up @ u_p + K_yp @ y_p + K_uf @ u_f.

    Args:
        T_p: Length of the past (initialisation) window.
        f1: Input stage cost. Signature: f1(u_f, t) -> cp.Expression.
        f2: Output stage cost. Signature: f2(y, t) -> cp.Expression.
        h: Constraint function. Signature: h(y) -> cp.Expression; the
            constraint h(y) <= 0 is enforced (hard or soft).
        H_up: Hankel matrix for past inputs,  shape (m*T_p, N_col).
        H_uf: Hankel matrix for future inputs, shape (m*T_f, N_col).
        H_yf: Hankel matrix for future outputs, shape (p*T_f, N_col).
        H_yp: Hankel matrix for past outputs,  shape (p*T_p, N_col).
        past_y: Initial output history, list of p-dim arrays, length T_p.
        past_u: Initial input history,  list of m-dim arrays, length T_p.
        U_bounds: Optional dict with keys 'umin' and 'umax' (box constraints).
        verbose: If True, pass verbose=True to the CVXPY solver.
        soft_constraint: If True, h(y) <= 0 is penalised softly.
        soft_penalty: Penalty weight for soft constraint violation.
    """

    def __init__(
        self,
        T_p: int,
        f1: Callable,
        f2: Callable,
        h: Callable,
        H_up: np.ndarray,
        H_uf: np.ndarray,
        H_yf: np.ndarray,
        H_yp: np.ndarray,
        past_y: list,
        past_u: list,
        U_bounds: dict = None,
        verbose: bool = False,
        soft_constraint: bool = True,
        soft_penalty: float = 1e5,
    ):
        self.T_p = T_p
        self.U_bounds = U_bounds
        self.verbose = verbose
        self.f1 = f1
        self.f2 = f2
        self.h = h
        self.soft_constraint = soft_constraint
        self.soft_penalty = soft_penalty

        self._first_call = True
        self._u_prev: Optional[np.ndarray] = None  # warm-start cache

        self.H_up = H_up.copy() if H_up is not None else None
        self.H_uf = H_uf.copy() if H_uf is not None else None
        self.H_yf = H_yf.copy() if H_yf is not None else None
        self.H_yp = H_yp.copy() if H_yp is not None else None
        self.past_y = past_y.copy() if past_y is not None else None
        self.past_u = past_u.copy() if past_u is not None else None

        if self._data_ready():
            self._build_problem()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _data_ready(self) -> bool:
        """Returns True when all four Hankel matrices are available."""
        return all(x is not None for x in [self.H_up, self.H_uf, self.H_yf, self.H_yp])

    def reset(self, past_y: list, past_u: list):
        """Reset the I/O history for a new episode.

        Args:
            past_y: New output history, list of p-dim arrays, length T_p.
            past_u: New input history,  list of m-dim arrays, length T_p.
        """
        self.past_y = copy.deepcopy(past_y)
        self.past_u = copy.deepcopy(past_u)
        self._first_call = True
        self._u_prev = None

    def update_params(self, **kwargs):
        """Update controller parameters and rebuild the CVXPY problem.

        Args:
            **kwargs: Any attribute of this class by name.

        Raises:
            ValueError: If an unknown parameter name is passed.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        if self._data_ready():
            self._build_problem()

    # ------------------------------------------------------------------
    # Problem construction
    # ------------------------------------------------------------------

    def _build_problem(self):
        """Estimate the SPC predictor and construct the CVXPY problem.

        Computes K_hat = H_yf @ pinv(M) and splits it into K_up, K_yp, K_uf.
        Unlike the DR variant, there are no residuals — the single deterministic
        prediction y_hat = K_up @ u_p + K_yp @ y_p + K_uf @ u_f is used
        directly in both the cost and the constraint.
        """
        # Infer dimensions from Hankel shapes
        m = self.H_up.shape[0] // self.T_p     # input  dim
        p = self.H_yp.shape[0] // self.T_p     # output dim
        T_f = self.H_uf.shape[0] // m          # prediction horizon
        self.m = m

        # SPC predictor (no residuals needed for the nominal case)
        M = np.vstack([self.H_up, self.H_yp, self.H_uf])
        K_hat = self.H_yf @ np.linalg.pinv(M)

        d_up = m * self.T_p
        d_yp = p * self.T_p
        self.K_up = K_hat[:, :d_up]                    # (p*T_f, m*T_p)
        self.K_yp = K_hat[:, d_up:d_up + d_yp]        # (p*T_f, p*T_p)
        self.K_uf = K_hat[:, d_up + d_yp:]            # (p*T_f, m*T_f)

        # Parameters updated cheaply at each solve
        self._u_p_param = cp.Parameter(m * self.T_p)
        self._y_p_param = cp.Parameter(p * self.T_p)

        # Decision variable: full future input sequence
        u_f = cp.Variable(m * T_f)
        self._u_f = u_f

        y_det = self.K_up @ self._u_p_param + \
            self.K_yp @ self._y_p_param + self.K_uf @ u_f

        constraints = []

        # Output constraint h(y) <= 0 (hard or stored for soft penalisation)
        if self.soft_constraint:
            # evaluated once; reused in objective
            self._h_expr = self.h(y_det)
        else:
            constraints.append(self.h(y_det) <= 0)

        # Input box constraints
        if self.U_bounds is not None:
            if "umin" in self.U_bounds and "umax" in self.U_bounds:
                constraints += [u_f >= self.U_bounds["umin"],
                                u_f <= self.U_bounds["umax"]]
            else:
                raise ValueError(
                    "U_bounds must contain both 'umin' and 'umax'.")

        self._prob = cp.Problem(cp.Minimize(0), constraints)

    # ------------------------------------------------------------------
    # Objective assembly
    # ------------------------------------------------------------------

    def _make_objective(self, t) -> cp.Minimize:
        """Assemble the SPC objective at time t.

        Args:
            t: Current time index, forwarded to f1 and f2.

        Returns:
            A cp.Minimize expression.
        """
        y_det = (
            self.K_up @ self._u_p_param
            + self.K_yp @ self._y_p_param
            + self.K_uf @ self._u_f
        )
        cost = self.f1(self._u_f, t) + self.f2(y_det, t)

        if self.soft_constraint:
            cost = cost + self.soft_penalty * \
                cp.sum(cp.square(cp.pos(self._h_expr)))

        return cp.Minimize(cost)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def _solve(self, u_p: np.ndarray, y_p: np.ndarray, t) -> dict:
        """Update I/O parameters and re-solve the cached problem.

        Args:
            u_p: Flattened past input vector of length m*T_p.
            y_p: Flattened past output vector of length p*T_p.
            t: Current time index forwarded to the objective.

        Returns:
            Dict with keys 'status', 'objective_value', and 'u_f' (None if
            infeasible).
        """
        self._u_p_param.value = u_p
        self._y_p_param.value = y_p

        self._prob = cp.Problem(self._make_objective(t),
                                self._prob.constraints)

        # Warm start: shift previous solution by one step, repeat last input
        if self._u_prev is not None:
            shifted = np.roll(self._u_prev, -self.m)
            shifted[-self.m:] = self._u_prev[-self.m:]
            self._u_f.value = shifted

        self._prob.solve(solver=cp.MOSEK, verbose=self.verbose)

        status = self._prob.status
        feasible = status in ["optimal", "optimal_inaccurate"]

        if feasible and self._u_f.value is not None:
            self._u_prev = self._u_f.value.copy()

        return {
            "status":          status,
            "objective_value": self._prob.value,
            "u_f":             self._u_f.value if feasible else None,
        }

    # ------------------------------------------------------------------
    # Controller interface
    # ------------------------------------------------------------------

    def __call__(self, dyn_system) -> np.ndarray:
        """Compute the next control input for the given system state.

        Args:
            dyn_system: Object exposing `.output` (current y_k) and `.t`
                (current time index).

        Returns:
            u_next: First predicted input, shape (m,).
        """
        # Append latest output (skipped on first call — history pre-loaded)
        if self._first_call:
            self._first_call = False
        else:
            self.past_y.append(dyn_system.output)
            self.past_y = self.past_y[-self.T_p:]

        u_p = np.hstack(self.past_u)
        y_p = np.hstack(self.past_y)

        result = self._solve(u_p, y_p, dyn_system.t)

        # Extract first step of the predicted input sequence
        if result["u_f"] is None:
            u_next = np.zeros(self.m)
        else:
            u_next = result["u_f"][: self.m]

        u_next = np.array(u_next).flatten()

        # Advance input history
        self.past_u.append(u_next)
        self.past_u = self.past_u[-self.T_p:]

        return u_next
