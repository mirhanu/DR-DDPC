# -*- coding: utf-8 -*-

import cvxpy as cp
import numpy as np
import copy
from typing import Callable, Optional
from src.util import *


class DRSPCController:
    """Distributionally Robust Subspace Predictive Controller (DR-SPC).

    Implements the DR-SPC method.

    Offline, the SPC predictor is estimated as:
        K_hat = H_yf @ pinv([H_up; H_yp; H_uf])

    and split into past-input, past-output, and future-input gain matrices.
    Prediction residuals Xi_hat = H_yf - K_hat @ M are precomputed and used
    to form the empirical disturbance distribution for robustification.

    Args:
        T_p: Length of the past (initialisation) window.
        f1: Input stage cost. Signature: f1(u_f, t) -> cp.Expression.
        f2: Output stage cost. Signature: f2(Y, t) -> cp.Expression,
            where Y has shape (p*T_f, N_samples).
        h: Constraint function. Signature: h(Y) -> cp.Expression of shape
            (h_dim, N_samples).
        H_up: Hankel matrix for past inputs,  shape (m*T_p, N_col).
        H_uf: Hankel matrix for future inputs, shape (m*T_f, N_col).
        H_yf: Hankel matrix for future outputs, shape (p*T_f, N_col).
        H_yp: Hankel matrix for past outputs,  shape (p*T_p, N_col).
        h_dim: Output dimension of the constraint function h.
        Lobj: Lipschitz constant of f2.
        Lcon: Lipschitz constant of h.
        beta: CVaR confidence level.
        epsilon1: Radius scaling coefficient for the Wasserstein ball.
        epsilon2: Constant offset added to the Wasserstein radius.
        epsilon_const: Wasserstein radius used in the CVaR constraint.
        N_obj_res: Number of residual samples used in the objective.
        N_const_res: Number of residual samples used in the CVaR constraint.
        norm_type: p-norm order used to compute the Wasserstein radius .
        past_y: Initial output history, list of p-dim arrays, length T_p.
        past_u: Initial input history,  list of m-dim arrays, length T_p.
        U_bounds: Optional dict with keys 'umin' and 'umax' (box constraints).
        verbose: If True, pass verbose=True to the CVXPY solver.
        soft_constraint: If True, the CVaR constraint is penalised softly.
        soft_penalty: Penalty weight for soft CVaR violation.
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
        h_dim: int,
        Lobj: float,
        Lcon: float,
        beta: float,
        epsilon1: float,
        epsilon2: float,
        epsilon_const: float,
        N_obj_res: float,
        N_const_res: float,
        norm_type: int,
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
        self.N_obj_res = N_obj_res
        self.N_const_res = N_const_res
        self.norm_type = norm_type
        self.epsilon_const = epsilon_const
        self.soft_constraint = soft_constraint
        self.soft_penalty = soft_penalty
        self.f1 = f1
        self.f2 = f2
        self.h = h
        self.h_dim = h_dim
        self.Lobj = Lobj
        self.Lcon = Lcon
        self.beta = beta
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2

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
        """Update controller hyperparameters and rebuild the CVXPY problem.

        Args:
            **kwargs: Any attribute of this class by name (e.g. Lobj=0.5).

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

        Computes K_hat = H_yf @ pinv(M) and splits it into past-input (K_up),
        past-output (K_yp), and future-input (K_uf) gain matrices. Residuals
        Xi_hat = H_yf - K_hat @ M form the empirical disturbance distribution.

        The predicted output for residual sample i is:
            y_hat_i = K_up @ u_p + K_yp @ y_p + K_uf @ u_f + xi_hat_i

        u_p and y_p enter as cp.Parameter so only their values need updating
        at each solve — the problem graph is reused across time steps.
        """
        # Infer dimensions from Hankel shapes
        m = self.H_up.shape[0] // self.T_p     # input  dim
        p = self.H_yp.shape[0] // self.T_p     # output dim
        T_f = self.H_uf.shape[0] // m          # prediction horizon
        self.m = m

        # SPC predictor and residuals
        M = np.vstack([self.H_up, self.H_yp, self.H_uf])
        K_hat = self.H_yf @ np.linalg.pinv(M)
        # prediction residuals (p*T_f, N_col)
        self.Xi_hat = self.H_yf - K_hat @ M

        # Split K_hat into past/future gain matrices
        d_up = m * self.T_p
        d_yp = p * self.T_p
        self.K_up = K_hat[:, :d_up]                     # (p*T_f, m*T_p)
        self.K_yp = K_hat[:, d_up:d_up + d_yp]         # (p*T_f, p*T_p)
        self.K_uf = K_hat[:, d_up + d_yp:]             # (p*T_f, m*T_f)

        # Parameters updated cheaply at each solve
        self._u_p_param = cp.Parameter(m * self.T_p)
        self._y_p_param = cp.Parameter(p * self.T_p)

        # Decision variable: full future input sequence
        u_f = cp.Variable(m * T_f)
        self._u_f = u_f

        y_det = self.K_up @ self._u_p_param + \
            self.K_yp @ self._y_p_param + self.K_uf @ u_f

        # --- Wasserstein radius for objective ---
        d_past = (m + p) * self.T_p    # row offset to future-input block in M
        if self.norm_type == 1:
            # 1-norm: only the future-input block contributes non-trivially
            uf_diffs = M[d_past:, :self.N_obj_res] - self._u_f[:, None]
            self._wass_radius = (
                self.epsilon1 / self.N_obj_res *
                cp.norm1(cp.reshape(uf_diffs, (-1,)))
                + self.epsilon2
            )
        else:
            # p-norm: full trajectory difference [u_p; y_p; u_f] vs each column of M
            m_f = cp.hstack([self._u_p_param, self._y_p_param, self._u_f])
            diff = M[:, :self.N_obj_res] - m_f[:, None]
            self._wass_radius = (
                self.epsilon1 *
                (1.0 / self.N_obj_res) ** (1.0 / self.norm_type)
                * cp.pnorm(diff, self.norm_type)
                + self.epsilon2
            )

        # --- CVaR constraint on h ---
        s = cp.Variable((self.h_dim, self.N_const_res), nonneg=True)    # slack
        # VaR level
        tau = cp.Variable((self.h_dim,))

        Y_samples = y_det[:, None] + self.Xi_hat[:, :self.N_const_res]
        cvar_lhs = tau[:, None] + self.h(Y_samples)

        cvar_violation = (
            (1.0 / self.N_const_res) * cp.sum(s, axis=1)
            - self.beta * tau
            + self.Lcon * self.epsilon_const
        )

        constraints = [cvar_lhs <= s]

        if self.soft_constraint:
            self._cvar_violation_expr = cvar_violation  # penalised in objective
        else:
            constraints.append(cvar_violation <= 0)
            self._cvar_violation_expr = None

        # --- Input box constraints ---
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
        """Assemble the DR-SPC objective at time t.

        The objective consists of:
          - f1: input stage cost (deterministic).
          - f2: sample-average output stage cost over N_obj_res residuals.
          - Wasserstein penalty: Lobj * wass_radius.
          - Soft CVaR penalty (only when soft_constraint=True).

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
        Y_samples = y_det[:, None] + self.Xi_hat[:, :self.N_obj_res]

        terms = [
            self.f1(self._u_f, t),
            (1.0 / self.N_obj_res) * self.f2(Y_samples, t=t),
            self.Lobj * self._wass_radius,
        ]

        if self.soft_constraint and self._cvar_violation_expr is not None:
            soft_pen = (self.soft_penalty / self.N_const_res) * cp.sum(
                cp.square(cp.pos(self._cvar_violation_expr))
            )
            terms.append(soft_pen)

        return cp.Minimize(cp.sum(terms))

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

        if status != "optimal":
            print(f"[DR-SPC] Solver status: {status}")

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

        On the first call the I/O history is taken from the pre-loaded
        past_u / past_y buffers (no output observation needed). On every
        subsequent call the latest system output is appended before solving.

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
