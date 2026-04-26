# -*- coding: utf-8 -*-

import cvxpy as cp
import numpy as np
import copy
from typing import Callable, Optional


class DeepCController:
    """Data-Enabled Predictive Controller (DeePC) — nominal baseline.

    Standard DeePC with L1 regularisation on the implicit coefficient g. 

    Args:
        T_p: Length of the past (initialisation) window.
        f1: Input stage cost. Signature: f1(u_f, t) -> cp.Expression.
        f2: Output stage cost. Signature: f2(y_f, t) -> cp.Expression.
        h: Constraint function. Signature: h(y_f) -> cp.Expression; the
            constraint h(y_f) <= 0 is enforced (hard or soft).
        H_up: Hankel matrix for past inputs,  shape (m*T_p, N_col).
        H_uf: Hankel matrix for future inputs, shape (m*T_f, N_col).
        H_yf: Hankel matrix for future outputs, shape (p*T_f, N_col).
        H_yp: Hankel matrix for past outputs,  shape (p*T_p, N_col).
        lambda_g: L1 regularisation weight on g (promotes sparsity / consistency).
        past_y: Initial output history, list of p-dim arrays, length T_p.
        past_u: Initial input history,  list of m-dim arrays, length T_p.
        U_bounds: Optional dict with keys 'umin' and 'umax' (box constraints).
        verbose: If True, pass verbose=True to the CVXPY solver.
        soft_constraint: If True, h(y_f) <= 0 is penalised softly.
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
        lambda_g: float,
        past_y: list,
        past_u: list,
        U_bounds: dict = None,
        verbose: bool = False,
        soft_constraint: bool = True,
        soft_penalty: float = 1e5,
    ):
        self.T_p = T_p
        self.f1 = f1
        self.f2 = f2
        self.h = h
        self.U_bounds = U_bounds
        self.verbose = verbose
        self.lambda_g = lambda_g
        self.soft_constraint = soft_constraint
        self.soft_penalty = soft_penalty

        self._first_call = True

        self.H_up = H_up.copy() if H_up is not None else None
        self.H_uf = H_uf.copy() if H_uf is not None else None
        self.H_yf = H_yf.copy() if H_yf is not None else None
        self.H_yp = H_yp.copy() if H_yp is not None else None
        self.past_y = copy.deepcopy(past_y) if past_y is not None else None
        self.past_u = copy.deepcopy(past_u) if past_u is not None else None

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

    def update_params(self, **kwargs):
        """Update controller parameters and rebuild the CVXPY problem.

        Args:
            **kwargs: Any attribute of this class by name (e.g. lambda_g=0.1).

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
        """Construct the CVXPY problem with parametric u_ini / y_ini.

        Unlike the DR controllers, no projection or residual computation is
        needed — the standard DeePC equality constraints H_up @ g == u_p and
        H_yp @ g == y_p directly pin the past trajectory.
        """
        n_g = self.H_uf.shape[1]
        self.m = self.H_up.shape[0] // self.T_p

        # Parameters updated cheaply at each solve
        self._u_p_param = cp.Parameter(self.H_up.shape[0])
        self._y_p_param = cp.Parameter(self.H_yp.shape[0])

        # Decision variable
        g = cp.Variable(n_g)
        self._g = g

        # Past-trajectory equality constraints
        constraints = [
            self.H_up @ g == self._u_p_param,
            self.H_yp @ g == self._y_p_param,
        ]

        # Input box constraints
        if self.U_bounds is not None:
            if "umin" in self.U_bounds and "umax" in self.U_bounds:
                constraints += [
                    self.H_uf @ g >= self.U_bounds["umin"],
                    self.H_uf @ g <= self.U_bounds["umax"],
                ]
            else:
                raise ValueError(
                    "U_bounds must contain both 'umin' and 'umax'.")

        # Output constraint h(y_f) <= 0 (hard or stored for soft penalisation)
        if self.soft_constraint:
            self._h_expr = self.h(self.H_yf @ g)
        else:
            constraints.append(self.h(self.H_yf @ g) <= 0)

        self._prob = cp.Problem(cp.Minimize(0), constraints)

    # ------------------------------------------------------------------
    # Objective assembly
    # ------------------------------------------------------------------

    def _make_objective(self, t) -> cp.Minimize:
        """Assemble the DeePC objective at time t.

        Args:
            t: Current time index, forwarded to f1 and f2.

        Returns:
            A cp.Minimize expression.
        """
        g = self._g
        cost = (
            self.f1(self.H_uf @ g, t)
            + self.f2(self.H_yf @ g, t)
            + self.lambda_g * cp.norm1(g)
        )

        if self.soft_constraint:
            cost = cost + self.soft_penalty * \
                cp.sum(cp.square(cp.pos(self._h_expr)))

        return cp.Minimize(cost)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def _solve(self, u_ini: np.ndarray, y_ini: np.ndarray, t) -> dict:
        """Update I/O parameters and re-solve the cached problem.

        Args:
            u_ini: Flattened past input vector of length m*T_p.
            y_ini: Flattened past output vector of length p*T_p.
            t: Current time index forwarded to the objective.

        Returns:
            Dict with keys 'status', 'objective_value', and 'g' (None if
            infeasible).
        """
        self._u_p_param.value = u_ini
        self._y_p_param.value = y_ini

        self._prob = cp.Problem(self._make_objective(t),
                                self._prob.constraints)
        self._prob.solve(solver=cp.MOSEK, verbose=self.verbose)

        status = self._prob.status
        feasible = status in ["optimal", "optimal_inaccurate"]

        return {
            "status":          status,
            "objective_value": self._prob.value,
            "g":               self._g.value if feasible else None,
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

        u_ini = np.hstack(self.past_u)
        y_ini = np.hstack(self.past_y)

        result = self._solve(u_ini, y_ini, dyn_system.t)

        # Extract first step of the predicted input sequence
        if result["g"] is None:
            u_next = np.zeros(self.m)
        else:
            u_next = (self.H_uf @ result["g"])[: self.m]

        u_next = np.array(u_next).flatten()

        # Advance input history
        self.past_u.append(u_next)
        self.past_u = self.past_u[-self.T_p:]

        return u_next
