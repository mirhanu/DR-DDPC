# -*- coding: utf-8 -*-
"""
cost_functions.py
=================
Reference signals, cost functions (f1, f2 variants), and constraint function h.
All functions are compatible with both NumPy arrays and CVXPY expressions.

NOTE ON HORIZON LENGTH
----------------------
No f2 function accepts or defaults to T_F.  The number of steps T is always
inferred from the input vector length:  T = len(y_f) // p.
This makes every function correct whether it is called with a prediction
horizon vector (length T_F * p) or a full trajectory vector (length T_RUN * p).

The same applies to f1: T is inferred from len(u_f) // m.
"""

import numpy as np
import cvxpy as cp
from config.config import (
    m, p, T_RUN,
    R_PERF, Q_PERF,
    Y_MIN_DEFAULT, Y_MAX_DEFAULT,
)

# =============================================================================
# REFERENCE SIGNALS
# =============================================================================


def zero_ref(times):
    """Constant zero reference."""
    return np.zeros_like(times, dtype=float)


def sin_ref(times, T_run=T_RUN):
    """Sinusoidal reference with period T_run."""
    return np.sin(2 * np.pi * (1 / T_run) * times)


# =============================================================================
# REFERENCE VECTOR BUILDERS
# =============================================================================

def u_ref_fn(t, T):
    """Zero input reference over T steps."""
    return np.zeros(m * T)


def y_ref_fn(t, T, ref_func):
    """
    Output reference vector [y_r(t), …, y_r(t+T-1)] tiled over p outputs.
    T is always passed explicitly — never defaulted.
    """
    y_r = ref_func(t + np.arange(T))
    return np.tile(y_r, p)


# =============================================================================
# INPUT COST  f1
# =============================================================================

def f1(u_f, t, R_perf=R_PERF):
    """
    Quadratic input cost  (u_f - u_ref)^T R (u_f - u_ref).
    T is inferred from the length of u_f.
    """
    T = u_f.shape[0] // m
    R_cost = np.kron(np.eye(T), R_perf)
    d = u_f - u_ref_fn(t, T)
    return d.T @ R_cost @ d


# =============================================================================
# OUTPUT COST  f2 VARIANTS
# =============================================================================

def f2_quadratic(y_f, t, ref_func=sin_ref, Q_perf=Q_PERF):
    """
    Quadratic tracking cost  ||Q^{1/2}(y_f - y_ref)||_2^2.
    T is inferred from the length of y_f.
    Works for 1-D numpy, 2-D numpy, and 1-D/2-D CVXPY expressions.
    """
    T = y_f.shape[0] // p
    y_r = y_ref_fn(t, T, ref_func)
    Q_cost = np.kron(np.eye(T), Q_perf)
    Q_half = np.linalg.cholesky(Q_cost)

    if isinstance(y_f, np.ndarray):
        if y_f.ndim == 1:
            d = y_f - y_r
            return np.sum((Q_half.T @ d) ** 2)
        else:
            D = y_f - y_r[:, None]
            return np.sum((Q_half.T @ D) ** 2)
    else:
        if y_f.ndim == 1:
            d = y_f - y_r
            return cp.sum_squares(Q_half.T @ d)
        else:
            D = y_f - y_r[:, None]
            return cp.sum_squares(Q_half.T @ D)


def f2_l1(y_f, t, ref_func=sin_ref):
    """
    L1 tracking cost  ||y_f - y_ref||_1.
    T is inferred from the length of y_f.
    """
    T = y_f.shape[0] // p
    y_r = y_ref_fn(t, T, ref_func)

    if isinstance(y_f, np.ndarray):
        if y_f.ndim == 1:
            return np.sum(np.abs(y_f - y_r))
        else:
            return np.sum(np.abs(y_f - y_r[:, None]))
    else:
        if y_f.ndim == 1:
            return cp.norm1(y_f - y_r)
        else:
            return cp.norm1(y_f - y_r[:, None])


def f2_asymmetric(y_f, t, alpha=2.0, beta=1.0,
                  ref_func=sin_ref, Q_perf=Q_PERF):
    """
    Asymmetric quadratic cost:  alpha * max(e, 0) + beta * max(-e, 0)
    where  e = Q^{1/2}(y_f - y_ref).
    T is inferred from the length of y_f.
    """
    T = y_f.shape[0] // p
    y_r = y_ref_fn(t, T, ref_func)
    Q_cost = np.kron(np.eye(T), Q_perf)
    Q_half = np.linalg.cholesky(Q_cost)

    if isinstance(y_f, np.ndarray):
        if y_f.ndim == 1:
            e = Q_half.T @ (y_f - y_r)
            return np.sum(alpha * np.maximum(e, 0) + beta * np.maximum(-e, 0))
        else:
            E = Q_half.T @ (y_f - y_r[:, None])
            return np.sum(alpha * np.maximum(E, 0) + beta * np.maximum(-E, 0))
    else:
        if y_f.ndim == 1:
            e = Q_half.T @ (y_f - y_r)
            return cp.sum(alpha * cp.pos(e) + beta * cp.pos(-e))
        else:
            E = Q_half.T @ (y_f - y_r[:, None])
            return cp.sum(alpha * cp.pos(E) + beta * cp.pos(-E))


# =============================================================================
# CONSTRAINT FUNCTION  h
# =============================================================================

def h(y_pred, y_min=Y_MIN_DEFAULT, y_max=Y_MAX_DEFAULT):
    """
    Stacked constraint  h(y) = [y - y_max ; y_min - y]  (<=0 means feasible).
    Accepts 1-D or 2-D arrays and CVXPY expressions.
    T is inferred from the length of y_pred.
    """
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    T = y_pred.shape[0] // p
    y_min_h = np.tile(y_min, T)[:, None]
    y_max_h = np.tile(y_max, T)[:, None]

    if isinstance(y_pred, np.ndarray):
        return np.vstack([y_pred - y_max_h, -y_pred + y_min_h])
    else:
        return cp.vstack([y_pred - y_max_h, -y_pred + y_min_h])


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def compute_cost(y, u, t0, f1_fn=f1, f2_fn=f2_quadratic):
    """
    Average per-step cost  (1/T) * [f1(u) + f2(y)].

    y : (p, T_run)  output trajectory
    u : (m, T_run)  input trajectory

    Flattens in Fortran order so the vector is [y_0, y_1, …, y_{T-1}]
    (same convention the controllers use), then calls f1_fn and f2_fn.
    Both functions infer T from the vector length, so no horizon
    argument needs to be passed here.
    """
    T = y.shape[1]
    u_flat = u.flatten(order='F')   # length m * T
    y_flat = y.flatten(order='F')   # length p * T
    return (1 / T) * (f1_fn(u_flat, t0) + f2_fn(y_flat, t0))


def compute_violation_rate(y, h_fn=h):
    """Fraction of time steps with at least one active constraint violation."""
    T = y.shape[1]
    const_vals = h_fn(y.T.flatten())
    return np.any(const_vals.reshape(T, -1) > 0, axis=1).mean()
