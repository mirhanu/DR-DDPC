# -*- coding: utf-8 -*-
"""
config.py
=========
Central configuration file for all system, simulation, and controller parameters.
Import this module in experiment files and the main simulation script.
"""

import numpy as np

# =============================================================================
# SYSTEM PARAMETERS
# =============================================================================

A = np.array([[0.7326, -0.0861],
              [0.1722,  0.9909]])
B = np.array([[0.0609],
              [0.0064]])
C = np.array([[0, 1.4142]])
D = np.array([[0]])
K = np.array([[-0.5], [0.5]])

# Derived dimensions
n = A.shape[0]   # state dimension
m = B.shape[1]   # input dimension
p = C.shape[0]   # output dimension

# Default measurement noise covariance
R_NOISE_DEFAULT = 0.012 * np.eye(p)

# =============================================================================
# INPUT / OUTPUT BOUNDS
# =============================================================================

Y_MIN_DEFAULT = np.array([-2.0])
Y_MAX_DEFAULT = np.array([2.0])
U_MIN = np.array([-2.0])
U_MAX = np.array([2.0])

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

T_P = 5    # past horizon length
T_F = 10   # future / prediction horizon length
T_RUN = 50   # online simulation steps per episode
T_OFFLINE = 200  # offline data length for Hankel construction

N_MC_DEFAULT = 50  # default number of Monte Carlo episodes

# =============================================================================
# CONTROLLER PARAMETERS
# =============================================================================

# --- Regularised DeePC ---
LAMBDA_G = 1.0

# --- Distributionally Robust parameters ---
EPSILON1 = 0.001    # Wasserstein radius — objective
EPSILON2 = 0.001    # Wasserstein radius — objective (secondary)
EPSILON_CONST = 0.0001   # Wasserstein radius — constraint
BETA = 0.2      # CVaR confidence level
NORM_TYPE = 2        # p-norm order used in Wasserstein ball

# Number of residual samples used inside DR controllers
N_OBJ_RES = T_OFFLINE - T_F - T_P + 1
N_CONST_RES = 20

# --- Performance cost weights ---
R_PERF = 0.05 * np.eye(m)
Q_PERF = 1.0 * np.eye(p)

# Input-bound dict (tiled over prediction horizon) — convenience helper


def make_u_bounds(u_min=U_MIN, u_max=U_MAX, T_f=T_F):
    return {
        "umin": np.tile(u_min, T_f),
        "umax": np.tile(u_max, T_f),
    }


U_BOUNDS = make_u_bounds()

# =============================================================================
# LIPSCHITZ CONSTANTS
# =============================================================================


def compute_lipschitz(T_f=T_F, p=p, Q_perf=Q_PERF,
                      y_min=Y_MIN_DEFAULT, y_max=Y_MAX_DEFAULT, r=NORM_TYPE):
    """
    Return (L_f2, L_con) — the Lipschitz constants required by the DR
    controllers for the objective and the constraint function respectively.
    """
    r_star = np.inf if r == 1 else (1 if r == np.inf else r / (r - 1))

    Q_cost_f = np.kron(np.eye(T_f), Q_perf)
    norm_Q_f = np.linalg.norm(Q_cost_f, ord=r)

    y_abs_max = np.max(np.maximum(np.abs(y_min), np.abs(y_max)))
    My_f = y_abs_max if np.isinf(r) else (T_f * p) ** (1.0 / r) * y_abs_max

    L_f2 = 2 * norm_Q_f * My_f
    L_con = 1.0 if np.isinf(r) else 2 ** (1.0 / r_star)

    return L_f2, L_con


L_F2, L_CON = compute_lipschitz()
