# -*- coding: utf-8 -*-
"""
experiments.py
==============
Experiment configurations.

Each entry in EXPERIMENTS is a dict that fully describes one experiment:
  - which controllers to use (and whether baselines are run once or swept)
  - the sweep grid (if any)
  - which reference signal and f2 cost to use
  - output / constraint bounds
  - innovation noise settings
  - N_MC episodes

Experiment index
----------------
  EXP_CONSTRAINT_SWEEP   — constraint satisfaction vs epsilon_const & beta
                            (y in [0, 2], zero reference, DR-SPC + DR-DDPC swept,
                             SPC + Reg-DeePC baselines run once)

  EXP_NOISE_ZERO_MEAN    — tracking cost vs noise covariance
                            (y in [-2, 2], sin reference, zero-mean Gaussian noise,
                             3 R_noise values, no sweep)

  EXP_NOISE_BIASED       — same as above but innovation mean = 0.05
                            (y in [-2, 2], sin reference, biased Gaussian noise)

  EXP_COST_L1            — alternative cost: L1 tracking
                            (y in [-2, 2], sin reference, zero-mean noise)

  EXP_COST_ASYMMETRIC    — alternative cost: asymmetric quadratic tracking
                            (y in [-2, 2], sin reference, zero-mean noise)
"""

import numpy as np
from config.config import (
    R_NOISE_DEFAULT, N_MC_DEFAULT,
    Y_MIN_DEFAULT, Y_MAX_DEFAULT,
)

# ---------------------------------------------------------------------------
# Shared sweep grids
# ---------------------------------------------------------------------------
EPSILON_CONST_SWEEP = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]
BETA_SWEEP = [0.1, 0.2, 0.5, 0.7, 0.9]
R_NOISE_SWEEP = [0.012, 0.0012, 0.000012]

# ---------------------------------------------------------------------------
# Experiment 1 — Constraint satisfaction sweep
# ---------------------------------------------------------------------------
EXP_CONSTRAINT_SWEEP = {
    "name": "constraint_sweep",
    "description": (
        "Constraint satisfaction performance for varying epsilon_const and beta. "
        "Output constrained to [0, 2]; reference is zero. "
        "DR-SPC and DR-DDPC are swept over the (epsilon_const, beta) grid. "
        "SPC and Reg-DeePC are run once as baselines (no sweep parameters)."
    ),

    # --- bounds ---
    "y_min": np.array([0.0]),
    "y_max": np.array([2.0]),

    # --- reference & cost ---
    "ref_func":  "zero",           # key looked up in runner: zero_ref
    "f2_variant": "quadratic",     # key looked up in runner: f2_quadratic

    # --- noise ---
    "innovation_mean": np.array([0.0]),   # zero-mean
    "R_noise": R_NOISE_DEFAULT,

    # --- Monte Carlo ---
    "N_MC": N_MC_DEFAULT,

    # --- initial state range ---
    # y in [0, 2], so initialise the state in [0.5, 1] to stay feasible
    "init_state_range": (0.5, 1.0),

    # --- controllers ---
    # Swept controllers: their epsilon_const and beta are replaced each iteration
    "swept_controllers": ["DR-SPC"],
    # Baseline controllers: run once with default parameters
    "baseline_controllers": ["SPC", "Reg-DeePC"],

    # --- sweep grid ---
    "sweep": {
        "epsilon_const": EPSILON_CONST_SWEEP,
        "beta":          BETA_SWEEP,
    },

    # --- output ---
    "save_prefix": "constraint_sweep",
    "plot": True,
    "plot_type": "heatmap",     # plot_sweep_heatmaps
}

# ---------------------------------------------------------------------------
# Experiment 2 — Noise sweep, zero-mean innovation
# ---------------------------------------------------------------------------
EXP_NOISE_ZERO_MEAN = {
    "name": "noise_sweep_zero_mean",
    "description": (
        "Tracking cost vs measurement noise covariance. "
        "Output constrained to [-2, 2]; sinusoidal reference. "
        "Zero-mean Gaussian innovation. "
        "All controllers compared across 3 R_noise levels."
    ),

    # --- bounds ---
    "y_min": Y_MIN_DEFAULT,
    "y_max": Y_MAX_DEFAULT,

    # --- reference & cost ---
    "ref_func":   "sin",
    "f2_variant": "quadratic",

    # --- noise ---
    "innovation_mean": np.array([0.0]),   # zero-mean
    "R_noise_sweep": R_NOISE_SWEEP,       # iterated by runner
    "snr_labels":      ["10", "20", "40"],

    # --- Monte Carlo ---
    "N_MC": N_MC_DEFAULT,

    # --- controllers ---
    "swept_controllers":    [],                          # no parameter sweep
    # all run at each noise level
    "baseline_controllers": ["SPC", "DR-SPC", "Reg-DeePC"],

    # --- sweep grid ---
    "sweep": None,   # no (epsilon_const, beta) sweep

    # --- output ---
    "save_prefix": "noise_sweep_zero_mean",
    "plot": True,
    "plot_type": "boxplot",
}

# ---------------------------------------------------------------------------
# Experiment 3 — Noise sweep, biased innovation  (mean = 0.05)
# ---------------------------------------------------------------------------
EXP_NOISE_BIASED = {
    "name": "noise_sweep_biased",
    "description": (
        "Same setup as EXP_NOISE_ZERO_MEAN but with a non-zero innovation mean "
        "of 0.05."
    ),

    # --- bounds ---
    "y_min": Y_MIN_DEFAULT,
    "y_max": Y_MAX_DEFAULT,

    # --- reference & cost ---
    "ref_func":   "sin",
    "f2_variant": "quadratic",

    # --- noise ---
    "innovation_mean": np.array([0.05]),  # biased
    "R_noise_sweep": R_NOISE_SWEEP,
    "snr_labels":      ["10", "20", "40"],

    # --- Monte Carlo ---
    "N_MC": N_MC_DEFAULT,

    # --- controllers ---
    "swept_controllers":    [],        # no parameter sweep
    "baseline_controllers": ["SPC", "DR-SPC", "Reg-DeePC"],

    # --- sweep grid ---
    "sweep": None,

    # --- output ---
    "save_prefix": "noise_sweep_biased",
    "plot": True,
    "plot_type": "boxplot",
}

# ---------------------------------------------------------------------------
# Experiment 4 — Alternative cost: L1 tracking
# ---------------------------------------------------------------------------
EXP_COST_L1 = {
    "name": "cost_l1",
    "description": (
        "Tracking performance with an L1 output cost. "
        "Output constrained to [-2, 2]; sinusoidal reference; "
        "zero-mean noise at the default covariance."
    ),

    # --- bounds ---
    "y_min": Y_MIN_DEFAULT,
    "y_max": Y_MAX_DEFAULT,

    # --- reference & cost ---
    "ref_func":   "sin",
    "f2_variant": "l1",           # f2_l1

    # --- noise ---
    "innovation_mean": np.array([0.0]),
    "R_noise_sweep": R_NOISE_SWEEP,
    "snr_labels":      ["10", "20", "40"],


    # --- Monte Carlo ---
    "N_MC": N_MC_DEFAULT,

    # --- controllers ---
    "swept_controllers":    [],         # no parameter sweep
    "baseline_controllers": ["SPC", "DR-SPC", "Reg-DeePC"],

    # --- sweep grid ---
    "sweep": None,

    # --- output ---
    "save_prefix": "cost_l1",
    "plot": True,
    "plot_type": "boxplot",
}

# ---------------------------------------------------------------------------
# Experiment 5 — Alternative cost: asymmetric quadratic tracking
# ---------------------------------------------------------------------------
EXP_COST_ASYMMETRIC = {
    "name": "cost_asymmetric",
    "description": (
        "Tracking performance with an asymmetric quadratic output cost "
        "(alpha=2, beta=1 — over-shooting penalised more heavily). "
        "Output constrained to [-2, 2]; sinusoidal reference; "
        "zero-mean noise at the default covariance."
    ),

    # --- bounds ---
    "y_min": Y_MIN_DEFAULT,
    "y_max": Y_MAX_DEFAULT,

    # --- reference & cost ---
    "ref_func":   "sin",
    "f2_variant": "asymmetric",   # f2_asymmetric

    # asymmetric cost hyper-parameters
    "f2_kwargs": {"alpha": 2.0, "beta": 1.0},

    # --- noise ---
    "innovation_mean": np.array([0.0]),
    "R_noise_sweep": R_NOISE_SWEEP,
    "snr_labels":      ["10", "20", "40"],

    # --- Monte Carlo ---
    "N_MC": N_MC_DEFAULT,

    # --- controllers ---
    "swept_controllers":    [],
    "baseline_controllers": ["SPC", "DR-SPC", "Reg-DeePC"],

    # --- sweep grid ---
    "sweep": None,

    # --- output ---
    "save_prefix": "cost_asymmetric",
    "plot": True,
    "plot_type": "boxplot",
}

# ---------------------------------------------------------------------------
# Master list — the runner iterates over this
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    EXP_CONSTRAINT_SWEEP,
    EXP_NOISE_ZERO_MEAN,
    EXP_NOISE_BIASED,
    EXP_COST_L1,
    EXP_COST_ASYMMETRIC,
]
