# -*- coding: utf-8 -*-
"""
run_experiments.py
==================
Main entry point.  Reads experiment configs from experiments.py and runs them.

Usage
-----
    # Run all experiments
    python run_experiments.py

    # Run a specific subset by name
    python run_experiments.py --experiments constraint_sweep noise_sweep_zero_mean
"""

import argparse
import datetime
import os

# --- project modules ---
from src.cost_functions import (
    zero_ref, sin_ref,
    f2_quadratic, f2_l1, f2_asymmetric,
    h as h_default,
)
from src.mc_runner import (
    build_lti, build_controllers
)
from src.experiment_runners import (
    run_constraint_sweep, run_noise_sweep, run_cost_comparison
)
from src.experiments import EXPERIMENTS


# =============================================================================
# LOOK-UP TABLES  (string keys used in experiment configs → callables)
# =============================================================================

REF_FUNCS = {
    "zero": zero_ref,
    "sin":  sin_ref,
}

F2_VARIANTS = {
    "quadratic":  f2_quadratic,
    "l1":         f2_l1,
    "asymmetric": f2_asymmetric,
}

# =============================================================================
# DISPATCH
# =============================================================================

_RUNNER_MAP = {
    "constraint_sweep":     run_constraint_sweep,
    "noise_sweep_zero_mean": run_noise_sweep,
    "noise_sweep_biased":   run_noise_sweep,
    "cost_l1":              run_cost_comparison,
    "cost_asymmetric":      run_cost_comparison,
}


def run_experiment(exp):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp['name']}")
    print(exp["description"])
    print('='*70)

    # --- resolve callables from string keys ---
    ref_func = REF_FUNCS[exp["ref_func"]]
    f2_base = F2_VARIANTS[exp["f2_variant"]]
    f2_kwargs = exp.get("f2_kwargs", {})

    # Bind ref_func (and any extra kwargs) into a single-argument callable
    def f2_fn(y_f, t):
        return f2_base(y_f, t, ref_func=ref_func, **f2_kwargs)

    # Build constraint function with experiment-specific bounds
    y_min = exp["y_min"]
    y_max = exp["y_max"]

    def h_fn(y_pred):
        return h_default(y_pred, y_min=y_min, y_max=y_max)

    # --- build system ---
    R_noise = exp.get("R_noise", exp.get("R_noise_sweep", [None])[
                      0] if exp.get("R_noise_sweep") else None)

    lti = build_lti(R_noise,
                    innovation_mean=exp.get("innovation_mean"))

    # --- build controllers ---
    controllers = build_controllers(exp, h_fn, f2_fn)

    # Update f2 on all controllers (in case factory used a different default)
    for ctrl in controllers.values():
        ctrl.update_params(f2=f2_fn)

    # --- output directory ---
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = "results"
    os.makedirs(results_root, exist_ok=True)  # create results/ if needed
    save_dir = os.path.join(results_root, f"{exp['save_prefix']}_{ts}")
    os.makedirs(save_dir, exist_ok=True)

    # --- dispatch to specific runner ---
    runner = _RUNNER_MAP[exp["name"]]
    return runner(exp, controllers, lti, h_fn, f2_fn, save_dir)


def main():
    parser = argparse.ArgumentParser(description="Run simulation experiments.")
    parser.add_argument(
        "--experiments", nargs="*", default=None,
        help="Experiment names to run (default: all). "
             "E.g. --experiments constraint_sweep noise_sweep_zero_mean"
    )
    args = parser.parse_args()

    selected = EXPERIMENTS
    if args.experiments:
        selected = [e for e in EXPERIMENTS if e["name"] in args.experiments]
        if not selected:
            print(f"No matching experiments found for: {args.experiments}")
            return

    for exp in selected:
        run_experiment(exp)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
