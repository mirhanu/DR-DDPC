# -*- coding: utf-8 -*-
"""
experiment_runners.py
=====================
One runner function per experiment type.

Contents
--------
  run_constraint_sweep   — Sweep (epsilon_const, beta) for DR controllers;
                           baselines are run once with default parameters.
  run_noise_sweep        — Sweep R_noise levels with a quadratic tracking cost.
  run_cost_comparison    — Same noise sweep but with an alternative f2 cost
                           (l1 or asymmetric).
"""

import numpy as np

from src.mc_runner import run_mc_episodes
from src.util import (
    plot_sweep_heatmaps,
    plot_noise_boxplots,
    _print_constraint_summary,
    _print_noise_summary,
    _save,
)

# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================


def run_constraint_sweep(exp, controllers, lti, h_fn, f2_fn, save_dir):
    """Sweep (epsilon_const, beta) for DR controllers; baselines run once.

    Args:
        exp:         Experiment config dict.
        controllers: Dict {name: controller_instance} for all controllers.
        lti:         Configured LTISystem instance.
        h_fn:        Constraint function h(y_pred) 
        f2_fn:       Output cost function f2(y_f, t)
        save_dir:    Directory to write results and plots to.

    Returns:
        Payload dict with keys ``sweep_results``, ``baseline_results``,
        ``epsilon_const_list``, ``beta_list``, ``N_MC``,
        ``controller_names``, ``baseline_names``.
    """
    eps_list = exp["sweep"]["epsilon_const"]
    beta_list = exp["sweep"]["beta"]
    N_MC = exp["N_MC"]

    baseline_ctrl = {k: v for k, v in controllers.items()
                     if k in exp["baseline_controllers"]}
    dr_ctrl = {k: v for k, v in controllers.items()
               if k in exp["swept_controllers"]}

    # --- baselines (run once) ---
    print("Running baselines …")
    baseline_results = run_mc_episodes(
        baseline_ctrl, lti, h_fn, f2_fn, N_MC, exp=exp)

    # --- DR sweep over (epsilon_const, beta) grid ---
    sweep_results = {
        name: {eps: {b: None for b in beta_list} for eps in eps_list}
        for name in dr_ctrl
    }

    print("Running DR controllers …")
    for eps in eps_list:
        for beta_val in beta_list:
            for ctrl in dr_ctrl.values():
                ctrl.update_params(epsilon_const=eps, beta=beta_val)
            ep_res = run_mc_episodes(dr_ctrl, lti, h_fn, f2_fn, N_MC, exp=exp)
            for name, res in ep_res.items():
                sweep_results[name][eps][beta_val] = res
            print(f"  eps={eps:.5f}, beta={beta_val:.1f} done")

    _print_constraint_summary(
        baseline_results, sweep_results, eps_list, beta_list)

    payload = {
        "sweep_results":      sweep_results,
        "baseline_results":   baseline_results,
        "epsilon_const_list": eps_list,
        "beta_list":          beta_list,
        "N_MC":               N_MC,
        "controller_names":   list(dr_ctrl.keys()),
        "baseline_names":     list(baseline_ctrl.keys()),
    }
    _save(payload, save_dir, "constraint_sweep_data.pkl")

    if exp.get("plot"):
        for name in dr_ctrl:
            plot_sweep_heatmaps(sweep_results, name,
                                eps_list, beta_list, save_dir)

    return payload


def run_noise_sweep(exp, controllers, lti, h_fn, f2_fn, save_dir):
    """Sweep R_noise levels; all controllers compared at each level.

    Args:
        exp:         Experiment config dict.  Must contain ``R_noise_sweep``
                     and optionally ``innovation_mean`` and ``snr_labels``.
        controllers: Dict {name: controller_instance}.
        lti:         Configured LTISystem instance.
        h_fn:        Constraint function h(y_pred) → constraint vector.
        f2_fn:       Output cost function f2(y_f, t) → scalar.
        save_dir:    Directory to write results and plots to.

    Returns:
        Payload dict with keys ``noise_sweep_results``, ``R_noise_values``,
        ``N_MC``, ``controller_names``, ``innovation_mean``.
    """
    r_noise_list = exp["R_noise_sweep"]
    N_MC = exp["N_MC"]
    innovation_mean = exp.get("innovation_mean", np.array([0.0]))
    names = list(controllers.keys())

    noise_results = {name: {rv: [] for rv in r_noise_list} for name in names}

    for rv in r_noise_list:
        lti.R = rv * np.eye(lti.p)
        lti.innovation_mean = innovation_mean
        print(f"  R_noise = {rv}")
        ep_res = run_mc_episodes(controllers, lti, h_fn, f2_fn, N_MC, exp=exp)
        for name, res in ep_res.items():
            noise_results[name][rv] = res

    _print_noise_summary(noise_results, r_noise_list, names)

    payload = {
        "noise_sweep_results": noise_results,
        "R_noise_values":      r_noise_list,
        "N_MC":                N_MC,
        "controller_names":    names,
        "innovation_mean":     innovation_mean,
    }
    _save(payload, save_dir, "noise_sweep_data.pkl")

    if exp.get("plot"):
        plot_noise_boxplots(
            noise_results, r_noise_list, names,
            snr_labels=exp.get("snr_labels"),
            save_dir=save_dir,
        )

    return payload


def run_cost_comparison(exp, controllers, lti, h_fn, f2_fn, save_dir):
    """Noise sweep with an alternative f2 cost — mirrors run_noise_sweep.

    Args:
        exp:         Experiment config dict.  Must contain ``R_noise_sweep``
                     and ``f2_variant``, and optionally ``snr_labels``.
        controllers: Dict {name: controller_instance}.
        lti:         Configured LTISystem instance.
        h_fn:        Constraint function h(y_pred) → constraint vector.
        f2_fn:       Output cost function f2(y_f, t) → scalar.
        save_dir:    Directory to write results and plots to.

    Returns:
        Payload dict with keys ``noise_sweep_results``, ``R_noise_values``,
        ``N_MC``, ``controller_names``, ``f2_variant``, ``innovation_mean``.
    """
    r_noise_list = exp["R_noise_sweep"]
    N_MC = exp["N_MC"]
    innovation_mean = exp.get("innovation_mean", np.array([0.0]))
    names = list(controllers.keys())

    noise_results = {name: {rv: [] for rv in r_noise_list} for name in names}

    for rv in r_noise_list:
        lti.R = rv * np.eye(lti.p)
        lti.innovation_mean = innovation_mean
        print(f"  R_noise = {rv}")
        ep_res = run_mc_episodes(controllers, lti, h_fn, f2_fn, N_MC, exp=exp)
        for name, res in ep_res.items():
            noise_results[name][rv] = res

    _print_noise_summary(noise_results, r_noise_list, names)

    payload = {
        "noise_sweep_results": noise_results,
        "R_noise_values":      r_noise_list,
        "N_MC":                N_MC,
        "controller_names":    names,
        "f2_variant":          exp["f2_variant"],
        "innovation_mean":     innovation_mean,
    }
    _save(payload, save_dir, f"{exp['save_prefix']}_data.pkl")

    if exp.get("plot"):
        plot_noise_boxplots(
            noise_results, r_noise_list, names,
            snr_labels=exp.get("snr_labels"),
            save_dir=save_dir,
        )

    return payload
