# -*- coding: utf-8 -*-
"""
mc_runner.py
============
System construction, controller instantiation, and Monte Carlo simulation.

Contents
--------
  build_lti            — Instantiate the LTI system with given noise settings.
  build_controllers    — Instantiate all controllers listed in an experiment config.
  run_mc_episodes      — Run N_MC episodes for a set of controllers and return
                         per-episode trajectories, costs, and violation rates.
"""

import numpy as np
# --- project modules ---
from config.config import (
    A, B, C, D, K,
    n, p,
    T_P, T_F, T_RUN, T_OFFLINE,
    EPSILON1, EPSILON2, EPSILON_CONST, BETA, NORM_TYPE,
    LAMBDA_G, Q_PERF,
    N_OBJ_RES, N_CONST_RES,
    make_u_bounds,
    compute_lipschitz,
)
from src.cost_functions import (
    f1,
    compute_cost, compute_violation_rate,
)
from src.util import (
    random_control_law,
    generate_hankel_datasets,
)

# --- controller imports ---
from src.controllers.reg_deepc import DeepCController
from src.controllers.dr_ddpc import DRDDPCController
from src.controllers.dr_spc import DRSPCController
from src.controllers.spc import SPCController
from src.controllers.lti import LTISystem

# =============================================================================
# SYSTEM
# =============================================================================


def build_lti(R_noise, innovation_mean=None):
    """Instantiate the LTI system with given noise covariance and mean.

    Args:
        R_noise:          Noise covariance — scalar (multiplied by I) or (p, p) matrix.
        innovation_mean:  Optional mean vector for the innovation noise, shape (p,).

    Returns:
        Configured LTISystem instance.
    """
    if np.isscalar(R_noise):
        R_noise = R_noise * np.eye(p)
    lti = LTISystem(A, B, C, D, R=R_noise, K=K, discrete=True, dt=1)
    if innovation_mean is not None:
        lti.innovation_mean = innovation_mean
    return lti


# =============================================================================
# CONTROLLERS
# =============================================================================


def build_controllers(exp, h_fn, f2_fn):
    """Instantiate all controllers listed in the experiment config.

    Args:
        exp:   Experiment config dict (from experiments.py).
        h_fn:  Constraint function  h(y_pred) → constraint vector.
        f2_fn: Output cost function f2(y_f, t) → scalar.

    Returns:
        Dict  {controller_name: controller_instance}.

    """
    U_bounds = make_u_bounds(T_f=T_F)
    h_dim = 2 * T_F * p

    L_f2_exp, L_con_exp = compute_lipschitz(
        T_f=T_F, p=p, Q_perf=Q_PERF,
        y_min=exp["y_min"], y_max=exp["y_max"], r=NORM_TYPE,
    )

    controllers = {}
    for name in exp["swept_controllers"] + exp["baseline_controllers"]:

        if name == "SPC":
            controllers[name] = SPCController(
                T_P, f1, f2_fn, h_fn,
                H_up=None, H_uf=None, H_yf=None, H_yp=None,
                past_y=None, past_u=None,
                U_bounds=U_bounds,
            )

        elif name == "Reg-DeePC":
            controllers[name] = DeepCController(
                T_P, f1, f2_fn, h_fn,
                H_up=None, H_uf=None, H_yf=None, H_yp=None,
                lambda_g=LAMBDA_G,
                past_y=None, past_u=None,
                U_bounds=U_bounds,
            )

        elif name == "DR-SPC":
            controllers[name] = DRSPCController(
                T_P, f1, f2_fn, h_fn,
                H_up=None, H_uf=None, H_yf=None, H_yp=None,
                h_dim=h_dim,
                Lobj=L_f2_exp, Lcon=L_con_exp,
                beta=BETA,
                epsilon1=EPSILON1, epsilon2=EPSILON2,
                epsilon_const=EPSILON_CONST,
                N_obj_res=N_OBJ_RES, N_const_res=N_CONST_RES,
                norm_type=NORM_TYPE,
                past_y=None, past_u=None,
                U_bounds=U_bounds,
            )

        elif name == "DR-DDPC":
            controllers[name] = DRDDPCController(
                T_P, f1, f2_fn, h_fn,
                H_up=None, H_uf=None, H_yf=None, H_yp=None,
                h_dim=h_dim,
                Lobj=L_f2_exp, Lcon=L_con_exp,
                beta=BETA,
                epsilon1=EPSILON1, epsilon2=EPSILON2,
                epsilon_const=EPSILON_CONST,
                N_obj_res=N_OBJ_RES, N_const_res=N_CONST_RES,
                norm_type=NORM_TYPE,
                past_y=None, past_u=None,
                U_bounds=U_bounds,
            )

        else:
            raise ValueError(f"Unknown controller name: '{name}'")

    return controllers


# =============================================================================
# MONTE CARLO RUNNER
# =============================================================================


def run_mc_episodes(controllers, lti, h_fn, f2_fn, N_MC, exp=None):
    """Run N_MC Monte Carlo episodes for every controller in the dict.


    Args:
        controllers: Dict  {name: controller_instance}.
        lti:         LTISystem instance (its state is overwritten each episode).
        h_fn:        Constraint function  h(y_pred) 
        f2_fn:       Output cost function f2(y_f, t) 
        N_MC:        Number of Monte Carlo episodes.
        exp:         Experiment config dict.  Used to read ``init_state_range``
                     (defaults to (-1, 1) if absent).

    Returns:
        Dict  {controller_name: [episode_dict, ...]}, where each episode_dict
        contains keys ``"y"``, ``"u"``, ``"x"``, ``"cost"``, ``"viol_rate"``.
    """
    results = {name: [] for name in controllers}

    for episode in range(N_MC):
        np.random.seed(episode)

        # --- offline data collection ---
        init_state_offline = np.ones(n)
        H_up, H_uf, H_yp, H_yf = generate_hankel_datasets(
            lti, init_state_offline, T_P, T_F, T_OFFLINE, random_control_law
        )

        # --- initial online window ---
        lo, hi = (exp or {}).get("init_state_range", (-1, 1))
        init_state_on = np.random.uniform(lo, hi, size=n)
        lti.state = init_state_on
        y_ini_arr, _, u_ini_arr = lti.simulate(
            random_control_law, T=T_P * lti.dt, t0=0.0
        )
        y_ini_list = [y_ini_arr[:, k] for k in range(y_ini_arr.shape[1])]
        u_ini_list = [u_ini_arr[:, k] for k in range(u_ini_arr.shape[1])]
        x0, y0, t0 = lti.state, lti.output, lti.t

        # --- simulate each controller from the same initial condition ---
        for name, ctrl in controllers.items():
            ctrl.reset(y_ini_list, u_ini_list)
            ctrl.update_params(H_up=H_up, H_uf=H_uf, H_yf=H_yf, H_yp=H_yp)

            np.random.seed(episode)
            lti.state = x0
            lti.output = y0
            lti.t = t0
            y, x, u = lti.simulate(ctrl, T=T_RUN)

            results[name].append({
                "y": y, "u": u, "x": x,
                "cost":      compute_cost(y, u, t0, f1, f2_fn),
                "viol_rate": compute_violation_rate(y, h_fn),
            })

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{N_MC} done")

    return results
