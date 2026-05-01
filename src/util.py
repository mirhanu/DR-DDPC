# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns

# ===========================================================================
# Data generation
# ===========================================================================


def generate_hankel_datasets(
    dyn_system,
    init_state: np.ndarray,
    T_ini: int,
    T_f: int,
    T_sim: int,
    control_law,
) -> tuple:
    """Simulate a system and split the resulting Hankel matrices into past/future blocks.

    Args:
        dyn_system: A DynamicSystem instance with attributes p, m, n, A, B, C, D.
        init_state: Initial state vector, shape (n,).
        T_ini: Past horizon length (number of block rows for u_p, y_p).
        T_f: Future horizon length (number of block rows for u_f, y_f).
        T_sim: Number of simulation steps.
        control_law: Callable used to generate inputs during simulation.

    Returns:
        Tuple (U_p, U_f, Y_p, Y_f), each a 2-D numpy array:
          - U_p: Past input Hankel block,   shape (m*T_ini, N_col).
          - U_f: Future input Hankel block, shape (m*T_f,   N_col).
          - Y_p: Past output Hankel block,  shape (p*T_ini, N_col).
          - Y_f: Future output Hankel block, shape (p*T_f,  N_col).
    """
    T_total = T_ini + T_f
    p, m = dyn_system.p, dyn_system.m

    dyn_system.state = init_state
    dyn_system.t = 0

    outputs, states, controls = dyn_system.simulate(
        control_law=control_law, T=T_sim * dyn_system.dt
    )

    # Build and split Hankel matrices
    Y_h = hankel_matrix(outputs,  T_total)
    U_h = hankel_matrix(controls, T_total)

    Y_p, Y_f = Y_h[:p * T_ini, :], Y_h[p * T_ini:, :]
    U_p, U_f = U_h[:m * T_ini, :], U_h[m * T_ini:, :]

    return U_p, U_f, Y_p, Y_f


def hankel_matrix(X: np.ndarray, s: int) -> np.ndarray:
    """Construct a block Hankel matrix from a multivariate time series.

    Args:
        X: Data array of shape (d, N), where d is the signal dimension
            and N is the number of time steps.
        s: Number of block rows (Hankel depth).

    Returns:
        Block Hankel matrix of shape (d*s, N-s+1).

    Raises:
        ValueError: If s > N.
    """
    X = np.atleast_2d(X)
    d, N = X.shape
    if s > N:
        raise ValueError(
            f"Hankel depth s={s} cannot exceed number of time steps N={N}.")

    cols = N - s + 1
    H = np.zeros((d * s, cols))
    for i in range(s):
        H[i * d:(i + 1) * d, :] = X[:, i:i + cols]
    return H


def random_control_law(dyn_sys):
    return np.random.randn(dyn_sys.m)


# ===========================================================================
# Plotting
# ===========================================================================


def plot_io_data(
    y: np.ndarray,
    u: np.ndarray,
    y_ref=None,
    u_ref=None,
    t: np.ndarray = None,
):
    """Plot output and input trajectories with optional references.

    Args:
        y: Output data, shape (p, N).
        u: Input data, shape (m, N).
        y_ref: Optional output reference. Can be a scalar (same for all
            outputs), shape (p,) for a constant per output, or shape (p, N)
            for a time-varying reference.
        u_ref: Optional input reference, shape (m,) constant per input.
        t: Time vector, shape (N,). Defaults to 0, 1, ..., N-1.
    """
    y = np.atleast_2d(y)
    u = np.atleast_2d(u)
    p, N = y.shape
    m, _ = u.shape

    if t is None:
        t = np.arange(N)

    fig, axes = plt.subplots(p + m, 1, sharex=True, figsize=(8, 2 * (p + m)))

    for i in range(p):
        axes[i].plot(t, y[i], label=f"y[{i}]")
        if y_ref is not None:
            axes[i].plot(t, np.asarray(y_ref[i]).flatten(),
                         "--", label=f"y_ref[{i}]")
        axes[i].set_ylabel(f"y[{i}]")
        axes[i].legend()
        axes[i].grid(True)

    for j in range(m):
        axes[p + j].plot(t, u[j], label=f"u[{j}]")
        if u_ref is not None:
            axes[p + j].plot(t, np.full_like(t, u_ref[j]),
                             "--", label=f"u_ref[{j}]")
        axes[p + j].set_ylabel(f"u[{j}]")
        axes[p + j].legend()
        axes[p + j].grid(True)

    axes[-1].set_xlabel("Time")
    plt.tight_layout()
    plt.show()


def plot_sweep_heatmaps(sweep_results, ctrl_name, eps_list, beta_list, save_dir="."):
    """Heat map of mean violation rate over (epsilon_const, beta) grid."""
    data_viol = np.zeros((len(eps_list), len(beta_list)))
    data_cost = np.zeros_like(data_viol)
    for i, eps in enumerate(eps_list):
        for j, b in enumerate(beta_list):
            data_viol[i, j] = np.mean(
                [e["viol_rate"] for e in sweep_results[ctrl_name][eps][b]]) * 100
            data_cost[i, j] = np.mean(
                [e["cost"] for e in sweep_results[ctrl_name][eps][b]])

    plt.rcParams.update({
        "font.size": 28,
        "axes.titlesize": 30,
        "axes.labelsize": 30,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
    })

    # ---- Violation heatmap ----
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        data_viol,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=beta_list,
        yticklabels=eps_list,
        annot_kws={"size": 22},
        cbar_kws={"shrink": 0.8}
    )
    plt.xlabel(r"$\beta$", labelpad=12)
    plt.ylabel(r"$\varepsilon_{\mathrm{con}}$", labelpad=12)
    plt.tight_layout()
    path = os.path.join(save_dir, f"violation_rate_{ctrl_name}.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    # ---- Cost heatmap ----
    plt.figure(figsize=(12, 9))
    sns.heatmap(
        data_cost,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=beta_list,
        yticklabels=eps_list,
        annot_kws={"size": 22},
        cbar_kws={"shrink": 0.8}
    )
    plt.xlabel(r"$\beta$", labelpad=12)
    plt.ylabel(r"$\varepsilon_{\mathrm{con}}$", labelpad=12)
    plt.tight_layout()
    path = os.path.join(save_dir, f"cost_{ctrl_name}.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap → {path}")


def plot_noise_boxplots(noise_results, r_noise_list, names, display_names=None,
                        save_dir=".", snr_labels=None,):
    """
    Grouped box plots of cost distribution across noise levels.

    Parameters
    ----------
    display_names : dict, optional
        Mapping from internal name to legend label,
        e.g. {"DR-SPC": "DR-DDPC", "DeePC": "Reg-DeePC"}.
        Defaults to identity.
    snr_labels : list of str, optional
        X-tick labels per noise level, e.g. ["10", "20", "40"] dB.
        Defaults to sqrt(R) values.
    """
    if display_names is None:
        display_names = {n: n for n in names}
    if snr_labels is None:
        snr_labels = [f"sqrt(R)={np.sqrt(rv):.4f}" for rv in r_noise_list]

    colors = ['steelblue', 'tomato', 'seagreen', 'orange']
    n_ctrl = len(names)
    width = 0.8 / n_ctrl
    x_base = np.arange(len(r_noise_list))

    fig, ax = plt.subplots(figsize=(7, 4))

    for i, (name, color) in enumerate(zip(names, colors)):
        offsets = x_base + (i - (n_ctrl - 1) / 2.0) * width
        for j, rv in enumerate(r_noise_list):
            data = [res["cost"] for res in noise_results[name][rv]]
            ax.boxplot(
                data,
                positions=[offsets[j]],
                widths=width * 0.9,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.7),
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                flierprops=dict(marker='o', color=color,
                                markersize=3, alpha=0.4),
                manage_ticks=False,
                showfliers=False,
            )

    # Legend — uncomment to enable
    # legend_elements = [Patch(facecolor=c, alpha=0.7, label=display_names[n])
    #                    for n, c in zip(names, colors)]
    # ax.legend(handles=legend_elements)

    ax.set_xticks(x_base)
    ax.set_xticklabels(snr_labels)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(r"$J_{\mathrm{test}}$")
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    path = os.path.join(save_dir, "noise_sweep_plot.pdf")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved boxplot -> {path}")


# =============================================================================
# I/O HELPERS
# =============================================================================

def _print_constraint_summary(baseline_results, sweep_results, eps_list, beta_list):
    print(f"\n{'Controller':<12} {'Mean Viol%':>12} {'Std Viol%':>12} {'Mean Cost':>12} {'Std Cost':>12}")
    print("-" * 62)
    for name, episodes in baseline_results.items():
        rates = [e["viol_rate"] for e in episodes]
        costs = [e["cost"] for e in episodes]
        print(f"{name:<12} {100*np.mean(rates):>11.2f}% {100*np.std(rates):>11.2f}%"
              f" {np.mean(costs):>12.4f} {np.std(costs):>12.4f}")

    print(f"\n{'Controller':<12} {'eps_const':>10} {'beta':>6} {'Mean Cost':>12} {'Std Cost':>12} {'Mean Viol%':>12} {'Std Viol%':>12}")
    print("-" * 56)
    for name in sweep_results:
        for eps in eps_list:
            for beta_val in beta_list:
                rates = [e["viol_rate"]
                         for e in sweep_results[name][eps][beta_val]]
                costs = [e["cost"]
                         for e in sweep_results[name][eps][beta_val]]
                print(f"{name:<12} {eps:>10.5f} {beta_val:>6.2f}"
                      f"{np.mean(costs):>12.4f} {np.std(costs):>12.4f}"
                      f" {100*np.mean(rates):>11.2f}% {100*np.std(rates):>11.2f}%")


def _print_noise_summary(noise_results, r_noise_list, names):
    print(f"\n{'Controller':<12} {'R_noise':>10} {'Mean Cost':>12} {'Std Cost':>12}")
    print("-" * 50)
    for name in names:
        for rv in r_noise_list:
            costs = [e["cost"] for e in noise_results[name][rv]]
            print(
                f"{name:<12} {rv:>10.7f} {np.mean(costs):>12.4f} {np.std(costs):>12.4f}")


def _save(payload, directory, filename):
    path = os.path.join(directory, filename)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved results → {path}")
