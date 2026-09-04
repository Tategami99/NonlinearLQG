"""
Regenerates Fig. 1 (utilization ratio vs. target-accuracy ratio) and Fig. 2 (empirical supermodularity
ratio vs. theoretical bounds, under C-scale and noise-scale sweeps) for the CDC/ variant (the in-place
per-j F_tilde_{G,j} fix to Theorem 2).

Parameters follow CDC2026.tex Section V as closely as the text specifies; where the paper doesn't pin
down a value (e.g. Fig. 1's C-scale, which the paper never states), a reasonable default is used and
noted below.

Presentation styling (fonts, colors, titles/labels aimed at a reader unfamiliar with the paper's
notation) follows the redesign documented in Timeline.md's "figure redesign for supervisor review"
entry -- colors are the CVD-validated palette from sensor_selection_sim.py's COLORS dict; see that
file's comment for the validation method (Claude's dataviz skill / scripts/validate_palette.js).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from sensor_selection_sim import (
    generate_spd, generate_M_stack, generate_C, per_sensor_info,
    greedy_select, greedy_linearized, brute_force_select,
    theorem2_bound, prior_bound_c19, empirical_gamma_h, F_of,
    perf_dir, COLORS,
)

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 17,
    'axes.linewidth': 1.0,
    'lines.linewidth': 2.4,
    'lines.markersize': 8,
})

N_FIG1 = 300          # paper's stated sample size (CDC2026.tex Sec. V: "300 simulations")
N_FIG2 = 300          # paper doesn't state Fig. 2's N explicitly; matched to Fig. 1's for consistency
N_STATE = 4
M_SENSORS = 7
SIGMA0_SCALE = 10.0
M_SCALE_RANGE = (1e-2, 1.0)
NOISE_SCALE_FIG1 = 1e2
C_SCALE_FIG1 = 1.0    # not specified by the paper for Fig. 1; chosen as a moderate default


def run_one_trial_fig1(R_ratio, seed):
    np.random.seed(seed)
    P = np.eye(N_STATE) * SIGMA0_SCALE
    Ix = np.linalg.inv(P)
    M = generate_M_stack(M_SENSORS, N_STATE, *M_SCALE_RANGE)
    C = generate_C(M_SENSORS, N_STATE, C_SCALE_FIG1)
    sigma2_vec = np.full(M_SENSORS, NOISE_SCALE_FIG1)
    Ii_list, ci_list = per_sensor_info(M, C, P, sigma2_vec)

    h0 = float(np.trace(P))
    R = h0 * R_ratio

    S_star, _ = brute_force_select(M_SENSORS, Ix, Ii_list, R)
    if len(S_star) == 0:
        return None  # R_ratio too loose to need any sensors; skip (ratio undefined)

    S_quad = greedy_select(M_SENSORS, Ix, Ii_list, R)
    S_lin = greedy_linearized(M_SENSORS, Ix, ci_list, sigma2_vec, R)

    return len(S_quad) / len(S_star), len(S_lin) / len(S_star)


def generate_fig1():
    R_ratios = np.logspace(-3, 0, 13)
    quad_mean, quad_std, lin_mean, lin_std = [], [], [], []
    rng = np.random.default_rng(0)

    for R_ratio in tqdm(R_ratios, desc="Fig 1 sweep"):
        quad_ratios, lin_ratios = [], []
        for t in range(N_FIG1):
            seed = int(rng.integers(0, 2**31 - 1))
            result = run_one_trial_fig1(R_ratio, seed)
            if result is None:
                continue
            qr, lr = result
            quad_ratios.append(qr)
            lin_ratios.append(lr)
        quad_mean.append(np.mean(quad_ratios) if quad_ratios else np.nan)
        quad_std.append(np.std(quad_ratios) if quad_ratios else 0.0)
        lin_mean.append(np.mean(lin_ratios) if lin_ratios else np.nan)
        lin_std.append(np.std(lin_ratios) if lin_ratios else 0.0)

    quad_mean, quad_std = np.array(quad_mean), np.array(quad_std)
    lin_mean, lin_std = np.array(lin_mean), np.array(lin_std)

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.axhline(1.0, color=COLORS['brute'], linestyle='--', linewidth=2,
               label='Optimal (brute force)', zorder=1)
    ax.plot(R_ratios, lin_mean, color=COLORS['lin'], marker='s',
            label='Baseline: linearized greedy [9]', zorder=3)
    ax.fill_between(R_ratios, lin_mean - lin_std, lin_mean + lin_std, color=COLORS['lin'], alpha=0.15)
    ax.plot(R_ratios, quad_mean, color=COLORS['quad'], marker='o',
            label='Proposed: quadratic-aware greedy', zorder=4)
    ax.fill_between(R_ratios, quad_mean - quad_std, quad_mean + quad_std, color=COLORS['quad'], alpha=0.15)
    ax.set_xscale('log')
    ax.set_xlabel(r'Target accuracy ratio $R_{\mathrm{ratio}}$  (smaller = stricter accuracy requirement)')
    ax.set_ylabel(r'Sensors used vs. optimal, $\ell/|S^\star|$  (1.0 = optimal)')
    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(alpha=0.3)
    fig.suptitle('Fewer sensors needed with quadratic-aware selection', y=0.98, fontsize=16)
    ax.set_title(f'N={N_FIG1} trials/point, shaded band = ±1 std. dev.', fontsize=11, color='#555555', pad=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(perf_dir + 'fig1.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig1.png")
    return R_ratios, quad_mean, lin_mean


def run_one_trial_fig2(C_scale, noise_scale, seed):
    np.random.seed(seed)
    P = np.eye(N_STATE) * SIGMA0_SCALE
    Ix = np.linalg.inv(P)
    M = generate_M_stack(M_SENSORS, N_STATE, *M_SCALE_RANGE)
    C = generate_C(M_SENSORS, N_STATE, C_scale)
    sigma2_vec = np.full(M_SENSORS, noise_scale)
    Ii_list, ci_list = per_sensor_info(M, C, P, sigma2_vec)

    g_true = empirical_gamma_h(M_SENSORS, Ix, Ii_list)
    g_bound = theorem2_bound(M_SENSORS, Ix, Ii_list, ci_list, M, sigma2_vec)
    F_full = F_of(list(range(M_SENSORS)), Ix, Ii_list)
    g_prior = prior_bound_c19(M_SENSORS, Ix, P, sigma2_vec, F_full)
    return g_true, g_bound, g_prior


def sweep_panel(param_name, param_values, fixed_C_scale, fixed_noise_scale, rng):
    emp_med, emp_lo, emp_hi = [], [], []
    bound_med, prior_med = [], []
    for val in tqdm(param_values, desc=f"Fig 2 sweep ({param_name})"):
        C_scale = val if param_name == 'C_scale' else fixed_C_scale
        noise_scale = val if param_name == 'noise_scale' else fixed_noise_scale
        trues, bounds, priors = [], [], []
        for t in range(N_FIG2):
            seed = int(rng.integers(0, 2**31 - 1))
            g_true, g_bound, g_prior = run_one_trial_fig2(C_scale, noise_scale, seed)
            trues.append(g_true)
            bounds.append(g_bound)
            if np.isfinite(g_prior):
                priors.append(g_prior)
        trues = np.array(trues)
        emp_med.append(np.median(trues))
        emp_lo.append(np.percentile(trues, 25))
        emp_hi.append(np.percentile(trues, 75))
        bound_med.append(np.median(bounds))
        prior_med.append(np.median(priors) if priors else np.nan)
    return (np.array(emp_med), np.array(emp_lo), np.array(emp_hi),
            np.array(bound_med), np.array(prior_med))


def generate_fig2():
    R_RATIO = 0.01  # fixed for Fig. 2, per the paper
    c_scales = np.logspace(-2, 1, 10)
    noise_scales = np.logspace(0, 4, 10)
    rng = np.random.default_rng(1)

    left = sweep_panel('C_scale', c_scales, fixed_C_scale=None, fixed_noise_scale=NOISE_SCALE_FIG1, rng=rng)
    right = sweep_panel('noise_scale', noise_scales, fixed_C_scale=10.0, fixed_noise_scale=None, rng=rng)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.3))
    for ax, xvals, data, xlabel, title in [
        (axes[0], c_scales, left, 'Linear-term magnitude (C scale)',
         'Varying the linear measurement term'),
        (axes[1], noise_scales, right, r'Measurement-noise scale ($\sigma^2$)',
         'Varying the measurement noise'),
    ]:
        emp_med, emp_lo, emp_hi, bound_med, prior_med = data
        ax.plot(xvals, emp_med, color=COLORS['empirical'], marker='o', linewidth=2.6,
                 label='True ratio (exhaustive computation)', zorder=4)
        ax.fill_between(xvals, emp_lo, emp_hi, color=COLORS['empirical'], alpha=0.12)
        ax.plot(xvals, bound_med, color=COLORS['thm2'], marker='^',
                 label='Proposed guarantee (Theorem 2)', zorder=3)
        ax.plot(xvals, prior_med, color=COLORS['prior'], marker='s', linestyle='--',
                 label="Prior-work guarantee [19]", zorder=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'Supermodularity ratio $\gamma_h$' + '\n(higher = tighter guarantee)')
        ax.set_title(title, fontsize=13)
        ax.legend(loc='lower right', framealpha=0.95, fontsize=10.5)
        ax.grid(alpha=0.3)
    fig.suptitle('How tight is the theoretical guarantee? (Per-sensor bound)', y=1.0)
    fig.text(0.5, 0.925,
              f'All three curves are lower bounds on how close greedy selection gets to optimal -- '
              f'higher is a tighter, more useful guarantee. N={N_FIG2} trials/point, '
              rf'$R_\mathrm{{ratio}}={R_RATIO}$.',
              ha='center', fontsize=10.5, color='#555555')
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(perf_dir + 'fig2.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2.png")
    return c_scales, noise_scales, left, right


if __name__ == "__main__":
    generate_fig1()
    generate_fig2()
