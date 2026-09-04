"""
Empirical check for Remark 1 (CDC2026.tex, CDCRevised/): "h(S)<=R is necessary, not sufficient" and
"the gap between h(S) and the realized MSE is governed by the tightness of the Van Trees bound, which
is asymptotically exact in high-SNR ... regimes."

This is a standalone supplementary check, NOT wired into the paper (no figure reference was added to
CDC2026.tex) -- see Timeline.md for why. It answers two questions about the fixed set S = full ground
set G under the model of sensor_selection_sim.py (y_i = 0.5 x^T M^(i) x + c_i^T x + v_i):

  1. Necessity: does an actual estimator's empirical MSE ever fall BELOW Tr(B_S)? (It shouldn't --
     Proposition 2 says Tr(B_S) lower-bounds every estimator's MSE.)
  2. Asymptotic tightness: does the gap between empirical MSE and Tr(B_S) shrink as measurement noise
     shrinks (SNR grows)?

Estimator used: the MAP estimate under the true (Gaussian prior + Gaussian-noise quadratic-likelihood)
model, computed by directly minimizing the negative log-posterior
    J(x) = 0.5 x^T Ix x + sum_i (1/(2 sigma_i^2)) (y_i - 0.5 x^T M^(i) x - c_i^T x)^2
via Newton-CG with an analytic gradient. This is the best-case estimator for this exact model (not an
EKF/UKF approximation), so it's the right choice for testing the bound itself rather than any particular
filter's suboptimality.
"""

import os
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from sensor_selection_sim import (
    generate_spd, generate_M_stack, generate_C, per_sensor_info,
    F_of, perf_dir,
)

N_STATE = 4
M_SENSORS = 7
SIGMA0_SCALE = 10.0
M_SCALE_RANGE = (1e-2, 1.0)
C_SCALE = 1.0
N_TRIALS = 300
NOISE_SCALES = np.logspace(-2, 4, 13)   # high-SNR (1e-2) -> low-SNR (1e4)


def neg_log_post_and_grad(x, y, M, C, sigma2_vec, Ix):
    val = 0.5 * x @ Ix @ x
    grad = Ix @ x
    for i in range(len(sigma2_vec)):
        Mx = M[i] @ x
        pred = 0.5 * x @ Mx + C[i] @ x
        r = y[i] - pred
        val += (r ** 2) / (2.0 * sigma2_vec[i])
        grad -= (r / sigma2_vec[i]) * (Mx + C[i])
    return val, grad


def linearized_start(y, C, sigma2_vec, Ix):
    """Closed-form MAP estimate under the linearized (quadratic term dropped) model -- used as one of
    several optimizer starting points, since the true posterior is a non-convex quartic in x and a
    single fixed start (e.g. x0=0) can get trapped in a local minimum far from the true optimum."""
    A = Ix.copy()
    b = np.zeros(Ix.shape[0])
    for i in range(len(sigma2_vec)):
        A = A + np.outer(C[i], C[i]) / sigma2_vec[i]
        b = b + C[i] * y[i] / sigma2_vec[i]
    return np.linalg.solve(A, b)


def map_estimate(y, M, C, sigma2_vec, Ix, rng, n_restarts=6):
    """Multi-start Newton-CG: the negative log-posterior is a non-convex quartic in x (quadratic
    measurement model), so a single fixed start can converge to a local, non-global minimum. Try the
    linearized closed-form start, x0=0, and several random perturbations around the linearized start;
    keep whichever converges to the lowest objective value."""
    n = Ix.shape[0]
    x_lin = linearized_start(y, C, sigma2_vec, Ix)
    starts = [x_lin, np.zeros(n)]
    for _ in range(n_restarts - len(starts)):
        starts.append(x_lin + rng.normal(scale=1.0, size=n))

    best_val, best_x = np.inf, None
    for x0 in starts:
        res = minimize(
            neg_log_post_and_grad, x0, args=(y, M, C, sigma2_vec, Ix),
            jac=True, method='Newton-CG',
            options={'xtol': 1e-10, 'maxiter': 200},
        )
        if res.fun < best_val:
            best_val, best_x = res.fun, res.x
    return best_x


def run_noise_level(noise_scale, seed):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    P = np.eye(N_STATE) * SIGMA0_SCALE
    Ix = np.linalg.inv(P)
    M = generate_M_stack(M_SENSORS, N_STATE, *M_SCALE_RANGE)
    C = generate_C(M_SENSORS, N_STATE, C_SCALE)
    sigma2_vec = np.full(M_SENSORS, noise_scale)

    Ii_list, ci_list = per_sensor_info(M, C, P, sigma2_vec)
    F_full = F_of(list(range(M_SENSORS)), Ix, Ii_list)
    B_S = np.linalg.inv(F_full)
    bound = float(np.trace(B_S))

    sq_errors = []
    for t in range(N_TRIALS):
        x_true = rng.multivariate_normal(np.zeros(N_STATE), P)
        y = np.zeros(M_SENSORS)
        for i in range(M_SENSORS):
            v = rng.normal(0.0, np.sqrt(sigma2_vec[i]))
            y[i] = 0.5 * x_true @ M[i] @ x_true + C[i] @ x_true + v
        x_hat = map_estimate(y, M, C, sigma2_vec, Ix, rng)
        sq_errors.append(np.sum((x_hat - x_true) ** 2))

    return bound, np.array(sq_errors)


def run_sweep():
    bounds, all_errors = [], []
    rng = np.random.default_rng(42)
    for noise_scale in tqdm(NOISE_SCALES, desc="MSE-vs-bound sweep"):
        seed = int(rng.integers(0, 2**31 - 1))
        bound, sq_errors = run_noise_level(noise_scale, seed)
        bounds.append(bound)
        all_errors.append(sq_errors)
    return np.array(bounds), all_errors


# A trial counts as a "MAP failure" (converged to a spurious local optimum of the non-convex quartic
# posterior, not a bound violation) if its squared error is a large outlier relative to Tr(B_S) --
# see Timeline.md for the diagnostic that established these are genuine local optima (near-zero
# gradient, lower objective than at x_true), not unconverged Newton-CG iterations.
FAILURE_THRESHOLD = 0.5  # squared-error cutoff in x-units (P has diagonal 10, so this is a generous cutoff)


def make_plot(bounds, all_errors):
    medians = np.array([np.median(e) for e in all_errors])
    lo = np.array([np.percentile(e, 25) for e in all_errors])
    hi = np.array([np.percentile(e, 75) for e in all_errors])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    ax.plot(NOISE_SCALES, bounds, color='#ff7f0e', marker='^', label=r'Van Trees bound $\mathrm{Tr}(B_S)$')
    ax.plot(NOISE_SCALES, medians, color='#1f77b4', marker='o', label='Median empirical SE (MAP estimator)')
    ax.fill_between(NOISE_SCALES, lo, hi, color='#1f77b4', alpha=0.15, label='IQR (25th-75th pct.)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Measurement-noise scale $\sigma^2$')
    ax.set_ylabel('Squared error')
    ax.set_title('Bound vs. achieved error (median), S = G', fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ratio = medians / bounds
    ax.plot(NOISE_SCALES, ratio, color='#2ca02c', marker='s')
    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Measurement-noise scale $\sigma^2$')
    ax.set_ylabel('Median SE / Tr(B_S)')
    ax.set_title('Tightness ratio (1.0 = exact)', fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(f'Remark 1 check (CDCRevised/, N={N_TRIALS}/point, MAP estimator, S=G, median-based)')
    fig.tight_layout()
    fig.savefig(perf_dir + 'remark1_mse_vs_bound.png', dpi=150)
    plt.close(fig)
    print(f"Saved {perf_dir}remark1_mse_vs_bound.png")


if __name__ == "__main__":
    bounds, all_errors = run_sweep()
    make_plot(bounds, all_errors)

    medians = np.array([np.median(e) for e in all_errors])
    means = np.array([np.mean(e) for e in all_errors])
    fail_frac = np.array([np.mean(e > FAILURE_THRESHOLD) for e in all_errors])
    violations = int(np.sum(medians < bounds))
    print(f"\nNecessity check (median): median SE < Tr(B_S) in {violations}/{len(NOISE_SCALES)} noise levels "
          f"(should be 0).")
    print("noise_scale, bound, median_SE, mean_SE, median/bound, MAP_failure_frac(SE>0.5)")
    for ns, b, med, mn, ff in zip(NOISE_SCALES, bounds, medians, means, fail_frac):
        print(f"{ns:.4g}, {b:.6g}, {med:.6g}, {mn:.6g}, {med/b:.4f}, {ff:.3f}")
