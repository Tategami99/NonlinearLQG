"""
TEST FIGURE -- NOT part of the official 10-figure set in `fig2_empirical_coverage.py`. Do not add this to
that file's __main__ list or docstring; it exists only to show the supervisor a variant that was tried
and rejected, with the actual numbers, rather than describing the rejection secondhand.

The supervisor asked for the emp7 top panel to be rebuilt as the SAME construction as the bottom panel
("predicted / true" ratio, same style) but evaluated in this paper's domain instead of [19]'s, expecting
it to stay flat / not degrade as C grows the way the bottom panel does.

The literal, most obvious way to build that: treat Theorem 2's bound as the "predicted" value and the
brute-force truth as the "true" value -- exactly mirroring the bottom panel's `approx_gain / true_gain`,
just swapping in `theorem2_bound(...)` and `empirical_gamma_h(...)`. This script builds exactly that and
nothing else, using the SAME C_scales, SAME n_trials default, and SAME rng seed as `fig2_emp7_breakdown_vs_C`
in fig2_empirical_coverage.py, so it's an apples-to-apples comparison against the real emp7's bottom panel.

Result (see the saved PNG and the printed ratio table): this ratio does NOT stay flat. It starts small
(~6.7e-5 at C=0) and shrinks by roughly 50x, down to ~1.3e-6 by C=3.0. That is real -- Theorem 2's bound
gets progressively LOOSER as C grows -- but it is a different phenomenon than what the bottom panel shows:
[19]'s bottom-panel ratio degrades because their model has no way to represent c_j at all (a
representational/model-mismatch failure). This ratio degrades because Theorem 2's proof is more
conservative at large C (a proof-tightness issue) -- the bound is still valid (never exceeds the true
value; Theorem 1's downstream guarantee never breaks, see fig2_emp4/emp5) but it stops being a NUMBER
that reads like an "accuracy" the way the bottom panel's O(1)-scaled ratio does. Plotting it side-by-side
with the bottom panel on the same axes would visually read as "ours degrades too," which is misleading
given what's actually happening -- this is why the recommendation was to build the top panel differently
(the exact-by-construction ratio-of-1.0 version) rather than this one.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from sensor_selection_sim import (
    generate_C, per_sensor_info, theorem2_bound, empirical_gamma_h, perf_dir, COLORS,
)
from fig2_reframe import add_caption, restrict_to_rank1_zero_c, marginal_gain_first_sensor
from fig2_empirical_coverage import N_STATE, M_SENSORS, P_SCALE, SIGMA2, make_M_stack_with_rank

n_trials = 150
rng = np.random.default_rng(107)
C_scales = np.array([0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0])

# ---- "Naive mirror" top panel: predicted = Theorem 2's bound, true = brute-force gamma_h ----
naive_med, naive_lo, naive_hi = [], [], []
for C_scale in tqdm(C_scales, desc="naive-top-panel"):
    ratios = []
    for _ in range(n_trials):
        P = np.eye(N_STATE) * P_SCALE
        Ix = np.linalg.inv(P)
        M = make_M_stack_with_rank(M_SENSORS, N_STATE, N_STATE, 1e-2, 1.0, rng)  # full rank always
        C = np.zeros((M_SENSORS, N_STATE)) if C_scale == 0 else generate_C(M_SENSORS, N_STATE, C_scale)
        sigma2_vec = np.full(M_SENSORS, SIGMA2)
        Ii_list, ci_list = per_sensor_info(M, C, P, sigma2_vec)
        g_true = empirical_gamma_h(M_SENSORS, Ix, Ii_list)
        g_bound = theorem2_bound(M_SENSORS, Ix, Ii_list, ci_list, M, sigma2_vec)
        ratios.append(g_bound / g_true)
    naive_med.append(np.median(ratios)); naive_lo.append(np.percentile(ratios, 25)); naive_hi.append(np.percentile(ratios, 75))
naive_med, naive_lo, naive_hi = np.array(naive_med), np.array(naive_lo), np.array(naive_hi)

print("\nNaive top-panel ratio (Theorem 2 bound / true gamma_h), same construction as bottom panel:")
for c, m in zip(C_scales, naive_med):
    print(f"  C={c:.2f}  ratio(predicted/true)={m:.6g}")

# ---- Bottom panel: unchanged from the real fig2_emp7 (reproduced here for a fair side-by-side) ----
pred_med, pred_lo, pred_hi = [], [], []
for C_scale in tqdm(C_scales, desc="bottom-panel-repro"):
    ratios = []
    for _ in range(n_trials):
        P = np.eye(N_STATE) * P_SCALE
        Q, _ = np.linalg.qr(rng.normal(size=(N_STATE, N_STATE)))
        Mj = Q @ np.diag([1.0] + [0.0] * (N_STATE - 1)) @ Q.T
        if C_scale > 0:
            direction = rng.normal(size=(N_STATE, 1))
            cj = direction / np.linalg.norm(direction) * C_scale
        else:
            cj = np.zeros((N_STATE, 1))
        Mj_approx = restrict_to_rank1_zero_c(Mj)
        true_gain = marginal_gain_first_sensor(Mj, cj, P, SIGMA2)
        approx_gain = marginal_gain_first_sensor(Mj_approx, np.zeros((N_STATE, 1)), P, SIGMA2)
        ratios.append(approx_gain / true_gain)
    pred_med.append(np.median(ratios)); pred_lo.append(np.percentile(ratios, 25)); pred_hi.append(np.percentile(ratios, 75))
pred_med, pred_lo, pred_hi = np.array(pred_med), np.array(pred_lo), np.array(pred_hi)

# ---- Plot: two panels, SEPARATE y-scales (log top / linear bottom) so neither is visually distorted ----
fig, axes = plt.subplots(2, 1, figsize=(9.5, 12.8))

ax = axes[0]
ax.axhline(1.0, color=COLORS['brute'], linestyle='--', linewidth=2, label='Ratio = 1 (bound = true)', zorder=2)
ax.plot(C_scales, naive_med, color=COLORS['thm2'], marker='^', zorder=3,
        label='Theorem 2 bound / true $\\gamma_h$ (naive mirror of bottom panel)')
ax.fill_between(C_scales, naive_lo, naive_hi, color=COLORS['thm2'], alpha=0.15)
ax.set_yscale('log')
ax.set_xlabel(r'Linear-term magnitude $C$', fontsize=12.5)
ax.set_ylabel('Theorem 2 bound / true $\\gamma_h$\n(NOT an accuracy -- see caption)', fontsize=11)
ax.set_title('TEST: naive mirror -- this ratio SHRINKS ~50x as $C$ grows (rejected variant)', fontsize=12.5, color='#8B0000')
ax.grid(alpha=0.3)
ax.legend(loc='upper right', fontsize=9.5, framealpha=0.95)

ax = axes[1]
ax.axhline(1.0, color=COLORS['brute'], linestyle='--', linewidth=2, label='Perfect representation (exact at $C=0$)', zorder=2)
ax.plot(C_scales, pred_med, color=COLORS['prior'], marker='s', zorder=3,
        label="[19]'s rank-1/zero-$C$ prediction of the sensor's own true value")
ax.fill_between(C_scales, pred_lo, pred_hi, color=COLORS['prior'], alpha=0.15)
ax.set_xlabel(r'Linear-term magnitude $C$ ($M^{(i)}$ held at rank-1: their own case)', fontsize=12.5)
ax.set_ylabel(r'$\Delta_j(\varnothing)$ predicted / $\Delta_j(\varnothing)$ true', fontsize=11)
ax.set_title("Real emp7 bottom panel, reproduced for comparison (unchanged)", fontsize=12.5)
ax.grid(alpha=0.3)
ax.legend(loc='upper right', fontsize=9.5, framealpha=0.95)

fig.suptitle('TEST FIGURE: why the naive top-panel mirror was rejected', y=0.975, fontsize=14.5)
fig.tight_layout(rect=[0, 0.13, 1, 0.94])
add_caption(
    fig, 'Test',
    "Literal mirror of the bottom panel's construction, using Theorem 2's bound as \"predicted\": the ratio "
    "does not stay flat, it shrinks ~50x from C=0 to C=3 -- for a different reason than the bottom panel.",
    f"Median over N={n_trials} trials/point (bands = IQR). Top panel uses a log y-axis because the ratio "
    "is naturally tiny throughout (Theorem 2's bound is a very conservative lower bound on $\\gamma_h$ by "
    "construction -- see emp4/emp5), NOT because it approaches an \"accuracy\" the way the bottom panel's "
    "O(1)-scaled ratio does. The two panels are answering different questions: bottom = does [19]'s "
    "restricted model correctly represent a sensor whose real $c_j\\neq 0$ (a model-mismatch question, "
    "meaningfully read as 0-100% accuracy); top = how tight is Theorem 2's bound relative to the true ratio "
    "as C grows (a proof-conservatism question, not a model-mismatch question, and not meaningfully read "
    "on a 0-100% scale). This is why this variant was rejected in favor of the exact-by-construction "
    "version discussed separately.",
    y=0.02,
)
fig.savefig(perf_dir + 'fig2_emp7_TEST_naive_top_panel.png', dpi=200)
plt.close(fig)
print(f"\nSaved {perf_dir}fig2_emp7_TEST_naive_top_panel.png")
