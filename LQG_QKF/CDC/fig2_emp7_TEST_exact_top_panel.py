"""
TEST FIGURE -- NOT part of the official 10-figure set in `fig2_empirical_coverage.py`.

This builds the user's exact request for the emp7 top panel: SAME y-axis as the bottom panel
(Delta_j(empty) predicted / Delta_j(empty) true -- one sensor's marginal information gain), SAME x-axis
(linear-term magnitude C), with the only change being full-rank M instead of rank-1 M, and "using our
paper" instead of [19]'s restricted view for the "predicted" side.

Since this paper's Delta_j(empty) = Tr(P) - Tr(B_j) formula (Proposition 2) was never actually restricted
to rank-1/C=0 -- that restriction only exists in the bottom panel because it's deliberately reproducing
[19]'s model, which has no linear-term parameter at all -- "predicted" (this paper, full-rank M_j, real
c_j) and "true" (the same M_j, same c_j) are the SAME calculation on the SAME inputs once you drop the
restriction. So the top panel below is not an empirical near-1 result, it is EXACTLY 1.0 at every C, with
zero variance (no IQR band -- there is nothing random left to bound once predicted and true are the same
function call). That flatness IS the result: this paper never approximates the linear term away, so there
is no representational gap for growing C to widen.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from sensor_selection_sim import perf_dir, COLORS
from fig2_reframe import add_caption, restrict_to_rank1_zero_c, marginal_gain_first_sensor
from fig2_empirical_coverage import N_STATE, P_SCALE, SIGMA2

n_trials = 150
rng = np.random.default_rng(107)
C_scales = np.array([0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0])

# ---- Top panel: full-rank M_j, this paper's own (unrestricted) computation for BOTH sides ----
ours_med, ours_lo, ours_hi = [], [], []
for C_scale in tqdm(C_scales, desc="exact-top-panel"):
    ratios = []
    for _ in range(n_trials):
        P = np.eye(N_STATE) * P_SCALE
        Q, _ = np.linalg.qr(rng.normal(size=(N_STATE, N_STATE)))
        Mj = Q @ np.diag(rng.uniform(0.3, 1.0, size=N_STATE)) @ Q.T  # full rank (all eigenvalues nonzero)
        if C_scale > 0:
            direction = rng.normal(size=(N_STATE, 1))
            cj = direction / np.linalg.norm(direction) * C_scale
        else:
            cj = np.zeros((N_STATE, 1))
        true_gain = marginal_gain_first_sensor(Mj, cj, P, SIGMA2)
        predicted_gain = marginal_gain_first_sensor(Mj, cj, P, SIGMA2)  # this paper: no restriction, same call
        ratios.append(predicted_gain / true_gain)
    ours_med.append(np.median(ratios)); ours_lo.append(np.percentile(ratios, 25)); ours_hi.append(np.percentile(ratios, 75))
ours_med, ours_lo, ours_hi = np.array(ours_med), np.array(ours_lo), np.array(ours_hi)

print("\nTop panel (this paper, full-rank M, no restriction) predicted/true ratio:")
for c, m, lo, hi in zip(C_scales, ours_med, ours_lo, ours_hi):
    print(f"  C={c:.2f}  ratio={m:.10g}  (IQR [{lo:.10g}, {hi:.10g}])")

# ---- Bottom panel: unchanged, [19]'s restricted view, rank-1 M_j ----
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

# ---- Plot: SAME y-axis scale/label on both panels, as requested ----
fig, axes = plt.subplots(2, 1, figsize=(9.5, 12.8), sharex=True, sharey=True)

ax = axes[0]
ax.axhline(1.0, color=COLORS['brute'], linestyle='--', linewidth=2, label='Ratio = 1 (predicted = true)', zorder=2)
ax.plot(C_scales, ours_med, color=COLORS['thm2'], marker='^', zorder=4,
        label='This paper: full-rank $M^{(i)}$, real $c_j$ (no restriction)')
ax.fill_between(C_scales, ours_lo, ours_hi, color=COLORS['thm2'], alpha=0.15)
ax.set_ylim(0.45, 1.05)
ax.set_ylabel(r'$\Delta_j(\varnothing)$ predicted / $\Delta_j(\varnothing)$ true', fontsize=11)
ax.set_title('TEST: this paper (full rank $M$, real $c_j$) -- exactly 1.0, by construction', fontsize=12.5)
ax.grid(alpha=0.3)
ax.legend(loc='lower left', fontsize=9.5, framealpha=0.95)

ax = axes[1]
ax.axhline(1.0, color=COLORS['brute'], linestyle='--', linewidth=2, label='Perfect representation (exact at $C=0$)', zorder=2)
ax.plot(C_scales, pred_med, color=COLORS['prior'], marker='s', zorder=3,
        label="[19]'s rank-1/zero-$C$ prediction of the sensor's own true value")
ax.fill_between(C_scales, pred_lo, pred_hi, color=COLORS['prior'], alpha=0.15)
ax.set_xlabel(r'Linear-term magnitude $C$', fontsize=12.5)
ax.set_ylabel(r'$\Delta_j(\varnothing)$ predicted / $\Delta_j(\varnothing)$ true', fontsize=11)
ax.set_title("Real emp7 bottom panel, reproduced for comparison (unchanged)", fontsize=12.5)
ax.grid(alpha=0.3)
ax.legend(loc='lower left', fontsize=9.5, framealpha=0.95)

fig.suptitle('TEST FIGURE: exact-by-construction top panel, as requested', y=0.975, fontsize=14.5)
fig.tight_layout(rect=[0, 0.15, 1, 0.94])
add_caption(
    fig, 'Test',
    "Same y-axis and x-axis on both panels, as requested: this paper's version (top) never restricts M's "
    "rank or drops c_j, so predicted and true are the same calculation -- exactly 1.0, no variance -- while "
    "[19]'s restricted view (bottom) degrades because it has no way to represent c_j at all.",
    f"Median over N={n_trials} trials/point; top panel's band is degenerate (predicted equals true by "
    "construction, not by estimation) -- it is drawn at zero width, not omitted, so the two panels stay "
    "visually comparable. Bottom panel unchanged from the real fig2_emp7_breakdown_vs_C. Both panels share "
    "the same y-axis scale/limits, as requested, so the flatness of the top curve against the decline of "
    "the bottom curve is directly readable at a glance.",
    y=0.02,
)
fig.savefig(perf_dir + 'fig2_emp7_TEST_exact_top_panel.png', dpi=200)
plt.close(fig)
print(f"\nSaved {perf_dir}fig2_emp7_TEST_exact_top_panel.png")
