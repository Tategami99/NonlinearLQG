"""
TEST FIGURE -- NOT part of the official 2-figure set in `fig2_empirical_coverage.py`.

Efficiency: |S*| / |S_selected|, optimal sensor count over actual count used. 1.0 = as good as
brute-force optimal. Reuses emp6's already-validated Monte Carlo setup (same greedy selection machinery,
same R_ratio=0.2 target, same reachability fix applied to emp6 itself on 2026-09-09) but looks at DECISION
QUALITY as a ratio rather than raw counts.

**Single panel, not two (revised 2026-09-09).** As with the applicability test figure, the original
version of this file paired this panel with a reproduction of emp7's bottom panel below it, copying that
figure's two-panel layout without the reason that layout exists: emp7's bottom panel deliberately holds M
at rank-1 ([19]'s domain), so a matching top panel needs full-rank M (this paper's domain) -- two
different underlying trial setups, hence two panels with clarifying text above each. This figure's "ours"
and "restricted-view" curves are already computed from the same experimental setup (same full-rank M
throughout, same C sweep, differing only in which selection policy is used on it) -- there was never a
domain split between them, so one panel is all this needs.

A second candidate (Jaccard set-overlap with the optimal set, a stricter check than count alone) was
tried alongside this one and turned out numerically IDENTICAL to the efficiency ratio at every C -- an
honest finding, not a bug: the restricted view's selected set is always a strict superset of the true
optimal set (it never picks a wrong sensor instead of a right one, just extra unnecessary ones on top).
Since it added no new information, it isn't rebuilt as its own figure here.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from sensor_selection_sim import (
    generate_C, per_sensor_info, h_val, greedy_select, brute_force_select, perf_dir, COLORS,
)
from fig2_reframe import add_caption, restrict_Ii_list
from fig2_empirical_coverage import N_STATE, M_SENSORS, P_SCALE, SIGMA2, make_M_stack_with_rank

n_trials = 150
R_ratio = 0.2
rng = np.random.default_rng(107)
C_scales = np.array([0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0])

eff_quad_med, eff_quad_lo, eff_quad_hi = [], [], []
eff_restr_med, eff_restr_lo, eff_restr_hi = [], [], []
skipped_total, n_total = 0, 0

for C_scale in tqdm(C_scales, desc="efficiency-vs-C"):
    eff_quad, eff_restr = [], []
    for _ in range(n_trials):
        P = np.eye(N_STATE) * P_SCALE
        Ix = np.linalg.inv(P)
        M = make_M_stack_with_rank(M_SENSORS, N_STATE, N_STATE, 1e-2, 1.0, rng)  # full rank always
        C = np.zeros((M_SENSORS, N_STATE)) if C_scale == 0 else generate_C(M_SENSORS, N_STATE, C_scale)
        sigma2_vec = np.full(M_SENSORS, SIGMA2)
        Ii_true, ci_true = per_sensor_info(M, C, P, sigma2_vec)
        Ii_restr = restrict_Ii_list(M, sigma2_vec, P)
        R = R_ratio * np.trace(P)
        n_total += 1

        if h_val(list(range(M_SENSORS)), Ix, Ii_true) > R:
            skipped_total += 1
            continue  # unreachable even with every sensor, true model -- see emp6's fix

        S_star, _ = brute_force_select(M_SENSORS, Ix, Ii_true, R)
        if len(S_star) == 0:
            continue  # target trivially met by the empty set -- ratio undefined, skip

        S_quad = greedy_select(M_SENSORS, Ix, Ii_true, R)
        S_restr = greedy_select(M_SENSORS, Ix, Ii_restr, R)
        eff_quad.append(len(S_star) / len(S_quad))
        eff_restr.append(len(S_star) / len(S_restr))

    eff_quad_med.append(np.median(eff_quad)); eff_quad_lo.append(np.percentile(eff_quad, 25)); eff_quad_hi.append(np.percentile(eff_quad, 75))
    eff_restr_med.append(np.median(eff_restr)); eff_restr_lo.append(np.percentile(eff_restr, 25)); eff_restr_hi.append(np.percentile(eff_restr, 75))

eff_quad_med, eff_quad_lo, eff_quad_hi = np.array(eff_quad_med), np.array(eff_quad_lo), np.array(eff_quad_hi)
eff_restr_med, eff_restr_lo, eff_restr_hi = np.array(eff_restr_med), np.array(eff_restr_lo), np.array(eff_restr_hi)

print(f"\nSkipped {skipped_total}/{n_total} trials (unreachable target)")
print("\nEfficiency |S*|/|S_selected|  (1.0 = matches optimal count):")
for c, q, r in zip(C_scales, eff_quad_med, eff_restr_med):
    print(f"  C={c:.2f}  ours={q:.4f}  restricted={r:.4f}")

fig, ax = plt.subplots(figsize=(9.5, 7.8))
ax.axhline(1.0, color=COLORS['brute'], linestyle='--', linewidth=2, label='Matches optimal exactly', zorder=2)
ax.plot(C_scales, eff_quad_med, color=COLORS['thm2'], marker='^', zorder=4,
        label="This paper: quadratic-aware greedy (full-rank $M$, real $c_j$)")
ax.fill_between(C_scales, eff_quad_lo, eff_quad_hi, color=COLORS['thm2'], alpha=0.15)
ax.plot(C_scales, eff_restr_med, color=COLORS['prior'], marker='s', zorder=3,
        label="Greedy driven by [19]'s rank-1/zero-$C$ view")
ax.fill_between(C_scales, eff_restr_lo, eff_restr_hi, color=COLORS['prior'], alpha=0.15)
ax.set_ylim(0.45, 1.08)
ax.set_xlabel(r'Linear-term magnitude $C$', fontsize=12.5)
ax.set_ylabel('Sensor-count efficiency\n(optimal count / count used)', fontsize=11)
ax.set_title('TEST: efficiency -- how close to the optimal sensor COUNT', fontsize=12.5)
ax.grid(alpha=0.3)
ax.legend(loc='lower left', fontsize=9.5, framealpha=0.95)

fig.suptitle('TEST FIGURE: efficiency framing (single panel, reuses emp6 data)', y=0.97, fontsize=14)
fig.tight_layout(rect=[0, 0.24, 1, 0.92])
add_caption(
    fig, 'Test',
    "Same target-satisfaction task as emp6, re-expressed as a ratio: this paper's greedy stays near-optimal "
    "as C grows; the restricted view needs increasingly more sensors relative to optimal.",
    f"Median over {n_total - skipped_total} of {n_total} sampled trials ({skipped_total} were unreachable "
    f"even using every sensor under the true model and were excluded; bands = IQR), R_ratio={R_ratio}, "
    "full-rank $M^{(i)}$ throughout. This paper's ratio is not exactly 1.0: greedy selection is a "
    "heuristic, not an exact solver, even when given the correct per-sensor information, so small real "
    "gaps from optimal are possible in principle (none were observed at this scale -- see the printed "
    "values). The restricted-view ratio declines because it underestimates each sensor's true information "
    "content (it cannot see the $c_j$ contribution), so it keeps adding sensors past the point actually "
    "needed before its own (too-low) estimate of h(S) crosses R.",
    y=0.03,
)
fig.savefig(perf_dir + 'fig2_emp7_TEST_efficiency_top_panel.png', dpi=200)
plt.close(fig)
print(f"\nSaved {perf_dir}fig2_emp7_TEST_efficiency_top_panel.png")
