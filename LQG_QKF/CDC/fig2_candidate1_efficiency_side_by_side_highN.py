"""
CANDIDATE FIGURE for CDC2026.tex's Figure 2 -- promoted from a test figure to a real candidate on
2026-09-09 (see `fig2_candidate1_efficiency_side_by_side.py` for the original N=150 version, kept
untouched).

High-trial-count rerun of `fig2_candidate1_efficiency_side_by_side.py`, saved to a SEPARATE .png (the
original, n_trials=150 version is left untouched). Purpose: the min-max bands in the original looked
"blocky" (few valid trials per C, especially at low C where most sampled trials are unreachable and get
filtered out -- see emp6's reachability fix), and the user wants to check whether the small real dip in
this paper's own efficiency at high C is reproducible with a much larger sample, not an artifact of a thin
sample at those C values.

n_trials bumped from 150 to 4000 (a ~27x increase in SAMPLED trials per C; the number of VALID trials
after reachability filtering increases by roughly the same factor, since the filtering rate is a property
of R_ratio/C, not sample size). Same seed (107), same R_ratio, same C_scales, same everything else as the
original -- only n_trials changed.
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

n_trials = 4000
R_ratio = 0.2
rng = np.random.default_rng(107)
C_scales = np.array([0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0])

quad_mean, quad_min, quad_max = [], [], []
restr_mean, restr_min, restr_max = [], [], []
skipped_total, n_total = 0, 0
quad_valid_total, quad_below_1_total = 0, 0

for C_scale in tqdm(C_scales, desc="efficiency-side-by-side-highN"):
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
            continue

        S_star, _ = brute_force_select(M_SENSORS, Ix, Ii_true, R)
        if len(S_star) == 0:
            continue

        S_quad = greedy_select(M_SENSORS, Ix, Ii_true, R)
        S_restr = greedy_select(M_SENSORS, Ix, Ii_restr, R)
        eff_quad.append(len(S_star) / len(S_quad))
        eff_restr.append(len(S_star) / len(S_restr))

    eff_quad, eff_restr = np.array(eff_quad), np.array(eff_restr)
    quad_mean.append(eff_quad.mean()); quad_min.append(eff_quad.min()); quad_max.append(eff_quad.max())
    restr_mean.append(eff_restr.mean()); restr_min.append(eff_restr.min()); restr_max.append(eff_restr.max())
    quad_valid_total += len(eff_quad)
    quad_below_1_total += int(np.sum(eff_quad < 0.999))

quad_mean, quad_min, quad_max = np.array(quad_mean), np.array(quad_min), np.array(quad_max)
restr_mean, restr_min, restr_max = np.array(restr_mean), np.array(restr_min), np.array(restr_max)

print(f"\nSkipped {skipped_total}/{n_total} trials (unreachable target)")
print("\nThis paper (mean, [min, max]):")
for c, m, lo, hi in zip(C_scales, quad_mean, quad_min, quad_max):
    print(f"  C={c:.2f}  mean={m:.4f}  [{lo:.4f}, {hi:.4f}]")
print("\n[19]-restricted view (mean, [min, max]):")
for c, m, lo, hi in zip(C_scales, restr_mean, restr_min, restr_max):
    print(f"  C={c:.2f}  mean={m:.4f}  [{lo:.4f}, {hi:.4f}]")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14.5, 7.6))

axL.axhline(1.0, color=COLORS['brute'], linestyle='--', linewidth=2, label='Matches optimal exactly', zorder=2)
axL.plot(C_scales, quad_mean, color=COLORS['thm2'], marker='^', zorder=4, label='Mean efficiency')
axL.fill_between(C_scales, quad_min, quad_max, color=COLORS['thm2'], alpha=0.15, label='Min-max range')
axL.set_ylim(0.75, 1.03)
axL.set_xlabel(r'Linear-term magnitude $C$', fontsize=12)
axL.set_ylabel('Sensor-count efficiency\n(optimal count / count used)', fontsize=11)
axL.set_title(f"This paper's domain: full-rank $M^{{(i)}}$, quadratic-aware greedy (N={n_trials}/point)\n(y-axis zoomed to [0.75, 1.03] -- note the different scale from the right panel)",
              fontsize=11.5)
axL.grid(alpha=0.3)
axL.legend(loc='lower left', fontsize=9.5, framealpha=0.95)

axR.axhline(1.0, color=COLORS['brute'], linestyle='--', linewidth=2, label='Matches optimal exactly', zorder=2)
axR.plot(C_scales, restr_mean, color=COLORS['prior'], marker='s', zorder=3, label='Mean efficiency')
axR.fill_between(C_scales, restr_min, restr_max, color=COLORS['prior'], alpha=0.15, label='Min-max range')
axR.set_ylim(0.45, 1.03)
axR.set_xlabel(r'Linear-term magnitude $C$', fontsize=12)
axR.set_ylabel('Sensor-count efficiency\n(optimal count / count used)', fontsize=11)
axR.set_title(f"[19]'s domain: greedy driven by their rank-1/zero-$C$ view (N={n_trials}/point)\n(y-axis at [0.45, 1.03] -- its own real data range)",
              fontsize=11.5)
axR.grid(alpha=0.3)
axR.legend(loc='lower left', fontsize=9.5, framealpha=0.95)

fig.suptitle(f'Figure 2 candidate: efficiency, side by side, high trial count (N={n_trials}/point)', y=0.99, fontsize=14.5)
fig.tight_layout(rect=[0, 0.24, 1, 0.93])
add_caption(
    fig, 'Test',
    f"Same figure as the N={150} version, rerun at N={n_trials} sampled trials/point to check whether this "
    "paper's small real dip at high C (and the blocky min-max bands) survive a much larger sample.",
    f"Mean over {n_total - skipped_total} of {n_total} sampled trials ({skipped_total} unreachable and "
    "excluded), bands = min-max range, R_ratio=0.2, full-rank $M^{(i)}$ throughout for both panels -- only "
    "the selection policy differs between them. Left panel's y-axis is zoomed to [0.75, 1.03]; right "
    "panel's is [0.45, 1.03] -- READ THE AXIS LABELS: the two panels are NOT on the same scale. This "
    f"paper's greedy hit the exact optimal count in {quad_valid_total - quad_below_1_total} of "
    f"{quad_valid_total} valid trials; the {quad_below_1_total} exception(s) needed one extra sensor. "
    "(Note: C is drawn from NumPy's unseeded global random state in this codebase, not the seeded "
    "generator used for M -- see CLAUDE.md's reproducibility caveat.)",
    y=0.03,
)
fig.savefig(perf_dir + 'fig2_candidate1_efficiency_side_by_side_highN.png', dpi=200)
plt.close(fig)
print(f"\nSaved {perf_dir}fig2_candidate1_efficiency_side_by_side_highN.png")
