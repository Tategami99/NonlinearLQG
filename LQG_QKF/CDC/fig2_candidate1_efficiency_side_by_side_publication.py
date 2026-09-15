"""
CANDIDATE FIGURE for CDC2026.tex's Figure 2 -- submission-style pass over
`fig2_candidate1_efficiency_side_by_side_highN.py`.

Same data and construction (same efficiency metric, same domain split, same independent y-axis scales), but
restyled to match the house style already used in the paper's own `fig1.png`/`fig2.png` (see
`generate_figures.py`): plain declarative suptitle, one small gray context line below it (not a full
baked-in caption -- that belongs in the LaTeX `\caption{}`, not the image), short plain-black panel titles,
ONE shared legend below both panels (not a separate legend per panel), no "TEST"/debug text anywhere. The
working/annotated version (`fig2_candidate1_efficiency_side_by_side_highN.py`) is kept as-is for internal
review -- this is a separate file so neither version has to compromise for the other's audience.
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
from fig2_reframe import restrict_Ii_list
from fig2_empirical_coverage import N_STATE, M_SENSORS, P_SCALE, SIGMA2, make_M_stack_with_rank

n_trials = 4000
R_ratio = 0.2
rng = np.random.default_rng(107)
C_scales = np.array([0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0])

quad_mean, quad_min, quad_max = [], [], []
restr_mean, restr_min, restr_max = [], [], []
skipped_total, n_total = 0, 0

for C_scale in tqdm(C_scales, desc="efficiency-side-by-side-publication"):
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

quad_mean, quad_min, quad_max = np.array(quad_mean), np.array(quad_min), np.array(quad_max)
restr_mean, restr_min, restr_max = np.array(restr_mean), np.array(restr_min), np.array(restr_max)

print(f"Skipped {skipped_total}/{n_total} trials (unreachable target)")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 7.0))

axL.axhline(1.0, color=COLORS['brute'], linestyle='--', linewidth=2, label='Optimal (brute force)', zorder=2)
axL.plot(C_scales, quad_mean, color=COLORS['thm2'], marker='^', label='This paper: quadratic-aware greedy', zorder=4)
axL.fill_between(C_scales, quad_min, quad_max, color=COLORS['thm2'], alpha=0.15)
axL.set_ylim(0.75, 1.03)
axL.set_xlabel(r'Linear-term magnitude $C$')
axL.set_ylabel('Sensor-count efficiency\n(optimal count / count used)')
axL.set_title("This paper's domain (full rank $M^{(i)}$)", fontsize=13)
axL.grid(alpha=0.3)

axR.axhline(1.0, color=COLORS['brute'], linestyle='--', linewidth=2, label='Optimal (brute force)', zorder=2)
axR.plot(C_scales, restr_mean, color=COLORS['prior'], marker='s', label="[19]: restricted (rank-1, zero-$C$) view", zorder=3)
axR.fill_between(C_scales, restr_min, restr_max, color=COLORS['prior'], alpha=0.15)
axR.set_ylim(0.45, 1.03)
axR.set_xlabel(r'Linear-term magnitude $C$')
axR.set_ylabel('Sensor-count efficiency\n(optimal count / count used)')
axR.set_title("[19]'s domain (rank-1 $M^{(i)}$, $C=0$)", fontsize=13)
axR.grid(alpha=0.3)

fig.suptitle('This paper matches the optimal sensor count; the restricted view does not', y=1.0)
fig.text(0.5, 0.925,
          f'N={n_trials} trials/point, shaded band = min-max range, $R_\\mathrm{{ratio}}={R_ratio}$. '
          'Left and right panels use different y-axis scales.',
          ha='center', fontsize=10.5, color='#555555')

handles = [
    plt.Line2D([0], [0], color=COLORS['brute'], linestyle='--', linewidth=2),
    plt.Line2D([0], [0], color=COLORS['thm2'], marker='^', linewidth=2.4),
    plt.Line2D([0], [0], color=COLORS['prior'], marker='s', linewidth=2.4),
]
labels = ['Optimal (brute force)', 'This paper: quadratic-aware greedy', "[19]: restricted (rank-1, zero-$C$) view"]
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=11, framealpha=0.95)
fig.tight_layout(rect=[0, 0.08, 1, 0.89])
fig.savefig(perf_dir + 'fig2_candidate1_efficiency_side_by_side_publication.png', dpi=200)
plt.close(fig)
print(f"Saved {perf_dir}fig2_candidate1_efficiency_side_by_side_publication.png")
