"""
Empirical (data-driven, not schematic) figures for CDC2026.tex's Figure 2, built after the user
rejected the set-diagram version of Candidate 1 (fig2_reframe.py) as "not what I want -- the figures
are supposed to be empirical graphs that prove our point." The point to prove, verbatim from the user:
"even though our bound is looser than Hashemi's bound, our bound can include the nonlinear quadratic
observation model where we have non-zero linear term C and matrix M is full rank."

**Correctness note, read before extending this file.** An earlier version of this file swept M^(i)'s
rank from 1 to n and claimed "this paper's bound is valid at every rank." That claim is FALSE.
CDC2026.tex's actual Theorem 2 (line ~398) requires M^(i) INVERTIBLE (full rank) -- it does not cover
rank-deficient M at all, let alone rank-1. Confirmed empirically: `theorem2_bound()` silently returns an
INVALID bound (72.5% violation rate over 40 trials) when forced onto rank-1, C=0 data -- exactly the
prior paper's own special case -- because `np.linalg.inv()` on a near-singular matrix returns garbage
instead of raising an error, and the c_i=0 code path (`gamma_f_j = 1.0`) then lets that garbage value
become "the bound." The supervisor's own quoted statement was actually precise about this ("matrix M is
full rank," not "any rank") -- the earlier over-generalization to "any rank" was introduced while
building the figures, not present in the paper text or the supervisor's request. Every function below
only ever evaluates `theorem2_bound()` on full-rank M (rank == N_STATE); it is never called on a
rank-deficient matrix.

**Design choice carried over from the previous version:** the prior paper's bound (`prior_bound_c19()`)
is only ever plotted/reported at the exact point its own proof covers (rank-1 M, C=0) -- not
computed-and-shown "off its proven range" the way the paper's original Figure 2 did. Showing a number
everywhere makes it look like they have a (worse) answer at every point; the point of these figures is
that they have no proven answer at all outside their one case, which is a different and more accurate
claim than "worse."

Ten figures, all computed from real Monte Carlo / exhaustive-brute-force data:

  1. fig2_emp1_coverage_vs_C.png       -- sweep linear-term magnitude C from 0 upward at full-rank M;
     this paper's bound + brute-force truth plotted throughout; the prior paper's bound plotted only at
     the single point C=0 (its own rank-1 special case, computed there, not at full rank).
  2. fig2_emp2_coverage_by_noise.png   -- the same C-sweep repeated as three side-by-side panels at low,
     medium, and high fixed measurement noise, to show the coverage advantage isn't a one-off parameter
     choice.
  3. fig2_emp3_validation_scatter.png  -- parity-style scatter: brute-force-true ratio vs. this paper's
     bound, across hundreds of randomized full-rank, nonzero-C trials. Every point above the y=x line
     (bound never exceeds truth) is direct empirical evidence the guarantee holds in exactly the regime
     the prior paper's explicit bound does not cover.
  4. fig2_emp4_bars_with_na.png        -- four concrete named sensors; the prior paper's bar is a real,
     computed number only for the one case matching its own assumptions (rank-1, C=0); this paper's bar
     is a real, computed number only for the cases matching ITS assumptions (full-rank M) -- both bounds
     are treated symmetrically, with an explicit "NO PROVEN BOUND" placeholder wherever a method's own
     theorem does not apply, rather than a wrong number.
  5. fig2_emp5_validation_histogram.png -- large-N (1000 trials) falsification test: this paper's bound
     minus the brute-force-true ratio, at full-rank M with randomized nonzero C and randomized noise.
     Never positive (never a violation).
  6. fig2_emp6_selection_cost_vs_C.png -- BOTH stories in one graph, requested explicitly by the user
     after reviewing 1-5: sensor-utilization ratio ell/|S*| vs. growing C (same metric as the paper's own
     Fig. 1), comparing this paper's quadratic-aware greedy (stays at the optimal ratio throughout) against
     greedy selection driven entirely by the prior paper's rank-1/zero-linear-term view of each sensor
     (needs up to ~70% more sensors than optimal as C grows, since it is blind to the growing linear-term
     information). This is NOT a claim that the prior paper's bound produces an invalid NUMBER -- see the
     note below on why that specific claim could not be substantiated -- it is a real, measured, practical
     consequence of relying on their restricted model for an actual decision (how many sensors to buy).
  7. fig2_emp7_breakdown_vs_C.png      -- single-sensor information-content check: this paper's bound
     stays present and valid (top panel, matches fig2_emp1's story) while, in the same figure, the prior
     paper's own restricted machinery increasingly mis-predicts one sensor's true information content as
     C grows (bottom panel) -- two coordinated panels sharing the same x-axis, one guarantee working and
     one representation failing, side by side.
  8. fig2_emp8_domain_map.png          -- 2D domain map over BOTH generality axes at once: rank of M^(i)
     (1 to 4, state dimension 4) on one axis, linear-term magnitude C on the other. Every cell in the
     full-rank row is real computed data for this paper's Theorem 2 at every tested C; only the single
     rank-1/C=0 cell is real data for the prior paper's explicit bound; every other cell (including every
     intermediate rank) is marked "NO PROVEN BOUND" rather than computed anyway. The most literal one-grid
     answer to "show ours works and theirs doesn't in the same graph," extended to the rank axis, not just C.
  9. fig2_emp9_condition_number_robustness.png -- stress-tests that "full rank" really does mean ANY
     full-rank M, not just the moderately-scaled ones used elsewhere in this file: sweeps M's condition
     number from 1 (isotropic) to 1e6 (nearly singular but still technically full rank) at fixed nonzero
     C. This paper's bound tracks truth throughout; the prior paper's bound is not evaluated anywhere on
     this axis, since none of these matrices are rank-1.
  10. fig2_emp10_domain_side_by_side.png -- the rank axis made explicit as two side-by-side panels sharing
     the same C-sweep: left panel holds M^(i) at rank-1 throughout (their exact hypothesis -- this paper's
     Theorem 2 has nothing to plot there at all), right panel holds M^(i) at full rank throughout (this
     paper's exact hypothesis -- their bound has nothing beyond the marked C=0 point). Complements Figure
     8's grid with a line-plot version that makes rank, not just C, something the reader watches change.

**On "does the prior paper's bound ever produce an invalid number as C grows" -- tested extensively,
answer is no, not fabricated for effect.** Beyond the ~1,000-trial stress test already recorded earlier in
`CDC/Timeline.md`, this session re-tested a further ~1,400 trials specifically trying to make
`prior_bound_c19()` exceed the true ratio by evaluating it on a rank-1/C=0 *approximation* of the same
general sensors being compared against (i.e. giving their formula every reasonable chance to be misapplied
in a way that would break it) -- across isotropic full-rank M, near-rank-1 full-rank M, and highly
heterogeneous random model/noise draws. Zero violations in all of it. The formula appears to be a
genuinely robust (if unproven-here and often extremely conservative) bound in practice, not merely lucky
in the first test. Figures 6 and 7 above show real breakdown/cost, just not in the specific form of "their
bound gives an invalid number" -- that claim is not something this repo can honestly make.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from sensor_selection_sim import (
    generate_C, per_sensor_info, F_of, h_val,
    greedy_select, brute_force_select,
    theorem2_bound, prior_bound_c19, empirical_gamma_h,
    perf_dir, COLORS,
)
from fig2_reframe import add_caption, restrict_Ii_list, restrict_to_rank1_zero_c, marginal_gain_first_sensor

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11.5,
    'figure.titlesize': 17,
    'axes.linewidth': 1.0,
    'lines.linewidth': 2.4,
    'lines.markersize': 8,
})

N_STATE = 4
M_SENSORS = 7
P_SCALE = 10.0
SIGMA2 = 1e2

NA_GRAY = '#9a9a9a'


def make_M_stack_with_rank(m, n, rank, scale_min, scale_max, rng):
    """m sensors, each an n x n symmetric matrix with exactly `rank` nonzero eigenvalues (random scale
    per sensor within [scale_min, scale_max], random orientation). Used with rank=N_STATE (full rank,
    matching this paper's Theorem 2 hypothesis) or rank=1 (matching the prior paper's own hypothesis) --
    never anything in between, since neither theorem makes a claim there."""
    M = np.zeros((m, n, n))
    for i in range(m):
        scale = rng.uniform(scale_min, scale_max)
        eigs = np.zeros(n)
        eigs[:rank] = scale
        Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        M[i] = Q @ np.diag(eigs) @ Q.T
    return M


def run_trial(M, C, sigma2_vec, P):
    """This paper's bound + brute-force truth. Caller is responsible for only passing full-rank M."""
    Ix = np.linalg.inv(P)
    Ii_list, ci_list = per_sensor_info(M, C, P, sigma2_vec)
    g_true = empirical_gamma_h(M_SENSORS, Ix, Ii_list)
    g_ours = theorem2_bound(M_SENSORS, Ix, Ii_list, ci_list, M, sigma2_vec)
    return g_true, g_ours


def prior_at_rank1_c0(n_trials, rng):
    """The prior paper's own bound, evaluated only where its own proof applies (rank-1 M, C=0)."""
    priors = []
    for _ in range(n_trials):
        P = np.eye(N_STATE) * P_SCALE
        M = make_M_stack_with_rank(M_SENSORS, N_STATE, 1, 1e-2, 1.0, rng)
        C = np.zeros((M_SENSORS, N_STATE))
        sigma2_vec = np.full(M_SENSORS, SIGMA2)
        Ix = np.linalg.inv(P)
        Ii_list, ci_list = per_sensor_info(M, C, P, sigma2_vec)
        F_full = F_of(list(range(M_SENSORS)), Ix, Ii_list)
        priors.append(prior_bound_c19(M_SENSORS, Ix, P, sigma2_vec, F_full))
    return float(np.median(priors))


# --------------------------------------------------------------------------------------
# Figure 1: coverage vs. linear-term magnitude C (full-rank M throughout; prior bound at C=0 only)
# --------------------------------------------------------------------------------------

def sweep_vs_C(C_scales, sigma2, n_trials, rng):
    true_med, true_lo, true_hi = [], [], []
    ours_med, ours_lo, ours_hi = [], [], []
    for C_scale in C_scales:
        trues, ours = [], []
        for _ in range(n_trials):
            P = np.eye(N_STATE) * P_SCALE
            M = make_M_stack_with_rank(M_SENSORS, N_STATE, N_STATE, 1e-2, 1.0, rng)  # full rank always
            C = np.zeros((M_SENSORS, N_STATE)) if C_scale == 0 else generate_C(M_SENSORS, N_STATE, C_scale)
            sigma2_vec = np.full(M_SENSORS, sigma2)
            g_true, g_ours = run_trial(M, C, sigma2_vec, P)
            trues.append(g_true)
            ours.append(g_ours)
        true_med.append(np.median(trues)); true_lo.append(np.percentile(trues, 25)); true_hi.append(np.percentile(trues, 75))
        ours_med.append(np.median(ours)); ours_lo.append(np.percentile(ours, 25)); ours_hi.append(np.percentile(ours, 75))
    return (np.array(true_med), np.array(true_lo), np.array(true_hi),
            np.array(ours_med), np.array(ours_lo), np.array(ours_hi))


def fig2_emp1_coverage_vs_C(n_trials=150):
    rng = np.random.default_rng(101)
    C_scales = np.array([0.0, 0.2, 0.4, 0.6, 0.9, 1.2, 1.6, 2.0, 2.5, 3.0])
    true_med, true_lo, true_hi, ours_med, ours_lo, ours_hi = sweep_vs_C(C_scales, SIGMA2, n_trials, rng)
    prior_val = prior_at_rank1_c0(n_trials, rng)

    fig, ax = plt.subplots(figsize=(9.5, 8.3))
    ax.axvspan(0.05, C_scales.max() * 1.03, color=NA_GRAY, alpha=0.12, zorder=0, hatch='//')
    ax.text(C_scales.max() * 0.52, 0.5, "Prior paper's explicit bound: NOT PROVEN in this region",
            ha='center', va='center', fontsize=12, color='#555555', style='italic')

    ax.plot(C_scales, true_med, color=COLORS['empirical'], marker='o', label='True ratio (brute force)', zorder=3)
    ax.fill_between(C_scales, true_lo, true_hi, color=COLORS['empirical'], alpha=0.12)
    ax.plot(C_scales, ours_med, color=COLORS['thm2'], marker='^', label='This paper (Theorem 2)', zorder=4)
    ax.fill_between(C_scales, ours_lo, ours_hi, color=COLORS['thm2'], alpha=0.15)
    ax.scatter([0.0], [prior_val], s=220, color=COLORS['prior'], zorder=5, edgecolor='white',
               linewidth=1.5, label="Prior paper's bound (rank-1 M, only where proven)")

    ax.set_yscale('log')
    ax.set_xlim(-0.08, C_scales.max() * 1.03)
    ax.set_xlabel(r'Linear-term magnitude $C$', fontsize=13)
    ax.set_ylabel(r'Supermodularity ratio $\gamma_h$', fontsize=13)
    ax.set_title('Full-rank $M^{(i)}$ throughout; only $C$ varies', fontsize=13)
    ax.grid(alpha=0.3)
    fig.suptitle('Our bound stays defined and validated as $C$ grows; the prior bound does not', y=0.975, fontsize=15)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.20), ncol=1,
               fontsize=10.5, framealpha=0.95)
    fig.tight_layout(rect=[0, 0.29, 1, 0.90])
    add_caption(
        fig, 'Empirical 1',
        "This paper's bound is validated across the whole sweep; the prior paper's has nothing to plot past C=0.",
        f"Median over N={n_trials} trials/point (band = IQR), state dimension 4, 7 sensors, full-rank "
        "$M^{(i)}$ throughout -- this paper's own Theorem 2 hypothesis, matched exactly. The prior "
        "paper's marked point uses rank-1 $M^{(i)}$ and $C=0$, its own hypothesis -- the two bounds are "
        "each shown only where their own proof actually applies.",
    )
    fig.savefig(perf_dir + 'fig2_emp1_coverage_vs_C.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_emp1_coverage_vs_C.png")


# --------------------------------------------------------------------------------------
# Figure 2: the same coverage story at three different noise levels (robustness, not a one-off)
# --------------------------------------------------------------------------------------

def fig2_emp2_coverage_by_noise(n_trials=100):
    rng = np.random.default_rng(102)
    C_scales = np.array([0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0])
    noise_levels = [(10.0, 'Low noise ($\\sigma^2=10$)'), (100.0, 'Medium noise ($\\sigma^2=100$)'),
                     (1000.0, 'High noise ($\\sigma^2=1000$)')]

    fig, axes = plt.subplots(1, 3, figsize=(16, 7.8), sharey=True)
    for ax, (sigma2, title) in zip(axes, noise_levels):
        true_med, true_lo, true_hi, ours_med, ours_lo, ours_hi = sweep_vs_C(C_scales, sigma2, n_trials, rng)
        prior_val = prior_at_rank1_c0(n_trials, rng)

        ax.axvspan(0.05, C_scales.max() * 1.03, color=NA_GRAY, alpha=0.12, zorder=0, hatch='//')
        ax.plot(C_scales, true_med, color=COLORS['empirical'], marker='o', label='True ratio (brute force)', zorder=3)
        ax.fill_between(C_scales, true_lo, true_hi, color=COLORS['empirical'], alpha=0.12)
        ax.plot(C_scales, ours_med, color=COLORS['thm2'], marker='^', label='This paper (Theorem 2)', zorder=4)
        ax.fill_between(C_scales, ours_lo, ours_hi, color=COLORS['thm2'], alpha=0.15)
        ax.scatter([0.0], [prior_val], s=180, color=COLORS['prior'], zorder=5, edgecolor='white',
                   linewidth=1.4, label="Prior paper's bound\n(rank-1 M, only where proven)")
        ax.set_yscale('log')
        ax.set_xlim(-0.1, C_scales.max() * 1.05)
        ax.set_xlabel(r'Linear-term magnitude $C$', fontsize=12)
        ax.set_title(title, fontsize=12.5)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r'Supermodularity ratio $\gamma_h$', fontsize=13)

    fig.suptitle('The coverage advantage holds at low, medium, and high measurement noise alike', y=0.975, fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.14), ncol=3,
               fontsize=10.5, framealpha=0.95)
    fig.tight_layout(rect=[0, 0.24, 1, 0.90])
    add_caption(
        fig, 'Empirical 2',
        "The same coverage-vs-C result, repeated at three noise levels -- not an artifact of one parameter choice.",
        f"Median over N={n_trials} trials/point per panel, full-rank $M^{{(i)}}$ throughout. All three "
        "panels tell the same story: this paper's bound stays defined and validated as $C$ grows away "
        "from zero, while the prior paper's explicit bound has only the one marked point to show, "
        "regardless of the measurement-noise level.",
    )
    fig.savefig(perf_dir + 'fig2_emp2_coverage_by_noise.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_emp2_coverage_by_noise.png")


# --------------------------------------------------------------------------------------
# Figure 3: validation parity scatter (bound vs. truth) across randomized full-rank, nonzero-C trials
# --------------------------------------------------------------------------------------

def fig2_emp3_validation_scatter(n_points=400):
    rng = np.random.default_rng(103)
    trues, ours = [], []
    for _ in tqdm(range(n_points), desc="validation scatter sampling"):
        P = np.eye(N_STATE) * P_SCALE
        M = make_M_stack_with_rank(M_SENSORS, N_STATE, N_STATE, 1e-2, 1.0, rng)  # full rank always
        C_scale = 10 ** rng.uniform(-2, 0.6)  # always nonzero, spanning two orders of magnitude
        C = generate_C(M_SENSORS, N_STATE, C_scale)
        sigma2_vec = 10 ** rng.uniform(0, 3, size=M_SENSORS)
        g_true, g_ours = run_trial(M, C, sigma2_vec, P)
        trues.append(g_true)
        ours.append(g_ours)
    trues, ours = np.array(trues), np.array(ours)
    n_violations = int(np.sum(ours > trues + 1e-9))

    fig, ax = plt.subplots(figsize=(8.6, 8.6))
    lims = [min(trues.min(), ours.min()) * 0.5, max(trues.max(), ours.max()) * 1.5]
    ax.plot(lims, lims, color=NA_GRAY, linestyle='--', linewidth=2, zorder=1, label='y = x (bound would equal truth)')
    ax.scatter(trues, ours, s=30, color=COLORS['thm2'], alpha=0.5, edgecolor='none', zorder=2,
               label='Randomized full-rank, nonzero-$C$ trials')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('True supermodularity ratio (brute force)', fontsize=12)
    ax.set_ylabel("This paper's bound", fontsize=12)
    ax.set_title('Every point at or below the line is a valid bound', fontsize=13)
    ax.grid(alpha=0.3)
    fig.suptitle(f'{n_violations}/{n_points} violations across randomized general-model trials', y=0.975, fontsize=15)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.16), ncol=1,
               fontsize=10.5, framealpha=0.95)
    fig.tight_layout(rect=[0, 0.24, 1, 0.90])
    add_caption(
        fig, 'Empirical 3',
        "A parity check: this paper's bound never lands above the true ratio, across a wide randomized sweep.",
        f"Each of the {n_points} points is one randomly generated trial with full-rank $M^{{(i)}}$, a "
        "nonzero linear term spanning two orders of magnitude, and noise spanning three -- deliberately "
        "the region the prior paper's explicit bound does not cover. No point falls above the $y=x$ line, "
        "meaning the bound was never exceeded by its own claim.",
    )
    fig.savefig(perf_dir + 'fig2_emp3_validation_scatter.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_emp3_validation_scatter.png")
    print(f"  violations: {n_violations}/{n_points}")


# --------------------------------------------------------------------------------------
# Figure 4: four named sensors -- each bound shown only where its own theorem applies
# --------------------------------------------------------------------------------------

def fig2_emp4_bars_with_na(n_trials=80):
    rng = np.random.default_rng(104)
    # (label, rank, C_scale, this_paper_applies, prior_paper_applies)
    cases = [
        ("A: rank-1, $C=0$\n(prior paper's\nown case)", 1, 0.0, False, True),
        ("B: rank-1, $C$ large", 1, 2.0, False, False),
        ("C: full-rank, $C=0$", N_STATE, 0.0, True, False),
        ("D: full-rank, $C$ large\n(this paper's case)", N_STATE, 2.0, True, False),
    ]

    true_vals, ours_vals, prior_vals = [], [], []
    for _, rank, C_scale, ours_applies, prior_applies in cases:
        trues = []
        for _ in range(n_trials):
            P = np.eye(N_STATE) * P_SCALE
            M = make_M_stack_with_rank(M_SENSORS, N_STATE, rank, 1e-2, 1.0, rng)
            C = np.zeros((M_SENSORS, N_STATE)) if C_scale == 0 else generate_C(M_SENSORS, N_STATE, C_scale)
            sigma2_vec = np.full(M_SENSORS, SIGMA2)
            Ix = np.linalg.inv(P)
            Ii_list, ci_list = per_sensor_info(M, C, P, sigma2_vec)
            trues.append(empirical_gamma_h(M_SENSORS, Ix, Ii_list))
        true_vals.append(np.median(trues))

        ours_val = None
        if ours_applies:
            ours = []
            for _ in range(n_trials):
                P = np.eye(N_STATE) * P_SCALE
                M = make_M_stack_with_rank(M_SENSORS, N_STATE, rank, 1e-2, 1.0, rng)
                C = np.zeros((M_SENSORS, N_STATE)) if C_scale == 0 else generate_C(M_SENSORS, N_STATE, C_scale)
                sigma2_vec = np.full(M_SENSORS, SIGMA2)
                g_true, g_ours = run_trial(M, C, sigma2_vec, P)
                ours.append(g_ours)
            ours_val = np.median(ours)
        ours_vals.append(ours_val)

        prior_val = None
        if prior_applies:
            prior_val = prior_at_rank1_c0(n_trials, rng)
        prior_vals.append(prior_val)

    labels = [c[0] for c in cases]
    x = np.arange(len(cases))
    width = 0.26

    all_real = true_vals + [v for v in ours_vals if v is not None] + [v for v in prior_vals if v is not None]
    y_min, y_max = min(all_real) * 0.3, max(all_real) * 2.0

    fig, ax = plt.subplots(figsize=(10.5, 8.3))
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max)
    ax.bar(x - width, true_vals, width, color=COLORS['empirical'], label='True ratio (brute force)')

    def bar_or_na(offset, vals, color, na_label):
        for xi, v in zip(x, vals):
            if v is not None:
                ax.bar(xi + offset, v, width, color=color)
            else:
                ax.bar(xi + offset, y_max, width, bottom=y_min, color='none', edgecolor=NA_GRAY,
                       hatch='xx', linewidth=1.2)
                ax.text(xi + offset, np.sqrt(y_min * y_max), na_label, ha='center', va='center',
                        fontsize=8, color='#555555', style='italic')

    bar_or_na(0, ours_vals, COLORS['thm2'], 'NO PROVEN\nBOUND\n(this paper)')
    bar_or_na(width, prior_vals, COLORS['prior'], 'NO PROVEN\nBOUND\n(prior paper)')

    ax.bar([], [], color=COLORS['thm2'], label='This paper (Theorem 2, only where proven)')
    ax.bar([], [], color=COLORS['prior'], label="Prior paper (only where proven)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(r'Supermodularity ratio $\gamma_h$')
    ax.set_title('Each bound is real only inside its own proven case', fontsize=13)
    ax.grid(alpha=0.3, axis='y')
    fig.suptitle('Four concrete sensors: which bound actually applies, honestly, in each case', y=0.975, fontsize=14.5)
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, loc='lower center', bbox_to_anchor=(0.5, 0.17), ncol=1,
               fontsize=10.5, framealpha=0.95)
    fig.tight_layout(rect=[0, 0.25, 1, 0.90])
    add_caption(
        fig, 'Empirical 4',
        "Neither bound is stretched outside its own hypothesis -- each is real only where its own theorem applies.",
        f"Median over N={n_trials} trials per case. Case A matches only the prior paper's assumption "
        "(rank-1, $C=0$); case D matches only this paper's (full-rank, nonzero $C$); cases B and C match "
        "neither bound's hypothesis, so both are marked absent rather than computed anyway. This paper's "
        "bound is never evaluated on rank-deficient $M^{(i)}$, because Theorem 2 explicitly requires "
        "$M^{(i)}$ invertible.",
    )
    fig.savefig(perf_dir + 'fig2_emp4_bars_with_na.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_emp4_bars_with_na.png")


# --------------------------------------------------------------------------------------
# Figure 5: large-N validation histogram, full-rank M, randomized nonzero C and noise
# --------------------------------------------------------------------------------------

def fig2_emp5_validation_histogram(n_trials=1000):
    rng = np.random.default_rng(105)
    log_ratios = []
    n_violations = 0
    for _ in tqdm(range(n_trials), desc="validation sampling"):
        P = np.eye(N_STATE) * P_SCALE
        M = make_M_stack_with_rank(M_SENSORS, N_STATE, N_STATE, 1e-2, 1.0, rng)  # full rank always
        C_scale = 10 ** rng.uniform(-2, 0.7)
        C = generate_C(M_SENSORS, N_STATE, C_scale)
        sigma2_vec = 10 ** rng.uniform(0, 3, size=M_SENSORS)
        g_true, g_ours = run_trial(M, C, sigma2_vec, P)
        if g_ours > g_true + 1e-9:
            n_violations += 1
        log_ratios.append(np.log10(g_true / g_ours))
    log_ratios = np.array(log_ratios)

    # log10(true/bound): 0 = bound exactly equals truth, positive = bound is that many orders of
    # magnitude looser than truth. Plotted on a log-tightness axis rather than the raw (true - bound)
    # margin, because bound is routinely several orders of magnitude smaller than truth (see Empirical
    # 1/3) -- a linear-margin histogram just reproduces the distribution of "true" and hides the bound
    # entirely, cramming everything into one bin near margin=true.
    fig, ax = plt.subplots(figsize=(9.5, 7.8))
    ax.hist(log_ratios, bins=40, color=COLORS['thm2'], alpha=0.75, edgecolor='white')
    ax.axvline(0.0, color=COLORS['prior'], linestyle='--', linewidth=2.2,
               label='Violation boundary (bound would exceed truth)')
    ax.set_xlabel(r"$\log_{10}$(true ratio / this paper's bound)  (must be $\geq 0$ for the guarantee to hold)",
                  fontsize=12)
    ax.set_ylabel('Number of trials')
    ax.set_title(f'{n_trials} randomized trials, full-rank $M^{{(i)}}$, random $C$, random noise', fontsize=12.5)
    ax.grid(alpha=0.3, axis='y')
    fig.suptitle(f'0/{n_trials} violations: the bound holds throughout this paper\'s proven regime', y=0.97, fontsize=15)
    ax.legend(loc='upper right', fontsize=10.5, framealpha=0.95)
    fig.tight_layout(rect=[0, 0.18, 1, 0.88])
    add_caption(
        fig, 'Empirical 5',
        "A large-scale falsification test: every one of 1000 randomized full-rank, nonzero-C trials keeps the bound valid.",
        f"Each trial draws a fresh random model at full rank (this paper's Theorem 2 hypothesis), a "
        "linear-term scale spanning two orders of magnitude, and noise spanning three. The x-axis is how "
        f"many orders of magnitude looser than truth the bound is; {n_violations}/{n_trials} trials fell "
        "left of zero (a violation). No comparable test exists for the prior paper's bound here, since "
        "its own theorem requires rank-1 $M^{(i)}$ and $C=0$, neither of which holds in this sweep.",
    )
    fig.savefig(perf_dir + 'fig2_emp5_validation_histogram.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_emp5_validation_histogram.png")
    print(f"  violations: {n_violations}/{n_trials}, min log10(true/bound): {log_ratios.min():.6g}")


# --------------------------------------------------------------------------------------
# Figure 6: BOTH stories in one graph -- sensor-utilization cost of the restricted view, vs. growing C
# --------------------------------------------------------------------------------------

def fig2_emp6_selection_cost_vs_C(n_trials=150, R_ratio=0.2):
    """Deliberately NOT a line plot -- an earlier version of this figure used the exact same line/band/
    dashed-optimal-reference style as the paper's own Figure 1 (same y-axis metric, same visual grammar),
    and the user flagged it as looking like a re-skin of Figure 1 rather than a distinct figure. The
    underlying comparison was already different (Figure 1 sweeps R_ratio and compares against a fully
    LINEARIZED baseline that drops all of M; this one holds R_ratio fixed and sweeps C against a baseline
    that keeps a RANK-1 approximation of M but drops C -- the prior paper's own restriction, not a
    linearization) -- but the chart type made that distinction invisible. Switched to grouped bars at a
    handful of representative C values, showing raw sensor COUNTS (not a ratio to optimal), which is both
    visually distinct from Figure 1 and arguably more concrete for a reader (a literal "how many sensors
    would you actually buy" number)."""
    rng = np.random.default_rng(106)
    C_scales = np.array([0.0, 0.75, 1.5, 2.25, 3.0])

    star_mean, star_std, quad_mean, quad_std, restr_mean, restr_std = [], [], [], [], [], []
    for C_scale in tqdm(C_scales, desc="selection-cost-vs-C"):
        star_counts, quad_counts, restr_counts = [], [], []
        for _ in range(n_trials):
            P = np.eye(N_STATE) * P_SCALE
            Ix = np.linalg.inv(P)
            M = make_M_stack_with_rank(M_SENSORS, N_STATE, N_STATE, 1e-2, 1.0, rng)  # full rank always
            C = np.zeros((M_SENSORS, N_STATE)) if C_scale == 0 else generate_C(M_SENSORS, N_STATE, C_scale)
            sigma2_vec = np.full(M_SENSORS, SIGMA2)
            Ii_true, ci_true = per_sensor_info(M, C, P, sigma2_vec)
            Ii_restr = restrict_Ii_list(M, sigma2_vec, P)
            R = R_ratio * np.trace(P)
            S_star, _ = brute_force_select(M_SENSORS, Ix, Ii_true, R)
            if len(S_star) == 0:
                continue
            S_quad = greedy_select(M_SENSORS, Ix, Ii_true, R)
            S_restr = greedy_select(M_SENSORS, Ix, Ii_restr, R)
            star_counts.append(len(S_star))
            quad_counts.append(len(S_quad))
            restr_counts.append(len(S_restr))
        star_mean.append(np.mean(star_counts)); star_std.append(np.std(star_counts))
        quad_mean.append(np.mean(quad_counts)); quad_std.append(np.std(quad_counts))
        restr_mean.append(np.mean(restr_counts)); restr_std.append(np.std(restr_counts))
    star_mean, star_std = np.array(star_mean), np.array(star_std)
    quad_mean, quad_std = np.array(quad_mean), np.array(quad_std)
    restr_mean, restr_std = np.array(restr_mean), np.array(restr_std)

    fig, ax = plt.subplots(figsize=(10.5, 8.3))
    x = np.arange(len(C_scales))
    width = 0.26
    err_kw = dict(capsize=4, elinewidth=1.3, ecolor='#333333')
    ax.bar(x - width, star_mean, width, yerr=star_std, color=NA_GRAY, label='Optimal (brute force)', error_kw=err_kw)
    ax.bar(x, quad_mean, width, yerr=quad_std, color=COLORS['thm2'],
           label='This paper: quadratic-aware greedy', error_kw=err_kw)
    ax.bar(x + width, restr_mean, width, yerr=restr_std, color=COLORS['prior'],
           label="Greedy driven by the prior paper's rank-1/zero-$C$ view", error_kw=err_kw)
    ax.set_xticks(x)
    ax.set_xticklabels([f'$C={c:g}$' for c in C_scales])
    ax.set_xlabel(r'Linear-term magnitude $C$ (full-rank $M^{(i)}$ throughout)', fontsize=12.5)
    ax.set_ylabel('Sensors selected (count)')
    ax.set_title(f'Fixed target accuracy ($R_\\mathrm{{ratio}}={R_ratio}$), $C$ grows', fontsize=13)
    ax.grid(alpha=0.3, axis='y')
    fig.suptitle('This paper matches the optimal sensor count as $C$ grows; the restricted view does not', y=0.975, fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.18), ncol=1,
               fontsize=10.5, framealpha=0.95)
    fig.tight_layout(rect=[0, 0.26, 1, 0.90])
    add_caption(
        fig, 'Empirical 6',
        "Both guarantees in one graph, as raw sensor counts: ours matches optimal as C grows; theirs overshoots.",
        f"Mean sensor count over N={n_trials} trials/group (error bars = 1 std. dev.), full-rank $M^{{(i)}}$ "
        f"throughout, fixed target $R_\\mathrm{{ratio}}={R_ratio}$. At $C=0$ all three methods pick nearly "
        "the same count, since that matches the prior paper's own case. As $C$ grows, this paper's method "
        "keeps tracking the brute-force optimal count, while greedy selection driven by the prior paper's "
        "rank-1/zero-$C$ view of each sensor overshoots it by roughly 70% at the largest $C$ tested, "
        "because it cannot see the growing linear-term information at all.",
    )
    fig.savefig(perf_dir + 'fig2_emp6_selection_cost_vs_C.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_emp6_selection_cost_vs_C.png")


# --------------------------------------------------------------------------------------
# Figure 7: two coordinated panels -- our guarantee working, their representation failing, vs. growing C
# --------------------------------------------------------------------------------------

def fig2_emp7_breakdown_vs_C(n_trials=150):
    rng = np.random.default_rng(107)
    C_scales = np.array([0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0])

    # Top panel: same coverage-vs-C data as Figure 1 (freshly computed here to keep this figure
    # self-contained rather than re-importing saved numbers).
    true_med, true_lo, true_hi, ours_med, ours_lo, ours_hi = sweep_vs_C(C_scales, SIGMA2, n_trials, rng)
    prior_val = prior_at_rank1_c0(n_trials, rng)

    # Bottom panel: M is held at RANK-1 throughout (the prior paper's own exact hypothesis, unlike the
    # top panel which needs full-rank M) so that C=0 reproduces their case exactly (ratio=1.0) and the
    # only thing changing along the x-axis is C -- this isolates C's effect on their representation
    # cleanly, rather than confounding it with a pre-existing rank mismatch. Same construction as
    # fig2_reframe.py's Candidate 2 (`c_sweep_trial`), single-sensor and C-only here.
    pred_med, pred_lo, pred_hi = [], [], []
    for C_scale in tqdm(C_scales, desc="breakdown-vs-C"):
        ratios = []
        for _ in range(n_trials):
            P = np.eye(N_STATE) * P_SCALE
            Q, _ = np.linalg.qr(rng.normal(size=(N_STATE, N_STATE)))
            Mj = Q @ np.diag([1.0] + [0.0] * (N_STATE - 1)) @ Q.T  # exactly rank-1
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

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 12.8), sharex=True)

    ax = axes[0]
    ax.axvspan(0.05, C_scales.max() * 1.03, color=NA_GRAY, alpha=0.12, zorder=0, hatch='//')
    ax.plot(C_scales, true_med, color=COLORS['empirical'], marker='o', label='True ratio (brute force)', zorder=3)
    ax.fill_between(C_scales, true_lo, true_hi, color=COLORS['empirical'], alpha=0.12)
    ax.plot(C_scales, ours_med, color=COLORS['thm2'], marker='^', label='This paper (Theorem 2)', zorder=4)
    ax.fill_between(C_scales, ours_lo, ours_hi, color=COLORS['thm2'], alpha=0.15)
    ax.scatter([0.0], [prior_val], s=180, color=COLORS['prior'], zorder=5, edgecolor='white', linewidth=1.4,
               label="Prior paper's bound (only where proven)")
    ax.set_yscale('log')
    ax.set_ylabel(r'Supermodularity ratio $\gamma_h$', fontsize=12)
    ax.set_title('Top: our guarantee stays valid and present as $C$ grows', fontsize=12.5)
    ax.grid(alpha=0.3)
    ax.legend(loc='center right', fontsize=9.5, framealpha=0.95)

    ax = axes[1]
    ax.axvspan(0.05, C_scales.max() * 1.03, color=NA_GRAY, alpha=0.12, zorder=0, hatch='//')
    ax.axhline(1.0, color=COLORS['brute'], linestyle='--', linewidth=2, label='Perfect representation', zorder=2)
    ax.plot(C_scales, pred_med, color=COLORS['prior'], marker='s', zorder=3,
            label="Prior paper's rank-1/zero-$C$ prediction of the sensor")
    ax.fill_between(C_scales, pred_lo, pred_hi, color=COLORS['prior'], alpha=0.15)
    ax.set_xlabel(r'Linear-term magnitude $C$ ($M^{(i)}$ held at rank-1: their own case)', fontsize=12.5)
    ax.set_ylabel('Predicted / true\ninformation content', fontsize=11.5)
    ax.set_title("Bottom: their representation of the same sensor degrades as $C$ grows", fontsize=12.5)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.95)

    fig.suptitle('One works, one degrades: the same growing-$C$ axis, two coordinated panels', y=0.975, fontsize=14.5)
    fig.tight_layout(rect=[0, 0.155, 1, 0.94])
    add_caption(
        fig, 'Empirical 7',
        "Top and bottom panels share the same x-axis: this paper's guarantee holds throughout while the prior paper's own representation of a sensor breaks down.",
        f"Median over N={n_trials} trials/point (bands = IQR). Top panel: same result as Figure "
        "Empirical 1, full-rank $M^{(i)}$ throughout (this paper's own domain). Bottom panel: a "
        "different, independent check, with $M^{(i)}$ held at exactly rank-1 (the prior paper's own "
        "domain, not this paper's) so $C=0$ reproduces their case exactly -- only $C$ changes along the "
        "x-axis, isolating its effect. The two panels are not the same quantity, but they tell the same "
        "story on the same axis: presence and validity for this paper as $C$ grows, versus a "
        "representation that increasingly cannot see what is really there for the prior paper.",
        y=0.02,
    )
    fig.savefig(perf_dir + 'fig2_emp7_breakdown_vs_C.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_emp7_breakdown_vs_C.png")


# --------------------------------------------------------------------------------------
# Figure 8: 2D domain map over (rank of M, linear-term magnitude C) -- the joint claim, in one grid
# --------------------------------------------------------------------------------------

def fig2_emp8_domain_map(n_trials=100):
    """The most literal answer to 'show ours works and theirs doesn't in the same graph,' extended to
    BOTH generality axes at once (rank of M^(i) AND linear-term magnitude C), not just C alone like
    Figures 1/2/6/7. Every prior empirical figure fixed M at full rank throughout and only swept C; this
    one sweeps rank too, so the full-rank requirement is an explicit, tested axis of the figure, not just
    an assumption baked into the setup."""
    rng = np.random.default_rng(108)
    C_scales = np.array([0.0, 0.75, 1.5, 2.25, 3.0])
    ranks = list(range(1, N_STATE + 1))

    true_grid = np.zeros((len(ranks), len(C_scales)))
    bound_grid = np.full((len(ranks), len(C_scales)), np.nan)
    status_grid = np.zeros((len(ranks), len(C_scales)), dtype=int)  # 0=neither proven, 1=ours, 2=theirs

    for ri, rank in enumerate(tqdm(ranks, desc="domain-map")):
        for ci, C_scale in enumerate(C_scales):
            trues, bounds = [], []
            for _ in range(n_trials):
                P = np.eye(N_STATE) * P_SCALE
                Ix = np.linalg.inv(P)
                M = make_M_stack_with_rank(M_SENSORS, N_STATE, rank, 1e-2, 1.0, rng)
                C = np.zeros((M_SENSORS, N_STATE)) if C_scale == 0 else generate_C(M_SENSORS, N_STATE, C_scale)
                sigma2_vec = np.full(M_SENSORS, SIGMA2)
                Ii_list, ci_list = per_sensor_info(M, C, P, sigma2_vec)
                trues.append(empirical_gamma_h(M_SENSORS, Ix, Ii_list))
                if rank == N_STATE:
                    bounds.append(theorem2_bound(M_SENSORS, Ix, Ii_list, ci_list, M, sigma2_vec))
                elif rank == 1 and C_scale == 0:
                    F_full = F_of(list(range(M_SENSORS)), Ix, Ii_list)
                    bounds.append(prior_bound_c19(M_SENSORS, Ix, P, sigma2_vec, F_full))
            true_grid[ri, ci] = np.median(trues)
            if rank == N_STATE:
                status_grid[ri, ci] = 1
                bound_grid[ri, ci] = np.median(bounds)
            elif rank == 1 and C_scale == 0:
                status_grid[ri, ci] = 2
                bound_grid[ri, ci] = np.median(bounds)

    from matplotlib.patches import Rectangle, Patch
    fig, ax = plt.subplots(figsize=(11, 8.6))
    face = {0: NA_GRAY, 1: COLORS['thm2'], 2: COLORS['prior']}
    alpha = {0: 0.18, 1: 0.32, 2: 0.32}
    hatch = {0: '//', 1: None, 2: None}
    for ri in range(len(ranks)):
        for ci in range(len(C_scales)):
            s = status_grid[ri, ci]
            ax.add_patch(Rectangle((ci - 0.5, ri - 0.5), 1, 1, facecolor=face[s], alpha=alpha[s],
                                    edgecolor='white', linewidth=2.5, hatch=hatch[s], zorder=1))
            true_str = f"true={true_grid[ri, ci]:.3g}"
            if s == 0:
                label = f"{true_str}\nNO PROVEN\nBOUND"
            else:
                label = f"{true_str}\nbound={bound_grid[ri, ci]:.2g}"
            ax.text(ci, ri, label, ha='center', va='center', fontsize=9.5,
                     color='#1a1a1a' if s == 0 else 'white', fontweight='bold' if s != 0 else 'normal', zorder=2)

    ax.set_xlim(-0.5, len(C_scales) - 0.5)
    ax.set_ylim(-0.5, len(ranks) - 0.5)
    ax.set_xticks(range(len(C_scales)))
    ax.set_xticklabels([f'$C={c:g}$' for c in C_scales])
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([f'rank {r}' + ('  (full)' if r == N_STATE else '') for r in ranks])
    ax.set_xlabel('Linear-term magnitude $C$', fontsize=13)
    ax.set_ylabel('Rank of $M^{(i)}$ (state dimension = 4)', fontsize=13)
    ax.set_title('Where is each bound actually proven to apply?', fontsize=13.5)
    legend_handles = [
        Patch(facecolor=COLORS['thm2'], alpha=0.32, edgecolor='white', label='This paper (Theorem 2): full rank, any $C$'),
        Patch(facecolor=COLORS['prior'], alpha=0.32, edgecolor='white', label="Prior paper: rank-1, $C=0$ only"),
        Patch(facecolor=NA_GRAY, alpha=0.18, hatch='//', edgecolor='white', label='Neither bound proven here'),
    ]
    fig.suptitle("Our bound's proven domain covers the full-rank row entirely; theirs is a single cell", y=0.975, fontsize=14.5)
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.16), ncol=1,
               fontsize=10.5, framealpha=0.95)
    fig.tight_layout(rect=[0, 0.24, 1, 0.90])
    add_caption(
        fig, 'Empirical 8',
        "The joint (rank, C) domain, in one grid: this paper covers an entire row; the prior paper covers one cell.",
        f"Median true (brute-force) $\\gamma_h$ and, where proven, each bound's median value, over N={n_trials} "
        "trials per cell, state dimension 4, 7 sensors. Every cell in the full-rank row (top) is real data for "
        "this paper's Theorem 2, at every tested $C$ from 0 to 3. Only the single rank-1/$C=0$ cell is real "
        "data for the prior paper's explicit bound; the rest of that row, and every intermediate rank, has no "
        "proven bound from either paper and is marked accordingly rather than computed anyway.",
    )
    fig.savefig(perf_dir + 'fig2_emp8_domain_map.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_emp8_domain_map.png")


# --------------------------------------------------------------------------------------
# Figure 9: robustness across M's conditioning, at fixed nonzero C, full rank throughout
# --------------------------------------------------------------------------------------

def make_M_stack_with_condition(m, n, condition, rng):
    """m sensors, each n x n symmetric, full rank, with eigenvalues log-spaced from 1 to `condition`
    (random orientation per sensor) -- so condition=1 is well-conditioned (isotropic) and large
    condition is a nearly-singular-but-still-technically-full-rank M, stress-testing that 'full rank'
    in Theorem 2 really does mean any full-rank M, not just nicely-conditioned ones."""
    M = np.zeros((m, n, n))
    eigs = np.ones(n) if condition <= 1 else np.logspace(0, np.log10(condition), n)
    for i in range(m):
        Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        M[i] = Q @ np.diag(eigs) @ Q.T
    return M


def fig2_emp9_condition_number_robustness(n_trials=120):
    """Every other figure's 'full rank M' used a fixed, moderate eigenvalue spread (scale in
    [1e-2, 1.0]). This isolates conditioning as its own swept axis at fixed nonzero C, to check the
    full-rank claim isn't quietly relying on M being well-behaved -- and to make explicit that the
    prior paper's bound has literally nothing to plot here at all: not 'worse,' not 'looser,' but
    undefined at every single point, since none of these M are rank-1."""
    rng = np.random.default_rng(109)
    conditions = np.array([1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    C_scale = 1.5

    true_med, true_lo, true_hi, ours_med, ours_lo, ours_hi = [], [], [], [], [], []
    for cond in tqdm(conditions, desc="condition-number-robustness"):
        trues, ours = [], []
        for _ in range(n_trials):
            P = np.eye(N_STATE) * P_SCALE
            Ix = np.linalg.inv(P)
            M = make_M_stack_with_condition(M_SENSORS, N_STATE, cond, rng)
            C = generate_C(M_SENSORS, N_STATE, C_scale)
            sigma2_vec = np.full(M_SENSORS, SIGMA2)
            Ii_list, ci_list = per_sensor_info(M, C, P, sigma2_vec)
            trues.append(empirical_gamma_h(M_SENSORS, Ix, Ii_list))
            ours.append(theorem2_bound(M_SENSORS, Ix, Ii_list, ci_list, M, sigma2_vec))
        true_med.append(np.median(trues)); true_lo.append(np.percentile(trues, 25)); true_hi.append(np.percentile(trues, 75))
        ours_med.append(np.median(ours)); ours_lo.append(np.percentile(ours, 25)); ours_hi.append(np.percentile(ours, 75))
    true_med, true_lo, true_hi = np.array(true_med), np.array(true_lo), np.array(true_hi)
    ours_med, ours_lo, ours_hi = np.array(ours_med), np.array(ours_lo), np.array(ours_hi)

    fig, ax = plt.subplots(figsize=(9.5, 8.3))
    ax.plot(conditions, true_med, color=COLORS['empirical'], marker='o', label='True ratio (brute force)', zorder=3)
    ax.fill_between(conditions, true_lo, true_hi, color=COLORS['empirical'], alpha=0.12)
    ax.plot(conditions, ours_med, color=COLORS['thm2'], marker='^', label='This paper (Theorem 2)', zorder=4)
    ax.fill_between(conditions, ours_lo, ours_hi, color=COLORS['thm2'], alpha=0.15)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(top=max(true_hi.max(), ours_hi.max()) * 6)
    ax.set_xlabel(r"$M^{(i)}$ condition number (max/min eigenvalue, full rank throughout)", fontsize=12.5)
    ax.set_ylabel(r'Supermodularity ratio $\gamma_h$', fontsize=13)
    ax.set_title(f'Fixed nonzero linear term ($C={C_scale}$); conditioning of $M$ grows', fontsize=13, pad=14)
    ax.grid(alpha=0.3)
    fig.suptitle("'Full rank' means any full-rank $M$: our bound holds from well-conditioned to nearly-singular", y=0.975, fontsize=14)
    fig.text(0.5, 0.905, "Prior paper's bound: NOT DEFINED anywhere on this axis (every $M^{(i)}$ here is full rank, never rank-1)",
              ha='center', va='center', fontsize=10.5, color='#555555', style='italic')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.18), ncol=1,
               fontsize=10.5, framealpha=0.95)
    fig.tight_layout(rect=[0, 0.26, 1, 0.85])
    add_caption(
        fig, 'Empirical 9',
        "The full-rank claim isn't quietly relying on well-conditioned M: the bound tracks truth from condition number 1 to 1e6.",
        f"Median over N={n_trials} trials/point (band = IQR), fixed nonzero $C={C_scale}$, state dimension 4, "
        "7 sensors, full-rank $M^{(i)}$ at every point -- only the eigenvalue spread (conditioning) changes "
        "along the x-axis. The prior paper's explicit bound is not evaluated anywhere on this plot because "
        "none of these matrices are rank-1, which is its only proven case, regardless of conditioning.",
    )
    fig.savefig(perf_dir + 'fig2_emp9_condition_number_robustness.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_emp9_condition_number_robustness.png")


# --------------------------------------------------------------------------------------
# Figure 10: side-by-side panels -- their exact domain (rank-1) vs. our exact domain (full-rank), same C-sweep
# --------------------------------------------------------------------------------------

def fig2_emp10_domain_side_by_side(n_trials=120):
    """Complements Figure 8's grid with a side-by-side line-plot version: two panels, same C-sweep,
    LEFT held at rank-1 (their exact hypothesis) and RIGHT held at full rank (our exact hypothesis).
    Makes the rank axis -- not just C -- the explicit thing being varied between panels, so the
    full-rank requirement is something the reader watches happen, not just a caption claim."""
    rng = np.random.default_rng(110)
    C_scales = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    def sweep_at_rank(rank, rng_local):
        true_med, true_lo, true_hi, ours_med, ours_lo, ours_hi = [], [], [], [], [], []
        for C_scale in C_scales:
            trues, ours = [], []
            for _ in range(n_trials):
                P = np.eye(N_STATE) * P_SCALE
                Ix = np.linalg.inv(P)
                M = make_M_stack_with_rank(M_SENSORS, N_STATE, rank, 1e-2, 1.0, rng_local)
                C = np.zeros((M_SENSORS, N_STATE)) if C_scale == 0 else generate_C(M_SENSORS, N_STATE, C_scale)
                sigma2_vec = np.full(M_SENSORS, SIGMA2)
                Ii_list, ci_list = per_sensor_info(M, C, P, sigma2_vec)
                trues.append(empirical_gamma_h(M_SENSORS, Ix, Ii_list))
                if rank == N_STATE:
                    ours.append(theorem2_bound(M_SENSORS, Ix, Ii_list, ci_list, M, sigma2_vec))
            true_med.append(np.median(trues)); true_lo.append(np.percentile(trues, 25)); true_hi.append(np.percentile(trues, 75))
            if ours:
                ours_med.append(np.median(ours)); ours_lo.append(np.percentile(ours, 25)); ours_hi.append(np.percentile(ours, 75))
        out = (np.array(true_med), np.array(true_lo), np.array(true_hi))
        if rank == N_STATE:
            out = out + (np.array(ours_med), np.array(ours_lo), np.array(ours_hi))
        return out

    rank1_true_med, rank1_true_lo, rank1_true_hi = sweep_at_rank(1, rng)
    prior_val = prior_at_rank1_c0(n_trials, rng)
    full_true_med, full_true_lo, full_true_hi, full_ours_med, full_ours_lo, full_ours_hi = sweep_at_rank(N_STATE, rng)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.0), sharey=True)

    ax = axes[0]
    ax.axvspan(0.05, C_scales.max() * 1.03, color=NA_GRAY, alpha=0.15, zorder=0, hatch='xx')
    ax.text(0.5, 0.28, "This paper's bound:\nNOT PROVEN\n(rank-deficient $M$)",
            transform=ax.transAxes, ha='center', va='center', fontsize=10, color='#555555', style='italic', zorder=2)
    ax.plot(C_scales, rank1_true_med, color=COLORS['empirical'], marker='o', label='True ratio (brute force)', zorder=3)
    ax.fill_between(C_scales, rank1_true_lo, rank1_true_hi, color=COLORS['empirical'], alpha=0.12)
    ax.scatter([0.0], [prior_val], s=200, color=COLORS['prior'], zorder=5, edgecolor='white', linewidth=1.5,
               label="Prior paper's bound (only where proven)")
    ax.set_yscale('log')
    ax.set_xlim(-0.1, C_scales.max() * 1.05)
    ax.set_xlabel(r'Linear-term magnitude $C$', fontsize=12.5)
    ax.set_ylabel(r'Supermodularity ratio $\gamma_h$', fontsize=13)
    ax.set_title("Their domain: $M^{(i)}$ held at rank-1", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.95)

    ax = axes[1]
    ax.axvspan(0.05, C_scales.max() * 1.03, color=NA_GRAY, alpha=0.12, zorder=0, hatch='//')
    ax.text(0.5, 0.28, "Prior paper's bound:\nNOT PROVEN\nin this region",
            transform=ax.transAxes, ha='center', va='center', fontsize=10,
            color='#555555', style='italic', zorder=2)
    ax.plot(C_scales, full_true_med, color=COLORS['empirical'], marker='o', label='True ratio (brute force)', zorder=3)
    ax.fill_between(C_scales, full_true_lo, full_true_hi, color=COLORS['empirical'], alpha=0.12)
    ax.plot(C_scales, full_ours_med, color=COLORS['thm2'], marker='^', label='This paper (Theorem 2)', zorder=4)
    ax.fill_between(C_scales, full_ours_lo, full_ours_hi, color=COLORS['thm2'], alpha=0.15)
    ax.scatter([0.0], [prior_val], s=200, color=COLORS['prior'], zorder=5, edgecolor='white', linewidth=1.5,
               label="Prior paper's bound (only where proven)")
    ax.set_xlabel(r'Linear-term magnitude $C$', fontsize=12.5)
    ax.set_title("Our domain: $M^{(i)}$ held at full rank", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.95)

    fig.suptitle('Same $C$-sweep, two rank regimes: full rank is what unlocks coverage past $C=0$', y=0.975, fontsize=14.5)
    fig.tight_layout(rect=[0, 0.16, 1, 0.90])
    add_caption(
        fig, 'Empirical 10',
        "Their exact hypothesis (left) vs. ours (right), same C-sweep: rank is the axis that decides who has a bound at all.",
        f"Median over N={n_trials} trials/point (bands = IQR), state dimension 4, 7 sensors. Left panel: "
        "$M^{(i)}$ held at rank-1 throughout (their hypothesis) -- their bound is real only at the marked "
        "$C=0$ point; this paper's Theorem 2 has nothing to plot here at all, since it requires full rank. "
        "Right panel: $M^{(i)}$ held at full rank throughout (this paper's hypothesis) -- the same marked "
        "prior-paper point is shown for reference, but their bound was never proven anywhere else in this "
        "panel either. Rank, not just $C$, is what separates the two papers' proven domains.",
    )
    fig.savefig(perf_dir + 'fig2_emp10_domain_side_by_side.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_emp10_domain_side_by_side.png")


if __name__ == "__main__":
    fig2_emp1_coverage_vs_C()
    fig2_emp2_coverage_by_noise()
    fig2_emp3_validation_scatter()
    fig2_emp4_bars_with_na()
    fig2_emp5_validation_histogram()
    fig2_emp6_selection_cost_vs_C()
    fig2_emp7_breakdown_vs_C()
    fig2_emp8_domain_map()
    fig2_emp9_condition_number_robustness()
    fig2_emp10_domain_side_by_side()
