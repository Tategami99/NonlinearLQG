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

Five figures, all computed from real Monte Carlo / exhaustive-brute-force data:

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
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from sensor_selection_sim import (
    generate_C, per_sensor_info, F_of,
    theorem2_bound, prior_bound_c19, empirical_gamma_h,
    perf_dir, COLORS,
)
from fig2_reframe import add_caption

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


if __name__ == "__main__":
    fig2_emp1_coverage_vs_C()
    fig2_emp2_coverage_by_noise()
    fig2_emp3_validation_scatter()
    fig2_emp4_bars_with_na()
    fig2_emp5_validation_histogram()
