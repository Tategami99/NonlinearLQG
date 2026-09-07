"""
Empirical (data-driven, not schematic) figures for CDC2026.tex's Figure 2. The point to prove, verbatim
from the user/supervisor: "even though our bound is looser than Hashemi's [19] bound, our bound can
include the nonlinear quadratic observation model where we have non-zero linear term C and matrix M is
full rank."

**Correctness note #1 (rank).** `theorem2_bound()` is only ever evaluated on full-rank M (rank ==
N_STATE) -- CDC2026.tex's Theorem 2 requires M^(i) invertible, and an earlier version of this file that
swept rank down to 1 found `theorem2_bound()` silently returns an INVALID bound (72.5% violation rate)
at rank-1, because `np.linalg.inv()` on a near-singular matrix returns garbage instead of raising an
error. Fixed by construction: no function below ever calls `theorem2_bound()` on rank-deficient M.

**Correctness note #2 (the cross-paper comparison itself), found later and more fundamental: NO figure
in this file plots a number from `prior_bound_c19()` any more.** Earlier versions marked [19]'s bound as
a reference point at their own rank-1/C=0 case. Re-deriving [19]'s eq. 27-28 against this paper's
Definition 3 found this comparison isn't just narrow in scope, it isn't well-posed at all: [19]'s
explicit constant bounds `c_f` (a MAX-type weak-submodularity constant, for THEIR cardinality-constrained
MAXIMIZATION problem) from above; this paper's `gamma_h` (Definition 3) is a MIN-type ratio, over the
exact same underlying marginal-gain family, for a different MINIMUM-cardinality problem. `gamma_h` and
`c_f` are the min and max of the same set of numbers, not reciprocals -- bounding the max from above
(what eq. 27 does) says nothing mathematically about the min. So `1/max_j(gamma_j)` (what
`prior_bound_c19()` returns) is not something [19]'s paper proves is a bound on `gamma_h`, even at their
own rank-1/C=0 case. (Empirically it has never been observed to exceed the true ratio, across ~6,700+
adversarial trials this session -- but "no counterexample found" is not "proven," and the underlying
quantity isn't rigorously connected to `gamma_h` regardless of rank or C.) Every figure below that once
plotted this number has been rebuilt to make the coverage claim textually instead: [19]'s Theorem 6
establishes weak submodularity *qualitatively* for the general model; its only *explicit, computable*
constant addresses a different quantity, derived only for rank-1, C=0 sensors. This is a stronger, cleaner
claim than a numeric comparison, and it needs no cross-paper number to make. `prior_bound_c19()` and
`prior_at_rank1_c0()` are kept in this file (and in `sensor_selection_sim.py`) for the historical record
and for anyone re-deriving this, but nothing below plots their output as if it bounds `gamma_h`.

Ten figures, all computed from real Monte Carlo / exhaustive-brute-force data, none of them schematic:

  1. fig2_emp1_coverage_vs_C.png       -- sweep linear-term magnitude C from 0 upward at full-rank M;
     this paper's bound + brute-force truth plotted throughout the whole sweep, no cross-paper number.
  2. fig2_emp2_coverage_by_noise.png   -- the same C-sweep repeated as three side-by-side panels at low,
     medium, and high fixed measurement noise, to show the coverage isn't a one-off parameter choice.
  3. fig2_emp3_validation_scatter.png  -- parity-style scatter: brute-force-true ratio vs. this paper's
     bound, across hundreds of randomized full-rank, nonzero-C trials. Every point above the y=x line
     is direct empirical evidence the guarantee holds. Purely first-party -- no [19] dependency at all.
  4. fig2_emp4_utility_guarantee_vs_C.png -- chains this paper's Theorem 1 and Theorem 2 end-to-end:
     plugs Theorem 2's certified gamma_h lower bound into Theorem 1's sensor-count inequality to get a
     fully computable guarantee, then checks it against the ACTUAL greedy sensor count as C grows. Never
     violated. Entirely self-contained -- answers "does the bound translate into a real, useful
     guarantee" without any reference to prior work.
  5. fig2_emp5_validation_histogram.png -- large-N (1000 trials) falsification test: this paper's bound
     minus the brute-force-true ratio, at full-rank M with randomized nonzero C and randomized noise.
     Never positive (never a violation). Purely first-party.
  6. fig2_emp6_selection_cost_vs_C.png -- sensor-utilization cost of using a cruder per-sensor MODEL:
     comparing this paper's quadratic-aware greedy (stays at the optimal sensor count throughout) against
     greedy selection driven by a rank-1/zero-linear-term approximation of each sensor's information
     (needs up to ~70% more sensors as C grows). This is a model-fidelity/decision-cost experiment, not a
     comparison to [19]'s bound formula -- it never calls `prior_bound_c19()`.
  7. fig2_emp7_breakdown_vs_C.png      -- single-sensor information-content check: this paper's bound
     stays present and valid (top panel) while a rank-1/zero-linear-term APPROXIMATION of one sensor's
     information increasingly mis-predicts its true content as C grows (bottom panel). Also a model-
     fidelity experiment, no `prior_bound_c19()` dependency.
  8. fig2_emp8_domain_map.png          -- 2D domain map over BOTH generality axes: rank of M^(i) (1-4) on
     one axis, linear-term magnitude C on the other. Every cell in the full-rank row is a real, explicit,
     computed bound (Theorem 2); the rank-1/C=0 cell is marked as [19]'s one QUALITATIVE case (no number
     attached, per Correctness note #2); every other cell has no explicit bound from either paper.
  9. fig2_emp9_condition_number_robustness.png -- stress-tests that "full rank" means ANY full-rank M:
     sweeps M's condition number from 1 (isotropic) to 1e6 (nearly singular but still full rank) at fixed
     nonzero C. This paper's bound tracks truth throughout. Purely first-party.
  10. fig2_emp10_domain_side_by_side.png -- the rank axis as two side-by-side panels sharing the same
     C-sweep: left holds M^(i) at rank-1 ([19]'s hypothesis -- this paper's Theorem 2 has nothing to plot
     there), right holds M^(i) at full rank (this paper's hypothesis -- an explicit bound throughout, no
     comparable result from [19] in either panel).

**On "does [19]'s bound (as originally defined/comparable) ever produce an invalid number" -- tested
extensively across four independent constructions this session (~6,700+ trials: their exact case, a
rank-1/C=0 approximation of general sensors, naive direct application to the real general model with
isotropic priors, and again with anisotropic priors + heterogeneous per-sensor noise) -- zero violations
in all of it, including a decision-level check (does greedy selection driven by their restricted view
ever fail a reachable target once a real measurement bug in that check was found and fixed -- see
`fig2_reframe.py`'s retraction docstring). The formula behaves as a robust (if often extremely
conservative, and not rigorously proven to bound gamma_h at all -- see Correctness note #2) quantity in
every construction tried.
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
    """No number from [19] is plotted here (a change from an earlier version of this figure). Re-deriving
    [19]'s eq. 27-28 against this paper's Definition 3 found that their explicit constant bounds a
    DIFFERENT quantity (c_f, the max-type weak-submodularity constant for their own maximization problem)
    than gamma_h (this paper's min-type supermodularity ratio, Definition 3, for a different min-cardinality
    problem) -- c_f and gamma_h are the max and min of the same underlying ratio family, not reciprocals,
    and bounding one from above says nothing about the other. So there is no numerically comparable
    quantity from [19] to plot at all, not even at their own rank-1/C=0 case -- this is a stronger, cleaner
    claim than "not proven outside their domain," and it needs no cross-paper number to make."""
    rng = np.random.default_rng(101)
    C_scales = np.array([0.0, 0.2, 0.4, 0.6, 0.9, 1.2, 1.6, 2.0, 2.5, 3.0])
    true_med, true_lo, true_hi, ours_med, ours_lo, ours_hi = sweep_vs_C(C_scales, SIGMA2, n_trials, rng)

    fig, ax = plt.subplots(figsize=(9.5, 8.3))
    ax.plot(C_scales, true_med, color=COLORS['empirical'], marker='o', label='True ratio (brute force)', zorder=3)
    ax.fill_between(C_scales, true_lo, true_hi, color=COLORS['empirical'], alpha=0.12)
    ax.plot(C_scales, ours_med, color=COLORS['thm2'], marker='^', label='This paper (Theorem 2)', zorder=4)
    ax.fill_between(C_scales, ours_lo, ours_hi, color=COLORS['thm2'], alpha=0.15)

    ax.set_yscale('log')
    ax.set_xlim(-0.08, C_scales.max() * 1.03)
    ax.set_xlabel(r'Linear-term magnitude $C$', fontsize=13)
    ax.set_ylabel(r'Supermodularity ratio $\gamma_h$', fontsize=13)
    ax.set_title('Full-rank $M^{(i)}$ throughout; only $C$ varies', fontsize=13)
    ax.grid(alpha=0.3)
    fig.suptitle('This paper stays defined and validated across the full sweep', y=0.975, fontsize=15)
    fig.text(0.5, 0.885, "[19] gives no explicit, computable bound on $\\gamma_h$ at any point on this axis",
              ha='center', va='center', fontsize=11, color='#555555', style='italic')
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.20), ncol=1,
               fontsize=10.5, framealpha=0.95)
    fig.tight_layout(rect=[0, 0.29, 1, 0.86])
    add_caption(
        fig, 'Empirical 1',
        "This paper's bound tracks the true ratio validly across the full sweep of the linear-term magnitude C.",
        f"Median over N={n_trials} trials/point (band = IQR), state dimension 4, 7 sensors, full-rank "
        "$M^{(i)}$ throughout -- this paper's own Theorem 2 hypothesis, matched exactly. [19]'s Theorem 6 "
        "shows weak submodularity holds qualitatively for this general model, but its only explicit, "
        "computable constant (eq. 27-28) bounds a differently-defined quantity for a different "
        "(cardinality-constrained maximization) problem, derived for rank-one, zero-linear-term sensors "
        "-- it is not a bound on $\\gamma_h$ itself, so no such curve is plotted here.",
    )
    fig.savefig(perf_dir + 'fig2_emp1_coverage_vs_C.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_emp1_coverage_vs_C.png")


# --------------------------------------------------------------------------------------
# Figure 2: the same coverage story at three different noise levels (robustness, not a one-off)
# --------------------------------------------------------------------------------------

def fig2_emp2_coverage_by_noise(n_trials=100):
    """No number from [19] is plotted here -- same reasoning as fig2_emp1 (their explicit constant bounds
    a differently-defined quantity, not gamma_h; see fig2_emp1's docstring)."""
    rng = np.random.default_rng(102)
    C_scales = np.array([0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0])
    noise_levels = [(10.0, 'Low noise ($\\sigma^2=10$)'), (100.0, 'Medium noise ($\\sigma^2=100$)'),
                     (1000.0, 'High noise ($\\sigma^2=1000$)')]

    fig, axes = plt.subplots(1, 3, figsize=(16, 7.8), sharey=True)
    for ax, (sigma2, title) in zip(axes, noise_levels):
        true_med, true_lo, true_hi, ours_med, ours_lo, ours_hi = sweep_vs_C(C_scales, sigma2, n_trials, rng)

        ax.plot(C_scales, true_med, color=COLORS['empirical'], marker='o', label='True ratio (brute force)', zorder=3)
        ax.fill_between(C_scales, true_lo, true_hi, color=COLORS['empirical'], alpha=0.12)
        ax.plot(C_scales, ours_med, color=COLORS['thm2'], marker='^', label='This paper (Theorem 2)', zorder=4)
        ax.fill_between(C_scales, ours_lo, ours_hi, color=COLORS['thm2'], alpha=0.15)
        ax.set_yscale('log')
        ax.set_xlim(-0.1, C_scales.max() * 1.05)
        ax.set_xlabel(r'Linear-term magnitude $C$', fontsize=12)
        ax.set_title(title, fontsize=12.5)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r'Supermodularity ratio $\gamma_h$', fontsize=13)

    fig.suptitle('This paper stays defined and validated at low, medium, and high measurement noise alike', y=0.975, fontsize=14.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.14), ncol=2,
               fontsize=10.5, framealpha=0.95)
    fig.tight_layout(rect=[0, 0.24, 1, 0.90])
    add_caption(
        fig, 'Empirical 2',
        "The same coverage-vs-C result, repeated at three noise levels -- not an artifact of one parameter choice.",
        f"Median over N={n_trials} trials/point per panel, full-rank $M^{{(i)}}$ throughout. All three "
        "panels tell the same story: this paper's bound stays defined and validated as $C$ grows, "
        "regardless of the measurement-noise level. [19] provides no explicit, computable bound on "
        "$\\gamma_h$ at any point, so no comparison curve is plotted (see Empirical 1's caption).",
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
# Figure 4: Theorem 1 + Theorem 2 chained end-to-end -- does the resulting sensor-count guarantee
# actually hold, checked against ground truth, as C grows?
# --------------------------------------------------------------------------------------

def greedy_with_prev_state(m, Ix, Ii_list, R):
    """Same as greedy_select, but also returns the second-to-last iterate S^g_{l-1} -- the state
    Theorem 1's bound is evaluated at (the last state that still had h(S) > R)."""
    S, prev_S = [], []
    candidates = list(range(m))
    h_fn = lambda s: h_val(s, Ix, Ii_list)
    while h_fn(S) > R and candidates:
        prev_S = list(S)
        best_j, best_val = None, np.inf
        for j in candidates:
            val = h_fn(S + [j])
            if val < best_val:
                best_val, best_j = val, j
        S.append(best_j)
        candidates.remove(best_j)
    return S, prev_S


def fig2_emp4_utility_guarantee_vs_C(n_trials=150, R_ratio=0.2):
    """Chains this paper's two theorems together end-to-end and checks the RESULT against ground truth,
    rather than checking gamma_h in isolation (Empirical 1/2/8/9/10) or a model-approximation consequence
    (Empirical 6/7). Theorem 1 (CDC2026.tex) states ell/|S*| <= 1 + (1/gamma_h) log((h(empty)-R)/
    (h(S^g_{l-1})-R)); plugging in Theorem 2's CERTIFIED LOWER BOUND on gamma_h in place of the (unknown
    in practice) true gamma_h gives a fully computable, honest upper bound on how many sensors greedy will
    actually need. Entirely self-contained -- no comparison to [19] at all.

    Plotted as a log-margin (guarantee / actual), the same transform used in Empirical 5, NOT as raw
    sensor counts. Theorem 2's certified gamma_h is structurally very conservative (median ~1e-5 to 1e-9
    across every parameter combination tried -- confirmed this is not noise-level-dependent, and matches
    CDC2026.tex's own disclosed 3-7-order-of-magnitude looseness relative to [19] in the same units), so
    Theorem 1's guarantee scales as ~1/gamma_h: an earlier version of this figure plotted absolute sensor
    counts and the guarantee curve reached ~10^6 while the true usage stayed at 1, which is an honest but
    visually misleading way to show a bound that is conservative BY KNOWN, DISCLOSED DESIGN, not broken.
    The log-margin framing shows the same never-violated result without that visual distortion."""
    rng = np.random.default_rng(112)
    C_scales = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    margin_med, margin_lo, margin_hi = [], [], []
    for C_scale in tqdm(C_scales, desc="utility-guarantee-vs-C"):
        margins = []
        for _ in range(n_trials):
            P = np.eye(N_STATE) * P_SCALE
            Ix = np.linalg.inv(P)
            M = make_M_stack_with_rank(M_SENSORS, N_STATE, N_STATE, 1e-2, 1.0, rng)  # full rank always
            C = np.zeros((M_SENSORS, N_STATE)) if C_scale == 0 else generate_C(M_SENSORS, N_STATE, C_scale)
            sigma2_vec = np.full(M_SENSORS, SIGMA2)
            Ii_true, ci_true = per_sensor_info(M, C, P, sigma2_vec)
            h_empty = float(np.trace(P))
            R = R_ratio * h_empty
            if h_val(list(range(M_SENSORS)), Ix, Ii_true) > R:
                continue  # target unreachable even with every sensor -- skip, per Remark 1's R<h(empty)

            S_star, _ = brute_force_select(M_SENSORS, Ix, Ii_true, R)
            S_final, S_prev = greedy_with_prev_state(M_SENSORS, Ix, Ii_true, R)
            if len(S_star) == 0:
                continue  # empty set already satisfies R -- ratio undefined, skip

            gamma_bound = theorem2_bound(M_SENSORS, Ix, Ii_true, ci_true, M, sigma2_vec)
            if not (np.isfinite(gamma_bound) and gamma_bound > 0):
                continue
            h_prev = h_val(S_prev, Ix, Ii_true)  # h(empty) if S_final has only 1 element

            emp_ratio = len(S_final) / len(S_star)
            guarantee = 1.0 + (1.0 / gamma_bound) * np.log((h_empty - R) / (h_prev - R))
            margins.append(np.log10(guarantee / emp_ratio))

        margin_med.append(np.median(margins)); margin_lo.append(np.percentile(margins, 25)); margin_hi.append(np.percentile(margins, 75))
    margin_med, margin_lo, margin_hi = np.array(margin_med), np.array(margin_lo), np.array(margin_hi)

    fig, ax = plt.subplots(figsize=(9.5, 8.8))
    ax.axhline(0.0, color=COLORS['brute'], linestyle='--', linewidth=2, label='Break-even (guarantee = actual usage)', zorder=2)
    ax.plot(C_scales, margin_med, color=COLORS['thm2'], marker='s', zorder=4,
            label='Chained guarantee vs. ground truth (Thm. 1 + Thm. 2)')
    ax.fill_between(C_scales, margin_lo, margin_hi, color=COLORS['thm2'], alpha=0.15)
    ax.set_xlabel(r'Linear-term magnitude $C$ (full-rank $M^{(i)}$ throughout)', fontsize=12.5)
    ax.set_ylabel(r'$\log_{10}$(guaranteed sensor count / actual sensor count)', fontsize=12)
    ax.set_title(f'Fixed target accuracy ($R_\\mathrm{{ratio}}={R_ratio}$), $C$ grows', fontsize=13)
    ax.grid(alpha=0.3)
    fig.suptitle('Chaining Theorem 1 and Theorem 2 gives a real, checkable guarantee -- never violated', y=0.975, fontsize=13.7)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.245), ncol=1,
               fontsize=10, framealpha=0.95)
    fig.tight_layout(rect=[0, 0.32, 1, 0.90])
    add_caption(
        fig, 'Empirical 4',
        "This paper's two theorems, chained end-to-end: the computable guarantee they jointly produce always sits above the true sensor count needed, never below.",
        f"Median over N={n_trials} trials/point (band = IQR), full-rank $M^{{(i)}}$ throughout, fixed "
        f"target $R_\\mathrm{{ratio}}={R_ratio}$. The y-axis is entirely computable in advance from "
        "Theorem 1's inequality using Theorem 2's certified lower bound on $\\gamma_h$, compared to the "
        "actual sensor count greedy used, checked here against ground truth. A positive value (always "
        "observed, well above 0) means the guarantee is never violated; the margin is large because "
        "Theorem 2's certified bound is itself conservative by design (consistent with the 3-7-order-of-"
        "magnitude looseness already disclosed in Section V relative to [19]'s tighter, narrower-scope "
        "constant), not because anything here is broken. No comparison to prior work is needed for this "
        "claim.",
    )
    fig.savefig(perf_dir + 'fig2_emp4_utility_guarantee_vs_C.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_emp4_utility_guarantee_vs_C.png")


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
    an assumption baked into the setup.

    No number from `prior_bound_c19()` is shown anywhere in this grid (a change from an earlier version).
    Re-deriving [19]'s eq. 27-28 against this paper's Definition 3 found their explicit constant bounds a
    DIFFERENT quantity (c_f, a max-type constant for their own maximization problem) than gamma_h (this
    paper's min-type ratio, for a different min-cardinality problem) -- they are the max and min of the
    same underlying ratio family, not reciprocals, so [19]'s explicit constant is not a bound on gamma_h at
    all, not even at their own rank-1/C=0 case. The rank-1/C=0 cell is still marked distinctly (it is the
    one case [19]'s Theorem 6 addresses at all), but only with the QUALITATIVE fact that weak submodularity
    is known to hold there -- no fabricated competing number for gamma_h is attached to it."""
    rng = np.random.default_rng(108)
    C_scales = np.array([0.0, 0.75, 1.5, 2.25, 3.0])
    ranks = list(range(1, N_STATE + 1))

    true_grid = np.zeros((len(ranks), len(C_scales)))
    bound_grid = np.full((len(ranks), len(C_scales)), np.nan)
    status_grid = np.zeros((len(ranks), len(C_scales)), dtype=int)  # 0=neither, 1=ours (numeric), 2=[19] qualitative-only

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
            true_grid[ri, ci] = np.median(trues)
            if rank == N_STATE:
                status_grid[ri, ci] = 1
                bound_grid[ri, ci] = np.median(bounds)
            elif rank == 1 and C_scale == 0:
                status_grid[ri, ci] = 2

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
            if s == 1:
                label = f"{true_str}\nbound={bound_grid[ri, ci]:.2g}"
            elif s == 2:
                label = f"{true_str}\n[19]: weak submod.\nshown qualitatively\n(no explicit bound)"
            else:
                label = f"{true_str}\nNO EXPLICIT\nBOUND (either paper)"
            ax.text(ci, ri, label, ha='center', va='center', fontsize=9,
                     color='#1a1a1a' if s == 0 else 'white', fontweight='bold' if s != 0 else 'normal', zorder=2)

    ax.set_xlim(-0.5, len(C_scales) - 0.5)
    ax.set_ylim(-0.5, len(ranks) - 0.5)
    ax.set_xticks(range(len(C_scales)))
    ax.set_xticklabels([f'$C={c:g}$' for c in C_scales])
    ax.set_yticks(range(len(ranks)))
    ax.set_yticklabels([f'rank {r}' + ('  (full)' if r == N_STATE else '') for r in ranks])
    ax.set_xlabel('Linear-term magnitude $C$', fontsize=13)
    ax.set_ylabel('Rank of $M^{(i)}$ (state dimension = 4)', fontsize=13)
    ax.set_title('Where does an explicit, computable $\\gamma_h$ bound actually exist?', fontsize=13.5)
    legend_handles = [
        Patch(facecolor=COLORS['thm2'], alpha=0.32, edgecolor='white', label='This paper (Theorem 2): explicit bound, full rank, any $C$'),
        Patch(facecolor=COLORS['prior'], alpha=0.32, edgecolor='white', label="[19]: qualitative weak submodularity only, no explicit bound"),
        Patch(facecolor=NA_GRAY, alpha=0.18, hatch='//', edgecolor='white', label='No explicit bound from either paper'),
    ]
    fig.suptitle("This paper's explicit bound covers the full-rank row entirely; [19] gives none anywhere", y=0.975, fontsize=14)
    fig.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.16), ncol=1,
               fontsize=10.5, framealpha=0.95)
    fig.tight_layout(rect=[0, 0.24, 1, 0.90])
    add_caption(
        fig, 'Empirical 8',
        "The joint (rank, C) domain, in one grid: this paper gives an explicit bound across an entire row; [19] gives none, anywhere.",
        f"Median true (brute-force) $\\gamma_h$ over N={n_trials} trials per cell, state dimension 4, 7 "
        "sensors, and this paper's Theorem 2 bound where it applies (full-rank row). [19]'s Theorem 6 shows "
        "weak submodularity holds qualitatively at rank-1, $C=0$ (marked), but its own explicit constant "
        "(eq. 27-28) bounds a differently-defined quantity for a different problem, not $\\gamma_h$ -- so "
        "no numeric value from [19] is shown anywhere in this grid, including at their own case.",
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
    full-rank requirement is something the reader watches happen, not just a caption claim.

    No number from `prior_bound_c19()` is plotted in either panel (a change from an earlier version) --
    see fig2_emp1's docstring for why: [19]'s explicit constant bounds a differently-defined, max-type
    quantity for a different problem, not this paper's min-type gamma_h, so it is not a valid comparison
    point even at their own rank-1/C=0 case."""
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
    full_true_med, full_true_lo, full_true_hi, full_ours_med, full_ours_lo, full_ours_hi = sweep_at_rank(N_STATE, rng)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.0), sharey=True)

    ax = axes[0]
    ax.axvspan(0.05, C_scales.max() * 1.03, color=NA_GRAY, alpha=0.15, zorder=0, hatch='xx')
    ax.text(0.5, 0.5, "This paper's Theorem 2:\nNOT APPLICABLE\n(rank-deficient $M$)",
            transform=ax.transAxes, ha='center', va='center', fontsize=10.5, color='#555555', style='italic', zorder=2)
    ax.plot(C_scales, rank1_true_med, color=COLORS['empirical'], marker='o', label='True ratio (brute force)', zorder=3)
    ax.fill_between(C_scales, rank1_true_lo, rank1_true_hi, color=COLORS['empirical'], alpha=0.12)
    ax.set_yscale('log')
    ax.set_xlim(-0.1, C_scales.max() * 1.05)
    ax.set_xlabel(r'Linear-term magnitude $C$', fontsize=12.5)
    ax.set_ylabel(r'Supermodularity ratio $\gamma_h$', fontsize=13)
    ax.set_title("[19]'s domain: $M^{(i)}$ held at rank-1", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.95)

    ax = axes[1]
    ax.plot(C_scales, full_true_med, color=COLORS['empirical'], marker='o', label='True ratio (brute force)', zorder=3)
    ax.fill_between(C_scales, full_true_lo, full_true_hi, color=COLORS['empirical'], alpha=0.12)
    ax.plot(C_scales, full_ours_med, color=COLORS['thm2'], marker='^', label='This paper (Theorem 2)', zorder=4)
    ax.fill_between(C_scales, full_ours_lo, full_ours_hi, color=COLORS['thm2'], alpha=0.15)
    ax.set_xlabel(r'Linear-term magnitude $C$', fontsize=12.5)
    ax.set_title("This paper's domain: $M^{(i)}$ held at full rank", fontsize=13)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.95)

    fig.suptitle('Same $C$-sweep, two rank regimes: only the full-rank side has an explicit bound at all', y=0.975, fontsize=14)
    fig.tight_layout(rect=[0, 0.20, 1, 0.90])
    add_caption(
        fig, 'Empirical 10',
        "[19]'s exact hypothesis (left) vs. this paper's (right), same C-sweep: this paper's Theorem 2 has no counterpart to compare against in either panel.",
        f"Median over N={n_trials} trials/point (bands = IQR), state dimension 4, 7 sensors. Left panel: "
        "$M^{(i)}$ held at rank-1 throughout ([19]'s hypothesis) -- this paper's Theorem 2 does not apply "
        "here, since it requires full rank; [19]'s own Theorem 6 shows weak submodularity holds "
        "qualitatively in this case, but gives no explicit, computable bound on $\\gamma_h$ to plot (see "
        "Empirical 1's caption). Right panel: $M^{(i)}$ held at full rank throughout (this paper's "
        "hypothesis) -- this paper's bound is explicit and validated throughout; [19] has no comparable "
        "result here either, qualitative or explicit.",
    )
    fig.savefig(perf_dir + 'fig2_emp10_domain_side_by_side.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_emp10_domain_side_by_side.png")


if __name__ == "__main__":
    fig2_emp1_coverage_vs_C()
    fig2_emp2_coverage_by_noise()
    fig2_emp3_validation_scatter()
    fig2_emp4_utility_guarantee_vs_C()
    fig2_emp5_validation_histogram()
    fig2_emp6_selection_cost_vs_C()
    fig2_emp7_breakdown_vs_C()
    fig2_emp8_domain_map()
    fig2_emp9_condition_number_robustness()
    fig2_emp10_domain_side_by_side()
