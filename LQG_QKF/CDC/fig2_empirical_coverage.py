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
claim than a numeric comparison, and it needs no cross-paper number to make. `prior_bound_c19()` is kept
in `sensor_selection_sim.py` for the historical record and for anyone re-deriving this, but nothing below
plots its output as if it bounds `gamma_h`.

**Cleanup (2026-09-09).** This file originally built ten candidate figures while the right story for
Figure 2 was still being worked out with the supervisor. Two were chosen (`fig2_emp6_selection_cost_vs_C`
and `fig2_emp7_breakdown_vs_C`) and the other eight (emp1/2/3/4/5/8/9/10) were deleted, along with their
now-orphaned helpers (`prior_at_rank1_c0`, `greedy_with_prev_state`, `make_M_stack_with_condition`) and
their rendered PNGs -- all still recoverable from git history if any of them are needed again. Two
standalone TEST scripts (`fig2_emp7_TEST_naive_top_panel.py`, `fig2_emp7_TEST_exact_top_panel.py`, both
outside this file) explore alternative constructions for emp7's top panel that were tried and set aside;
see their own docstrings for why.

Two figures remain, both computed from real Monte Carlo / exhaustive-brute-force data:

  6. fig2_emp6_selection_cost_vs_C.png -- sensor-utilization cost of using a cruder per-sensor MODEL:
     comparing this paper's quadratic-aware greedy (stays at the optimal sensor count throughout) against
     greedy selection driven by a rank-1/zero-linear-term approximation of each sensor's information
     (needs up to ~70% more sensors as C grows). This is a model-fidelity/decision-cost experiment, not a
     comparison to [19]'s bound formula -- it never calls `prior_bound_c19()`.
  7. fig2_emp7_breakdown_vs_C.png      -- single-sensor information-content check: this paper's bound
     stays present and valid (top panel) while a rank-1/zero-linear-term APPROXIMATION of one sensor's
     information increasingly mis-predicts its true content as C grows (bottom panel). Also a model-
     fidelity experiment, no `prior_bound_c19()` dependency.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from sensor_selection_sim import (
    generate_C, per_sensor_info, h_val,
    greedy_select, brute_force_select,
    theorem2_bound, empirical_gamma_h,
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
    would you actually buy" number).

    **Fixed 2026-09-09: unreachable-target trials were not being excluded.** `brute_force_select` silently
    returns the full sensor set when no subset meets R, instead of signaling "unreachable" -- the same bug
    class retracted from `fig2_reframe.py`'s Candidate 4 earlier this session, present here too because
    this function never had the reachability guard added. At this function's own R_ratio=0.2, the vast
    majority of trials at low C are unreachable even using every sensor under the TRUE model (98% at C=0,
    93% at C=0.75, down to 3% at C=3.0) -- when unreachable, optimal/ours/restricted-view all saturate at
    the full 7-sensor set together, which mechanically hid the real gap between methods at low C. Fixed by
    skipping trials where `h_val(full set) > R` under the true model, matching the retraction fix. Effect:
    the headline overshoot at the largest C tested is basically unchanged (~65% -> ~69%), but the low-C
    story changes materially -- corrected data shows a REAL ~17% overshoot already present at C=0 (previously
    shown as ~0%), because restricting M to rank-1 is lossy on its own, independent of C. That's a stronger
    and more accurate story for this paper's "full rank, not just nonzero C" generality claim."""
    rng = np.random.default_rng(106)
    C_scales = np.array([0.0, 0.75, 1.5, 2.25, 3.0])

    star_mean, star_std, quad_mean, quad_std, restr_mean, restr_std = [], [], [], [], [], []
    n_skipped, n_total = 0, 0
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
            n_total += 1
            if h_val(list(range(M_SENSORS)), Ix, Ii_true) > R:
                n_skipped += 1
                continue  # unreachable even with every sensor under the true model -- skip explicitly
                          # rather than let brute_force_select's fallback silently saturate all three
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

    fig, ax = plt.subplots(figsize=(10.5, 8.8))
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
    fig.suptitle("This paper's guarantee gets easier to meet as $C$ grows; the restricted view is maxed out throughout", y=0.975, fontsize=13.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.25), ncol=1,
               fontsize=10.5, framealpha=0.95)
    fig.tight_layout(rect=[0, 0.36, 1, 0.90])
    add_caption(
        fig, 'Empirical 6',
        "Both guarantees in one graph, as raw sensor counts: ours tracks the true optimal, which drops as C adds usable information; theirs needs every available sensor at every C tested, including C=0.",
        f"Mean sensor count over {n_total - n_skipped} of {n_total} sampled trials ({n_skipped} were "
        "unreachable even using every sensor under the true model and were excluded; error bars = 1 std. "
        "dev.), full-rank $M^{(i)}$ throughout, fixed target "
        f"$R_\\mathrm{{ratio}}={R_ratio}$. Even at $C=0$, the restricted view already needs all 7 available "
        "sensors -- about 17% more than the true optimal at that point -- because approximating each "
        "full-rank $M^{(i)}$ by its dominant rank-1 component is lossy on its own, independent of $C$; this "
        "is the rank axis of this paper's generality claim. As $C$ grows, the true optimal count keeps "
        "DROPPING (more usable information per sensor means fewer sensors are needed), but the restricted "
        "view cannot see any of that growing linear-term information, so it stays pinned at all 7 sensors "
        "throughout -- widening the gap to roughly 70% more sensors than optimal at the largest $C$ tested. "
        "This paper's quadratic-aware greedy tracks the shrinking true optimal the entire time.",
    )
    fig.savefig(perf_dir + 'fig2_emp6_selection_cost_vs_C.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_emp6_selection_cost_vs_C.png")


# --------------------------------------------------------------------------------------
# Figure 7: two coordinated panels -- our guarantee working, their representation failing, vs. growing C
# --------------------------------------------------------------------------------------

def fig2_emp7_breakdown_vs_C(n_trials=150):
    """No number from `prior_bound_c19()` is plotted in the top panel (a change from an earlier version,
    fixed alongside fig2_emp1/2/8/10 -- this one was missed in that pass and caught only when the user
    asked a follow-up question about the bottom panel's exact formula). See fig2_emp1's docstring for why:
    [19]'s explicit constant bounds a differently-defined quantity for a different problem, not gamma_h,
    so it is not a valid comparison point even at their own rank-1/C=0 case.

    Bottom panel: BOTH curves are the SAME per-sensor marginal-gain formula, `marginal_gain_first_sensor`
    -- Delta_j(empty) = Tr(P) - Tr(B_j) (Proposition 2's Van Trees bound, one candidate sensor added to
    the empty set) -- evaluated on two different inputs. "True" uses the real M_j (genuinely rank-1,
    freshly randomized per trial -- matches [19]'s own hypothesis on the quadratic side exactly) and the
    real c_j at the current C. "Predicted" uses `restrict_to_rank1_zero_c(M_j)` (a no-op here, since M_j
    is already exactly rank-1) and c=0, REGARDLESS of the true C being swept -- i.e. what a user of [19]'s
    framework would compute for this sensor, since their model has no parameter for a linear term at all.
    At C=0 the two are the IDENTICAL quantity (not just close), since nothing is being approximated away
    yet -- confirming ratio=1.0 there is an exact reproduction of their case, not a coincidence."""
    rng = np.random.default_rng(107)
    C_scales = np.array([0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0])

    # Top panel: same coverage-vs-C data as Figure 1 (freshly computed here to keep this figure
    # self-contained rather than re-importing saved numbers).
    true_med, true_lo, true_hi, ours_med, ours_lo, ours_hi = sweep_vs_C(C_scales, SIGMA2, n_trials, rng)

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
    ax.plot(C_scales, true_med, color=COLORS['empirical'], marker='o', label='True ratio (brute force)', zorder=3)
    ax.fill_between(C_scales, true_lo, true_hi, color=COLORS['empirical'], alpha=0.12)
    ax.plot(C_scales, ours_med, color=COLORS['thm2'], marker='^', label='This paper (Theorem 2)', zorder=4)
    ax.fill_between(C_scales, ours_lo, ours_hi, color=COLORS['thm2'], alpha=0.15)
    ax.set_yscale('log')
    ax.set_ylabel(r'Supermodularity ratio $\gamma_h$', fontsize=12)
    ax.set_title('Top: our guarantee stays valid and present as $C$ grows', fontsize=12.5)
    ax.grid(alpha=0.3)
    ax.legend(loc='center right', fontsize=9.5, framealpha=0.95)

    ax = axes[1]
    ax.axhline(1.0, color=COLORS['brute'], linestyle='--', linewidth=2, label='Perfect representation (exact at $C=0$)', zorder=2)
    ax.plot(C_scales, pred_med, color=COLORS['prior'], marker='s', zorder=3,
            label="[19]'s rank-1/zero-$C$ prediction of the sensor's own true value")
    ax.fill_between(C_scales, pred_lo, pred_hi, color=COLORS['prior'], alpha=0.15)
    ax.set_xlabel(r'Linear-term magnitude $C$ ($M^{(i)}$ held at rank-1: their own case)', fontsize=12.5)
    ax.set_ylabel(r'$\Delta_j(\varnothing)$ predicted / $\Delta_j(\varnothing)$ true' + '\n(one sensor\'s marginal information gain)', fontsize=11)
    ax.set_title("Bottom: their prediction of the same sensor's own value degrades as $C$ grows", fontsize=12.5)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9.5, framealpha=0.95)

    fig.suptitle('One works, one degrades: the same growing-$C$ axis, two coordinated panels', y=0.975, fontsize=14.5)
    fig.tight_layout(rect=[0, 0.185, 1, 0.94])
    add_caption(
        fig, 'Empirical 7',
        "Top and bottom panels share the same x-axis: this paper's guarantee holds throughout while [19]'s own prediction of a sensor's value breaks down.",
        f"Median over N={n_trials} trials/point (bands = IQR). Top panel: same result as Empirical 1, "
        "full-rank $M^{(i)}$ throughout (this paper's own domain); no comparison to [19] is plotted, since "
        "their explicit constant is not a bound on $\\gamma_h$ (see Empirical 1's caption). Bottom panel: "
        "a different, independent check -- the marginal information gain $\\Delta_j(\\varnothing) = "
        "\\mathrm{Tr}(P)-\\mathrm{Tr}(B_j)$ of adding ONE candidate sensor to the empty set, computed two "
        "ways: with the sensor's real rank-1 $M_j$ and real $c_j$ (\"true\"), vs. with the same $M_j$ but "
        "$c_j$ fixed at zero regardless of its real value (\"predicted\" -- what [19]'s framework computes, "
        "since it has no parameter for a linear term at all). At $C=0$ these are the identical quantity, "
        "not merely close, since $M_j$ is already exactly rank-1 and $c_j$ is already zero -- confirming "
        "the ratio is exactly 1.0 there. As $C$ grows, real information enters only the true side, so the "
        "prediction increasingly understates the sensor's actual value.",
        y=0.02,
    )
    fig.savefig(perf_dir + 'fig2_emp7_breakdown_vs_C.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_emp7_breakdown_vs_C.png")




if __name__ == "__main__":
    fig2_emp6_selection_cost_vs_C()
    fig2_emp7_breakdown_vs_C()
