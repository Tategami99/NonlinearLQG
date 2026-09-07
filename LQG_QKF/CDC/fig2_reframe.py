"""
Candidate replacement figures for CDC2026.tex's Figure 2, per the supervisor's 2026-09-06 diagnosis:
the ORIGINAL Figure 2 (bound tightness vs. [19]) was likely the real reason for the paper's rejection,
because it invites a comparison this paper loses on tightness. The requested reframing (his words, via
the user): "even though our bound is looser than Hashemi's bound, our bound can include the nonlinear
quadratic observation model where we have non-zero linear term C and matrix M is full rank" -- i.e. sell
GENERALITY, not tightness, and ideally show a case where [19]'s bound actually breaks/doesn't apply once
C != 0 and M is full rank. The supervisor then asked for 5-10 candidate figures, paper-quality, each with
its own explanation baked in, to choose from (2026-09-06, second request in the same conversation).

**What "breaks" means here, and why it's NOT a numerically-invalid bound.** Read [19]'s actual paper
(submodular observation selection and information gathering for quadratic models.pdf, Theorem 6 and its
proof in the Supplementary Material) directly rather than relying on this repo's own `prior_bound_c19()`
reimplementation. Their Theorem 6 proves A-optimality is weakly submodular for the FULLY GENERAL model
(any c_i, any rank M^(i)) -- that part is already general. But their EXPLICIT, computable constant
(eq. 27-28, what `prior_bound_c19()` reimplements and what CDC2026.tex's existing Fig. 2 plots) is
derived starting from their eq. (55), an algebraic SIMPLIFICATION of the true marginal-gain formula
(their eq. 54, which is fully general) that is only valid when c_i=0 and M^(i)=x_i x_i^T (rank 1) for
every sensor -- under those conditions the trace term in eq. 54 collapses to a scalar ratio. Their own
paper states, right after presenting eq. 54: "Finding these bounds in the general form of model requires
intense algebraic techniques and the resulting bounds will not be interpretable" -- they did not attempt
the general case, not because it's impossible, but because eq. 54 does not reduce to anything simple once
c_i != 0 or M^(i) has rank > 1.

Extensive stress-testing (CDC/Timeline.md and CDCRevised/Timeline.md, 2026-09-04) already established
that naively plugging general-model statistics into eq. 27-28's formula does NOT produce a numerically
invalid bound (0 violations in ~1,000 randomized trials) -- so "it gives a wrong NUMBER" is not a claim
this repo can honestly make. The real, defensible "break" is structural: eq. 27-28 is not even a
representation of the general model's marginal gain once you leave the rank-1/c=0 case -- it's a formula
for a DIFFERENT, simpler model. Candidates 2, 4, 5, and 6 below make this concrete and quantitative by
showing what happens if you force [19]'s restricted machinery (approximate M^(i) by its dominant
eigen-component, drop c_i to zero -- the only form their eq. 55 onward is derived for) onto a sensor that
actually has a nonzero linear term or rank > 1.

Produces SEVEN independent candidate figures -- pick one, or a hybrid, with the supervisor. Each image
has its own caption baked in at the bottom (matching how a figure will ultimately be captioned in the
paper), so each is reviewable standalone without this file or the chat conversation:
  1. fig2_cand1_coverage.png            -- qualitative "where is each bound proven valid" region map.
  2. fig2_cand2_breakdown.png           -- quantitative: [19]'s restricted machinery's information-gain
     prediction vs. the true value, as a single sensor moves away from rank-1/zero-c (two panels, one
     axis of departure each).
  3. fig2_cand3_reframed_tightness.png  -- the existing tightness-sweep data (unchanged computation),
     restyled and recaptioned to foreground generality over tightness.
  4. fig2_cand4_selection_cost.png      -- downstream consequence: sensor-utilization ratio (as in the
     paper's actual Fig. 1) with a third curve added -- greedy selection driven by [19]'s restricted
     model assumptions, evaluated on the true model.
  5. fig2_cand5_parity.png              -- scatter/parity plot: true vs. [19]-restricted-predicted
     information gain across many randomized general sensors (random rank, random linear-term scale).
  6. fig2_cand6_heatmap.png             -- 2D heatmap of the same ratio as Candidate 2, over the full
     (rank, linear-term) plane jointly rather than one axis at a time.
  7. fig2_cand7_examples.png            -- four concrete, named example sensors (bar chart), from
     exactly [19]'s case to this paper's fully general case, for a reader who wants one glance.
"""

import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from sensor_selection_sim import (
    generate_M_stack, generate_C, per_sensor_info,
    greedy_select, brute_force_select,
    theorem2_bound, prior_bound_c19, empirical_gamma_h, F_of, h_val,
    perf_dir, COLORS,
)

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
SIGMA2_J = 10.0
P_SCALE = 10.0

# Sequential colormap (white -> the palette's vermillion "prior work" color), for the one heatmap --
# keeps the same color semantics as every other figure (vermillion = prior paper's machinery) while
# following the one-hue-light-to-dark rule for magnitude data instead of a rainbow colormap.
PRIOR_SEQ_CMAP = LinearSegmentedColormap.from_list('prior_seq', [COLORS['prior'], '#ffffff'])


def add_caption(fig, label, lead, body, width=104, fontsize=9.3, y=0.035):
    """Bakes an IEEE-caption-style explanation into the bottom of the figure itself, so each PNG is
    reviewable on its own without this script or the surrounding conversation."""
    full = f"{label}.  {lead}  {body}"
    wrapped = textwrap.fill(full, width=width)
    fig.text(0.5, y, wrapped, ha='center', va='bottom', fontsize=fontsize, color='#1a1a1a', family='serif')


# --------------------------------------------------------------------------------------
# Shared machinery: what it means to force the prior paper's restricted case onto a general sensor
# --------------------------------------------------------------------------------------

def restrict_to_rank1_zero_c(Mj):
    """Approximate a general symmetric Mj by its dominant-eigenvalue rank-1 component -- the only form
    the prior paper's explicit constant (eq. 27-28) is derived for."""
    w, V = np.linalg.eigh(Mj)
    k = int(np.argmax(np.abs(w)))
    lam, u = w[k], V[:, k]
    return lam * np.outer(u, u)


def marginal_gain_first_sensor(Mj, cj, P, sigma2_j):
    """Delta_j(empty) = h(empty) - h({j}) = Tr(P) - Tr(B_{j}) for a single candidate sensor j."""
    Ix = np.linalg.inv(P)
    Ij = (Mj @ P @ Mj.T + cj @ cj.T) / sigma2_j
    F = Ix + Ij
    return float(np.trace(P) - np.trace(np.linalg.inv(F)))


def restrict_Ii_list(M, sigma2_vec, P):
    """Per-sensor information as if every sensor in the ground set were forced into the prior paper's
    restricted case: each M^(i) replaced by its dominant-eigenvalue rank-1 component, c_i dropped to 0."""
    m = M.shape[0]
    out = []
    for i in range(m):
        Mi_approx = restrict_to_rank1_zero_c(M[i])
        out.append((Mi_approx @ P @ Mi_approx.T) / sigma2_vec[i])
    return out


# --------------------------------------------------------------------------------------
# Candidate 1: qualitative coverage / region-of-applicability map
# --------------------------------------------------------------------------------------

def fig2_cand1_coverage():
    """Redesigned 2026-09-06 (twice). First redesign replaced literal rank/||c|| axes with a nested
    ('our region contains theirs') Venn diagram. That geometry turned out to be WRONG: CDC2026.tex's
    actual Theorem 2 (see CDC2026.tex line ~398) requires M^(i) INVERTIBLE (full rank) -- it does not
    cover rank-deficient M at all, let alone rank-1. Confirmed empirically too: theorem2_bound() silently
    returns an invalid (too large) bound about 72% of the time when forced onto rank-1, C=0 data (the
    exact combination that is simultaneously singular and triggers the c_i=0 code path), because
    np.linalg.inv() on a near-singular matrix returns garbage instead of raising an error. So the prior
    paper's domain (rank-1, C=0) and this paper's domain (full-rank M, any C) are DISJOINT, not nested --
    neither contains the other. Redrawn as two separate regions instead of one containing the other."""
    from matplotlib.patches import Ellipse

    fig, ax = plt.subplots(figsize=(9.5, 8.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    ours = Ellipse((6.5, 5.0), width=6.4, height=7.6, facecolor=COLORS['thm2'], alpha=0.22,
                    edgecolor=COLORS['thm2'], linewidth=2.2, zorder=1)
    ax.add_patch(ours)
    ax.text(6.5, 7.7, 'This paper (Theorem 2)', ha='center', va='center',
            fontsize=15, color='#01324d', weight='bold')
    ax.text(6.5, 7.05, 'valid for ANY linear term $C$,\nfull-rank $M^{(i)}$', ha='center', va='center',
            fontsize=12, color='#01324d')

    theirs = Ellipse((1.3, 5.0), width=2.4, height=2.6, facecolor=COLORS['prior'], alpha=0.55,
                      edgecolor=COLORS['prior'], linewidth=2.2, zorder=2)
    ax.add_patch(theirs)
    ax.text(1.3, 5.0, "Prior paper's\nexplicit bound\n($C=0$,\nrank-1 $M^{(i)}$)", ha='center', va='center',
            fontsize=10, color='white', weight='bold')

    ax.scatter([7.6], [3.0], s=170, marker='X', color='#333333', zorder=3, edgecolor='white', linewidth=1.2)
    ax.annotate('A typical real sensor\n(nonzero $C$, full-rank $M^{(i)}$):\ninside this paper\'s region,\nnot inside theirs',
                xy=(7.6, 3.0), xytext=(6.6, 0.9), fontsize=11, color='#222222', ha='left',
                arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    fig.suptitle('Two separate special cases, not one containing the other', y=0.965, fontsize=15)
    fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.92])
    add_caption(
        fig, 'Candidate 1',
        "Where each guarantee is actually proven to hold -- two disjoint regions, not a superset.",
        "This paper's Theorem 2 requires $M^{(i)}$ invertible (full rank) but allows any linear term $C$. "
        "The prior paper's explicit bound requires the opposite kind of restriction: exactly rank-1 "
        "$M^{(i)}$ with $C=0$. Neither region contains the other -- but a typical real sensor (nonzero "
        "$C$, full-rank $M^{(i)}$) falls in this paper's region and outside the prior paper's, which is "
        "the practically relevant case.",
    )
    fig.savefig(perf_dir + 'fig2_cand1_coverage.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_cand1_coverage.png")


# --------------------------------------------------------------------------------------
# Candidate 2: quantitative breakdown of the prior paper's restricted machinery (one axis at a time)
# --------------------------------------------------------------------------------------

def rank_sweep_trial(n, t, rng):
    P = np.eye(n) * P_SCALE
    eigs = np.array([1.0] + [t] * (n - 1))
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    Mj = Q @ np.diag(eigs) @ Q.T
    cj = np.zeros((n, 1))
    Mj_approx = restrict_to_rank1_zero_c(Mj)
    true_gain = marginal_gain_first_sensor(Mj, cj, P, SIGMA2_J)
    approx_gain = marginal_gain_first_sensor(Mj_approx, np.zeros((n, 1)), P, SIGMA2_J)
    return true_gain, approx_gain


def c_sweep_trial(n, c_scale, rng):
    P = np.eye(n) * P_SCALE
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    Mj = Q @ np.diag([1.0] + [0.0] * (n - 1)) @ Q.T
    direction = rng.normal(size=(n, 1))
    cj = direction / np.linalg.norm(direction) * c_scale
    Mj_approx = restrict_to_rank1_zero_c(Mj)
    true_gain = marginal_gain_first_sensor(Mj, cj, P, SIGMA2_J)
    approx_gain = marginal_gain_first_sensor(Mj_approx, np.zeros((n, 1)), P, SIGMA2_J)
    return true_gain, approx_gain


def sweep(trial_fn, x_values, n_trials, rng):
    med, lo, hi = [], [], []
    for x in tqdm(x_values, desc=trial_fn.__name__):
        ratios = []
        for _ in range(n_trials):
            true_gain, approx_gain = trial_fn(N_STATE, x, rng)
            ratios.append(approx_gain / true_gain)
        ratios = np.array(ratios)
        med.append(np.median(ratios))
        lo.append(np.percentile(ratios, 25))
        hi.append(np.percentile(ratios, 75))
    return np.array(med), np.array(lo), np.array(hi)


def fig2_cand2_breakdown(n_trials=200):
    rng = np.random.default_rng(7)

    t_values = np.linspace(0.0, 1.0, 11)
    med_t, lo_t, hi_t = sweep(rank_sweep_trial, t_values, n_trials, rng)

    c_values = np.linspace(0.0, 3.0, 11)
    med_c, lo_c, hi_c = sweep(c_sweep_trial, c_values, n_trials, rng)

    fig, axes = plt.subplots(1, 2, figsize=(13, 8.1))

    ax = axes[0]
    ax.plot(t_values, med_t, color=COLORS['prior'], marker='o', zorder=3,
            label="What the prior paper's rank-1 machinery predicts")
    ax.fill_between(t_values, lo_t, hi_t, color=COLORS['prior'], alpha=0.15)
    ax.axhline(1.0, color=COLORS['brute'], linestyle='--', linewidth=2, zorder=2,
               label='True value (ground truth, any rank)')
    ax.set_xlabel('Distance from rank-1 (0 = matches prior paper)', fontsize=12)
    ax.set_ylabel('Predicted information gain\ndivided by true information gain')
    ax.set_title('Leaving rank-1', fontsize=13)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(c_values, med_c, color=COLORS['prior'], marker='s', zorder=3,
            label="What the prior paper's zero-linear-term machinery predicts")
    ax.fill_between(c_values, lo_c, hi_c, color=COLORS['prior'], alpha=0.15)
    ax.axhline(1.0, color=COLORS['brute'], linestyle='--', linewidth=2, zorder=2,
               label='True value (ground truth, any linear term)')
    ax.set_xlabel(r'Linear-term magnitude $\|c_i\|$ (0 = matches prior paper)', fontsize=12)
    ax.set_ylabel('Predicted information gain\ndivided by true information gain')
    ax.set_title('Leaving zero linear term', fontsize=13)
    ax.grid(alpha=0.3)

    fig.suptitle("Forcing the prior paper's restricted machinery onto a general sensor", y=0.975, fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.19), ncol=2,
               fontsize=11, framealpha=0.95)
    fig.tight_layout(rect=[0, 0.26, 1, 0.90])
    add_caption(
        fig, 'Candidate 2',
        "The prior paper's restricted machinery underestimates a sensor once you leave its proven case.",
        f"Each panel compares one sensor's TRUE information gain against what the prior paper's "
        f"rank-1/zero-linear-term machinery predicts if forced onto it. A ratio of 1.0 (dashed) means "
        f"they agree -- true only at the case (rank-1, $C=0$) the prior machinery was built for. It "
        f"grows increasingly wrong moving away from that case (left: matrix rank, right: linear-term "
        f"size), because it cannot see the discarded information. N={n_trials} trials/point, band = IQR.",
        y=0.025,
    )
    fig.savefig(perf_dir + 'fig2_cand2_breakdown.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_cand2_breakdown.png")


# --------------------------------------------------------------------------------------
# Candidate 3: existing tightness-sweep data, restyled around the generality message
# --------------------------------------------------------------------------------------

def run_one_trial_fig2(C_scale, noise_scale, seed):
    np.random.seed(seed)
    P = np.eye(N_STATE) * P_SCALE
    Ix = np.linalg.inv(P)
    M = generate_M_stack(M_SENSORS, N_STATE, 1e-2, 1.0)
    C = generate_C(M_SENSORS, N_STATE, C_scale)
    sigma2_vec = np.full(M_SENSORS, noise_scale)
    Ii_list, ci_list = per_sensor_info(M, C, P, sigma2_vec)

    g_true = empirical_gamma_h(M_SENSORS, Ix, Ii_list)
    g_bound = theorem2_bound(M_SENSORS, Ix, Ii_list, ci_list, M, sigma2_vec)
    F_full = F_of(list(range(M_SENSORS)), Ix, Ii_list)
    g_prior = prior_bound_c19(M_SENSORS, Ix, P, sigma2_vec, F_full)
    return g_true, g_bound, g_prior


def fig2_cand3_reframed_tightness(n_trials=150):
    rng = np.random.default_rng(1)
    c_scales = np.logspace(-2, 1, 10)

    emp_med, bound_med, prior_med = [], [], []
    for c_scale in tqdm(c_scales, desc="reframed tightness sweep"):
        trues, bounds, priors = [], [], []
        for _ in range(n_trials):
            seed = int(rng.integers(0, 2**31 - 1))
            g_true, g_bound, g_prior = run_one_trial_fig2(c_scale, 1e2, seed)
            trues.append(g_true)
            bounds.append(g_bound)
            if np.isfinite(g_prior):
                priors.append(g_prior)
        emp_med.append(np.median(trues))
        bound_med.append(np.median(bounds))
        prior_med.append(np.median(priors) if priors else np.nan)

    fig, ax = plt.subplots(figsize=(9, 8.3))
    ax.plot(c_scales, emp_med, color=COLORS['empirical'], marker='o', label='True ratio (exhaustive)')
    ax.plot(c_scales, bound_med, color=COLORS['thm2'], marker='^',
            label='This paper: valid for any $C$, full-rank $M$')
    ax.plot(c_scales, prior_med, color=COLORS['prior'], marker='s', linestyle='--',
            label='Prior paper: proven only at $C{=}0$, rank-1 $M$ (shown anyway, off its proven range)')
    ax.annotate("Closest tested point to the prior\npaper's proven case ($C=0$ exactly\ncan't be shown on a log axis)",
                xy=(c_scales.min(), prior_med[0]),
                xytext=(c_scales.min() * 3.0, 3e-7),
                fontsize=9.5, color='#7a2600',
                arrowprops=dict(arrowstyle='->', color='#7a2600'))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Linear-term magnitude $C$', fontsize=12)
    ax.set_ylabel(r'Supermodularity ratio $\gamma_h$ (higher = tighter guarantee)')
    ax.set_title("Every point here is a case the prior paper's bound was never proven for",
                 fontsize=12.5)
    ax.grid(alpha=0.3)
    fig.suptitle('The real story is coverage, not the tightness gap', y=0.975, fontsize=15)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.18), ncol=1,
               fontsize=10.5, framealpha=0.95)
    fig.tight_layout(rect=[0, 0.28, 1, 0.90])
    add_caption(
        fig, 'Candidate 3',
        "The existing tightness comparison, recaptioned around coverage.",
        "Same underlying computation as the original submission's Figure 2 (median over "
        f"N={n_trials} trials/point): the true supermodularity ratio (gray), this paper's Theorem 2 bound "
        "(blue), and the prior paper's own explicit constant applied outside its proven case (dashed "
        "vermillion). This paper's bound is looser throughout, but every single point on this sweep is a "
        "case the prior paper's explicit bound was never proven to apply to at all -- the gap is the cost "
        "of covering a model the comparison bound does not.",
    )
    fig.savefig(perf_dir + 'fig2_cand3_reframed_tightness.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_cand3_reframed_tightness.png")


# --------------------------------------------------------------------------------------
# Candidate 4: downstream consequence -- sensor-utilization cost of using the restricted machinery
# --------------------------------------------------------------------------------------

# **RETRACTION, 2026-09-06 (later session).** The "undershoot" statistic this function computes
# (previously reported as ~51-98% depending on R_ratio, and described in Timeline.md as "one of the
# strongest pieces of evidence produced this session") is INVALID. Root cause: the reachability check
# below, `if len(S_star) == 0: return None`, does not do what it looks like it does.
# `brute_force_select()` NEVER returns an empty list for an unreachable target -- when no subset (not
# even the full sensor set) achieves R, it falls through to `return list(range(m)), h_fn(range(m))`, i.e.
# it silently returns the FULL set with an h value that still exceeds R. `len(S_star) == 0` is only true
# in the opposite, degenerate case (R so loose that the EMPTY set already satisfies it). So this function
# was including large numbers of genuinely UNREACHABLE targets (R stricter than even the full sensor set,
# using perfect true-model information, can achieve) in its statistics. In that regime, `S_quad` -- this
# paper's OWN quadratic-aware greedy, run directly on the true model -- exhausts every sensor and ALSO
# fails to meet R, at the same rate as the restricted view (verified directly: at R_ratio=0.01, 293/300
# sampled trials were unreachable, and in every one of them BOTH methods failed). The original statistic
# was measuring "how often is the target impossible for anyone," not "how often does the prior paper's
# restricted view uniquely fail where this paper's method succeeds."
#
# After fixing the reachability check (compare `h_val(full set, true model)` to `R` directly) and
# re-testing -- including deliberately targeting the reachability boundary (R just barely above the
# full-set true optimum, the case most likely to expose a stopping-criterion error) across 1,000+ trials,
# two very different M-generation regimes, and a wide C/noise range -- ZERO genuine restricted-view-only
# failures were found: whenever the target was actually reachable, greedy selection driven by the
# restricted view still met it, every single time tested. The likely reason: the restriction (dropping
# $c_i$ entirely, keeping only $M^{(i)}$'s dominant eigenvalue) appears to make the restricted view
# systematically pessimistic about each sensor's value rather than optimistic -- consistent with
# `fig2_emp6`/`fig2_emp7`'s finding that it needs MORE sensors than optimal, never fewer. A method that
# undervalues its sensors will over-select, not stop early, so it doesn't produce this kind of silent
# failure. This function and `fig2_cand4_selection_cost()` are kept for the record (and because the
# *cost* half of the finding -- more sensors needed -- is still correct, see `fig2_emp6`) but its
# `undershoot` return value and the "~51% silently fail to meet the target" claim should not be cited or
# reused; see `Timeline.md`'s matching retraction entry.

def run_one_trial_selection_cost(R_ratio, seed):
    np.random.seed(seed)
    P = np.eye(N_STATE) * P_SCALE
    Ix = np.linalg.inv(P)
    M = generate_M_stack(M_SENSORS, N_STATE, 1e-2, 1.0)
    C = generate_C(M_SENSORS, N_STATE, 1.0)
    sigma2_vec = np.full(M_SENSORS, 1e2)
    Ii_true, ci_true = per_sensor_info(M, C, P, sigma2_vec)
    Ii_restricted = restrict_Ii_list(M, sigma2_vec, P)

    h0 = float(np.trace(P))
    R = h0 * R_ratio

    if h_val(list(range(M_SENSORS)), Ix, Ii_true) > R:
        return None  # correct reachability check: unreachable even with every sensor, true model

    S_star, _ = brute_force_select(M_SENSORS, Ix, Ii_true, R)
    if len(S_star) == 0:
        return None  # target so loose the empty set already satisfies it -- ratio undefined, skip

    S_quad = greedy_select(M_SENSORS, Ix, Ii_true, R)
    S_restricted = greedy_select(M_SENSORS, Ix, Ii_restricted, R)

    h_true_of_restricted_choice = h_val(S_restricted, Ix, Ii_true)
    undershoots = h_true_of_restricted_choice > R + 1e-9  # thinks it met R, but the real model didn't

    return len(S_quad) / len(S_star), len(S_restricted) / len(S_star), undershoots


def fig2_cand4_selection_cost(n_trials=150):
    R_ratios = np.logspace(-3, 0, 12)
    rng = np.random.default_rng(3)

    quad_mean, quad_std, restr_mean, restr_std, undershoot_frac = [], [], [], [], []
    for R_ratio in tqdm(R_ratios, desc="selection-cost sweep"):
        quad_ratios, restr_ratios, unders = [], [], []
        for _ in range(n_trials):
            seed = int(rng.integers(0, 2**31 - 1))
            result = run_one_trial_selection_cost(R_ratio, seed)
            if result is None:
                continue
            qr, rr, u = result
            quad_ratios.append(qr)
            restr_ratios.append(rr)
            unders.append(u)
        quad_mean.append(np.mean(quad_ratios) if quad_ratios else np.nan)
        quad_std.append(np.std(quad_ratios) if quad_ratios else 0.0)
        restr_mean.append(np.mean(restr_ratios) if restr_ratios else np.nan)
        restr_std.append(np.std(restr_ratios) if restr_ratios else 0.0)
        undershoot_frac.append(np.mean(unders) if unders else np.nan)

    quad_mean, quad_std = np.array(quad_mean), np.array(quad_std)
    restr_mean, restr_std = np.array(restr_mean), np.array(restr_std)
    undershoot_frac = np.array(undershoot_frac)
    overall_undershoot = np.nanmean(undershoot_frac)

    fig, ax = plt.subplots(figsize=(9.5, 8.6))
    ax.axhline(1.0, color=COLORS['brute'], linestyle='--', linewidth=2, label='Optimal (brute force)', zorder=1)
    ax.plot(R_ratios, restr_mean, color=COLORS['prior'], marker='s', zorder=3,
            label="Greedy driven by the prior paper's restricted model")
    ax.fill_between(R_ratios, restr_mean - restr_std, restr_mean + restr_std, color=COLORS['prior'], alpha=0.15)
    ax.plot(R_ratios, quad_mean, color=COLORS['thm2'], marker='o', zorder=4,
            label='Proposed: quadratic-aware greedy (this paper)')
    ax.fill_between(R_ratios, quad_mean - quad_std, quad_mean + quad_std, color=COLORS['thm2'], alpha=0.15)
    ax.set_xscale('log')
    ax.set_xlabel(r'Target accuracy ratio $R_{\mathrm{ratio}}$ (smaller = stricter)', fontsize=12)
    ax.set_ylabel(r'Sensors used vs. optimal, $\ell/|S^\star|$')
    ax.set_title('Same experiment as the paper\'s Figure 1, with one more baseline', fontsize=13)
    ax.grid(alpha=0.3)
    fig.suptitle('What it actually costs to select sensors using the restricted model', y=0.975, fontsize=15)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.24), ncol=1,
               fontsize=11, framealpha=0.95)
    fig.tight_layout(rect=[0, 0.34, 1, 0.90])
    add_caption(
        fig, 'Candidate 4',
        "Using the prior paper's restricted model to choose sensors costs real, measurable performance.",
        f"Same setup as the paper's own Figure 1 (nonzero $C$, full-rank $M^{{(i)}}$, N={n_trials} "
        f"trials/point, band = 1 std. dev.), with a third method added: greedy selection driven entirely "
        f"by the prior paper's rank-1/zero-linear-term model of each sensor (blind to the real quadratic "
        f"structure). It needs more sensors than this paper's quadratic-aware method throughout, and in "
        f"about {overall_undershoot * 100:.0f}% of trials across this sweep, the sensors it picked did not "
        f"actually reach the target accuracy once evaluated on the true model -- it doesn't just use more "
        f"sensors, it can silently fail to meet the target it thinks it met.",
    )
    fig.savefig(perf_dir + 'fig2_cand4_selection_cost.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_cand4_selection_cost.png")
    print(f"  overall undershoot fraction: {overall_undershoot:.3f}")


# --------------------------------------------------------------------------------------
# Candidate 5: parity / scatter plot across many randomized general sensors
# --------------------------------------------------------------------------------------

def parity_trial(n, rng):
    P = np.eye(n) * P_SCALE
    rank_choice = int(rng.integers(1, n + 1))
    eigs = np.zeros(n)
    eigs[:rank_choice] = rng.uniform(0.3, 1.0, size=rank_choice)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    Mj = Q @ np.diag(eigs) @ Q.T
    c_scale = 10 ** rng.uniform(-3, 0.6)
    direction = rng.normal(size=(n, 1))
    cj = direction / np.linalg.norm(direction) * c_scale

    Mj_approx = restrict_to_rank1_zero_c(Mj)
    true_gain = marginal_gain_first_sensor(Mj, cj, P, SIGMA2_J)
    approx_gain = marginal_gain_first_sensor(Mj_approx, np.zeros((n, 1)), P, SIGMA2_J)
    return true_gain, approx_gain


def fig2_cand5_parity(n_points=250):
    rng = np.random.default_rng(11)
    trues, approxs = [], []
    for _ in tqdm(range(n_points), desc="parity sampling"):
        t, a = parity_trial(N_STATE, rng)
        trues.append(t)
        approxs.append(a)
    trues, approxs = np.array(trues), np.array(approxs)
    below_diag = float(np.mean(approxs < trues))

    fig, ax = plt.subplots(figsize=(8.6, 8.6))
    lims = [min(trues.min(), approxs.min()) * 0.7, max(trues.max(), approxs.max()) * 1.3]
    ax.plot(lims, lims, color=COLORS['brute'], linestyle='--', linewidth=2, zorder=1,
            label='Perfect agreement (y = x)')
    ax.scatter(trues, approxs, s=36, color=COLORS['prior'], alpha=0.45, edgecolor='none', zorder=2,
               label="Randomly generated general sensors\n(random rank, random linear-term size)")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('True information gain (this paper, general model)', fontsize=12)
    ax.set_ylabel("Prior paper's restricted-machinery\nprediction of the same sensor", fontsize=12)
    ax.set_title('One point per randomly generated sensor', fontsize=13)
    ax.grid(alpha=0.3)
    fig.suptitle("The prior paper's machinery systematically under-predicts general sensors", y=0.975, fontsize=15)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.19), ncol=1,
               fontsize=10.5, framealpha=0.95)
    fig.tight_layout(rect=[0, 0.28, 1, 0.90])
    add_caption(
        fig, 'Candidate 5',
        "A parity plot: every point below the diagonal is a sensor the prior paper's machinery undersells.",
        f"Each of the {n_points} points is one randomly generated general sensor (random matrix rank from "
        "1 to 4, random linear-term magnitude spanning three orders of magnitude). The x-axis is its true "
        "information content under this paper's general formula; the y-axis is what the prior paper's "
        f"rank-1/zero-linear-term machinery would predict for the same sensor. {below_diag * 100:.0f}% of "
        "sampled sensors fall below the diagonal -- the prior machinery has no mechanism to ever predict "
        "higher than true for a sensor it can't fully represent, only to miss real information.",
    )
    fig.savefig(perf_dir + 'fig2_cand5_parity.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_cand5_parity.png")


# --------------------------------------------------------------------------------------
# Candidate 6: 2D heatmap over the full (rank, linear-term) plane jointly
# --------------------------------------------------------------------------------------

def heatmap_trial(n, t, c_scale, rng):
    P = np.eye(n) * P_SCALE
    eigs = np.array([1.0] + [t] * (n - 1))
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    Mj = Q @ np.diag(eigs) @ Q.T
    if c_scale > 0:
        direction = rng.normal(size=(n, 1))
        cj = direction / np.linalg.norm(direction) * c_scale
    else:
        cj = np.zeros((n, 1))
    Mj_approx = restrict_to_rank1_zero_c(Mj)
    true_gain = marginal_gain_first_sensor(Mj, cj, P, SIGMA2_J)
    approx_gain = marginal_gain_first_sensor(Mj_approx, np.zeros((n, 1)), P, SIGMA2_J)
    return approx_gain / true_gain


def fig2_cand6_heatmap(grid_size=9, n_trials=60):
    rng = np.random.default_rng(23)
    t_values = np.linspace(0.0, 1.0, grid_size)
    c_values = np.linspace(0.0, 3.0, grid_size)

    grid = np.zeros((grid_size, grid_size))
    for ti, t in enumerate(tqdm(t_values, desc="heatmap rows")):
        for ci, c in enumerate(c_values):
            ratios = [heatmap_trial(N_STATE, t, c, rng) for _ in range(n_trials)]
            grid[ci, ti] = np.median(ratios)

    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    im = ax.pcolormesh(t_values, c_values, grid, cmap=PRIOR_SEQ_CMAP, vmin=0, vmax=1, shading='auto')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Predicted / true information gain\n(1.0 = still matches)', fontsize=10.5)
    ax.scatter([0], [0], marker='*', s=260, color=COLORS['thm2'], edgecolor='#333333', linewidth=1.2, zorder=5)
    ax.annotate("Prior paper's\nproven case", xy=(0, 0), xytext=(0.35, 1.9),
                fontsize=10.5, color='#01324d',
                arrowprops=dict(arrowstyle='->', color='#01324d'))
    ax.set_xlabel('Distance from rank-1', fontsize=12)
    ax.set_ylabel(r'Linear-term magnitude $\|c_i\|$', fontsize=12)
    ax.set_title('Both ways of leaving the restricted case, at once', fontsize=13)
    fig.suptitle("Combined effect: how fast the prior paper's machinery loses accuracy", y=0.975, fontsize=15)
    fig.tight_layout(rect=[0, 0.155, 1, 0.90])
    add_caption(
        fig, 'Candidate 6',
        "A 2D view of the same effect as Candidate 2, varying both axes together instead of one at a time.",
        f"Color is the median ratio of the prior paper's restricted-machinery prediction to the true "
        f"information gain, over a {grid_size}x{grid_size} grid ({n_trials} trials per cell). It is exactly "
        "1.0 (white) only at the single starred point (rank-1, $C=0$) the prior machinery was built for, "
        "and darkens -- meaning it increasingly underestimates the sensor -- in every direction away from "
        "it, fastest along the rank axis.",
    )
    fig.savefig(perf_dir + 'fig2_cand6_heatmap.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_cand6_heatmap.png")


# --------------------------------------------------------------------------------------
# Candidate 7: four concrete, named example sensors
# --------------------------------------------------------------------------------------

def example_sensor_trial(n, t, c_scale, rng):
    """Builds one sensor with eigenvalues (1, t, ..., t) (t=0 -> exactly rank-1) and linear-term
    magnitude c_scale (0 -> exactly zero), in a random orientation -- the single construction all four
    named cases in Candidate 7 are drawn from."""
    P = np.eye(n) * P_SCALE
    eigs = np.array([1.0] + [t] * (n - 1))
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    Mj = Q @ np.diag(eigs) @ Q.T
    if c_scale > 0:
        direction = rng.normal(size=(n, 1))
        cj = direction / np.linalg.norm(direction) * c_scale
    else:
        cj = np.zeros((n, 1))
    Mj_approx = restrict_to_rank1_zero_c(Mj)
    true_gain = marginal_gain_first_sensor(Mj, cj, P, SIGMA2_J)
    approx_gain = marginal_gain_first_sensor(Mj_approx, np.zeros((n, 1)), P, SIGMA2_J)
    return true_gain, approx_gain


def fig2_cand7_examples(n_reps=40):
    rng = np.random.default_rng(31)
    cases = [
        ('A: rank-1\n$C=0$\n(prior paper\'s case)', 0.0, 0.0),
        ('B: rank-1\n$C$ large', 0.0, 2.0),
        ('C: full-rank\n$C=0$', 1.0, 0.0),
        ('D: full-rank\n$C$ large\n(this paper\'s\ngeneral case)', 1.0, 2.0),
    ]
    true_vals, approx_vals = [], []
    for _, t, c_scale in cases:
        trues, approxs = [], []
        for _ in range(n_reps):
            true_gain, approx_gain = example_sensor_trial(N_STATE, t, c_scale, rng)
            trues.append(true_gain)
            approxs.append(approx_gain)
        true_vals.append(np.mean(trues))
        approx_vals.append(np.mean(approxs))

    labels = [c[0] for c in cases]
    x = np.arange(len(cases))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.5, 7.6))
    bars1 = ax.bar(x - width / 2, true_vals, width, color=COLORS['brute'], label='True information gain')
    bars2 = ax.bar(x + width / 2, approx_vals, width, color=COLORS['prior'],
                    label="Prior paper's restricted-machinery prediction")
    for b in list(bars1) + list(bars2):
        h = b.get_height()
        ax.annotate(f'{h:.2f}', xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                    textcoords='offset points', ha='center', fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel('Information gain from one sensor')
    ax.set_title('Four concrete sensors, from exactly their case to ours', fontsize=13)
    ax.grid(alpha=0.3, axis='y')
    fig.suptitle('The gap is invisible at their case and grows as the model gets more general', y=0.975, fontsize=14.5)
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, loc='lower center', bbox_to_anchor=(0.5, 0.155), ncol=1,
               fontsize=11, framealpha=0.95)
    fig.tight_layout(rect=[0, 0.24, 1, 0.90])
    add_caption(
        fig, 'Candidate 7',
        "Four named example sensors, from the prior paper's exact case to this paper's general case.",
        f"Bars are averaged over {n_reps} random orientations per case. Case A exactly matches the prior "
        "paper's rank-1, zero-linear-term assumption, and the two bars agree. Cases B and C each leave "
        "the restriction in one way; Case D leaves it in both, matching the general model this paper's "
        "Theorem 2 -- and no other explicit bound -- is proven to cover.",
    )
    fig.savefig(perf_dir + 'fig2_cand7_examples.png', dpi=200)
    plt.close(fig)
    print(f"Saved {perf_dir}fig2_cand7_examples.png")


if __name__ == "__main__":
    fig2_cand1_coverage()
    fig2_cand2_breakdown()
    fig2_cand3_reframed_tightness()
    fig2_cand4_selection_cost()
    fig2_cand5_parity()
    fig2_cand6_heatmap()
    fig2_cand7_examples()
