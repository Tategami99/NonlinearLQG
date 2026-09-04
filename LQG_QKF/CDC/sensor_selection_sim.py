"""
Sensor/measurement selection under the quadratic observation model of CDC2026.tex (Eq. 1-5).

Static, single-shot A-optimality Van Trees bound selection: x ~ N(0, P), and for each i in the ground
set G, y_i = 1/2 x^T M^(i) x + c_i^T x + v_i, v_i ~ N(0, sigma_i^2). No dynamics (this is not the LQG
tracking problem in LQG_QKF.py in this same directory -- that is a separate, later-stage experiment).

Implements:
  - h(S) = Tr(B_S) (Proposition 2's closed-form Van Trees bound) and its greedy/brute-force minimizers
    (Algorithm 1 / Problem (5))
  - the linearized greedy baseline (drops all quadratic M^(i) terms, keeping only the linear c_i)
  - the corrected Theorem 2 lower bound on the supermodularity ratio gamma_h (per-j F_tilde_{G,j},
    explicit min over j, c_j=0 case)
  - [19]'s own restricted-case (rank-1, c_i=0) explicit weak-submodularity constant, for the Fig. 2
    reference-baseline curve
  - exhaustive (brute-force) computation of the true empirical gamma_h (Definition 3), for validation
"""

import os
import numpy as np
from itertools import combinations

os.chdir(os.path.dirname(os.path.abspath(__file__)))

test_dir = 'sensor_selection_test/'
pkl_dir = test_dir + 'pkl/'
perf_dir = test_dir + 'perf/'
cache_dir = test_dir + 'cache/'
for d in (pkl_dir, perf_dir, cache_dir):
    os.makedirs(d, exist_ok=True)

# Okabe-Ito-based, colorblind-validated (see LQG_QKF/CDC/Timeline.md "figure redesign" entry --
# scripts/validate_palette.js from Claude's dataviz skill flagged the old orange/green pair used
# together in Fig. 2 as CVD-indistinguishable, deltaE 0.7 for protanopia). Meaning is consistent
# across figures: dark gray = ground truth/reference, blue = the proposed method or bound, vermillion
# = the comparison baseline or prior-work bound.
COLORS = {
    'quad': '#0072B2',      # blue: proposed greedy quadratic method
    'lin': '#D55E00',       # vermillion: linearized greedy baseline
    'brute': '#333333',     # dark gray (dashed): brute-force optimal reference
    'thm2': '#0072B2',      # blue: proposed Theorem 2 bound
    'empirical': '#333333',  # dark gray: empirical (true) gamma_h
    'prior': '#D55E00',     # vermillion: [19]'s restricted-case bound
}


# --------------------------------------------------------------------------------------
# Model generation
# --------------------------------------------------------------------------------------

def generate_spd(n, scale=1.0):
    """Random symmetric positive-definite n x n matrix (used as prior covariance P or as M^(i))."""
    A = np.random.randn(n, n)
    return scale * (A.T @ A) + np.eye(n) * 1e-3


def generate_M_stack(m, n, scale_min, scale_max):
    """Stack of m full-rank symmetric M^(i), each with an independently sampled scale."""
    M = np.zeros((m, n, n))
    for i in range(m):
        scale = np.random.uniform(scale_min, scale_max) if scale_max > scale_min else scale_min
        M[i] = generate_spd(n, scale)
    return M


def generate_C(m, n, scale):
    return np.random.randn(m, n) * scale


# --------------------------------------------------------------------------------------
# Core Van Trees / A-optimality machinery (Proposition 2, Problem (5))
# --------------------------------------------------------------------------------------

def per_sensor_info(M, C, P, sigma2_vec):
    """
    I_i := (M^(i) P M^(i).T + c_i c_i.T) / sigma_i^2  (Theorem 2's per-sensor information term).
    Returns (Ii_list, ci_list), each a length-m list of (n,n) / (n,1) arrays.
    """
    m = M.shape[0]
    Ii_list, ci_list = [], []
    for i in range(m):
        ci = C[i, :].reshape(-1, 1)
        Ii = (M[i] @ P @ M[i].T + ci @ ci.T) / sigma2_vec[i]
        Ii_list.append(Ii)
        ci_list.append(ci)
    return Ii_list, ci_list


def per_sensor_info_linearized(ci_list, sigma2_vec):
    """Linearized-model per-sensor information: drops the M^(i) P M^(i).T term entirely."""
    return [(ci_list[i] @ ci_list[i].T) / sigma2_vec[i] for i in range(len(ci_list))]


def F_of(S, Ix, Ii_list):
    F = Ix.copy()
    for i in S:
        F = F + Ii_list[i]
    return F


def h_val(S, Ix, Ii_list):
    """h(S) = Tr(B_S) = Tr(F_S^{-1})."""
    F = F_of(S, Ix, Ii_list)
    return float(np.trace(np.linalg.inv(F)))


def greedy_select(m, Ix, Ii_list, R, stop_h=None):
    """
    Algorithm 1: greedily add the sensor with the largest marginal decrease in h(S) until h(S) <= R.
    stop_h, if given, is a *different* h used only for the stopping check (still ranks by h from
    Ii_list) -- unused here (see greedy_linearized, which is self-contained instead).
    """
    S = []
    candidates = list(range(m))
    h_fn = lambda s: h_val(s, Ix, Ii_list)
    while h_fn(S) > R and candidates:
        best_j, best_val = None, np.inf
        for j in candidates:
            val = h_fn(S + [j])
            if val < best_val:
                best_val, best_j = val, j
        S.append(best_j)
        candidates.remove(best_j)
    return S


def greedy_linearized(m, Ix, ci_list, sigma2_vec, R):
    """
    The linearized greedy baseline of Section V: greedy selection applied entirely to the locally
    linearized surrogate (drop all quadratic terms, keep only the linear c_i). Self-contained -- both
    the ranking and the R-ratio stopping check use h_lin, exactly mirroring how the true quadratic
    method is self-contained w.r.t. h. Since h_lin(emptyset) = h(emptyset) = Tr(P) as well, R is the
    same absolute threshold for both methods.
    """
    Ii_lin = per_sensor_info_linearized(ci_list, sigma2_vec)
    return greedy_select(m, Ix, Ii_lin, R)


def brute_force_select(m, Ix, Ii_list, R):
    """Exact min-cardinality solution to Problem (5) via combinatorial search over subset sizes."""
    h_fn = lambda s: h_val(list(s), Ix, Ii_list)
    for size in range(m + 1):
        best_comb, best_val = None, np.inf
        for comb in combinations(range(m), size):
            v = h_fn(comb)
            if v < best_val:
                best_val, best_comb = v, comb
        if best_val <= R:
            return list(best_comb), best_val
    return list(range(m)), h_fn(range(m))


# --------------------------------------------------------------------------------------
# Theorem 2 (fixed) and reference bounds
# --------------------------------------------------------------------------------------

def theorem2_bound(m, Ix, Ii_list, ci_list, M, sigma2_vec, buggy=False):
    """
    Corrected Theorem 2: gamma_h >= min_{j in G} min{gamma_f(j), gamma_g(j)}, with the per-j
    F_tilde_{G,j} := F_{G\\{j}} + sigma_j^2 I_j (matches CDC2026.tex post-fix).

    If buggy=True, reproduces the *original* (pre-fix) bound instead -- using the single global
    F_tilde_G := F_G = I_x + sum_{i in G} I_i (plain +I_j at index j, not +sigma_j^2 I_j) for every j.
    Exists only to numerically demonstrate the bug (see self_test_theorem2 below); not used for Fig. 2.
    """
    G = list(range(m))
    if buggy:
        F_G = F_of(G, Ix, Ii_list)  # old, single global object -- reused for every j below

    best = np.inf
    for j in G:
        G_minus_j = [i for i in G if i != j]
        sigma2_j = float(sigma2_vec[j])
        Ij = Ii_list[j]

        if buggy:
            F_tilde_Gj = F_G
        else:
            F_Gminusj = F_of(G_minus_j, Ix, Ii_list)
            F_tilde_Gj = F_Gminusj + sigma2_j * Ij

        lam_min_Ix = np.linalg.eigvalsh(Ix).min()
        lam_max_FGj = np.linalg.eigvalsh(F_tilde_Gj).max()

        cj = ci_list[j]
        norm_cj_sq = float((cj.T @ cj).squeeze())
        if norm_cj_sq > 1e-30:
            num_f = 1.0 + sigma2_j * lam_min_Ix / norm_cj_sq
            den_f = 1.0 + sigma2_j * lam_max_FGj / norm_cj_sq
            gamma_f_j = (lam_min_Ix / lam_max_FGj) * (num_f / den_f)
        else:
            gamma_f_j = 1.0  # c_j = 0 case: Delta_j(S) = g_j(S) directly (see proof)

        Mj = M[j]
        try:
            Mj_inv = np.linalg.inv(Mj)
            MjT_inv = np.linalg.inv(Mj.T)
            P_from_Ix = np.linalg.inv(Ix)  # = P (Gaussian prior: I_x = P^{-1})
            Vj = MjT_inv @ (sigma2_j * Ix) @ Mj_inv

            Us_bound = Vj + P_from_Ix          # V_j + I_x^{-1}
            Ut_bound = Vj + np.linalg.inv(F_tilde_Gj)  # V_j + F_tilde_{G,j}^{-1}

            tr_Us_inv = float(np.trace(np.linalg.inv(Us_bound)))
            tr_Ut_inv = float(np.trace(np.linalg.inv(Ut_bound)))

            ratio_lam2 = (lam_min_Ix / lam_max_FGj) ** 2
            gamma_g_j = ratio_lam2 * (tr_Us_inv / max(tr_Ut_inv, 1e-300))
        except np.linalg.LinAlgError:
            gamma_g_j = np.inf  # M^(j) singular -- shouldn't happen (M generated full rank)

        best = min(best, min(gamma_f_j, gamma_g_j))
    return best


def prior_bound_c19(m, Ix, P, sigma2_vec, F_full):
    """
    [19]'s own explicit weak-submodularity constant (their eq. 27-28), valid only for their restricted
    case (rank-1 M^(i), c_i=0). Applied here as-is (naively, on our general model) purely as the
    reference "prior work" curve for Fig. 2 -- exactly as the paper's own Fig. 2 caption describes it.
    Converted to the gamma_h in [0,1] convention via gamma_prev = 1 / c_fA (see Timeline.md pass-2 notes).
    """
    lam_max_P = np.linalg.eigvalsh(P).max()
    lam_min_BG = np.linalg.eigvalsh(np.linalg.inv(F_full)).min()  # B_G = F_full^{-1}
    vals = []
    for i in range(m):
        s2 = float(sigma2_vec[i])
        eig_sP = np.linalg.eigvalsh(s2 * P)
        lam_max_sP, lam_min_sP = eig_sP.max(), eig_sP.min()
        num = (lam_max_P ** 2) * (lam_max_sP + 1.0)
        den = (lam_min_BG ** 2) * (lam_min_sP + 1.0)
        if den > 0 and num > 0 and np.isfinite(num) and np.isfinite(den):
            vals.append(1.0 / (num / den))
    return min(vals) if vals else np.nan


def empirical_gamma_h(m, Ix, Ii_list):
    """
    Exhaustive computation of the true gamma_h (Definition 3) via full enumeration over
    S1 subseteq S2 subseteq G, j in G\\S2. Feasible for small m (m<=~10; 3^m triples).
    """
    h_cache = {}
    for mask in range(1 << m):
        S = tuple(i for i in range(m) if mask & (1 << i))
        h_cache[S] = h_val(list(S), Ix, Ii_list)

    best = np.inf
    for s2_mask in range(1 << m):
        S2 = tuple(i for i in range(m) if s2_mask & (1 << i))
        G_minus_S2 = [i for i in range(m) if not (s2_mask & (1 << i))]
        if not G_minus_S2:
            continue
        for j in G_minus_S2:
            S2j = tuple(sorted(S2 + (j,)))
            denom = h_cache[S2] - h_cache[S2j]
            if denom <= 1e-12:
                continue
            s2_list = list(S2)
            full_mask = (1 << len(s2_list)) - 1
            for s1_mask in range(1 << len(s2_list)):
                if s1_mask == full_mask:
                    continue
                S1 = tuple(s2_list[k] for k in range(len(s2_list)) if s1_mask & (1 << k))
                S1j = tuple(sorted(S1 + (j,)))
                numer = h_cache[S1] - h_cache[S1j]
                if numer < 0:
                    continue  # numerical artifact from matrix inversion
                ratio = numer / denom
                if ratio < best:
                    best = ratio
    return best


# --------------------------------------------------------------------------------------
# Quick self-test: does the FIXED Theorem 2 bound actually lower-bound the true gamma_h,
# including at sigma_j^2 > 1 where the OLD (buggy) bound is expected to fail?
# --------------------------------------------------------------------------------------

def self_test_theorem2(n_trials=200, m=5, n=3, sigma2_low=1e-2, sigma2_high=10.0, seed=0):
    rng = np.random.default_rng(seed)
    fixed_violations = 0
    buggy_violations = 0
    fixed_gaps = []
    for t in range(n_trials):
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        P = generate_spd(n, scale=1.0)
        Ix = np.linalg.inv(P)
        M = generate_M_stack(m, n, 0.1, 2.0)
        C = generate_C(m, n, 1.0)
        # deliberately include sigma_i^2 > 1 (the regime where the old bound breaks)
        sigma2_vec = np.random.uniform(sigma2_low, sigma2_high, size=m)

        Ii_list, ci_list = per_sensor_info(M, C, P, sigma2_vec)

        gamma_true = empirical_gamma_h(m, Ix, Ii_list)
        gamma_fixed = theorem2_bound(m, Ix, Ii_list, ci_list, M, sigma2_vec, buggy=False)
        gamma_buggy = theorem2_bound(m, Ix, Ii_list, ci_list, M, sigma2_vec, buggy=True)

        if gamma_fixed > gamma_true + 1e-9:
            fixed_violations += 1
        if gamma_buggy > gamma_true + 1e-9:
            buggy_violations += 1
        fixed_gaps.append(gamma_true - gamma_fixed)

    print(f"n_trials={n_trials}, m={m}, n={n}, sigma2 in [{sigma2_low},{sigma2_high}]")
    print(f"FIXED bound violates gamma_h >= bound in {fixed_violations}/{n_trials} trials")
    print(f"BUGGY (pre-fix) bound violates gamma_h >= bound in {buggy_violations}/{n_trials} trials")
    print(f"FIXED bound gap (gamma_true - bound): mean={np.mean(fixed_gaps):.4f}, "
          f"min={np.min(fixed_gaps):.4f}")
    return fixed_violations, buggy_violations


if __name__ == "__main__":
    self_test_theorem2()
