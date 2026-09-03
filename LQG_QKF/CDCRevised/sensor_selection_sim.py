"""
Sensor/measurement selection under the quadratic observation model of CDC2026.tex (Eq. 1-5).

This is the CDCRevised variant: Theorem 2 is the Woodbury-identity-based bound (see CDC2026.tex in this
folder), which requires NO rank or invertibility assumption on M^(i). This file is otherwise identical
to LQG_QKF/CDC/sensor_selection_sim.py -- the model, greedy/brute-force/linearized selection, and
empirical gamma_h machinery are unaffected by which Theorem 2 proof is used; only theorem2_bound() and
its docstring/self-test differ from the CDC/ (per-j F_tilde_{G,j}) variant.

Static, single-shot A-optimality Van Trees bound selection: x ~ N(0, P), and for each i in the ground
set G, y_i = 1/2 x^T M^(i) x + c_i^T x + v_i, v_i ~ N(0, sigma_i^2). No dynamics.

Implements:
  - h(S) = Tr(B_S) (Proposition 2's closed-form Van Trees bound) and its greedy/brute-force minimizers
    (Algorithm 1 / Problem (5))
  - the linearized greedy baseline (drops all quadratic M^(i) terms, keeping only the linear c_i)
  - the Woodbury-based Theorem 2 lower bound on the supermodularity ratio gamma_h -- a single closed
    form, no per-j object, no rank/invertibility assumption, no c_j=0 case split
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

COLORS = {
    'quad': '#1f77b4',   # blue: proposed greedy quadratic method
    'lin': '#d62728',    # red: linearized greedy baseline
    'brute': '#2ca02c',  # green (dashed): brute-force optimal
    'thm2': '#ff7f0e',   # orange: proposed Theorem 2 bound
    'empirical': '#1f77b4',  # blue: empirical gamma_h
    'prior': '#2ca02c',  # green (dashed): [19]'s restricted-case bound
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


def generate_rank_deficient_M_stack(m, n, scale_min, scale_max, rank):
    """Stack of m rank-<=rank (possibly singular) symmetric PSD M^(i). Used to demonstrate that the
    Woodbury bound (unlike the CDC/ per-j fixed bound) does not require M^(i) invertible."""
    M = np.zeros((m, n, n))
    for i in range(m):
        scale = np.random.uniform(scale_min, scale_max)
        A = np.random.randn(n, rank)
        M[i] = scale * (A @ A.T)
    return M


def generate_C(m, n, scale):
    return np.random.randn(m, n) * scale


# --------------------------------------------------------------------------------------
# Core Van Trees / A-optimality machinery (Proposition 2, Problem (5)) -- identical to CDC/
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


def greedy_select(m, Ix, Ii_list, R):
    """Algorithm 1: greedily add the sensor with the largest marginal decrease in h(S) until h(S)<=R."""
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
    the ranking and the R-ratio stopping check use h_lin.
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
# Theorem 2 (Woodbury-based) and reference bounds
# --------------------------------------------------------------------------------------

def theorem2_bound_woodbury(m, Ix, Ii_list, P):
    """
    Woodbury-based Theorem 2 (CDCRevised): a single closed form,
        gamma_h >= lam_min(B_G)^2 / [lam_max(P)^2 (1 + lam_max(P) * max_i lam_max(I_i))],
    with B_G = (I_x + sum_{i in G} I_i)^{-1}. No per-j object, no min_j needed beyond the max inside the
    formula, no rank/invertibility assumption on M^(i), no c_j=0 case split -- see CDC2026.tex's Theorem
    2 proof in this folder for the derivation (Woodbury identity applied directly to B_S, matching
    TI_sensor_selection/CASE's draft).
    """
    F_full = F_of(list(range(m)), Ix, Ii_list)
    B_G = np.linalg.inv(F_full)
    lam_min_BG = np.linalg.eigvalsh(B_G).min()
    lam_max_P = np.linalg.eigvalsh(P).max()
    max_lam_Ii = max(np.linalg.eigvalsh(Ii_list[i]).max() for i in range(m))
    return (lam_min_BG ** 2) / ((lam_max_P ** 2) * (1.0 + lam_max_P * max_lam_Ii))


def prior_bound_c19(m, Ix, P, sigma2_vec, F_full):
    """
    [19]'s own explicit weak-submodularity constant (their eq. 27-28), valid only for their restricted
    case (rank-1 M^(i), c_i=0). Applied here as-is (naively, on our general model) purely as the
    reference "prior work" curve for Fig. 2.
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
# Self-test: validity of the Woodbury bound, INCLUDING at rank-deficient M and c_i=0
# (the two cases the CDC/ per-j fixed bound cannot handle without its M-invertible assumption)
# --------------------------------------------------------------------------------------

def self_test_theorem2_woodbury(n_trials=200, m=5, n=3, sigma2_low=1e-2, sigma2_high=1e2, seed=0):
    rng = np.random.default_rng(seed)
    violations = 0
    gaps = []
    for t in range(n_trials):
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        P = generate_spd(n, scale=1.0)
        Ix = np.linalg.inv(P)
        M = generate_M_stack(m, n, 0.1, 2.0)
        C = generate_C(m, n, 1.0)
        sigma2_vec = np.random.uniform(sigma2_low, sigma2_high, size=m)

        Ii_list, ci_list = per_sensor_info(M, C, P, sigma2_vec)
        gamma_true = empirical_gamma_h(m, Ix, Ii_list)
        gamma_bound = theorem2_bound_woodbury(m, Ix, Ii_list, P)

        if gamma_bound > gamma_true + 1e-9:
            violations += 1
        gaps.append(gamma_true - gamma_bound)

    print(f"[general model] n_trials={n_trials}, violations={violations}/{n_trials}, "
          f"mean gap={np.mean(gaps):.4f}")

    # Rank-deficient M (rank 1, n=4) -- CDC/'s fixed bound requires M invertible and cannot run here.
    n2, m2, rank = 4, 5, 1
    violations_rd = 0
    for t in range(50):
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        P = generate_spd(n2, scale=1.0)
        Ix = np.linalg.inv(P)
        M = generate_rank_deficient_M_stack(m2, n2, 0.1, 2.0, rank)
        C = generate_C(m2, n2, 1.0)
        sigma2_vec = np.random.uniform(sigma2_low, sigma2_high, size=m2)
        Ii_list, ci_list = per_sensor_info(M, C, P, sigma2_vec)
        gamma_true = empirical_gamma_h(m2, Ix, Ii_list)
        gamma_bound = theorem2_bound_woodbury(m2, Ix, Ii_list, P)
        if gamma_bound > gamma_true + 1e-9:
            violations_rd += 1
    print(f"[rank-1 M, n=4] violations={violations_rd}/50")

    # c_i = 0 for all i (pure quadratic, no linear term at all)
    violations_c0 = 0
    for t in range(50):
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        P = generate_spd(n, scale=1.0)
        Ix = np.linalg.inv(P)
        M = generate_M_stack(m, n, 0.1, 2.0)
        C = np.zeros((m, n))
        sigma2_vec = np.random.uniform(sigma2_low, sigma2_high, size=m)
        Ii_list, ci_list = per_sensor_info(M, C, P, sigma2_vec)
        gamma_true = empirical_gamma_h(m, Ix, Ii_list)
        gamma_bound = theorem2_bound_woodbury(m, Ix, Ii_list, P)
        if gamma_bound > gamma_true + 1e-9:
            violations_c0 += 1
    print(f"[c_i=0 for all i] violations={violations_c0}/50")

    return violations, violations_rd, violations_c0


if __name__ == "__main__":
    self_test_theorem2_woodbury()
