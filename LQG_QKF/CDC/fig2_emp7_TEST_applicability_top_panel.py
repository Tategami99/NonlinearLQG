"""
TEST FIGURE -- NOT part of the official 2-figure set in `fig2_empirical_coverage.py`.

Idea #1 from the "what else could Figure 2 be" discussion: instead of an accuracy-style ratio, plot
APPLICABILITY -- at each C, what fraction of trials have a valid, explicit, computable bound at all.

This paper's Theorem 2 needs only full-rank M -- it never needs C=0 -- so it is computable in 100% of
trials at every C tested. Hashemi's explicit constant (eq. 27-28) is only DEFINED for rank-1 M and C=0 --
their formula has no c_j parameter at all -- so the instant C > 0, "is their explicit constant computable
for this sensor" is definitionally False, every single trial. That makes this a hard step function: 100%
at C=0, 0% for every C>0, for the SAME reason on every trial (not sampling noise averaging out to a
gradual curve). Built to show exactly that shape, since it was requested explicitly.

**Single panel, not two (revised 2026-09-09).** The original version of this file paired the top panel
with a reproduction of the real emp7 bottom panel below it, copying that figure's two-panel layout. The
user pointed out this made no sense here: emp7's two panels exist because the bottom panel deliberately
uses [19]'s domain (M held at rank-1) while a matching top panel would use this paper's domain (full-rank
M) -- two DIFFERENT underlying trial setups, so two panels with clarifying text above each is the right
way to keep that domain distinction visible. This figure's "ours" and "theirs" curves are computed from
the SAME experimental setup already (same C sweep, no domain split between the two curves) -- they were
always plottable on one axis, and the second panel was just unmotivated visual duplication.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sensor_selection_sim import perf_dir, COLORS
from fig2_reframe import add_caption

n_trials = 150
C_scales = np.array([0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0])

# Fraction of trials where each theorem's explicit constant is computable at all.
# Ours: full-rank M_j -> always computable, regardless of C. Theirs: only defined at rank-1, C=0.
ours_coverage = []
theirs_coverage = []
for C_scale in C_scales:
    ours_hits, theirs_hits = 0, 0
    for _ in range(n_trials):
        # This paper's Theorem 2: applicable iff M_j is full rank. Always true by construction here.
        ours_hits += 1
        # [19]'s explicit constant (eq. 27-28): applicable iff M_j is rank-1 AND c_j == 0.
        theirs_hits += 1 if C_scale == 0.0 else 0
    ours_coverage.append(ours_hits / n_trials)
    theirs_coverage.append(theirs_hits / n_trials)
ours_coverage, theirs_coverage = np.array(ours_coverage), np.array(theirs_coverage)

fig, ax = plt.subplots(figsize=(9.5, 7.8))
ax.plot(C_scales, ours_coverage, color=COLORS['thm2'], marker='^', linewidth=2.4, zorder=4,
        label="This paper (Theorem 2): needs only full-rank $M^{(i)}$")
ax.plot(C_scales, theirs_coverage, color=COLORS['prior'], marker='s', linewidth=2.4, zorder=3,
        label="[19]'s explicit constant: only defined at rank-1, $C=0$")
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel(r'Linear-term magnitude $C$', fontsize=12.5)
ax.set_ylabel('Fraction of trials with a valid,\nexplicit, computable bound', fontsize=11)
ax.set_title('TEST: applicability, not accuracy -- a hard step, not a gradual curve', fontsize=12.5, color='#8B0000')
ax.grid(alpha=0.3)
ax.legend(loc='center right', fontsize=9.5, framealpha=0.95)

fig.suptitle('TEST FIGURE: applicability framing (idea #1, single panel)', y=0.97, fontsize=14)
fig.tight_layout(rect=[0, 0.24, 1, 0.92])
add_caption(
    fig, 'Test',
    "Answers a different question than an accuracy ratio would -- not 'how close' but 'does a number "
    "exist at all.' Ours: yes, always. Theirs: only at their own C=0 case -- a hard step, not a decline.",
    "[19]'s explicit weak-submodularity constant (eq. 27-28) has no parameter for a linear term -- it is "
    "undefined, not merely inaccurate, for any $C\\neq 0$, so its coverage is exactly 0% there on every "
    "trial (not an average over noisy cases). This paper's Theorem 2 requires only that $M^{(i)}$ be full "
    "rank, so its coverage is 100% throughout. Both curves come from the same experimental setup -- there "
    "is no domain split to justify a second panel here.",
    y=0.03,
)
fig.savefig(perf_dir + 'fig2_emp7_TEST_applicability_top_panel.png', dpi=200)
plt.close(fig)
print(f"Saved {perf_dir}fig2_emp7_TEST_applicability_top_panel.png")
