#!/usr/bin/env python3
# validate_bwcont.py ── sanity check for BWcont / continuous-Gaussian HMM
# ===============================================================

import itertools
import numpy as np

from initHMMcont import initHMMcont
from BWcont import BWcont
from forwardcont import forwardcont
from log_sum_exp import log_sum_exp

# ─── 1. Ground-truth continuous HMM (2 states, 1-D Gaussian emissions) ──
STATES = ["S0", "S1"]

TRUE_START = np.array([0.65, 0.35])
TRUE_TRANS = np.array([[0.80, 0.20],
                       [0.25, 0.75]])
TRUE_EM_PARAMS = [
    {"mean": -3.0, "sd": 1.0},
    {"mean":  3.0, "sd": 1.2},
]

TRUE_HMM = initHMMcont(
    STATES,
    start_probs=TRUE_START,
    trans_probs=TRUE_TRANS,
    emission_params=TRUE_EM_PARAMS,
)

# ─── 2. Utilities ──────────────────────────────────────────────────────
def sample_hmm_cont(hmm, *, n_seq=1000, length=100, rng):
    """Generate synthetic sequences of real numbers from a Gaussian-HMM."""
    n_states = len(hmm["states"])
    out = []
    for _ in range(n_seq):
        seq = np.empty(length, dtype=float)
        s = rng.choice(n_states, p=hmm["start_probs"])
        mu, sd = hmm["emission_params"][s]["mean"], hmm["emission_params"][s]["sd"]
        seq[0] = rng.normal(mu, sd)
        for t in range(1, length):
            s = rng.choice(n_states, p=hmm["trans_probs"][s])
            mu, sd = hmm["emission_params"][s]["mean"], hmm["emission_params"][s]["sd"]
            seq[t] = rng.normal(mu, sd)
        out.append(seq)
    return out


def random_cont_hmm(n_states, rng):
    """Random fully-dense 1-D Gaussian HMM."""
    sv = rng.dirichlet(np.ones(n_states))
    tm = np.stack([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)])
    em = [{"mean": rng.normal(0, 5), "sd": rng.uniform(0.5, 2.0)}
          for _ in range(n_states)]
    return initHMMcont(list(range(n_states)), start_probs=sv,
                       trans_probs=tm, emission_params=em)


def max_abs_err(a, b):
    return np.max(np.abs(a - b))


def min_error_permuted(t_start, t_trans, t_mu, t_sd,
                       e_start, e_trans, e_mu, e_sd):
    """Minimal max-error after state permutation."""
    n = len(t_start)
    best = np.inf
    for perm in itertools.permutations(range(n)):
        P = np.eye(n)[list(perm)]            # permutation matrix
        s_err = max_abs_err(t_start, e_start @ P)
        t_err = max_abs_err(t_trans, P.T @ e_trans @ P)
        m_err = max_abs_err(t_mu,  (P.T @ e_mu))
        d_err = max_abs_err(t_sd,  (P.T @ e_sd))
        best  = min(best, max(s_err, t_err, m_err, d_err))
    return best


# ─── 3. Prepare synthetic data ─────────────────────────────────────────
RNG = np.random.default_rng(42)
OBS_LIST = sample_hmm_cont(TRUE_HMM, n_seq=1000, length=100, rng=RNG)

# ─── 4. Fit with BWcont (multiple random restarts) ─────────────────────
RESTARTS = 7
best_ll  = -np.inf
BEST_HMM = None

for r in range(RESTARTS):
    init = random_cont_hmm(2, rng=np.random.default_rng(100 + r))
    # init = TRUE_HMM
    trained = BWcont(init, OBS_LIST, maxIterations=120, delta=1e-4)["hmm"]

    ll = sum(
        log_sum_exp(forwardcont(trained, seq)[:, -1])
        for seq in OBS_LIST
    )
    if ll > best_ll:
        best_ll, BEST_HMM = ll, trained

print(f"Best log-likelihood over {RESTARTS} restarts: {best_ll:,.1f}")

# ─── 5. Evaluate against ground truth (up to permutation) ──────────────
est_start = BEST_HMM["start_probs"]
est_trans = BEST_HMM["trans_probs"]
est_mu    = np.array([p["mean"] for p in BEST_HMM["emission_params"]])
est_sd    = np.array([p["sd"]   for p in BEST_HMM["emission_params"]])

true_mu = np.array([p["mean"] for p in TRUE_EM_PARAMS])
true_sd = np.array([p["sd"]   for p in TRUE_EM_PARAMS])

max_err = min_error_permuted(
    TRUE_START, TRUE_TRANS, true_mu, true_sd,
    est_start,  est_trans,  est_mu, est_sd
)
print(f"max |error|  after permutation: {max_err:.4f}")

# ─── 6. Verdict ────────────────────────────────────────────────────────
TOL = 0.15          # ≤15 ppt / units  tolerance for parameters
if max_err < TOL:
    print("PASS – learned parameters are within tolerance.")
else:
    print("FAIL – maximum error exceeds tolerance.")
print('\nTrue vs Best Start Probs:')
print(TRUE_HMM['start_probs'])
print(np.round(BEST_HMM['start_probs'],4))
print('\nTrue vs Best Transition Probs:')
print(TRUE_HMM['trans_probs'])
print(np.round(BEST_HMM['trans_probs'],4))
print('\nTrue vs Best Emission Params:')
print(TRUE_HMM['emission_params'])
formatted_em = [{'mean': np.round(item['mean'],4), 'sd': np.round(item['sd'],2)} for item in BEST_HMM['emission_params']]
print(formatted_em)
