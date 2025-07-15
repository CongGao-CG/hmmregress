#!/usr/bin/env python3
# validate_bwcont_reg.py – unit-test for BWcont_Reg (Gaussian-regression HMM)
# ==========================================================================

import itertools
import numpy as np
from pathlib import Path

# ── import user-supplied modules ─────────────────────────────────────────
from initHMMcont_Reg import initHMMcont_Reg
from BWcont_Reg       import BWcont_Reg
from forwardcont_Reg  import forwardcont_Reg
from log_sum_exp      import log_sum_exp
from softmax          import softmax

# ------------------------------------------------------------------------
# 1.  Ground-truth model definition (2 states, small feature sets)
# ------------------------------------------------------------------------
STATES = ["S0", "S1"]

F_S   = 2          # #features for start logits
F_T   = 3          # #features for transition logits
F_E   = 2          # #features for emission mean

TRUE_startCoefs = np.array([[ 2.0, -1.5],     # logits favour S0
                            [-1.0,  0.5]])    # logits favour S1

TRUE_transCoefs = {
    "S0": np.array([[ 2.5,  0.0, -1.0],       # S0→S0 strong
                    [-1.5,  0.5,  1.0]]),     # S0→S1 weaker
    "S1": np.array([[ 0.0, -1.0,  1.5],       # S1→S0 weaker
                    [ 2.0,  0.5, -0.5]]),     # S1→S1 strong
}

TRUE_emissionCoefs = np.array([[ 1.0,  0.5],   # mean = 1*X₁ + 0.5*X₂  (state 0)
                               [-1.0,  0.3]])  #        -1*X₁ + 0.3*X₂ (state 1)

TRUE_sds = np.array([0.8, 1.0])

TRUE_HMM = initHMMcont_Reg(
    STATES,
    startCoefs   = TRUE_startCoefs,
    transCoefs   = TRUE_transCoefs,
    emissionCoefs= TRUE_emissionCoefs,
    sds          = TRUE_sds,
)

# ------------------------------------------------------------------------
# 2.  Synthetic sequence generator
# ------------------------------------------------------------------------
def sample_sequence(hmm, length, rng):
    """Return (obs, Xs, Xt, Xe) for a single sequence."""
    F_S = hmm["startCoefs"].shape[1]
    F_T = next(iter(hmm["transCoefs"].values())).shape[1]
    F_E = hmm["emissionParams"][STATES[0]]["coefs"].size

    Xs = rng.normal(size=F_S)
    Xt = rng.normal(size=(F_T, length - 1))
    Xe = rng.normal(size=(F_E, length))

    # start
    start_p = softmax(hmm["startCoefs"] @ Xs)
    s = rng.choice(len(STATES), p=start_p)
    obs = np.empty(length)
    mu = hmm["emissionParams"][STATES[s]]["coefs"] @ Xe[:, 0]
    sd = hmm["emissionParams"][STATES[s]]["sd"]
    obs[0] = rng.normal(mu, sd)

    for t in range(1, length):
        trans_p = softmax(hmm["transCoefs"][STATES[s]] @ Xt[:, t-1])
        s = rng.choice(len(STATES), p=trans_p)
        mu = hmm["emissionParams"][STATES[s]]["coefs"] @ Xe[:, t]
        sd = hmm["emissionParams"][STATES[s]]["sd"]
        obs[t] = rng.normal(mu, sd)

    return obs, Xs, Xt, Xe


def make_dataset(hmm, n_seq=150, length=60, seed=0):
    rng = np.random.default_rng(seed)
    obs_list, Xs_list, Xt_list, Xe_list = [], [], [], []
    for _ in range(n_seq):
        o, xs, xt, xe = sample_sequence(hmm, length, rng)
        obs_list.append(o)
        Xs_list.append(xs)
        Xt_list.append(xt)
        Xe_list.append(xe)
    return obs_list, Xs_list, Xt_list, Xe_list


OBS_LIST, XS_LIST, XT_LIST, XE_LIST = make_dataset(TRUE_HMM)

# ------------------------------------------------------------------------
# 3.  Helper – error after state permutation
# ------------------------------------------------------------------------
def permute_params(params, perm):
    """Apply permutation array 'perm' (len=nStates) to parameter containers."""
    startC = params["startCoefs"][perm]
    transC = {STATES[i]: params["transCoefs"][STATES[perm[i]]][perm] for i in range(len(perm))}
    emisC  = np.array([params["emissionParams"][STATES[perm[i]]]["coefs"] for i in range(len(perm))])
    sds    = np.array([params["emissionParams"][STATES[perm[i]]]["sd"]    for i in range(len(perm))])
    return startC, transC, emisC, sds


def max_err(true_p, est_p):
    return np.max(np.abs(true_p - est_p))


def min_error_up_to_perm(est):
    best = np.inf
    for perm in itertools.permutations(range(len(STATES))):
        t_start, t_trans, t_emis, t_sd = permute_params(TRUE_HMM,  perm)
        e_start, e_trans, e_emis, e_sd = permute_params(est, perm)

        # stack all coefficients for a single max-abs difference
        err = max(
            max_err(t_start, e_start),
            max_err(np.vstack(list(t_trans.values())),
                    np.vstack(list(e_trans.values()))),
            max_err(t_emis, e_emis),
            max_err(t_sd,   e_sd),
        )
        best = min(best, err)
    return best

# ------------------------------------------------------------------------
# 4.  Training with BWcont_Reg (random restarts)
# ------------------------------------------------------------------------
def random_hmm(seed):
    rng = np.random.default_rng(seed)
    startC = rng.normal(scale=0.5, size=TRUE_startCoefs.shape)
    transC = {st: rng.normal(scale=0.5, size=mat.shape)
              for st, mat in TRUE_transCoefs.items()}
    emisC  = rng.normal(scale=0.5, size=TRUE_emissionCoefs.shape)
    sds    = rng.uniform(0.5, 2.0, size=TRUE_sds.shape)
    return initHMMcont_Reg(STATES, startC, transC, emisC, sds)


BEST_LL  = -np.inf
BEST_HMM = None
RESTARTS = 5

for r in range(RESTARTS):
    init = random_hmm(100 + r)
    trained = BWcont_Reg(init, OBS_LIST, XS_LIST, XT_LIST, XE_LIST,
                         maxIterations=80, delta=1e-6)["hmm"]

    ll = sum(
        log_sum_exp(forwardcont_Reg(trained, o, xs, xt, xe)[:, -1])
        for o, xs, xt, xe in zip(OBS_LIST, XS_LIST, XT_LIST, XE_LIST)
    )
    if ll > BEST_LL:
        BEST_LL, BEST_HMM = ll, trained

print(f"Best log-likelihood over {RESTARTS} restarts: {BEST_LL:,.1f}")

# ------------------------------------------------------------------------
# 5.  Compare to truth (up to permutation)
# ------------------------------------------------------------------------
max_error = min_error_up_to_perm(BEST_HMM)
print(f"max |error|  after permutation: {max_error:.3f}")

TOL = 0.30          # coefficients can differ more yet yield same probs
if max_error < TOL:
    print("PASS – learned parameters are within tolerance.")
else:
    print("FAIL – maximum error exceeds tolerance.")
print('\nTrue vs Best Start Coefs:')
print(TRUE_HMM['startCoefs'])
print(BEST_HMM['startCoefs'])
print('\nTrue vs Best Transition Coefs:')
print(TRUE_HMM['transCoefs'])
print(BEST_HMM['transCoefs'])
print('\nTrue vs Emission Params:')
print(TRUE_HMM['emissionParams'])
print(BEST_HMM['emissionParams'])
