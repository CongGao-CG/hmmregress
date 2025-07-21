#!/usr/bin/env python3
# validate_bwcont_reg.py  ––  test BWcont_Reg on a synthetic dataset
# ================================================================

import itertools
import numpy as np

from initHMMcont_Reg import initHMMcont_Reg
from BWcont_Reg       import BWcont_Reg
from forwardcont_Reg  import forwardcont_Reg
from log_sum_exp      import log_sum_exp
from softmax          import softmax

# ─── 1. Ground-truth regression-HMM definition ──────────────────────────
STATES = ["S0", "S1"]


F_S, F_T, F_E = 2, 3, 2          # #features for start, transition, emission

TRUE_startCoefs = np.array([[ 0.0,  0.0],
                            [-1.0,  0.5]])

TRUE_transCoefs = {
    "S0": np.array([[ 0.0,  0.0,  0.0],
                    [-1.5,  0.5,  1.0]]),
    "S1": np.array([[ 0.0,  0.0,  0.0],
                    [ 2.0,  0.5, -0.5]]),
}

TRUE_emissionCoefs = np.array([[ 1.0,  0.5],
                               [-1.0,  0.3]])

TRUE_sds = np.array([0.8, 1.0])

TRUE_HMM = initHMMcont_Reg(
    STATES,
    startCoefs    = TRUE_startCoefs,
    transCoefs    = TRUE_transCoefs,
    emissionCoefs = TRUE_emissionCoefs,
    sds           = TRUE_sds,
)

# ─── 2.  Synthetic-sequence generator ───────────────────────────────────
def sample_sequence(hmm, length, rng):
    """Generate one sequence plus its predictor matrices/vectors."""
    F_S = hmm["startCoefs"].shape[1]
    F_T = next(iter(hmm["transCoefs"].values())).shape[1]
    F_E = hmm["emissionParams"][STATES[0]]["coefs"].size

    # Initialize with first element/row as 1 for bias term
    Xs = np.ones(F_S)                             # (F_S,)
    if F_S > 1:
        Xs[1:] = rng.normal(size=F_S - 1)
    
    Xt = np.ones((F_T, length - 1))               # (F_T, T-1)
    if F_T > 1:
        Xt[1:, :] = rng.normal(size=(F_T - 1, length - 1))
    
    Xe = np.ones((F_E, length))                   # (F_E, T)
    if F_E > 1:
        Xe[1:, :] = rng.normal(size=(F_E - 1, length))

    # initial state
    p0 = softmax(hmm["startCoefs"] @ Xs)
    s  = rng.choice(len(STATES), p=p0)

    obs = np.empty(length)
    mu0 = hmm["emissionParams"][STATES[s]]["coefs"] @ Xe[:, 0]
    sd0 = hmm["emissionParams"][STATES[s]]["sd"]
    obs[0] = rng.normal(mu0, sd0)

    for t in range(1, length):
        pt = softmax(hmm["transCoefs"][STATES[s]] @ Xt[:, t-1])
        s  = rng.choice(len(STATES), p=pt)
        mu = hmm["emissionParams"][STATES[s]]["coefs"] @ Xe[:, t]
        sd = hmm["emissionParams"][STATES[s]]["sd"]
        obs[t] = rng.normal(mu, sd)

    return obs, Xs, Xt, Xe


def make_dataset(hmm, *, n_seq=1000, length=100, seed=0):
    rng = np.random.default_rng(seed)
    obs_list, xs_list, xt_list, xe_list = [], [], [], []
    for _ in range(n_seq):
        o, xs, xt, xe = sample_sequence(hmm, length, rng)
        obs_list.append(o)
        xs_list.append(xs)
        xt_list.append(xt)
        xe_list.append(xe)
    return obs_list, xs_list, xt_list, xe_list


OBS_LIST, XS_LIST, XT_LIST, XE_LIST = make_dataset(TRUE_HMM)

# ─── 3.  Helper – evaluate up to state permutation ──────────────────────
def permute_params(hmm_dict, perm):
    startC = hmm_dict["startCoefs"][perm, :]
    transC = {}
    for i in range(len(perm)):
        transC[STATES[i]] = hmm_dict["transCoefs"][STATES[perm[i]]][perm, :]
    emisC = np.array([
        hmm_dict["emissionParams"][STATES[perm[i]]]["coefs"] 
        for i in range(len(perm))
    ])
    sds = np.array([
        hmm_dict["emissionParams"][STATES[perm[i]]]["sd"] 
        for i in range(len(perm))
    ])
    return startC, transC, emisC, sds


def max_abs_err(a, b):          # helper
    return np.max(np.abs(a - b))


def min_error_permuted(est_hmm):
    best = np.inf
    t_s = TRUE_startCoefs
    t_t = TRUE_transCoefs
    t_e = TRUE_emissionCoefs
    t_d = TRUE_sds
    for perm in itertools.permutations(range(len(STATES))):
        e_s, e_t, e_e, e_d = permute_params(est_hmm, perm)
        e_s = e_s - e_s[0,:]
        e_t = {key: arr - arr[0] for key, arr in e_t.items()}
        err = max(
            max_abs_err(t_s, e_s),
            max_abs_err(np.vstack(list(t_t.values())),
                        np.vstack(list(e_t.values()))),
            max_abs_err(t_e, e_e),
            max_abs_err(t_d, e_d),
        )
        best = min(best, err)
    return best

# ─── 4.  Train via BWcont_Reg (handful of random restarts) ──────────────
def random_hmm(seed):
    rng = np.random.default_rng(seed)

    startC = np.zeros_like(TRUE_startCoefs)
    startC[1:] = rng.normal(scale=3, size=startC[1:].shape)

    transC = {}
    for st, tpl in TRUE_transCoefs.items():
        mat = np.zeros_like(tpl)
        mat[1:] = rng.normal(scale=3, size=mat[1:].shape)   # keep row-0 = 0
        transC[st] = mat

    emisC = rng.normal(scale=3, size=TRUE_emissionCoefs.shape)
    sds   = rng.uniform(0.5, 2.0, size=TRUE_sds.shape)

    return initHMMcont_Reg(STATES, startC, transC, emisC, sds)

BEST_LL, BEST_HMM = -np.inf, None
RESTARTS = 5

for r in range(RESTARTS):
    init = random_hmm(100 + r)
    # init = TRUE_HMM
    trained = BWcont_Reg(init, OBS_LIST, XS_LIST, XT_LIST, XE_LIST,
                         maxIterations=100, delta=1e-2)["hmm"]
    ll = sum(
        log_sum_exp(
            forwardcont_Reg(trained, o, xs, xt, xe)[:, -1]
        )
        for o, xs, xt, xe in zip(OBS_LIST, XS_LIST, XT_LIST, XE_LIST)
    )
    if ll > BEST_LL:
        BEST_LL, BEST_HMM = ll, trained

print(f"Best log-likelihood over {RESTARTS} restarts: {BEST_LL:,.1f}")

# ─── 5.  Compare to truth (allowing label switches) ─────────────────────
max_error = min_error_permuted(BEST_HMM)
print(f"max |error|  after permutation : {max_error:.3f}")

TOL = 0.30         # generous tolerance – coefficients can differ yet give identical probs
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
