#!/usr/bin/env python3
# validate_bw.py ── synthetic-data sanity check for BW/BWR
# =======================================================

import itertools
import numpy as np
from initHMM import initHMM
from BW import BW
from forward import forward
from log_sum_exp import log_sum_exp

# ─── ground-truth HMM ────────────────────────────────────────────────────
STATES  = ['S0', 'S1']
SYMBOLS = ['A', 'B']

TRUE_START = np.array([0.6, 0.4])
TRUE_TRANS = np.array([[0.7, 0.3],
                       [0.2, 0.8]])
TRUE_EMIS  = np.array([[0.5, 0.5],
                       [0.1, 0.9]])

TRUE_HMM = initHMM(
    STATES, SYMBOLS,
    start_probs=TRUE_START,
    trans_probs=TRUE_TRANS,
    emission_probs=TRUE_EMIS,
)

# ─── helpers ─────────────────────────────────────────────────────────────
def sample_hmm(hmm, n_seq=400, length=200, rng=np.random.default_rng()):
    '''Generate observation sequences from an HMM.'''
    n_states  = len(hmm['states'])
    n_symbols = len(hmm['symbols'])
    obs_list  = []
    for _ in range(n_seq):
        s = rng.choice(n_states, p=hmm['start_probs'])
        seq = [rng.choice(n_symbols, p=hmm['emission_probs'][s])]
        for _ in range(length - 1):
            s = rng.choice(n_states, p=hmm['trans_probs'][s])
            seq.append(rng.choice(n_symbols, p=hmm['emission_probs'][s]))
        obs_list.append(seq)
    return obs_list


def random_hmm(n_states, n_symbols, rng):
    '''Return a random fully-dense HMM of given size.'''
    sv = rng.dirichlet(np.ones(n_states))
    tm = np.stack([rng.dirichlet(np.ones(n_states)) for _ in range(n_states)])
    em = np.stack([rng.dirichlet(np.ones(n_symbols)) for _ in range(n_states)])
    return initHMM(
        list(range(n_states)), list(range(n_symbols)),
        start_probs=sv, trans_probs=tm, emission_probs=em
    )


def max_abs_err(true, est):
    return np.max(np.abs(true - est))


def min_error_up_to_permutation(t_start, t_trans, t_emis,
                                e_start, e_trans, e_emis):
    '''Return the smallest max-|error| achievable by permuting state labels.'''
    n = len(t_start)
    best = np.inf
    for perm in itertools.permutations(range(n)):
        P = np.eye(n)[list(perm)]      # permutation matrix
        s_err = max_abs_err(t_start, e_start @ P)
        t_err = max_abs_err(t_trans, P.T @ e_trans @ P)
        e_err = max_abs_err(t_emis,  P.T @ e_emis)
        best  = min(best, max(s_err, t_err, e_err))
    return best


# ─── main routine ────────────────────────────────────────────────────────
if __name__ == '__main__':
    rng = np.random.default_rng(0)
    OBS_LIST = sample_hmm(TRUE_HMM, n_seq=400, length=200, rng=rng)
    BEST_r   = np.nan
    BEST_LL  = -np.inf
    BEST_HMM = None
    RESTARTS = 5
    for r in range(RESTARTS):
        init = random_hmm(2, 2, rng=np.random.default_rng(r + 123))
        trained = BW(init, OBS_LIST, maxIterations=100)['hmm']
        ll = sum(
            log_sum_exp(forward(trained, obs)[:, -1])
            for obs in OBS_LIST
        )
        if ll > BEST_LL:
            BEST_r, BEST_LL, BEST_HMM = r, ll, trained
    
    print(f'Best log-likelihood over {RESTARTS} restarts: {BEST_LL:,.1f}')

    max_err = min_error_up_to_permutation(
        TRUE_START, TRUE_TRANS, TRUE_EMIS,
        BEST_HMM['start_probs'],
        BEST_HMM['trans_probs'],
        BEST_HMM['emission_probs'],
    )
    print(f'max |error|  after permutation: {max_err:.4f}')

    TOL = 0.05
    if max_err < TOL:
        print('PASS – learned parameters are within tolerance.')
    else:
        print('FAIL – maximum error exceeds tolerance.')
    print('\nTrue vs Best Start Probs:')
    print(TRUE_HMM['start_probs'])
    print(np.round(BEST_HMM['start_probs'],2))
    print('\nTrue vs Best Transition Probs:')
    print(TRUE_HMM['trans_probs'])
    print(np.round(BEST_HMM['trans_probs'],2))
    print('\nTrue vs Best Emission Probs:')
    print(TRUE_HMM['emission_probs'])
    print(np.round(BEST_HMM['emission_probs'],2))
