import numpy as np


def initHMM(states, symbols, start_probs=None, trans_probs=None, emission_probs=None):
    n_states = len(states)
    n_symbols = len(symbols)
    sv = np.full(n_states, 1.0 / n_states)
    tm = 0.5 * np.eye(n_states) + (0.5 / n_states) * np.ones((n_states, n_states))
    em = np.full((n_states, n_symbols), 1.0 / n_symbols)
    if start_probs is not None:
        sv = np.array(start_probs, dtype=float)
    if trans_probs is not None:
        tm = np.array(trans_probs, dtype=float)
    if emission_probs is not None:
        em = np.array(emission_probs, dtype=float)
    return {
        'states': states,
        'symbols': symbols,
        'start_probs': sv,
        'trans_probs': tm,
        'emission_probs': em
    }
