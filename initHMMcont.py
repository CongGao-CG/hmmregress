import numpy as np


def initHMMcont(states, start_probs=None, trans_probs=None, emission_params=None):
    n_states = len(states)
    sv = np.full(n_states, 1.0 / n_states)
    tm = 0.5 * np.eye(n_states) + (0.5 / n_states) * np.ones((n_states, n_states))
    if emission_params is None:
        emission_params = [{'mean': 0.0, 'sd': 1.0} for _ in states]
    if start_probs is not None:
        sv = np.array(start_probs, dtype=float)
    if trans_probs is not None:
        tm = np.array(trans_probs, dtype=float)
    return {'states': states,
            'start_probs': sv,
            'trans_probs': tm,
            'emission_params': emission_params
    }
