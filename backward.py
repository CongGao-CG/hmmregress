import numpy as np


def backward(hmm, observation):
    n_obs = len(observation)
    n_states = len(hmm['states'])
    trans = np.nan_to_num(hmm['trans_probs'], nan=0.0)
    emis = np.nan_to_num(hmm['emission_probs'], nan=0.0)
    b = np.empty((n_states, n_obs))
    for i in range(n_states):
        b[i, n_obs - 1] = 0.0
    for k in range(n_obs - 2, -1, -1):
        for i in range(n_states):
            logsum = -np.inf
            for j in range(n_states):
                temp = (
                    b[j, k + 1]
                    + np.log(trans[i, j])
                    + np.log(emis[j, observation[k + 1]])
                )
                logsum = max(logsum, temp) + np.log1p(np.exp(-abs(logsum - temp)))
            b[i, k] = logsum
    return b
