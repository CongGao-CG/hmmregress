import numpy as np


def forward(hmm, observation):
    n_obs = len(observation)
    n_states = len(hmm['states'])
    trans = np.nan_to_num(hmm['trans_probs'], nan=0.0)
    emis = np.nan_to_num(hmm['emission_probs'], nan=0.0)
    f = np.empty((n_states, n_obs))
    f[:] = np.nan
    for i in range(n_states):
        f[i, 0] = np.log(hmm['start_probs'][i] * emis[i, observation[0]])
    for k in range(1, n_obs):
        for i in range(n_states):
            logsum = -np.inf
            for j in range(n_states):
                temp = f[j, k-1] + np.log(trans[j, i])
                logsum = max(logsum, temp) + np.log1p(np.exp(-abs(logsum - temp)))
            f[i, k] = np.log(emis[i, observation[k]]) + logsum
    return f
