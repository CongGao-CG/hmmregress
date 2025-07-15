import numpy as np
from log_norm_pdf import log_norm_pdf


def backwardcont(hmm, observation):
    n_obs = len(observation)
    n_states = len(hmm['states'])
    trans = np.nan_to_num(hmm['trans_probs'], nan=0.0)
    b = np.empty((n_states, n_obs))
    b[:, n_obs-1] = 0.0
    for k in range(n_obs-2, -1, -1):
        for i in range(n_states):
            logsum = -np.inf
            for j in range(n_states):
                mu = hmm['emission_params'][j]['mean']
                sd = hmm['emission_params'][j]['sd']
                temp = b[j, k+1] + np.log(trans[i, j]) + log_norm_pdf(observation[k+1], mu, sd)
                logsum = max(temp, logsum) + np.log1p(np.exp(-abs(temp - logsum)))
            b[i, k] = logsum
    return b
