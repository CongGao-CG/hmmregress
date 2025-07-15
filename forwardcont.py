import numpy as np
from log_norm_pdf import log_norm_pdf


def forwardcont(hmm, observation):
    n_obs = len(observation)
    n_states = len(hmm['states'])
    trans = np.nan_to_num(hmm['trans_probs'], nan=0.0)
    f = np.empty((n_states, n_obs))
    for i in range(n_states):
        mu = hmm['emission_params'][i]['mean']
        sd = hmm['emission_params'][i]['sd']
        f[i, 0] = np.log(hmm['start_probs'][i]) + log_norm_pdf(observation[0], mu, sd)
    for k in range(1, n_obs):
        for i in range(n_states):
            logsum = -np.inf
            for j in range(n_states):
                temp = f[j, k-1] + np.log(trans[j, i])
                logsum = max(temp, logsum) + np.log1p(np.exp(-abs(temp - logsum)))
            mu = hmm['emission_params'][i]['mean']
            sd = hmm['emission_params'][i]['sd']
            f[i, k] = log_norm_pdf(observation[k], mu, sd) + logsum
    return f
