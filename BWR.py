import numpy as np
from forward import forward
from backward import backward
from log_sum_exp import log_sum_exp


def BWR(hmm, obs_list):
    n_seq = len(obs_list)
    n_states = len(hmm['states'])
    n_symbols = len(hmm['symbols'])
    logstart_Y = np.empty((n_seq, n_states))
    TransitionMatrix = np.zeros((n_states, n_states))
    logxi_list = [[[] for _ in range(n_states)] for _ in range(n_states)]
    loggamma_list = [[] for _ in range(n_states)]
    EmissionMatrix = np.zeros((n_states, n_symbols))
    for idx, observation in enumerate(obs_list):
        n_obs = len(observation)
        f = forward(hmm, observation)
        b = backward(hmm, observation)
        likelihood = log_sum_exp(f[:, n_obs-1])
        for i in range(n_states):
            for j in range(n_states):
                for t in range(n_obs-1):
                    logxi_list[i][j].append(
                        f[i, t]
                        + np.log(hmm['trans_probs'][i, j])
                        + np.log(hmm['emission_probs'][j, observation[t+1]])
                        + b[j, t+1]
                        - likelihood
                    )
        for i in range(n_states):
            gamma_vals = f[i, :] + b[i, :] - likelihood
            loggamma_list[i].extend(gamma_vals.tolist())
        for i in range(n_states):
            logstart_Y[idx, i] = f[i, 0] + b[i, 0] - likelihood
    for i in range(n_states):
        denom = log_sum_exp(np.array([xi for row in logxi_list[i] for xi in row]))
        for j in range(n_states):
            num = log_sum_exp(np.array(logxi_list[i][j]))
            TransitionMatrix[i, j] = np.exp(num - denom)
    obs_flat = [sym for seq in obs_list for sym in seq]
    for i in range(n_states):
        gamma_flat = loggamma_list[i]
        denom = log_sum_exp(np.array(gamma_flat))
        for s_idx in range(n_symbols):
            positions = [k for k, sym in enumerate(obs_flat) if sym == s_idx]
            if positions:
                vals = [gamma_flat[k] for k in positions]
                EmissionMatrix[i, s_idx] = np.exp(log_sum_exp(np.array(vals)) - denom)
    startVec = np.exp([log_sum_exp(logstart_Y[:, i]) for i in range(n_states)])
    return {'TransitionMatrix': TransitionMatrix,
            'EmissionMatrix': EmissionMatrix,
            'startVec': startVec}