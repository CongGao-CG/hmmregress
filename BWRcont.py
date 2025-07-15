import numpy as np
from forwardcont import forwardcont
from backwardcont import backwardcont
from log_norm_pdf import log_norm_pdf
from log_sum_exp import log_sum_exp


def BWRcont(hmm, obs_list):
    n_seq = len(obs_list)
    n_states = len(hmm['states'])
    logstart_Y = np.empty((n_seq, n_states))
    TM = np.zeros((n_states, n_states))
    logxi_list = [[[] for _ in range(n_states)] for _ in range(n_states)]
    loggamma_list = [[] for _ in range(n_states)]
    for idx, obs in enumerate(obs_list):
        n_obs = len(obs)
        f = forwardcont(hmm, obs)
        b = backwardcont(hmm, obs)
        likelihood = log_sum_exp(f[:, n_obs-1])
        for i in range(n_states):
            for j in range(n_states):
                for t in range(n_obs-1):
                    logxi_list[i][j].append(
                        f[i, t] + np.log(hmm['trans_probs'][i, j]) +
                        log_norm_pdf(obs[t+1], hmm['emission_params'][j]['mean'], hmm['emission_params'][j]['sd']) +
                        b[j, t+1] - likelihood
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
            TM[i, j] = np.exp(num - denom)
    flat_obs = [x for seq in obs_list for x in seq]
    EM_params = []
    for i in range(n_states):
        vals = loggamma_list[i]
        weights = np.exp(np.array(vals) - max(vals))
        mean = np.dot(weights, flat_obs) / weights.sum()
        var = np.dot(weights, np.square(flat_obs)) / weights.sum() - mean**2
        sd = np.sqrt(var) if var > 1e-8 else 1e-4
        EM_params.append({'mean': mean, 'sd': sd})
    start_vec = np.exp([log_sum_exp(logstart_Y[:, i]) for i in range(n_states)])
    return {'TransitionMatrix': TM, 'EmissionParams': EM_params, 'startVec': start_vec}
