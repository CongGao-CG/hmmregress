import numpy as np
import copy


def myinitHMMcont(states, start_probs=None, trans_probs=None, emission_params=None):
    n_states = len(states)
    sv = np.full(n_states, 1.0 / n_states)
    tm = 0.5 * np.eye(n_states) + (0.5 / n_states) * np.ones((n_states, n_states))
    if emission_params is None:
        emission_params = [{'mean': 0.0, 'sd': 1.0} for _ in states]
    if start_probs is not None:
        sv = np.array(start_probs, dtype=float)
    if trans_probs is not None:
        tm = np.array(trans_probs, dtype=float)
    return {'states': states, 'start_probs': sv, 'trans_probs': tm, 'emission_params': emission_params}


def log_sum_exp(x):
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))


def log_norm_pdf(x, mu, sd):
    return -0.5 * (np.log(2 * np.pi * sd**2) + ((x - mu)**2) / (sd**2))


def myforwardcont(hmm, observation):
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


def mybackwardcont(hmm, observation):
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


def myBWRcont(hmm, obs_list):
    n_seq = len(obs_list)
    n_states = len(hmm['states'])
    logstart_Y = np.empty((n_seq, n_states))
    TM = np.zeros((n_states, n_states))
    logxi_list = [[[] for _ in range(n_states)] for _ in range(n_states)]
    loggamma_list = [[] for _ in range(n_states)]
    for idx, obs in enumerate(obs_list):
        n_obs = len(obs)
        f = myforwardcont(hmm, obs)
        b = mybackwardcont(hmm, obs)
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


def myBWcont(hmm, obs_list, maxIterations=100, delta=1e-9, pseudoCount=0):
    tempHmm = copy.deepcopy(hmm)
    mask_trans = ~np.isnan(hmm['trans_probs'])
    mask_sv = ~np.isnan(hmm['start_probs'])
    diff = []
    for _ in range(maxIterations):
        bw = myBWRcont(tempHmm, obs_list)
        TM = bw['TransitionMatrix']
        EM = bw['EmissionParams']
        SV = bw['startVec']
        TM[mask_trans] += pseudoCount
        SV[mask_sv] += pseudoCount
        TM = TM / TM.sum(axis=1)[:, None]
        SV = SV / SV.sum()
        d = np.sqrt(np.sum((tempHmm['trans_probs'] - TM)**2)) + \
            np.sqrt(np.sum((np.array([p['mean'] for p in tempHmm['emission_params']]) - np.array([p['mean'] for p in EM]))**2 +
                           (np.array([p['sd'] for p in tempHmm['emission_params']]) - np.array([p['sd'] for p in EM]))**2)) + \
            np.sqrt(np.sum((tempHmm['start_probs'] - SV)**2))
        diff.append(d)
        tempHmm['trans_probs'] = TM.copy()
        tempHmm['emission_params'] = EM
        tempHmm['start_probs'] = SV.copy()
        if d < delta:
            break
    tempHmm['trans_probs'][~mask_trans] = np.nan
    tempHmm['start_probs'][~mask_sv] = np.nan
    return {'hmm': tempHmm, 'difference': diff}
