import numpy as np
import copy


def myinitHMM(states, symbols, start_probs=None, trans_probs=None, emission_probs=None):
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


def myforward(hmm, observation):
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


def mybackward(hmm, observation):
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



def log_sum_exp(x):
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))


def myBWR(hmm, obs_list):
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
        f = myforward(hmm, observation)
        b = mybackward(hmm, observation)
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


def myBW(hmm, obs_list, maxIterations=100, delta=1e-9, pseudoCount=0):
    tempHmm = copy.deepcopy(hmm)
    mask_trans = ~np.isnan(hmm['trans_probs'])
    mask_emis = ~np.isnan(hmm['emission_probs'])
    mask_sv = ~np.isnan(hmm['start_probs'])
    diff = []
    for _ in range(maxIterations):
        bw = myBWR(tempHmm, obs_list)
        TM = bw['TransitionMatrix']
        EM = bw['EmissionMatrix']
        SV = bw['startVec']
        TM[mask_trans] += pseudoCount
        EM[mask_emis] += pseudoCount
        SV[mask_sv] += pseudoCount
        TM = TM / TM.sum(axis=1)[:, None]
        EM = EM / EM.sum(axis=1)[:, None]
        SV = SV / SV.sum()
        d = (np.sqrt(np.sum((tempHmm['trans_probs'] - TM)**2)) +
             np.sqrt(np.sum((tempHmm['emission_probs'] - EM)**2)) +
             np.sqrt(np.sum((tempHmm['start_probs'] - SV)**2)))
        diff.append(d)
        tempHmm['trans_probs'] = TM.copy()
        tempHmm['emission_probs'] = EM.copy()
        tempHmm['start_probs'] = SV.copy()
        if d < delta:
            break
    tempHmm['trans_probs'][~mask_trans] = np.nan
    tempHmm['emission_probs'][~mask_emis] = np.nan
    tempHmm['start_probs'][~mask_sv] = np.nan
    return {'hmm': tempHmm, 'difference': diff}
