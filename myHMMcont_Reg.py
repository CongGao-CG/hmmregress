import numpy as np
import copy
import statsmodels.api as sm


def myinitHMMcont_Reg(States, startCoefs, transCoefs, emissionCoefs, sds):
    startCoefs = np.asarray(startCoefs)
    if startCoefs.ndim != 2 or startCoefs.shape[0] != len(States):
        raise ValueError("startCoefs must be matrix with nStates rows")
    if not isinstance(transCoefs, dict) or len(transCoefs) != len(States):
        raise ValueError("transCoefs must be dict of nStates elements")
    for st in States:
        mat = np.asarray(transCoefs[st])
        if mat.ndim != 2 or mat.shape[0] != len(States):
            raise ValueError(f"transCoefs[{st}] must be matrix with nStates rows")
    emissionCoefs = np.asarray(emissionCoefs)
    if emissionCoefs.ndim != 2 or emissionCoefs.shape[0] != len(States):
        raise ValueError("emissionCoefs must be matrix with nStates rows")
    sds = np.asarray(sds)
    if sds.ndim != 1 or sds.size != len(States):
        raise ValueError("sds must be vector of nStates elements")
    emissionParams = {
        States[i]: {'coefs': emissionCoefs[i, :], 'sd': sds[i]}
        for i in range(len(States))
    }
    return {
        'States': States,
        'startCoefs': startCoefs,
        'transCoefs': {st: np.asarray(transCoefs[st]) for st in States},
        'emissionParams': emissionParams
    }


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


def log_sum_exp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def norm_logpdf(x, mean, sd):
    return -0.5 * (np.log(2 * np.pi * sd**2) + ((x - mean)**2) / (sd**2))


def myforwardcont_Reg(hmm, observation, Xs, Xt, Xe):
    States = hmm['States']
    nStates = len(States)
    Xs = np.asarray(Xs)
    Xt = np.asarray(Xt)
    Xe = np.asarray(Xe)
    start_logits = hmm['startCoefs'] @ Xs
    start_probs = softmax(start_logits)
    emission_coefs = np.vstack([hmm['emissionParams'][st]['coefs'] for st in States])
    emission_means = (emission_coefs @ Xe).T
    transCoefs = {st: np.asarray(hmm['transCoefs'][st]) for st in States}
    nObs = len(observation)
    trans_probs_list = []
    for k in range(nObs - 1):
        TM = np.zeros((nStates, nStates))
        for i, st in enumerate(States):
            logits = transCoefs[st] @ Xt[:, k]
            TM[i, :] = softmax(logits)
        trans_probs_list.append(TM)
    f = np.zeros((nStates, nObs))
    for i, st in enumerate(States):
        f[i, 0] = np.log(start_probs[i]) + norm_logpdf(
            observation[0],
            emission_means[0, i],
            hmm['emissionParams'][st]['sd']
        )
    for k in range(1, nObs):
        for i, st in enumerate(States):
            logsum = -np.inf
            for j, ps in enumerate(States):
                temp = f[j, k - 1] + np.log(trans_probs_list[k - 1][j, i])
                logsum = max(temp, logsum) + np.log1p(np.exp(-abs(temp - logsum)))
            f[i, k] = norm_logpdf(
                observation[k],
                emission_means[k, i],
                hmm['emissionParams'][st]['sd']
            ) + logsum
    return f


def mybackwardcont_Reg(hmm, observation, Xs, Xt, Xe):
    States = hmm['States']
    nStates = len(States)
    Xt = np.asarray(Xt)
    Xe = np.asarray(Xe)
    emission_coefs = np.vstack([hmm['emissionParams'][st]['coefs'] for st in States])
    emission_means = (emission_coefs @ Xe).T
    transCoefs = {st: np.asarray(hmm['transCoefs'][st]) for st in States}
    nObs = len(observation)
    trans_probs_list = []
    for k in range(nObs - 1):
        TM = np.zeros((nStates, nStates))
        for i, st in enumerate(States):
            logits = transCoefs[st] @ Xt[:, k]
            TM[i, :] = softmax(logits)
        trans_probs_list.append(TM)
    b = np.zeros((nStates, nObs))
    b[:, nObs - 1] = 0.0
    for k in range(nObs - 2, -1, -1):
        for i, st in enumerate(States):
            logsum = -np.inf
            for j, ns in enumerate(States):
                temp = (
                    b[j, k + 1]
                    + np.log(trans_probs_list[k][i, j])
                    + norm_logpdf(
                        observation[k + 1],
                        emission_means[k + 1, j],
                        hmm['emissionParams'][ns]['sd']
                    )
                )
                logsum = max(temp, logsum) + np.log1p(np.exp(-abs(temp - logsum)))
            b[i, k] = logsum
    return b


def myBWRcont_Reg(hmm, obs_list, Xs_list, Xt_list, Xe_list):
    States = hmm['States']
    nStates = len(States)
    nSeq = len(obs_list)
    logstart_Y = np.zeros((nSeq, nStates))
    logxi_list = {st: {ns: [] for ns in States} for st in States}
    loggamma_list = {st: [] for st in States}
    Xs_all = np.column_stack(Xs_list).T
    Xt_all = np.column_stack([Xt for Xt in Xt_list]).T
    Xe_all = np.column_stack([Xe for Xe in Xe_list]).T
    emission_Y = np.concatenate(obs_list)
    for idx, obs in enumerate(obs_list):
        Xs = Xs_list[idx]
        Xt = Xt_list[idx]
        Xe = Xe_list[idx]
        f = myforwardcont_Reg(hmm, obs, Xs, Xt, Xe)
        b = mybackwardcont_Reg(hmm, obs, Xs, Xt, Xe)
        likelihood = log_sum_exp(f[:, -1])
        logstart_Y[idx, :] = f[:, 0] + b[:, 0] - likelihood
        for i, st in enumerate(States):
            for j, ns in enumerate(States):
                for k in range(len(obs) - 1):
                    temp = (
                        f[i, k]
                        + np.log(
                            softmax(hmm['transCoefs'][st] @ Xt[:, k])[j]
                        )
                        + norm_logpdf(
                            obs[k + 1],
                            hmm['emissionParams'][ns]['coefs'] @ Xe[:, k + 1],
                            hmm['emissionParams'][ns]['sd']
                        )
                        + b[j, k + 1]
                        - likelihood
                    )
                    logxi_list[st][ns].append(temp)
        for i, st in enumerate(States):
            gamma_vals = f[i, :] + b[i, :] - likelihood
            loggamma_list[st].extend(gamma_vals.tolist())
    for idx, st in enumerate(States[1:], start=1):
        y = 1 / (1 + np.exp(logstart_Y[:, 0] - logstart_Y[:, idx]))
        model = sm.GLM(y, Xs_all, family=sm.families.Binomial()).fit()
        hmm['startCoefs'][idx, :] = model.params
    for st in States:
        for j, ns in enumerate(States[1:], start=1):
            y = 1 / (1 + np.exp(
                np.array(logxi_list[st][States[0]]) - np.array(logxi_list[st][ns])
            ))
            model = sm.GLM(y, Xt_all, family=sm.families.Binomial()).fit()
            hmm['transCoefs'][st][j, :] = model.params
    for st in States:
        weights = np.exp(np.array(loggamma_list[st]) - np.max(loggamma_list[st]))
        model = sm.WLS(emission_Y, Xe_all, weights=weights).fit()
        coefs = model.params
        resid = model.resid
        sd = np.sqrt(np.sum(weights * resid**2) / np.sum(weights))
        hmm['emissionParams'][st]['coefs'] = coefs
        hmm['emissionParams'][st]['sd'] = max(sd, 1e-4)
    return hmm


def myBWcont_Reg(hmm, obs_list, Xs_list, Xt_list, Xe_list, maxIterations=100, delta=1e-9, pseudoCount=0):
    tempHmm = copy.deepcopy(hmm)
    diff = []
    for _ in range(maxIterations):
        bw = myBWRcont_Reg(tempHmm, obs_list, Xs_list, Xt_list, Xe_list)
        TC = bw['transCoefs']
        EP = bw['emissionParams']
        SC = bw['startCoefs']
        States = hmm['States']
        d = (
            np.sqrt(np.sum(
                (np.vstack([tempHmm['transCoefs'][st] for st in States]) -
                 np.vstack([TC[st] for st in States]))**2
            )) +
            np.sqrt(np.sum(
                (np.concatenate([np.append(tempHmm['emissionParams'][st]['coefs'], tempHmm['emissionParams'][st]['sd']) for st in States]) -
                 np.concatenate([np.append(EP[st]['coefs'], EP[st]['sd']) for st in States]))**2                 
            )) +
            np.sqrt(np.sum((tempHmm['startCoefs'] - SC)**2))
        )
        diff.append(d)
        tempHmm['transCoefs'] = TC
        tempHmm['emissionParams'] = EP
        tempHmm['startCoefs'] = SC
        if d < delta:
            break
    return {'hmm': tempHmm, 'difference': diff}
