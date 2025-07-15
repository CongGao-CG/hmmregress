import numpy as np
from softmax import softmax
from log_norm_pdf import log_norm_pdf


def forwardcont_Reg(hmm, observation, Xs, Xt, Xe):
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
        f[i, 0] = np.log(start_probs[i]) + log_norm_pdf(
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
            f[i, k] = log_norm_pdf(
                observation[k],
                emission_means[k, i],
                hmm['emissionParams'][st]['sd']
            ) + logsum
    return f
