import numpy as np
from softmax import softmax
from log_norm_pdf import log_norm_pdf


def backwardcont_Reg(hmm, observation, Xs, Xt, Xe):
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
                    + log_norm_pdf(
                        observation[k + 1],
                        emission_means[k + 1, j],
                        hmm['emissionParams'][ns]['sd']
                    )
                )
                logsum = max(temp, logsum) + np.log1p(np.exp(-abs(temp - logsum)))
            b[i, k] = logsum
    return b
