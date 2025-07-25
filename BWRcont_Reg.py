import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from forwardcont_Reg import forwardcont_Reg
from backwardcont_Reg import backwardcont_Reg
from softmax import softmax
from log_norm_pdf import log_norm_pdf
from log_sum_exp import log_sum_exp


def BWRcont_Reg(hmm, obs_list, Xs_list, Xt_list, Xe_list):
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
        f = forwardcont_Reg(hmm, obs, Xs, Xt, Xe)
        b = backwardcont_Reg(hmm, obs, Xs, Xt, Xe)
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
                        + log_norm_pdf(
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
    y = np.repeat(np.arange(nStates), Xs_all.shape[0])
    w = np.exp(logstart_Y.flatten('F') - logstart_Y.flatten('F').max())
    model = LogisticRegression(solver='lbfgs', max_iter=1000, fit_intercept=False, penalty=None).fit(np.tile(Xs_all, [nStates, 1]), y, sample_weight=w)
    coef = model.coef_
    if nStates > 2:
        coef = coef - coef[0]
        coef = coef[1:]
    hmm['startCoefs'][1:] = coef
    for st in States:
         y = np.repeat(np.arange(nStates), Xt_all.shape[0])
         w = np.exp(np.concatenate(list(logxi_list[st].values())) - np.concatenate(list(logxi_list[st].values())).max())
         model = LogisticRegression(solver='lbfgs', max_iter=1000, fit_intercept=False, penalty=None).fit(np.tile(Xt_all, [nStates, 1]), y, sample_weight=w)
         coef = model.coef_
         if nStates > 2:
             coef = coef - coef[0]
             coef = coef[1:]
         hmm['transCoefs'][st][1:] = coef
    for st in States:
        weights = np.exp(np.array(loggamma_list[st]) - np.max(loggamma_list[st]))
        model = sm.WLS(emission_Y, Xe_all, weights=weights).fit()
        coefs = model.params
        resid = model.resid
        sd = np.sqrt(np.sum(weights * resid**2) / np.sum(weights))
        hmm['emissionParams'][st]['coefs'] = coefs
        hmm['emissionParams'][st]['sd'] = max(sd, 1e-4)
    return hmm
