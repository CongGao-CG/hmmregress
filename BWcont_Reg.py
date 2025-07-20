import numpy as np
import copy
from BWRcont_Reg import BWRcont_Reg


def BWcont_Reg(hmm, obs_list, Xs_list, Xt_list, Xe_list, maxIterations=100, delta=1e-9, pseudoCount=0):
    tempHmm = copy.deepcopy(hmm)
    diff = []
    for _ in range(maxIterations):
        working_hmm = copy.deepcopy(tempHmm)
        bw = BWRcont_Reg(working_hmm, obs_list, Xs_list, Xt_list, Xe_list)
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
        print(d)
        tempHmm['transCoefs'] = TC
        tempHmm['emissionParams'] = EP
        tempHmm['startCoefs'] = SC
        if d < delta:
            break
    return {'hmm': tempHmm, 'difference': diff}
