import numpy as np
import copy
from BWRcont import BWRcont


def BWcont(hmm, obs_list, maxIterations=100, delta=1e-9, pseudoCount=0):
    tempHmm = copy.deepcopy(hmm)
    mask_trans = ~np.isnan(hmm['trans_probs'])
    mask_sv = ~np.isnan(hmm['start_probs'])
    diff = []
    for _ in range(maxIterations):
        bw = BWRcont(tempHmm, obs_list)
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
    
    order = np.argsort(-tempHmm['start_probs'])
    tempHmm['start_probs'] = tempHmm['start_probs'][order]
    tempHmm['trans_probs'] = tempHmm['trans_probs'][order][:, order]
    tempHmm['emission_params'] = [tempHmm['emission_params'][i] for i in order]
    tempHmm['states'] = [tempHmm['states'][i] for i in order]
    
    return {'hmm': tempHmm, 'difference': diff}
