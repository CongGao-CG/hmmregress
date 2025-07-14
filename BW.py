import numpy as np
import copy
from BWR import BWR


def BW(hmm, obs_list, maxIterations=100, delta=1e-9, pseudoCount=0):
    tempHmm = copy.deepcopy(hmm)
    mask_trans = ~np.isnan(hmm['trans_probs'])
    mask_emis = ~np.isnan(hmm['emission_probs'])
    mask_sv = ~np.isnan(hmm['start_probs'])
    diff = []
    for _ in range(maxIterations):
        bw = BWR(tempHmm, obs_list)
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
    
    order = np.argsort(-tempHmm['start_probs'])
    tempHmm['start_probs'] = tempHmm['start_probs'][order]
    tempHmm['trans_probs'] = tempHmm['trans_probs'][order][:, order]
    tempHmm['emission_probs'] = tempHmm['emission_probs'][order]
    tempHmm['states'] = [tempHmm['states'][i] for i in order]
    
    return {'hmm': tempHmm, 'difference': diff}
