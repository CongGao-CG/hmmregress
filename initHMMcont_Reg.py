import numpy as np


def initHMMcont_Reg(States, startCoefs, transCoefs, emissionCoefs, sds):
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
