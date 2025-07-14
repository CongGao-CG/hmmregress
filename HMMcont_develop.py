import numpy as np
from myHMMcont import myinitHMMcont
from myHMMcont import myBWcont
from myHMMcont import myforwardcont
from myHMMcont import log_sum_exp


hmmcont = myinitHMMcont(
    states=["Rainy", "Sunny"],
    start_probs=[0.99, 0.01],
    trans_probs=[
        [0.99, 0.01],
        [0.99, 0.01]
    ],
    emission_params=[
        {'mean': 0.2, 'sd': 0.8},
        {'mean': 1.2, 'sd': 0.07}
    ]
)


single_obscont = [
    [0.99, -1.88, -0.56, -0.72, 1.02, 0.74, 0.44, -1.02]
]
multip_obscont = [
    [0.99, -1.88, -0.56, -0.72, 1.02, 0.74, 0.44, -1.02],
    [0.39, -0.37, 0.74, -0.043, -0.28, 1.01, 0.09, 1.23],
    [0.38, 0.11, -0.52, -0.44, 2.14, -0.94, 1.44, 0.48]
]


mytrainedHMMcont = myBWcont(hmmcont, single_obscont, maxIterations = 6)
log_sum_exp(myforwardcont(hmmcont, single_obscont[0])[:,7])
log_sum_exp(myforwardcont(mytrainedHMMcont['hmm'], single_obscont[0])[:,7])


mytrainedHMMcont = myBWcont(hmmcont, multip_obscont, maxIterations = 3)
sum(log_sum_exp(myforwardcont(hmmcont, obs)[:, 7]) for obs in multip_obscont)
sum(log_sum_exp(myforwardcont(mytrainedHMMcont['hmm'], obs)[:, 7]) for obs in multip_obscont)
