import numpy as np
from myHMM import myinitHMM
from myHMM import myBW
from myHMM import myforward
from myHMM import log_sum_exp


hmm = myinitHMM(
    states=["Rainy", "Sunny"],
    symbols=["walk", "shop", "clean"],
    start_probs=[0.6, 0.4],
    trans_probs=[
        [0.7, 0.3],
        [0.4, 0.6]
    ],
    emission_probs=[
        [0.1, 0.4, 0.5],
        [0.6, 0.3, 0.1]
    ]
)


symbol_to_index = {sym: i for i, sym in enumerate(hmm['symbols'])}
single_obs = [[symbol_to_index[s] for s in ["walk", "shop", "clean", "walk", "shop", "shop", "clean", "walk"]]]
multip_obs = [
    [symbol_to_index[s] for s in seq]
    for seq in [
        ["walk", "shop", "clean", "walk", "shop", "shop", "clean", "walk"],
        ["shop", "shop", "clean", "walk", "walk", "shop", "clean", "walk"],
        ["walk", "shop", "clean", "walk", "clean", "walk", "shop", "shop"]
    ]
]


mytrainedHMM = myBW(hmm, single_obs, maxIterations = 44)
log_sum_exp(myforward(hmm, single_obs[0])[:,7])
log_sum_exp(myforward(mytrainedHMM['hmm'], single_obs[0])[:,7])


mytrainedHMM = myBW(hmm, multip_obs, maxIterations = 172)
sum(log_sum_exp(myforward(hmm, obs)[:, 7]) for obs in multip_obs)
sum(log_sum_exp(myforward(mytrainedHMM['hmm'], obs)[:, 7]) for obs in multip_obs)
