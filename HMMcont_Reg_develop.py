import numpy as np
from myHMMcont_Reg import myinitHMMcont_Reg
from myHMMcont_Reg import myBWcont_Reg
from myHMMcont_Reg import myforwardcont_Reg
from myHMMcont_Reg import log_sum_exp


hmmcont_reg = myinitHMMcont_Reg(
    States=["A", "B", "C"],
    startCoefs=np.array([[0.0], [0.2], [-0.3]]),
    transCoefs={
        "A": np.array([[0.0, 0.0],
                       [0.1, -0.1],
                       [-0.2, 0.3]]),
        "B": np.array([[0.0, 0.0],
                       [0.3, 0.2],
                       [-0.1, 0.1]]),
        "C": np.array([[0.0, 0.0],
                       [-0.2, 0.2],
                       [0.5, -0.3]])
    },
    emissionCoefs=np.array([[1.0, 2.0],
                            [3.0, 4.0],
                            [5.0, 6.0]]),
    sds=[0.5, 1.0, 0.8]
)


single_obscont = [
    [0.99, -1.88, -0.56, -0.72, 1.02, 0.74, 0.44, -1.02]
]
multip_obscont = [
    [0.99, -1.88, -0.56, -0.72, 1.02, 0.74, 0.44, -1.02],
    [0.39, -0.37, 0.74, -0.043, -0.28, 1.01, 0.09, 1.23],
    [0.38, 0.11, -0.52, -0.44, 2.14, -0.94, 1.44, 0.48]
]


Xs_list = [
    np.array([[1]])
]
Xt_list = [
    np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [3, 4, 5, 6, 7, 1, 2]
    ])
]
Xe_list = [
    np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],
        [3, 4, 5, 6, 9, 7, 1, 2]
    ])
]


mytrainedHMMcont_Reg = myBWcont_Reg(
    hmmcont_reg,
    single_obscont,
    Xs_list=Xs_list,
    Xt_list=Xt_list,
    Xe_list=Xe_list,
    maxIterations=2
)
sum(
    log_sum_exp( 
        myforwardcont_Reg(hmmcont_reg, obs, Xs, Xt, Xe)[:, 7]
    )
    for obs, Xs, Xt, Xe in zip(single_obscont, Xs_list, Xt_list, Xe_list)
)
sum(
    log_sum_exp( 
        myforwardcont_Reg(mytrainedHMMcont_Reg['hmm'], obs, Xs, Xt, Xe)[:, 7]
    )
    for obs, Xs, Xt, Xe in zip(single_obscont, Xs_list, Xt_list, Xe_list)
)


mytrainedHMMcont_Reg = myBWcont_Reg(
    hmmcont_reg,
    multip_obscont,
    Xs_list=Xs_list * 3,
    Xt_list=Xt_list * 3,
    Xe_list=Xe_list * 3,
    maxIterations=2
)
sum(
    log_sum_exp(
        myforwardcont_Reg(hmmcont_reg, obs, Xs, Xt, Xe)[:, 7]
    )
    for obs, Xs, Xt, Xe in zip(multip_obscont, Xs_list * 3, Xt_list * 3, Xe_list * 3)
)
sum(
    log_sum_exp(
        myforwardcont_Reg(mytrainedHMMcont_Reg['hmm'], obs, Xs, Xt, Xe)[:, 7]
    )
    for obs, Xs, Xt, Xe in zip(multip_obscont, Xs_list * 3, Xt_list * 3, Xe_list * 3)
)
