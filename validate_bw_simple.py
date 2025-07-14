import numpy as np
from initHMM import initHMM
from BW import BW

STATES  = ['S0', 'S1']
SYMBOLS = ['A', 'B']

TRUE_START = np.array([0.6, 0.4])
TRUE_TRANS = np.array([[0.7, 0.3],
                       [0.2, 0.8]])
TRUE_EMIS  = np.array([[0.5, 0.5],
                       [0.1, 0.9]])

TRUE_HMM = initHMM(
    STATES, SYMBOLS,
    start_probs=TRUE_START,
    trans_probs=TRUE_TRANS,
    emission_probs=TRUE_EMIS,
)


def generate_sequences(hmm, n_sequences=400, seq_length=200):
    sequences = []
    rng = np.random.default_rng(0)
    for _ in range(n_sequences):
        seq = []
        # Sample initial state
        state = rng.choice(len(hmm['states']), p=hmm['start_probs'])
        symbol = rng.choice(len(hmm['symbols']),
                                p=hmm['emission_probs'][state])
        seq.append(symbol)
        for _ in range(seq_length - 1):
            state = rng.choice(len(hmm['states']),
                                   p=hmm['trans_probs'][state])
            symbol = rng.choice(len(hmm['symbols']), 
                                    p=hmm['emission_probs'][state])
            seq.append(symbol)
        sequences.append(seq)
    return sequences

synthetic_data = generate_sequences(TRUE_HMM)
print(synthetic_data[0][0:5])
rng = np.random.default_rng(123 + 2)
init_hmm = initHMM(
    states=['A', 'B'],
    symbols=[0, 1],
    start_probs=list(rng.dirichlet(np.ones(2))),
    trans_probs=[list(rng.dirichlet(np.ones(2))),
                 list(rng.dirichlet(np.ones(2)))],
    emission_probs=[list(rng.dirichlet(np.ones(2))),
                    list(rng.dirichlet(np.ones(2)))]
)
print(init_hmm)
result = BW(init_hmm, synthetic_data, maxIterations=100)
# print(result['difference'])
learned_hmm = result['hmm']

print('\nTrue vs Learned Start Probs:')
print(TRUE_HMM['start_probs'])
print(learned_hmm['start_probs'])
print('\nTrue vs Learned Transition Probs:')
print(TRUE_HMM['trans_probs'])
print(learned_hmm['trans_probs'])
print('\nTrue vs Learned Emission Probs:')
print(TRUE_HMM['emission_probs'])
print(learned_hmm['emission_probs'])
