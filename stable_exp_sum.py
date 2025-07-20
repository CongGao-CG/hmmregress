import numpy as np


def stable_exp_sum(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    m = np.maximum(x, y)
    return np.exp(m) * (np.exp(x - m) + np.exp(y - m))
