import numpy as np


def log_norm_pdf(x, mu, sd):
    return -0.5 * (np.log(2 * np.pi * sd**2) + ((x - mu)**2) / (sd**2))
