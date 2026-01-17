import numpy as np

def generate_traffic(n_ues, lam):
    return np.random.poisson(lam, size=n_ues)