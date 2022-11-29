
import numpy as np

# sys.path.append('c:\\Users\\fwanguem\\Desktop\\PyQEA_package')

from PyQEA import QuantumEvAlgorithm
from PyQEA.utils.cost_functions import f,g, rastrigin

n_dims = 10
up = 5*np.ones(n_dims)
low = -5*np.ones(n_dims)
integrals = np.full(n_dims, False)

optimizer = QuantumEvAlgorithm(g, n_dims=n_dims, upper_bound=up,
                                     lower_bound=low, integral_id=integrals,
                                     sigma_scaler=1.00002,
                                     mu_scaler=30, elitist_level=2,
                                     restrictions=[])

results = optimizer.training(N_iterations=500_000, sample_size=5, save=False,
                             filename='q11.npz')


def test_training():

    assert float(results['cost']) <= 1e-7
