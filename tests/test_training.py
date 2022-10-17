
import numpy as np

# sys.path.append('c:\\Users\\fwanguem\\Desktop\\PyQEA_package')

from PyQEA import QuantumEvAlgorithm
from PyQEA.utils.cost_functions import f,g, rastrigin

n_dims = 2
up = 5*np.ones(n_dims)
low = -5*np.ones(n_dims)
integrals = np.full(n_dims, False)

optimizer = QuantumEvAlgorithm(g, n_dims=n_dims, upper_bound=up,
                                     lower_bound=low, integral_id=integrals,
                                     sigma_scaler=1.03,
                                     mu_scaler=10, elitist_level=10,
                                     restrictions=[])

results = optimizer.training(N_iterations=1500, sample_size=20, save=False,
                             filename='q11.npz')


def test_training():

    assert float(results['cost']) <= 1e-7
