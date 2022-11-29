
import numpy as np
import os
# sys.path.append('c:\\Users\\fwanguem\\Desktop\\PyQEA_package')

from PyQEA import QuantumEvAlgorithm
from PyQEA.utils.cost_functions import f

n_dims = 10
up = 5*np.ones(n_dims)
low = -5*np.ones(n_dims)
integrals = np.full(n_dims, False)
integrals[0:3] = True

optimizer = QuantumEvAlgorithm(f, n_dims=n_dims, upper_bound=up,
                                     lower_bound=low, integral_id=integrals,
                                     sigma_scaler=1.0051,
                                     mu_scaler=10, elitist_level=2,
                                     restrictions=[])

results = optimizer.training(N_iterations=30000, sample_size=20, save=False)

print(results['min'])
def test_integral_training():

    assert float(results['cost']) <= 1