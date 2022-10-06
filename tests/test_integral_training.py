
import numpy as np

# sys.path.append('c:\\Users\\fwanguem\\Desktop\\PyQEA_package')

import PyQEA

n_dims = 10
up = 5*np.ones(n_dims)
low = -5*np.ones(n_dims)
integrals = np.full(n_dims, False)
integrals[0:3] = True

optimizer = PyQEA.QuantumEvAlgorithm(PyQEA.f, n_dims=n_dims, upper_bound=up,
                                     lower_bound=low, integral_id=integrals,
                                     sigma_scaler=1.003,
                                     mu_scaler=20, elitist_level=6,
                                     restrictions=[])

results = optimizer.training(N_iterations=4000, sample_size=20, save=False,
                             filename='q11.npz')


def test_integral_training():

    assert float(results['cost']) <= 1