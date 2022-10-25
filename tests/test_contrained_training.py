import numpy as np
import sys
# sys.path.append('c:\\Users\\fwanguem\\Desktop\\PyQEA_package')

from PyQEA import QuantumEvAlgorithm
from PyQEA.utils.cost_functions import f

def h(x: np.ndarray):
    """Definition of the restriction to be applied to the opt problem"""

    if x.ndim == 1:
        x = x[None]

    n_dims = x.shape[1]

    return  -x[:,1 ] - x[:,0] + 6 + sys.float_info.epsilon

n_dims = 100
up = 5*np.ones(n_dims)
low = -5*np.ones(n_dims)
integrals = np.full(n_dims, False)
integrals[0:2] = True

optimizer = QuantumEvAlgorithm(f, n_dims=n_dims, upper_bound=up,
                                     lower_bound=low, integral_id=integrals,
                                     sigma_scaler=1.002,
                                     mu_scaler=20, elitist_level=6,
                                     restrictions=[h])

results = optimizer.training(N_iterations=10000, sample_size=20, save=False,
                             filename='q11.npz')
print(results['min'])
def test_integral_training():

    assert float(results['cost']) <= 2