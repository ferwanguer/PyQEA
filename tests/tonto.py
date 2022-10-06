
import numpy as np 
import os
import sys

sys.path.append('c:\\Users\\fwanguem\\Desktop\\PyQEA\\src')

from PyQEA import qea, cost_functions





n_dims = 10
up = 5*np.ones(n_dims)
low = -5*np.ones(n_dims)
integrals = np.full(n_dims,False)

# np.array([True,True,False, False,False,False,True])

optimizer = qea.QuantumEvAlgorithm(cost_functions.f, n_dims = n_dims,upper_bound= up, lower_bound= low, integral_id=integrals, sigma_scaler = 1.003,
                                   mu_scaler = 20, elitist_level = 6, restrictions=[])

results = optimizer.training(N_iterations=5000, sample_size= 20, save= False ,filename='q11.npz')

print(results["min"])