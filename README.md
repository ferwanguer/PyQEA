# PyQEA
> Research toolkit for Quantum Inspired optimization in python


PyQEA is an extensive research library for Quantum inspired hyper-normal based
optimization in python. 

It is intended for the solution of global optimization problems where conventional 
genetic algorithms or PSO yield sub-optimal results. The current implementation of 
the algorithm allows for a fast deployment of any optimization problem, regardless of the non-linearity of its 
constraints or the complexity of the cost function. The library has the following features:

## Features
* High level module for Quantum Inspired optimization
* Built-in set of objective cost functions to test the optimization algorithm
* Capacity to implement non-linear restrictions 
* Capcity to implement integral-only variables

To install PyQEA, run this command in your terminal (Currently in test PyPI):

```shell
$ pip install -i https://test.pypi.org/simple/ PyQEA
```

To install the development version as it is, clone the development branch of the repo and then run:

```shell
$ cd PyQEA
$ python setup.py install
```

or: 

```shell
$ cd PyQEA
$ pip install .
```
### Basic Usage: 
PyQEA provides a high level implementation of  the proposed Quantum Inspired algorithm that allows a fast implementation and usage.
It aims to be user-friendly despite the non-trivial nature of its hyper-parameters. We now show the optimization process of a paraboloid (Sphere function)
of input dimension `n` centered in the vector: `[3.8, 3.8, 3.8, 3.8, ...]`. 

### Use case example: 

The optimizer setup is as follows:
```python
import numpy as np

import PyQEA

n_dims = 10 # Input dimensions of f(x)
up = 5.12 *np.ones(n_dims) # Upper bound defined for the input variables
low = -5*np.ones(n_dims)  # Lower bound defined for the input variables

integrals = np.full(n_dims, False) #Boolean vector defining which variables are integral

cost_function = PyQEA.f

optimizer = PyQEA.QuantumEvAlgorithm(cost_function, n_dims=n_dims, upper_bound=up,
                                     lower_bound=low, integral_id=integrals,
                                     sigma_scaler=1.003,
                                     mu_scaler=20, elitist_level=6,
                                     restrictions=[])

results = optimizer.training(N_iterations=4000, sample_size=20)
```
