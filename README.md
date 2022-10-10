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


