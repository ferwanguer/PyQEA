import numpy as np

"""This script defines the functions that are to be optimized"""


# x = np.zeros(4)

def f(x: np.ndarray):
    """n-dimensional paraboloid definition. For the first test of the
       optimization algorithm."""
    if x.ndim == 1:
        x = x[None]

    n_dims = x.shape[1]

    min_value = 3.8 * np.ones((1, n_dims))
    f = np.sum(np.square(x - min_value), axis=1)

    return f


def mse(x: np.ndarray):
    if x.ndim == 1:
        x = x[None]

    n_dims = x.shape[1]
    return np.sqrt(np.divide(f(x), n_dims))


def g(x: np.ndarray):
    """n-dimensional Ackley function definition. For the second test of the
       optimization algorithm. """
    if x.ndim == 1:
        x = x[None]

    n_dims = x.shape[1]

    a = 20.0
    b = 0.2
    c = 2 * np.pi

    auxiliar = -b * np.sqrt(np.divide(np.sum(np.square(x), axis=1), n_dims))
    auxiliar_2 = np.sum(np.cos(c * x), axis=1) / n_dims
    g = -a * np.exp(auxiliar) - np.exp(auxiliar_2) + a + np.exp(1.0)

    return g


def rastrigin(x: np.ndarray):
    """n-dimensional rastrigin function, testing purposes"""
    if x.ndim == 1:
        x = x[None]

    n_dims = x.shape[1]

    a = 10
    x_squared = np.square(x)
    x_sin = a * np.cos(2 * np.pi * x)
    rastrigin = a * n_dims + np.sum(x_squared - x_sin, axis=1)

    return rastrigin


def rosenbrock(x: np.ndarray):
    if x.ndim == 1:
        x = x[None]

    x_plus = x[:, 1:]
    x_i = x[:, 0:-1]
    aux_1 = 100 * np.square(x_plus - np.square(x_i))
    aux_2 = np.square(x_i - 1)
    rosenbrock = np.sum(aux_1 + aux_2, axis=1)
    return rosenbrock


def griewank(x: np.ndarray):
    if x.ndim == 1:
        x = x[None]

    n_dims = x.shape[1]
    dims = np.arange(1, n_dims + 1)
    x_squared = np.square(x)
    aux = np.cos(np.divide(x, np.sqrt(dims)))
    griewank = np.sum(x_squared, axis=1) / 4000 - np.prod(aux, axis=1) + 1

    return griewank


def michael(x: np.ndarray):
    if x.ndim == 1:
        x = x[None]

    m = 10
    n_dims = x.shape[1]
    x_squared = np.square(x)
    dims = np.arange(1, n_dims + 1)
    aux = np.sin(dims * x_squared / np.pi) ** (2 * m)
    return -np.sum(np.sin(x) * aux, axis=1)


def schwefel(x: np.ndarray):
    if x.ndim == 1:
        x = x[None]
    n_dims = x.shape[1]
    a = 418.9829 * n_dims

    return a - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)


def dropwave(x: np.ndarray):
    if x.ndim == 1:
        x = x[None]
    num = -1 - np.cos(12 * np.sqrt(np.sum(np.square(x), axis=1)))
    den = 2 + 0.5 * (np.sum(np.square(x), axis=1))
    return (num / den)


def schaffer_2(x: np.ndarray):
    if x.ndim == 1:
        x = x[None]
    x_1 = x[0, 0]
    x_2 = x[0, 1]

    num = np.sin(x_1 ** 2 - x_2 ** 2) ** 2 - 0.5
    den = (1 + 0.001 * (x_1 ** 2 + x_2 ** 2)) ** 2
    return 0.5 + num / den


A = np.random.rand(20, 20)
b = np.random.rand(20, 1)


def equation(x):
    if x.ndim == 1:
        x = x[None]

    return np.sum(np.square((np.matmul(A, x.T) - b)))
