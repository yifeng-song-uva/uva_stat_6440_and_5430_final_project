import numpy as np

def normalize_vector(arr):
    arr_sum = np.sum(arr)
    return arr / arr_sum

def normalize_by_row(arr):
    return (arr.T / arr.sum(axis=1)).T

def exp_normalize(arr):
    # exponentiate, and then normalize to 1 (using the idea of log-sum-exp)
    M = np.max(arr)
    arr = arr - M
    exp_arr = np.exp(arr)
    return exp_arr / np.sum(exp_arr)

def linear_time_natural_gradient(g, h, z):
    # Gradient g, Hessian H = diag(h) + 1 * z * 1^T (both already scaled by batch size if needed)
    # based on the special structure of Hessian matrix w.r.t alpha or xi, using Woodbury matrix identity
    c = np.sum(g/h) / (1/z + np.sum(1/h))
    return (g-c)/h

def stochastic_variational_update(old_value, value_hat, rho):
    new_value = (1 - rho) * old_value + rho * value_hat
    return new_value

def stochastic_hyperparameter_update(old_value, value_hat, rho):
    new_value = old_value - rho * value_hat
    return new_value