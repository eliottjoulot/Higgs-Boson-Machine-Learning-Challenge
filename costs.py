""" Function used to compute the loss. """

import numpy as np


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))


def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    return calculate_mse(e)


def compute_gradient(y, tx, w):
    """Compute the gradient using MSE"""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err