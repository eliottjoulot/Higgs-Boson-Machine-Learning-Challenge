# -*- coding: utf-8 -*-
"""
Created on Sat Oct  6 17:26:51 2018

@author: Kevin
"""
import numpy as np

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_gradient(y, tx, w):
    """Compute the gradient using MSE"""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss

    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(y,tx,w)
        # gradient w by descent update
        w = w - gamma * grad

        print("SGD Stochastic Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))

    # Will return the last values of loss and w
    return loss, w




def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""

    # implement stochastic gradient descent.


    w = initial_w
    for n_iter in range(max_iters):

        # compute gradient and loss
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad = compute_stoch_gradient(minibatch_y,minibatch_tx,w)
            w = w - gamma*grad
            # loss not on batch but on all
            #loss = compute_loss(minibatch_y,minibatch_tx,w)
            loss = compute_loss(y, tx, w)

        # store w and loss
        print("SGD Stochastic Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))


    # Return last loss and w
    return loss, w

def least_squares(y, tx):
    """calculate the least squares."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return compute_mse(y,tx,w), w

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return compute_mse(y,tx,w), w
