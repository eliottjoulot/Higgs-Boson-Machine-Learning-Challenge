# -*- coding: utf-8 -*-
"""implement a least square function."""

import numpy as np
from costs import *


def least_squares(y, tx):
    """calculate the least squares."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return compute_loss(y,tx,w), w