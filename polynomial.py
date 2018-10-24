# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree, crossterm=True):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))

    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]



    # Add cross term column product CT
    if(crossterm):
        temp_col = x[:,0]
        for j in range(1,x.shape[1]):
            temp_col = np.multiply(temp_col,x[:,j])

        poly = np.c_[poly, np.transpose(temp_col)]


        poly = np.c_[poly, np.transpose(np.power(temp_col,2))]
        #poly = np.c_[poly, np.transpose(np.power(temp_col,0.5))]



    return poly

    #for j in range(x.shape[1]):
#        temp_col = numpy.transpose([x[:,j]])
        #temp_matrix = x[:,j+1:-1]
        #np.hstack(poly, np.multiply())
    #return poly
