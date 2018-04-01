#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@brief

@namespace ...
@authors Alessandro Luongo <alessandro.luongo@atos.net>
@copyright 2018  Bull S.A.S.  -  All rights reserved.
           This code is released under LGPL license.
           Please contact Bull SAS for details about its license.
           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois

Description
This library is meant to find the parameters of a dataset that will dictate the running 
time of a QRAM query in a quantum computer. 


"""

import itertools
import logging

import numpy as np
from scipy import linalg

import numpy as np
from scipy.optimize import minimize

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics.pairwise import paired_distances


class QramUtils():
    def __init__(self, dataset, logging_handler=None):
        """
        This library assume that the sample of the trainig set
        are stored along the rows of the matrix.

        :param dataset: a numpy array
        :return:
        """
        self.dataset = dataset
        self.logging= logging_handler
        pass

    def find_qubits(self):
        """
        Calculates how many qubits we need to index this matrix in a quantum register
        :return:
        """
        index_register = np.ceil(np.log2(self.dataset.shape[0]))
        content_register = np.ceil(np.log2(self.dataset.shape[1]))
        qubits=index_register+content_register        
        return qubits

    def sparsity(self):
        """
        Return the ration (0,1) of nonzero elements in the matrix
        :return:
        """
        sparsity = np.count_nonzero(self.dataset) / np.prod(self.dataset.shape)
        return 1. - sparsity

    def frobenius(self):
        """
        Calculate the frobenius norm of the matrix with eigenvalues scaled between 0 and 1

        :param self:
        :return:
        """

        eigval = linalg.svd(self.dataset, compute_uv=False)

        eigval = np.square(eigval)

        normalized_eigenvalues = np.true_divide(eigval, max(eigval))

        frob_norm = np.sqrt(sum(normalized_eigenvalues))
        return frob_norm

    def __mu(self, p, args):
        """
        The function to minimize

        :param x: the matrix representing the dataset
        :param p: the parameter [0,1]
        :return:
        """

        # logging.info("Executing mu with p {}".format(p))
        # input("run mu")

        def s(p, X):

            norms = np.power(np.linalg.norm(X, p, axis=1), np.full(X.shape[0], p))
            vector_of_powers = np.full(len(norms), p)
            # print(vector_of_powers)
            exp_norms = np.power(norms, vector_of_powers)
            max_norms = max(exp_norms)

            # logging.debug("Norms {}".format(norms))
            # logging.debug("Exp_norms {}".format(exp_norms))
            # logging.debug("max_norms {}".format(max_norms))

            return max_norms

        s1 = s(2 * p, args['X'])
        s2 = s(2 * (1 - p), args['X'].T)
        mu = np.sqrt(s1 * s2)
        self.logging.debug("mu = sqrt( s1(), s2()) = {}".format(mu))

        return mu

    def __hack(self):
        """

        """
        domain = [i for i in np.arange(0.0001, 1.0, 0.05)]
        self.logging.debug("domain of mu: {}".format(domain))
        values = [self.__mu(i, {'X': self.dataset}) for i in domain]
        self.logging.debug("calculated values of p are: {}".format(values))
        best_p = domain[values.index(min(values))]
        self.logging.info('best p {}'.format(best_p))
        return best_p, min(values)

    def find_p(self):
        """
        The normalization  factors are µ_p(A) = sqrt( s_(2p)(A)* s(2(1−p))(A^t) ), where
        we denote  by  s_p(A):= max_(i \in [m]) ||a_i||^p_p  the maximum  L_p norm of  the  row vectors,
        and by  s_p(A^t) the maximum  l_p norm of  the  column  vectors.
        """
        return self.__hack()

        cons = ({'type': 'ineq', 'fun': lambda p: p + 1},
                {'type': 'ineq', 'fun': lambda p: 1 - p})

        res = minimize(self.__mu, 0.5, method='SLSQP', args={'X': self.dataset}, options={'ftol': 1e-4, 'disp': True},
                       bounds=((0.0000000001, 1),))
        # constraints=cons)

        return res.x, res.fun


if "__main__" == __name__:
    pass
