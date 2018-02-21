#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@brief 

@namespace ...
@authors Alessandro Luongo <alessandro.luongo@atos.net>
@copyright 2018  Bull S.A.S.  -  All rights reserved.
           This is not Free or Open Source software.
           Please contact Bull SAS for details about its license.
           Bull - Rue Jean Jaurès - B.P. 68 - 78340 Les Clayes-sous-Bois


Description 
This code simply plot the frobenius norm of various instances of the MNIST dataset.
For k increasing PCA dimension, the data is first projected in the k subspace, and then
normalized and polynomially expanded. Since this will be the matrix stored in QRAM, 
we want to plot the Frobenius norm of the matrices. 

Overview
=========


"""

import sys
import random
import argparse
import pickle
import itertools
import logging

import numpy as np
from scipy import linalg

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mnist import MNIST

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics.pairwise import paired_distances

import utils

import pickle

#from libQSFA import QSFA


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-4s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    handlers=[logging.StreamHandler() ])

np.set_printoptions(precision=3, threshold=np.nan, formatter={'float': '{: 0.3f}'.format})


def create_x_dot(X, labels_indices):
    """

    :param X:
    :param labels_indices:
    :return:
    """
    number_of_derivatives = 14000
    X_dot = np.array([])
    for label, indices in labels_indices.items():
        X_= X[np.random.choice(indices, number_of_derivatives, replace=True)]
        if len(X_dot)==0:    # lazy!
            X_dot=X_[:-1]-X_[1:]
        else:
            X_dot=np.concatenate((X_dot, X_[:-1]-X_[1:]), axis=0)
    return X_dot











if __name__ == "__main__":
    """
    main of the file.
    """

    logging.info("Let's plot how sparsity change with PCA dimension.")

    mndata = MNIST("/home/scinawa/workspaces/qsfa/data")
    train_img, train_labels = mndata.load_training()
    train4pca = np.array(train_img)
    labels_indices, label_frequencies = utils.counting_indices(train_labels)

    pca_dim = range(10, 100, 10)
    y_X = []
    y_X_dot = []
    x = []

    for i in pca_dim:
        logging.info("Doing PCA bringing data to dimension {}".format(i))
        pca = PCA(n_components=i, svd_solver='randomized').fit(train4pca)
        x_train = pca.transform(train4pca)

        poly = preprocessing.PolynomialFeatures(2)
        x_train = poly.fit_transform(x_train)
        x_train = x_train[:, 1:]
        x.append(len(x_train[0]))
        logging.debug("Normalizing data to 0 mean and  1 variance")
        x_train = preprocessing.scale(x_train)

        logging.debug("Creating X_dot")
        x_dot = create_x_dot(x_train, labels_indices)
        
        sparsity_x = np.count_nonzero(x_train)/np.prod(x_train.shape)
        sparsity_x_dot = np.count_nonzero(x_dot)/np.prod(x_dot.shape)

        logging.info("Dimension x_train {} x_dot {}".format(x_train.shape, x_dot.shape))
        logging.info("Sparsity: {} and {}".format(sparsity_x, sparsity_x_dot))

        
        #y_X.append(frob_x)
        #y_X_dot.append(frob_x_dot)




