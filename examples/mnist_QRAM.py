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
"""
import sys
import argparse
import itertools
import logging

import numpy as np
from scipy import linalg

import numpy as np
from scipy.optimize import minimize

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mnist import MNIST

from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics.pairwise import paired_distances

from libQRAM import libQRAM

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-4s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    handlers=[logging.StreamHandler()])

np.set_printoptions(precision=4, threshold=np.nan, formatter={'float': '{: 0.4f}'.format})



def create_x_dot(X, labels_indices):
    """
    This is the same function used in QSFA to create the matrix X_dot.

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


def menu():
    """
    Create the nice *NIX like menu
    """

    parser = argparse.ArgumentParser(add_help=True, description='Analyze a dataset and model QRAM parameters')

    parser.add_argument("--db", help="path of the mnist database",
                        action='store', dest='db', default='data')

    parser.add_argument("--generateplot", help="run experiment with various dimension",
                        action="store_true", dest="generateplot", default=False)

    parser.add_argument("--analize", help="Run all the analysis of the matrix",
                        action="store_true", dest="analize", default=False)

    # if not generate plot, you can specify these
    parser.add_argument("--pca-dim", help='pca dimension', action='store',
                        dest='pcadim', type=int, default=39)
    parser.add_argument("--polyexp", help='degree of polynomial expansion', action='store',
                        dest='polyexp', type=int, default=2)

    parser.add_argument("--loglevel", help='set log level', action='store',
                        dest='loglevel',  choices=['DEBUG','INFO'] )

    if len(sys.argv)==1:
      parser.print_help(sys.stderr)
      sys.exit(1)
    parsed_args = parser.parse_args()

    return parsed_args


def analize(X):
    """

    :return:
    """

    libq = libQRAM(X)

    sparsity = libq.sparsity()
    logging.info("Sparsity (0=dense 1=empty): {}".format(sparsity))

    frob_norm = libq.frobenius()
    logging.info("The Frobenius norm: {}".format(frob_norm))

    best_p = libq.find_p()
    logging.info("Best p value: {}".format(best_p))

    logging.info("The \\mu value is {} ".format(min(frob_norm, best_p)))

    #qubits_used = libq.find_qubits()
    #loggin.info("Qubits needed ceil(log_2(nd)) = {} ".format(qubits))
    return

def set_logging_level(parsed_args):
    """

    :return:
    """

    if parsed_args.loglevel == 'DEBUG':
        logging.getLogger().setLevel(logging.DEBUG)
    if parsed_args.loglevel == 'INFO':
        logging.getLogger().setLevel(logging.INFO)

if "__main__" == __name__:

    parsed_args = menu()
    set_logging_level(parsed_args)

    try:
      mndata = MNIST(parsed_args.db)
      train_img, train_labels = mndata.load_training()
      test_img, test_labels = mndata.load_testing()
      X = np.array(train_img)
    except Exception as e:
      logging.error("Issues while opening database file: {}".format(e))

    if parsed_args.analize:
      analize(X)
      sys.exit() 
    
    if parsed_args.generateplot:
      logging.info("Finding QRAM parameters with PCA and polynomial expansion")
      polyexp = dict()
      polyexp[2] = [20, 30, 40, 50, 70, 80, 90]
      polyexp[3] = [20, 30, 40, 50, 70, 90, 100, 120] #,25, 30] #, 40,50, 70, 90, 110, 150, 200]

      for polydeg in polyexp:
          for dim in polyexp[polydeg]:
            logging.info("Polydeg: {} - PCA: {}".format(polydeg, dim))
            pca = PCA(n_components=dim, svd_solver='full').fit(X)
            x_train = pca.transform(X)
            analize(x_train)
      sys.exit()
      
