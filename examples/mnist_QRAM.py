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
import os
import itertools
import logging

import numpy as np
import pickle

from mnist import MNIST

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.metrics.pairwise import paired_distances

from qramutils import qramutils

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-4s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    handlers=[logging.StreamHandler()])

np.set_printoptions(precision=4, threshold=np.nan, formatter={'float': '{: 0.4f}'.format})

def set_logging_level(parsed_args):
    if parsed_args.loglevel == 'DEBUG':
        logging.getLogger().setLevel(logging.DEBUG)
    if parsed_args.loglevel == 'INFO':
        logging.getLogger().setLevel(logging.INFO)

def menu():
    """
    Create the nice *NIX like menu
    """

    parser = argparse.ArgumentParser(add_help=True, description='Analyze a dataset and model QRAM parameters')

    parser.add_argument("--db", help="path of the mnist database",
                        action='store', dest='db', default='data')

    parser.add_argument("--generateplot", help="run experiment with various dimension",
                        action="store_true", dest="generateplot", default=False)

    parser.add_argument("--frobnormplot", help="Plot Frobenius norm of data before and after polyexp",
                        action="store_true", dest="frobnormplot", default=False)

    parser.add_argument("--svplot", help="Plot singular values after polyexp after polyexp",
                        action="store_true", dest="svplot", default=False)

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


def create_x_dot(X, labels_indices):
    """
    This is the same function used in QSFA to create the matrix X_dot.

    :param X:
    :param labels_indices:
    :return:
    """
    X_dot = np.empty(shape=(0,X.shape[1]))
    for label, indices in labels_indices.items():
        X_= X[np.random.choice(indices, 14000, replace=True)]
        X_dot = np.concatenate (( X_dot, X_[:-1]-X_[1:] ) )
    return X_dot




    number_of_derivatives = 14000
    X_dot = np.array()
    for label, indices in labels_indices.items():
        X_= X[np.random.choice(indices, number_of_derivatives, replace=True)]
        if len(X_dot)==0:    # lazy!
            X_dot=X_[:-1]-X_[1:]
        else:
            X_dot=np.concatenate((X_dot, X_[:-1]-X_[1:]), axis=0)
    return X_dot


def analize(X):
    """
    Print the parameters of a given numpy matrix

    :return:
    """

    libq = qramutils.QramUtils(X, logging_handler=logging)

    logging.info("Matrix dimension {}".format(X.shape))

    sparsity = libq.sparsity()
    logging.info("Sparsity (0=dense 1=empty): {}".format(sparsity))

    frob_norm = libq.frobenius()
    logging.info("The Frobenius norm: {}".format(frob_norm))

    best_p, min_sqrt_p = libq.find_p()
    logging.info("Best p value: {}".format(best_p))

    logging.info("The \\mu value is: {}".format(min(frob_norm, min_sqrt_p)))

    qubits_used = libq.find_qubits()
    logging.info("Qubits needed to index+data register: {} ".format(qubits_used))
    return


def preprocessing_data(X, pca_dim, polyexp):
    logging.info("Calculating parameters for default configuration: PCA dim 39, polyexp 2")
    # PCA
    pca = PCA(n_components=pca_dim, svd_solver='full').fit(X)
    x_train = pca.transform(X)

    # scale
    x_train = preprocessing.scale(x_train)

    # poly exp
    poly = preprocessing.PolynomialFeatures(polyexp)
    x_train = poly.fit_transform(x_train)
    x_train = x_train[:, 1:]

    return x_train

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


    label_indices = {l: [i for i, x in enumerate(train_labels) if x == l] for l in
                   set(train_labels)}



    if parsed_args.analize:
      logging.info("Finding QRAM parameters with PCA and polynomial expansion")

      x_train = preprocessing_data(parsed_args.pcadim, parsed_args.polyexp)
      analize(x_train)
      sys.exit()


    if parsed_args.generateplot:
      logging.info("Analize QRAM parameters with PCA and polynomial expansion")

      polyexp = dict()
      polyexp[2] = [20, 30] #, 40, 50, 70, 80, 90]
      polyexp[3] = [20, 30] #, 40, 50, 70, 90, 100, 120] #,25, 30] #, 40,50, 70, 90, 110, 150, 200]

      for polydeg in polyexp:
          for dim in polyexp[polydeg]:
            logging.info("Poly degree: {} - PCA dimension: {}".format(polydeg, dim))

            x_train = preprocessing_data(X, dim, polydeg)
            analize(x_train)

      sys.exit()


    if parsed_args.svplot:
        logging.info("Plotting sve of polyexp")
        
        logging.info("PCA")
        pca = PCA(n_components=50, svd_solver='full').fit(X)
        x_pca = pca.transform(X)

        logging.info("Polyexp")
        poly = preprocessing.PolynomialFeatures(2)
        x_polyexp = poly.fit_transform(x_pca)
        x_polyexp = x_polyexp[:, 1:]

        logging.info("Find sve of polyexp matrix")
        _, sv, _ = np.linalg.svd(x_polyexp, full_matrices=False)

        sv = np.true_divide(sv, max(sv)) 
   `    qramutils.QramUtils.plot_singular_values(sv, 'PCA=50 PolyExp=2')

        return None


    if  parsed_args.frobnormplot:
        logging.info("Plotting: frobnorm=f(pca+polyexp)")

        polyexp = dict()
        polyexp[2] = [1, 10, 15,  20, 25, 30, 40, 70, 90, 110, 130] #, 40, 50, ]  # , 40, 50, 70, 80, 90]
        #polyexp[3] = [20, 30]  # , 40, 50, 70, 90, 100, 120] #,25, 30] #, 40,50, 70, 90, 110, 150, 200]

        fb_x=[1.0, 2.631461205347267, 3.014750250807214, 3.30245528768108, 3.5329254271493187, 3.7273982866202804, 4.035126723593193, 5, 6, 7]
        fb_x_polyexp = [1.000450826717045, 3.63873803591727, 4.571147605466965, 5.371257219395459, 6.065145285198597, 6.679874641337981, 7.734203137909173, 8, 9, 10]
        fb_x_dot=[1.0, 2.682166681602912, 3.1202306066670102, 3.4516985296140184, 3.7240875773740116, 3.9507441146385296, 4.314716158996903, 5, 6, 7]
        fb_x_dot_polyexp = [1.000441443427627, 3.835120699131673, 4.933300346915373, 5.87783024450804, 6.730190901151586, 7.484233107661058, 8.797614028341483, 9, 11, 12]
        fb_x=[]
        fb_x_polyexp=[]
        fb_x_dot=[]
        fb_x_dot_polyexp=[]

        X_dot = create_x_dot(X, label_indices)

        for dim in polyexp[2]:
            logging.info("Poly degree: {} - PCA dimension: {}".format(2, dim))

            pca = PCA(n_components=dim, svd_solver='full').fit(X)
            x_pca = pca.transform(X)

            libq = qramutils.QramUtils(x_pca, logging_handler=logging)
            frob_norm = libq.frobenius()
            fb_x.append(frob_norm)

            poly = preprocessing.PolynomialFeatures(2)
            x_polyexp = poly.fit_transform(x_pca)
            x_polyexp = x_polyexp[:, 1:]

            libq = qramutils.QramUtils(x_polyexp, logging_handler=logging)
            frob_norm_polyexp = libq.frobenius()
            fb_x_polyexp.append(frob_norm_polyexp)



            pca = PCA(n_components=dim, svd_solver='full').fit(X_dot)
            x_dot_pca = pca.transform(X_dot)

            libq = qramutils.QramUtils(x_dot_pca, logging_handler=logging)
            frob_norm = libq.frobenius()
            fb_x_dot.append(frob_norm)

            x_dot_polyexp = poly.fit_transform(x_dot_pca)
            x_dot_polyexp = x_dot_polyexp[:, 1:]

            libq = qramutils.QramUtils(x_dot_polyexp, logging_handler=logging)
            frob_norm_dot_polyexp = libq.frobenius()
            fb_x_dot_polyexp.append(frob_norm_dot_polyexp)


        pickle.dump([fb_x, fb_x_polyexp, fb_x_dot, fb_x_dot_polyexp], open("frobenius_norms.pickle", "wb"))
        #[fb_x, fb_x_polyexp, fb_x_dot, fb_x_dot_polyexp] = pickle.load(open("frobenius_norms.pickle", "rb"))
        print(fb_x)
        print(fb_x_polyexp)

        print(fb_x_dot)
        print(fb_x_dot_polyexp)

        fig1 = plt.figure(figsize=(7.6,7.6))
        ax = fig1.add_subplot(111)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(10)

        plt.grid()

        ax.plot(polyexp[2], np.power(fb_x, 2), label="Frobenius norm of X + PCA")
        ax.plot(polyexp[2], np.power(fb_x_polyexp, 2), label="Frobenius norm of X + PCA+polyexp")

        ax.plot(polyexp[2], np.power(fb_x_dot, 2), label="Frobenius norm of X_dot + PCA")
        ax.plot(polyexp[2], np.power(fb_x_dot_polyexp, 2), label="Frobenius norm of X_dot + PCA + polyexp(x)")
        
        ax.set_xlabel("PCA dimension")
        ax.set_ylabel("Normalized Frb. norm (2**x)")

        #ax.set_ylim(ymin=0, ymax=20)
        #ax.set_xlim(xmin=0, xmax=max(polyexp[2])+5)
        #ax.set_xscale('log')

        plt.legend(loc='upper left', prop={'size': 13})

        dst_path=os.path.dirname(os.path.abspath(__file__))+'/'+'frobenius_norm_vs_polyexp_frobenius_norm.png'
        print("Plot {0} created".format(dst_path))
        fig1.savefig(dst_path)

