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
                        action="store_true", dest="analize", default=True)

    # if not generate plot, you can specify these
    parser.add_argument("--pca-dim", help='pca dimension', action='store',
                        dest='pcadim', type=int, default=39)
    parser.add_argument("--polyexp", help='degree of polynomial expansion', action='store',
                        dest='polyexp', type=int, default=2)

    parsed_args = parser.parse_args()

    return parsed_args


def analize():
    """

    :return:
    """

    libq = libQRAM(X)

    sparsity = libq.sparsity()
    logging.info("Sparsity: {}".format(sparsity))

    frob_norm = libq.frobenius()
    logging.info("The frobenius norm of the matrix is {}".format(frob_norm))

    p_value_for_norm = libq.find_p()
    logging.info("Best p value for data is {}".format(p_value_for_norm.x))

    logging.info("The mu value is {} ".format(min(frob_norm, p_value_for_norm.fun)))

    qubits_used = libq.find_qubits()
    loggin.info("Qubits needed ceil(log_2(nd)) = {} ".format(qubits))
    return


if "__main__" == __name__:

    parsed_args = menu()

    # if args.csv:
    #    parse_csv

    mndata = MNIST(parsed_args.db)
    train_img, train_labels = mndata.load_training()
    test_img, test_labels = mndata.load_testing()
    X = np.array(train_img)

    if parsed_args.analzie:


    if parsed_args.generateplot:
        generateplots()



