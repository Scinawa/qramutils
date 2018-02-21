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


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-4s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    handlers=[logging.StreamHandler()])

np.set_printoptions(precision=4, threshold=np.nan, formatter={'float': '{: 0.4f}'.format})



def tests():
    X = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 5, 6]])
    X = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1],[1, 1, 1],[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 5, 6]])
    return

class libQRAM():
    def __init__(self, dataset):
        """
        This library assume that the sample of the trainig set
        are stored along the rows of the matrix.

        :param dataset: a numpy array
        :return:
        """
        self.dataset = dataset
        pass

    def find_qubits(self):
        """
        Calculates how many qubits we need to index this matrix in a quantum register
        :return:
        """
        index_register = np.ceil(np.log2())
        content_register = np.ceil(np.log2())

        return qubits

    def sparsity(self):
        """
        Return the ration (0,1) of nonzero elements in the matrix
        :return:
        """
        sparsity = np.count_nonzero(self.dataset) / np.prod(self.dataset.shape)
        return 1.-sparsity

    def frobenius(self):
        """
        Calculate the frobenius norm of the matrix with eigenvalues scaled between 0 and 1

        :param self:
        :return:
        """

        eigval_x = linalg.svd(self.dataset, compute_uv=False)

        normalized_eigenvalues_x = np.true_divide(eigval_x, max(eigval_x))
        frob_norm = np.sqrt(sum(np.square(normalized_eigenvalues_x)))

        return frob_norm

    def find_p(self):
        """
        The normalization  factors are µ_p(A) = sqrt( s_(2p)(A)* s(2(1−p))(A^t) ), where
        we denote  by  s_p(A):= max_(i \in [m]) ||a_i||^p_p  the maximum  L_p norm of  the  row vectors,
        and by  s_p(A^t) the maximum  l_p norm of  the  column  vectors.
        """

        def mu(p, args):
            """
            The function to minimize

            :param x: the matrix representing the dataset
            :param p: the parameter [0,1]
            :return:
            """
            #logging.info("Executing mu with p {}".format(p))
            #input("run mu")

            def s(p, X):
                p = p[0]
                norms = np.linalg.norm(X, p, axis=1)
                vector_of_powers = np.full(len(norms), p)
                #print(vector_of_powers)
                exp_norms = np.power(norms, vector_of_powers)
                max_norms = max(exp_norms)

                #logging.debug("Norms {}".format(norms))
                #logging.debug("Exp_norms {}".format(exp_norms))
                #logging.debug("max_norms {}".format(max_norms))

                return max_norms

            mu = np.sqrt(s(2 * p, args['X']) * s(2 * (1 - p), args['X'].T))
            logging.debug("mu = sqrt( s(), s() = {}".format(mu))

            return mu

        cons = ({'type': 'ineq', 'fun': lambda p: p + 1},
                {'type': 'ineq', 'fun': lambda p: 1 - p})

        res = minimize(mu, 0.1, method='SLSQP', args={'X': self.dataset}, options={'ftol': 1e-4, 'disp': True},
                       bounds=((0.0000000001, 1),) )
        # constraints=cons)

        return res


def menu():
    """
    Create the nice *NIX like menu
    """

    parser = argparse.ArgumentParser(add_help=True, description='Analyze a dataset and model QRAM parameters')

    parser.add_argument("--db", help="path of the mnist database",
                        action='store', dest='db', default='/home/scinawa/workspaces/libQRAM/data')

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


if "__main__" == __name__:
    pass