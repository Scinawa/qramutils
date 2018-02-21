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

import utils
import pickle

#from libQSFA import QSFA


logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(name)-4s %(levelname)-8s %(message)s',
                                                datefmt='%m-%d %H:%M',
                                                                    handlers=[logging.StreamHandler() ])

np.set_printoptions(precision=3, threshold=np.nan, formatter={'float': '{: 0.3f}'.format})



  def plot_graph():
    fig1 = plt.figure(figsize=(7.6,7.6))
    ax = fig1.add_subplot(111)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)

    plt.grid()

    ax.plot(x, y_1, label="Norm of X")
    ax.plot(x, y_2, label="Norm of X_dot")
    ax.set_xlabel("PCA dim (pixels)")
    ax.set_ylabel("Frobenius norm")

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
   # ax.set_yscale('log')

    plt.legend(loc='lower right', prop={'size': 18})

    dst_path='/home/scinawa/graphs/frobenius_norm.png'
    print("File saved in {0}".format(dst_path))
    fig1.savefig(dst_path)

    return None


