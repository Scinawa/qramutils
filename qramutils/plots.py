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





  def plot_graph(plot_name, x, y_1, label_1, y_2, label_2, x_axis_label, y_axis_label):
    """

    """ 
    fig1 = plt.figure(figsize=(7.6,7.6))
    ax = fig1.add_subplot(111)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)

    plt.grid()

    ax.plot(x, y_1, label=label_1)
    ax.plot(x, y_2, label=label_2)
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_axis_label)

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=0)
   # ax.set_yscale('log')

    plt.legend(loc='lower right', prop={'size': 18})

    dst_path=os.path.dirname(os.path.abspath(__file__))+'/'+plotname+'frobenius_norm.png'
    print("Plot {0} created".format(dst_path))
    fig1.savefig(dst_path)

    return None


