
# coding: utf-8

# In[6]:

import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from mnist import MNIST
import logging
import pickle
import pickle
import matplotlib

from qramutils import qramutils

matplotlib.use('Agg')
import matplotlib.pyplot as plt



def preprocessing_data(X, pca_dim, polyexp):
    # PCA
    pca = PCA(n_components=pca_dim, svd_solver='full').fit(X)
    x_train = pca.transform(X)

    # poly exp
    poly = preprocessing.PolynomialFeatures(polyexp)
    x_train = poly.fit_transform(x_train)
    x_train = x_train[:, 1:]

    # scale
    x_train = preprocessing.scale(x_train)
    print("Shape of data after preprocessing: {}".format(x_train.shape))
    return x_train


def create_x_dot(X, labels_indices, derivatives):
    """
    This is the same function used in QSFA to create the matrix X_dot.

    :param X:
    :param labels_indices:
    :return:
    """
    X_dot = np.empty(shape=(0,X.shape[1]))
    for label, indices in labels_indices.items():
        X_= X[np.random.choice(indices, derivatives, replace=True)]
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



def __real_plot(frob_x_dot, frob_x, x_axis, ymax, exp=0, xlabel='', ylabel='', filename='default_filename.png'):
    fig1 = plt.figure(figsize=(7.6, 7.6))
    ax = fig1.add_subplot(111)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)
    
    ax.plot(x_axis, np.array(frob_x[2]), label="$polyexp_2(X)$")
    ax.plot(x_axis, np.array(frob_x[3]), label="$polyexp_3(X)$")
    ax.plot(x_axis, np.array(frob_x_dot[2]), label="$polyexp_2(\dot{X})$")
    ax.plot(x_axis, np.array(frob_x_dot[3]), label="$polyexp_3(\dot{X})$")
    plt.legend(loc='upper left', prop={'size': 12})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(ymin=0, ymax=ymax)

    plt.grid()
    fig1.savefig(filename)

    print("Plot {0} created".format(filename))



mndata = MNIST("/home/scinawa/workspaces/qsfa_mnist/data/")
train_img, train_labels = mndata.load_training()
test_img, test_labels = mndata.load_testing()
X = np.array(train_img)
X_test = np.array(test_img)


frob_x = {2: [], 3: []}
cond_x = {2: [], 3: []}
frob_x_dot = {2: [], 3: []}
cond_x_dot = {2: [], 3: []}


x_new = np.concatenate((X, X_test))
y_new = np.concatenate((train_labels, test_labels))


for pexp in [2,3]:
    for tranche in range(30000, 70000, 5000):
        if pexp==2:
            pcadim=80
        else:
            pcadim=36
        x_new = x_new[:tranche]
        y_new = y_new[:tranche]

        x_signal = preprocessing_data(x_new, pcadim, pexp)

        libq = qramutils.QramUtils(x_signal)
        frob_norm = libq.frobenius()
        cond_numb = np.linalg.cond(x_signal)
        print("X_signal: Shape: {}, Frobenius {} Cond number {}, pcadim {} polyexp {}".format(x_signal.shape, frob_norm, cond_numb, pcadim, pexp))
        frob_x[pexp].append(frob_norm)
        cond_x[pexp].append(cond_numb)

        label_indices = {l: [i for i, x in enumerate(y_new) if x == l] for l in  set(y_new)}

        derivatives = int(x_new.shape[0]/12)
        x_dot = create_x_dot(x_signal, label_indices, derivatives=derivatives)

        libq_2 = qramutils.QramUtils(x_dot)
        frob_norm_2 = libq_2.frobenius()
        cond_numb_2 = np.linalg.cond(x_dot)
        print("X_dot: Shape: {} Frobenius {} Cond number {}, pcadim {} polyexp {} derivatives {} ".format(x_dot.shape, frob_norm_2, cond_numb_2, pcadim, pexp, derivatives))
        frob_x_dot[pexp].append(frob_norm_2)
        cond_x_dot[pexp].append(cond_numb_2)

pickle.dump([frob_x_dot[2], frob_x[2], frob_x_dot[3], frob_x[3] ], open("26-aprile-frob_norm.pickle", "wb"))
pickle.dump([cond_x_dot[2], cond_x[2], cond_x_dot[3], cond_x[3] ], open("26-aprile-conditioning_number.pickle", "wb"))

[frob_x_dot[2], frob_x[2], frob_x_dot[3], frob_x[3] ] = pickle.load(open("26-aprile-frob_norm.pickle", "rb"))
[cond_x_dot[2], cond_x[2], cond_x_dot[3], cond_x[3] ] = pickle.load(open("26-aprile-conditioning_number.pickle", "rb"))

try:
    print("Frobenius norms X_dot poly2")
    print (frob_x_dot[2])
    print("Frobenius norm X poly2")
    print(frob_x[2])

    print("Frobenius norms X_dot poly3")
    print (frob_x_dot[3])
    print("Frobenius norms X poly 3")
    print(frob_x[3])

    print("Cond number X_dot poly2")
    print(cond_x_dot[2])
    print("Cond number X poly2")
    print(cond_x[2])

    print("Cond number X_dot poly3")
    print(cond_x_dot[3])
    print("Cond number X poly3")
    print(cond_x[3])

except Exception as e:
    print("non benissimo")


#    "Frob x_dot 2: {} \n Frob x {} \n Frob_x_dot 3 {} \n Frob_x 3 {}  \n xaxis2 {} \n xaxis3 {}".format(frob_x_dot[2],
#                                                                                                        frob_x[2],
#                                                                                                        frob_x_dot[3],
#              libaaaaaaaaaaaaaaaa                                                      libaaaaaaaaaaaaaaaa                                    frob_x[2]))
#print(
#    "Cond x_dot 2: {} \n Cond x {} \n Cond_x_dot 3 {} \n Cond_x 3 {}  \n xaxis2 {} \n xaxis3 {}".format(cond_x_dot[2],
#                                                                                                        cond_x[2],
#                                                                                                        cond_x_dot[3],
#                                                                                                        cond_x[3]  ))

__real_plot(cond_x_dot, cond_x, [i for i in range(30000, 70000, 5000)], ymax=1.8*max(np.concatenate((cond_x[2], cond_x_dot[2], cond_x[3], cond_x_dot[3]))), exp=0, xlabel='Training set dimension', ylabel='$\kappa$', filename='incremental-conditionin_number.png')
__real_plot(frob_x_dot, frob_x, [i for i in range(30000, 70000, 5000)], ymax=1.8*max(np.concatenate((frob_x[3], frob_x_dot[3], frob_x[3], frob_x_dot[3]))), exp=0, xlabel='Training set dimension', ylabel='Norm', filename='incremental-frobenius_norm.png')


