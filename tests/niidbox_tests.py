import numpy as np
import cvxopt as co

from nose.tools import *

from functools import partial

from test_setup import get_1d_toy_data
from tcrfr_lbpa import TCRFR_lbpa
from tcrfr_lbpa_iset import TCRFR_lbpa_iset

x = None  # contains the (normalized) data
y = None  # regression labels for training subset
z = None  # ground truth latent states

A = None  # Adjacency matrix

labeled_inds = None    # indices of labeled examples
unlabeled_inds = None  # indices of unlabeled examples


def setup(fun=None, exms=10, train=5, deps=1, add_intercept=True):
    global A, labeled_inds, unlabeled_inds, x, y, z
    print "Generate {0} examples, {1} labeled, {2} edges/exm. Add intercept: {3}".format(exms, train, deps, add_intercept)
    x, y, z = get_1d_toy_data(exms, plot=False)
    # normalize data
    y -= np.mean(y, axis=0)
    y /= np.max(np.abs(y))
    x -= np.mean(x, axis=0)
    x /= np.max(np.abs(x))
    if add_intercept:
        # adding bias term for intercept
        x = np.hstack([x, np.ones((exms, 1))])
    # ...and corresponding transition matrix
    A = co.spmatrix(0, [], [], (exms, exms), tc='d')
    for j in range(1, deps):
        for i in range(j, exms):
            A[i-j, i] = 1
            A[i, i-j] = 1
    inds = np.random.permutation(exms)
    unlabeled_inds = inds[train:]
    labeled_inds = inds[:train]
    return fun


@with_setup(setup=partial(setup, exms=4, train=0, deps=2, add_intercept=True))
def test_toy_data_setting():
    print "Test the data generation process."
    vals = np.array(A.V, dtype=np.int)
    assert A.size == (4, 4)  # we expect a squared matrix
    assert np.sum(vals) == 2*4-2  # chain-model, hence nodes*2-2 edges
    assert sum( A[co.matrix([0, 5, 10, 15])] ) == 0. # diagonal should be empty


@with_setup(setup=partial(setup, exms=16, train=0, deps=2, add_intercept=True))
def test_cluster_setting():
    print "Test Cluster setting."
    # setup cluster 0 & 2 = unlabeled, 1 = labeled (2 latent states)
    cluster = [np.arange(0, 6), np.arange(6, 10), np.arange(10, 16)]
    assert any(z[6:10]==1) and any(z[6:10]==0)
    labeled_inds = np.arange(6, 10)
    unlabeled_inds = np.setdiff1d(np.arange(16), labeled_inds)
    lbpa = TCRFR_lbpa_iset(cluster, x.T, y[labeled_inds], labeled_inds, unlabeled_inds, states=2, A=A, \
                           reg_theta=0.9, reg_gamma=1., trans_sym=[1])
    lbpa.fit(use_grads=False)
    y_pred, lat_pred = lbpa.predict()
    print y_pred