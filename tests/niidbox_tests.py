import numpy as np
import cvxopt as co

from nose.tools import *

from functools import partial

from test_setup import get_1d_toy_data

x = None  # contains the (normalized) data
y = None  # regression labels for training subset
z = None  # ground truth latent states

A = None  # Adjacency matrix

labeled_inds = None    # indices of labeled examples
unlabeled_inds = None  # indices of unlabeled examples


def setup(fun=None, exms=10, train=5, deps=1, add_intercept=True):
    global A, labeled_inds, unlabeled_inds, x, y, z
    print "Generate data and adjacency matrix."
    print exms
    print train
    x, y, z = get_1d_toy_data(exms, plot=False)
    # normalize data
    y -= np.mean(y, axis=0)
    y /= np.max(np.abs(y))
    x -= np.mean(x, axis=0)
    x /= np.max(np.abs(x))
    if add_intercept:
        # adding bias term for intercept
        x = np.hstack([x, np.ones((exms, 1))])
    print x.shape
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
    print "I RAN!!"
    vals = np.array(A.V, dtype=np.int)
    assert A.size == (4, 4)  # we expect a squared matrix
    assert np.sum(vals) == 2*4-2  # chain-model, hence nodes*2-2 edges
    assert sum( A[co.matrix([0, 5, 10, 15])] ) == 0. # diagonal should be empty
