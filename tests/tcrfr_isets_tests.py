from functools import partial

import cvxopt as co
import numpy as np
from nose.tools import *

from scripts.utils_experiment import get_1d_toy_data
from tcrfr_lbpa_iset import TCRFR_lbpa_iset
from tcrfr_qp import TCRFR_QP

x = None  # contains the (normalized) niidbox-data
y = None  # regression labels for training subset
z = None  # ground truth latent states

A = None  # Adjacency matrix

labeled_inds = None    # indices of labeled examples
unlabeled_inds = None  # indices of unlabeled examples


def setup(fun=None, exms=10, train=5, deps=1, add_intercept=True):
    global A, labeled_inds, unlabeled_inds, x, y, z
    print "Generate {0} examples, {1} labeled, {2} edges/exm. Add intercept: {3}".format(exms, train, deps, add_intercept)
    x, y, z = get_1d_toy_data(exms, plot=False)
    # normalize niidbox-data
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
def test_toy_data():
    print "Test the niidbox-data generation process."
    vals = np.array(A.V, dtype=np.int)
    assert A.size == (4, 4)  # we expect a squared matrix
    assert np.sum(vals) == 2*4-2  # chain-model, hence nodes*2-2 edges
    assert sum( A[co.matrix([0, 5, 10, 15])] ) == 0. # diagonal should be empty


@with_setup(setup=partial(setup, exms=100, train=0, deps=2, add_intercept=True))
def test_transition_conversion():
    print "Test transition conversion."
    num_exms = 100
    num_edges = 200

    # create a (valid) random transition matrix
    edges = np.zeros((num_edges, 3), dtype=np.int64)
    neighbors = np.zeros(num_edges, dtype=np.int32)
    A = co.spmatrix(0, [], [], (num_exms, num_exms), tc='d')
    cnt_edges = 0
    while cnt_edges < num_edges:
        e1 = np.random.randint(0, num_exms)
        e2 = np.random.randint(0, num_exms)
        if e1 != e2 and A[e1, e2] == 0:
            val = np.random.randint(1, 3)
            A[e1, e2] = val
            A[e2, e1] = val
            edges[cnt_edges, :] = (e1, e2, val)
            neighbors[e1] += 1
            neighbors[e2] += 1
            cnt_edges += 1

    qp = TCRFR_QP(x.T, y[labeled_inds], labeled_inds, states=2, A=A, \
                           reg_theta=0.9, reg_gamma=1., trans_sym=[1])

    assert qp.N.shape[1]==np.max(neighbors)
    assert qp.E.shape[0]==num_edges  # correct number of edges detected?

    # check if the correct edges are in the edge-array
    for (e1, e2, t) in edges:
        cnt_1 = 0
        cnt_2 = 0
        for e in range(qp.E.shape[0]):
            # check edges in both direction (only one should be in there)
            if qp.E[e, 0]==e1 and qp.E[e, 1]==e2 and qp.E[e, 2]==t:
                cnt_1 += 1
            if qp.E[e, 0]==e2 and qp.E[e, 1]==e1 and qp.E[e, 2]==t:
                cnt_2 += 1
        assert cnt_1 == 1 or cnt_2 == 1
        assert cnt_1+cnt_2 == 1

    for i in range(num_exms):
        assert neighbors[i] == np.sum(qp.N_weights[i, :])
        for (e1, e2, t) in edges:
            if e1 == i or e2 == i:
                if e2 == i:
                    e2 = e1
                    e1 = i
                assert e2 in qp.N[i, :neighbors[i]]
                ind = np.where(qp.N[i, :neighbors[i]] == e2)[0][0]
                # print qp.N_inv[i, ind]
                assert qp.N[e2, qp.N_inv[i, ind]] == i


@with_setup(setup=partial(setup, exms=16, train=0, deps=2, add_intercept=True))
def test_cluster_setting():
    print "Test Cluster setting."
    # setup cluster 0 = labeled, 1 = unlabeled (2 latent states)
    cluster = [np.arange(0, 10), np.arange(10, 16)]
    A[9, 10] = 0
    A[10, 9] = 0
    print z
    assert any(z[:6]==0) and any(z[6:16]==1)
    labeled_inds = np.arange(0, 10)

    qp = TCRFR_QP(x.T, y[labeled_inds], labeled_inds, states=2, A=A, \
                           reg_theta=0.6, reg_gamma=1., trans_sym=[1])
    qp.fit(use_grads=False)

    lbpa_iset = TCRFR_lbpa_iset(cluster, x.T, y[labeled_inds], labeled_inds, states=2, A=A, \
                           reg_theta=0.6, reg_gamma=1., trans_sym=[1])
    lbpa_iset.fit(use_grads=False)

    print '------------'
    inds = np.zeros(y.size, dtype=np.int8)
    inds[labeled_inds] = 1
    print inds
    print '------------'
    print lbpa_iset.latent
    print qp.latent
    print z
    print '------------'
    print lbpa_iset.log_partition_pl(qp.unpack_v(qp.v))
    print qp.log_partition_pl(qp.unpack_v(qp.v))
    print '------------'


@with_setup(setup=partial(setup, exms=5000, train=0, deps=2, add_intercept=True))
def test_cluster_large_setting():
    print "Test Cluster setting."
    # setup cluster 0 = labeled, 1 = unlabeled (2 latent states)
    cluster = [np.arange(0, 2501), np.arange(2501, 5000)]
    A[2500, 2501] = 0
    A[2501, 2500] = 0
    labeled_inds = np.random.permutation(y.size)
    labeled_inds = labeled_inds[:1500]

    qp = TCRFR_QP(x.T, y[labeled_inds], labeled_inds, states=2, A=A, \
                           reg_theta=0.9, reg_gamma=1., trans_sym=[1])
    #qp.fit(use_grads=False)

    lbpa_iset = TCRFR_lbpa_iset(cluster, x.T, y[labeled_inds], labeled_inds, states=2, A=A, \
                           reg_theta=0.9, reg_gamma=1., trans_sym=[1])
    lbpa_iset.fit(use_grads=False)

    print '------------'
    print np.sum(np.abs(lbpa_iset.latent-z))
    print np.sum(np.abs((1-lbpa_iset.latent)-z))
    print '------------'
    print np.sum(np.abs(qp.latent-z))
    print np.sum(np.abs((1-qp.latent)-z))
    print '------------'

@with_setup(setup=partial(setup, exms=16, train=0, deps=2, add_intercept=True))
def test_cluster_fixed_latent_states():
    print "Test Cluster setting."
    # setup cluster 0 = labeled, 1 = unlabeled (2 latent states)
    cluster = [np.arange(0, 16)]
    labeled_inds = np.random.permutation(16)
    labeled_inds = labeled_inds[:8]

    qp = TCRFR_QP(x.T, y[labeled_inds], labeled_inds, states=2, A=A, \
                           reg_theta=0.6, reg_gamma=1., trans_sym=[1])
    qp.fit(use_grads=False)

    lbpa_iset = TCRFR_lbpa_iset(cluster, x.T, y[labeled_inds], labeled_inds, states=2, A=A, \
                           reg_theta=0.6, reg_gamma=1., trans_sym=[1])
    fixed = -np.ones(y.size, dtype=np.int)
    fixed[4:10] = z[4:10]
    lbpa_iset.set_fixed_latent_states(fixed)
    lbpa_iset.fit(use_grads=False)

    print '------------'
    inds = np.zeros(y.size, dtype=np.int8)
    inds[labeled_inds] = 1
    print inds
    print '------------'
    print fixed
    print '------------'
    print 'Truth: ', z
    print 'Lbpa : ', lbpa_iset.latent
    print 'Qp   : ', qp.latent
    print '------------'
    print lbpa_iset.log_partition_pl(qp.unpack_v(qp.v))
    print qp.log_partition_pl(qp.unpack_v(qp.v))
    print '------------'

