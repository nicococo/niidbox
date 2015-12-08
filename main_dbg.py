import numpy as np
import matplotlib.pyplot as plt
import cvxopt as co
import time

from tcrfr_bf import TCRFR_BF
from tcrfr_qp import TCRFR_QP
from tcrfr_lbpa import TCRFR_lbpa

from test_setup import get_1d_toy_data

def get_test_data(exms, train):
    # generate toy data
    x, y, z = get_1d_toy_data(exms, plot=False)
    x -= np.mean(x, axis=0)
    x = np.hstack([x, np.ones((exms, 1))])
    print x.shape
    # ...and corresponding transition matrix
    A = np.zeros((exms, exms), dtype=np.int32)
    for j in range(1, 3):
        for i in range(j, exms):
            A[i-j, i] = 1
            A[i, i-j] = 1
    A = co.sparse(co.matrix(A))
    inds = np.random.permutation(exms)
    uinds = inds[train:]
    linds = inds[:train]
    lbpa = TCRFR_lbpa(x.T.copy(), y[linds].copy(), linds, uinds, states=2, A=A, reg_theta=0.8, trans_sym=[1])
    qp   = TCRFR_QP(x.T.copy(), y[linds].copy(), linds, uinds, states=2, A=A, reg_theta=0.8, trans_sym=[1])
    bf   = TCRFR_BF(x.T.copy(), y[linds].copy(), linds, uinds, states=2, A=A, reg_theta=0.8, trans_sym=[1])
    return lbpa, qp, bf, x, y, z


def test_bf():
    lbpa, qp, bf, x, y, z = get_test_data(10, 2)
    # lbpa.fix_lbl_map = True
    lbpa.fit(use_grads=False)
    qp.fit(use_grads=False)
    bf.fit(use_grads=True)
    print 'STATES ------------------------'
    foo = np.zeros(lbpa.samples, dtype=np.int8)
    foo[lbpa.label_inds] = 1
    print 'Train  = ', foo
    print 'True   = ', z
    print 'BF     = ', bf.latent
    print 'QP     = ', qp.latent
    print 'LBPA   = ', lbpa.latent


def test_constr_speed():
    exms = 2000
    train = 100
    x, y, z = get_1d_toy_data(exms, plot=False)
    x -= np.mean(x, axis=0)
    x = np.hstack([x, np.ones((exms, 1))])
    print x.shape
    # ...and corresponding transition matrix
    A = np.zeros((exms, exms), dtype=np.int32)
    for j in range(1, 20):
        for i in range(j, exms):
            A[i-j, i] = 1
            A[i, i-j] = 1
    A = co.sparse(co.matrix(A, tc='i'))
    inds = np.random.permutation(exms)
    uinds = inds[train:]
    linds = inds[:train]

    lbpa = TCRFR_lbpa(x.T.copy(), y[linds].copy(), linds, uinds, states=2, A=A, reg_theta=0.5, trans_sym=[1])
    t = time.time()
    for i in range(10):
        lbpa = TCRFR_lbpa(x.T.copy(), y[linds].copy(), linds, uinds, states=2, A=A, reg_theta=0.5, trans_sym=[1])
    print time.time()-t


if __name__ == '__main__':
    test_bf()
    # test_constr_speed()

    #
    # lbpa, qp, x, y, z = get_test_data(2000, 100)
    #
    # lbpa.get_joint_feature_maps(lbpa.latent)
    # qp.get_joint_feature_maps(lbpa.latent)
    #
    # # fit
    # # t = time.time()
    # # lbpa.fit(use_grads=False)
    # # lbpa_train_time = time.time()-t
    # #
    # # t = time.time()
    # qp.fit(use_grads=False)
    # # qp_train_time = time.time()-t
    # #
    # # _, _ = qp.map_inference(lbpa.u.copy(), lbpa.unpack_v(lbpa.v.copy()))
    # # qp_latent = qp.latent
    # #
    # # _, _ = lbpa.map_inference(lbpa.u.copy(), lbpa.unpack_v(lbpa.v.copy()))
    # # lbpa_latent = lbpa.latent
    #
    #
    # print 'LOG Z ------------------------'
    # #print np.log(part_value)
    # t = time.time()
    # for i in range(100):
    #     qp.log_partition(qp.unpack_v(qp.v))
    # print time.time()-t
    # # t = time.time()
    # # for i in range(10):
    # #     qp.log_partition_bak(lbpa.unpack_v(lbpa.v))
    # # print time.time()-t
    #
    # # print 'TIMES ------------------------'
    # # print qp_train_time
    # # print lbpa_train_time