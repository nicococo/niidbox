import numpy as np
import cvxopt as co

from tcrfr_bf import TCRFR_BF
from tcrfr_qp import TCRFR_QP
from tcrfr_lbpa import TCRFR_lbpa
from tcrfr_lbpa_iset import TCRFR_lbpa_iset

from utils_experiment import get_1d_toy_data, evaluate

from utils import profile, print_profiles

def get_test_data(exms, train):
    # generate toy niidbox-data
    x, y, z = get_1d_toy_data(exms, plot=False)
    y -= np.mean(y, axis=0)
    x -= np.mean(x, axis=0)
    x = np.hstack([x, np.ones((exms, 1))])
    print x.shape
    # ...and corresponding transition matrix
    A = np.zeros((exms, exms), dtype=np.int32)
    for j in range(1, 2):
        for i in range(j, exms):
            A[i-j, i] = 1
            A[i, i-j] = 1
    A = co.sparse(co.matrix(A))
    inds = np.random.permutation(exms)
    linds = inds[:train]
    lbpa = TCRFR_lbpa(x.T.copy(), y[linds].copy(), linds,  states=2, A=A, reg_gamma=10., reg_theta=0.8, trans_sym=[1])
    qp   = TCRFR_QP(x.T.copy(), y[linds].copy(), linds, states=2, A=A, reg_gamma=10., reg_theta=0.8, trans_sym=[1])
    bf   = TCRFR_BF(x.T.copy(), y[linds].copy(), linds, states=2, A=A, reg_gamma=10., reg_theta=0.8, trans_sym=[1])
    return lbpa, qp, bf, x, y, z


def test_smiley():
    # x = np.loadtxt('../../Projects/si.txt')
    # y = np.loadtxt('../../Projects/phi.txt')
    # z = np.loadtxt('../../Projects/facies.txt')
    # height, width = x.shape
    # exms = x.size
    #
    # x = x.reshape((x.size, 1), order='C')
    # y = y.reshape(y.size, order='C')
    # z = z.reshape(z.size, order='C')
    #
    # x[np.where(z==0)] -= 0.9
    #
    # y -= np.mean(y, axis=0)
    # x -= np.mean(x, axis=0)
    # y /= np.max(np.abs(y))
    # y *= 1.0
    # x /= np.max(np.abs(x))
    #
    # x = np.hstack([x, np.ones((exms, 1))])
    # print x.shape, y.shape, z.shape

    data = np.load('niidbox-data/data_smiley.npz')
    x = data['x']
    y = data['y']
    z = data['latent']
    width = data['width']
    height = data['height']
    exms = x.shape[0]
    linds = np.random.permutation(exms)[:np.int(0.3*exms)]

    A = co.spmatrix(0, [], [], (exms, exms), tc='d')
    for k in range(1, 3):
        for i in range(height):
            for j in range(width):
                idx = i*width + j
                idx1 = (i+k)*width + j
                idx2 = i*width + j + k
                if k == 1 or (k>1 and idx in linds) or (k>1 and idx1 in linds) or (k>1 and idx2 in linds):
                    if i < height-k:
                        A[idx, idx1] = 1
                        A[idx1, idx] = 1
                    if j < width-k:
                        A[idx, idx2] = 1
                        A[idx2, idx] = 1

    qp   = TCRFR_QP(x.T.copy(), y[linds].copy(), linds, states=2, A=A,
                    reg_gamma=10000., reg_theta=0.85, trans_sym=[1])
    u = np.random.randn(qp.get_num_feats()*qp.S)
    v = np.random.randn(qp.get_num_compressed_dims())
    #np.savez('../../Projects/hotstart.npz', start=(u, v, linds))
    #(u, v, linds) = np.load('../../Projects/hotstart.npz')['start']

    # qp.fit(use_grads=False, hotstart=(u, v), auto_adjust=False)
    # qp.fit(use_grads=False, hotstart=None, auto_adjust=True)

    lbpa = TCRFR_lbpa(x.T.copy(), y[linds].copy(), linds,  states=2, A=A,
                      reg_gamma=10000., reg_theta=0.995, trans_sym=[1], trans_regs=[[10., 2.]])
    lbpa.verbosity_level = 3
    lbpa.set_log_partition(lbpa.LOGZ_PL_MAP)
    lbpa.fit(use_grads=False, hotstart=(u, v), auto_adjust=False)
    #lbpa.fit(use_grads=False, hotstart=None, auto_adjust=True)
    #lbpa.map_inference(qp.u, lbpa.unpack_v(qp.v))

    # initialize all non-fixed latent variables with random states
    import sklearn.cluster as cl
    kmeans = cl.KMeans(n_clusters=2, init='random', n_init=4, max_iter=100, tol=0.0001)
    kmeans.fit(x)

    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.subplot(1, 7, 1)
    plt.imshow(y.reshape((height, width), order='C'))
    plt.subplot(1, 7, 2)
    plt.imshow(x[:, 0].reshape((height, width), order='C'))
    plt.subplot(1, 7, 3)
    plt.imshow(z.reshape((height, width), order='C'))
    plt.subplot(1, 7, 4)
    plt.imshow(lbpa.latent.reshape((height, width), order='C'))
    plt.title('LBP')
    plt.subplot(1, 7, 5)
    plt.imshow(kmeans.labels_.reshape((height, width), order='C'))
    plt.title('KMeans')
    plt.subplot(1, 7, 6)
    plt.imshow(qp.latent.reshape((height, width), order='C'))
    plt.title('QP')
    plt.subplot(1, 7, 7)
    res, _ = lbpa.predict()

    print 'Labels:', linds.size
    print 'RESULT:', np.sum((res - y)*(res - y))/y.size

    plt.imshow(res.reshape((height, width), order='C'))
    plt.show()


def test_bf():
    lbpa, qp, bf, x, y, z = get_test_data(10, 4)
    lbpa.verbosity_level = 2
    # lbpa.fix_lbl_map = True
    lbpa.fit(use_grads=False)
    # qp.fit(use_grads=False)
    # qp.fit(use_grads=False)
    #u = np.random.randn(qp.get_num_feats()*qp.S)
    #v = np.random.randn(qp.get_num_compressed_dims())
    qp.u, qp.v = lbpa.u, lbpa.v
    qp.map_inference(qp.u, lbpa.unpack_v(qp.v))
    bf.map_inference(qp.u, lbpa.unpack_v(qp.v))
    lbpa.map_inference(qp.u, lbpa.unpack_v(qp.v))

    #lbpa.fit()

    print 'STATES ------------------------'
    foo = np.zeros(lbpa.samples, dtype=np.int8)
    foo[lbpa.label_inds] = 1
    print 'Train  = ', foo
    print 'True   = ', z
    print 'BF     = ', bf.latent
    print 'QP     = ', qp.latent
    print 'LBPA   = ', lbpa.latent
    #

    logZ_bf = bf.log_partition(lbpa.unpack_v(qp.v))

    print 'logZ ------------------------'
    print 'True                    = ', logZ_bf
    print 'Pseudo-likelihood SUM   = ', lbpa.log_partition_pl(lbpa.unpack_v(qp.v))
    print 'Pseudo-likelihood MAP   = ', lbpa.log_partition_map(lbpa.unpack_v(qp.v))
    print 'Unary                   = ', lbpa.log_partition_unary(lbpa.unpack_v(qp.v))


def test_constr_speed():
    exms = 10000
    train = 1000
    x, y, z = get_1d_toy_data(exms, plot=False)
    x -= np.mean(x, axis=0)
    x = np.hstack([x, np.ones((exms, 1))])
    print x.shape
    # ...and corresponding transition matrix

    A = co.spmatrix(0, [], [], (exms, exms), tc='d')
    for j in range(1, 2):
        for i in range(j, exms):
            A[i-j, i] = 1
            A[i, i-j] = 1

    inds = np.random.permutation(exms)
    uinds = inds[train:]
    linds = inds[:train]

    print('Start:')
    cluster = [ np.arange(exms)]
    lbpa = TCRFR_lbpa_iset(cluster, x.T, y[linds], linds, states=2, A=A, \
                           reg_theta=0.9, reg_gamma=1., trans_sym=[1])
    lbpa.fit(use_grads=False)
    y_pred, lat_pred = lbpa.predict()
    print evaluate(y[uinds], y_pred[uinds], z[uinds], lat_pred[uinds], 0.0)


def test_lbp():
    exms = 30
    train = 10
    x, y, z = get_1d_toy_data(exms, plot=False)
    x -= np.mean(x, axis=0)
    x = np.hstack([x, np.ones((exms, 1))])
    print x.shape
    # ...and corresponding transition matrix
    A = np.zeros((exms, exms), dtype=np.int32)
    for j in range(1, 2):
        for i in range(j, exms):
            A[i-j, i] = 1
            A[i, i-j] = 1
    A = co.sparse(co.matrix(A, tc='i'))
    inds = np.random.permutation(exms)
    uinds = inds[train:]
    linds = inds[:train]
    cluster = [np.arange(exms)]

    qp = TCRFR_QP(x.T.copy(), y[linds].copy(), linds, states=2, A=A, reg_gamma=10., reg_theta=0.9, trans_sym=[1])
    lbpa = TCRFR_lbpa(x.T.copy(), y[linds].copy(), linds, states=2, A=A, reg_gamma=10., reg_theta=0.9, trans_sym=[1])
    lbpa_iset = TCRFR_lbpa_iset(cluster, x.T.copy(), y[linds].copy(), linds, states=2, A=A, reg_gamma=10., reg_theta=0.9, trans_sym=[1])
    qp.fit()
    lbpa.fit()
    lbpa_iset.fit()

    # lbpa_iset.map_inference(lbpa.u, lbpa.unpack_v(lbpa.v))
    lid = lbpa.map_indep(qp.u, lbpa.unpack_v(qp.v))

    print 'STATES ------------------------'
    foo = np.zeros(lbpa.samples, dtype=np.int8)
    foo[lbpa.label_inds] = 1
    print 'Train  = ', foo
    print 'True   = ', z
    print 'QP     = ', qp.latent
    print 'LBPA   = ', lbpa.latent
    print 'LBPAc  = ', lbpa_iset.latent
    print 'Indep  = ', lid


if __name__ == '__main__':
    test_bf()
    # test_smiley()
    # test_constr_speed()
    # test_lbp()

    print_profiles()