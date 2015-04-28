import numpy as np
import matplotlib.pyplot as plt
import cvxopt as co

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_svmlight_file
import sklearn.cluster as cl

from latent_svr import LatentSvr
from latent_ridge import LatentRidgeRegression
from latent_multiclass_regression_map import LatentMulticlassRegressionMap
from latent_mrf import LatentMrf


def get_2d_toy_data(grid_x, grid_y):
    # generate 2D grid
    gx = np.linspace(0, 1, grid_x)
    gy = np.linspace(0, 1, grid_y)
    (gx, gy) = np.meshgrid(gx, gy)

    # latent variable z
    z = np.zeros((grid_y, grid_x), dtype=np.uint8)
    (zx, zy) = np.where((gx > 0.3))
    z[zx, zy] = 1

    # inputs  x
    # x = 4.0*np.sign(z-0.5)*gy + 1.0*np.sign(z-0.5) + 0.5*np.random.randn(GRID_X, GRID_Y)
    x = 4.0*gy + 1.0*gx + (np.sign(z-0.5)*10.1) + 0.01*np.random.randn(grid_y, grid_x)

    # ..and corresponding target value y
    y = (np.sign(z-0.5)*10.1)*x + 0.01*np.random.randn(grid_y, grid_x)

    # scikit learns samples x features format
    vecX = x.reshape((grid_x*grid_y), 1)
    vecy = y.reshape((grid_x*grid_y))
    vecz = z.reshape((grid_x*grid_y))
    return vecX, vecy, vecz


if __name__ == '__main__':
    grid_x = 10
    grid_y = 10
    (vecX, vecy, vecz) = get_2d_toy_data(grid_x, grid_y)

    # normalize data
    vecy = vecy-np.mean(vecy)
    vecX = vecX-np.mean(vecX)
    vecX /= np.max(vecX)
    vecy /= np.max(vecy)
    vecX = np.hstack((vecX, np.ones((vecX.shape[0], 1))))

    # build neighbor list
    A = np.zeros((grid_x*grid_y, grid_x*grid_y))
    for i in range(grid_x*grid_y):
        row = np.floor(float(i) / float(grid_x))
        col = np.floor(float(i) % float(grid_x))
        # upper
        if row < (grid_y-1):
            A[i+grid_x, i] = 1
        # lower
        if row > 0:
            A[i-grid_x, i] = 1
        # right
        if col < (grid_x-1):
            A[i+1, i] = 1
        # left
        if col > 0:
            A[i-1, i] = 1
    print A

    mrf = LatentMrf(co.matrix(vecX.T), co.matrix(A.T), num_states=2, y=co.matrix(vecy.T))
    u = mrf.get_hotstart_sol()
    v = mrf.get_hotstart_sol()

    # (obj_lp, states_lp, psi_lp, res_lp) = mrf.lp_relax_max(v - u*(u.trans()*psi))
    (obj_lp, states_lp, psi_lp, res_lp) = mrf.lp_relax_max(v)
    (obj_qp, states_qp, psi_qp) = mrf.qp_relax_max(v, u)
    (obj_qpr, states_qpr, psi_qpr, res_qpr) = mrf.qp_relax_max_direct(v, u)
    obj = obj_qpr
    states = states_qpr
    if grid_x*grid_y < 18:
        print('Calculate real objective using brute-force...')
        (obj, states, psi) = mrf.simple_max(v, u)

    states_qpr = np.array(states_qpr.trans())
    states_qp = np.array(states_qp.trans())
    states_lp = np.array(states_lp.trans())
    states = np.array(states.trans())
    # print states
    # print states_lp
    print (obj[0, 0], obj_lp[0, 0], obj_qp[0, 0], obj_qpr[0, 0])

    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.pcolor(states.reshape(grid_x, grid_y))
    plt.title('Original')

    plt.subplot(2, 2, 2)
    plt.pcolor(states_lp.reshape(grid_x, grid_y))
    plt.title('LP')

    plt.subplot(2, 2, 3)
    plt.pcolor(states_qp.reshape(grid_x, grid_y))
    plt.title('QP cutting plane')

    plt.subplot(2, 2, 4)
    plt.pcolor(states_qpr.reshape(grid_x, grid_y))
    plt.title('QP')
    plt.show()

    # ..and stop
    print('Finish!')
