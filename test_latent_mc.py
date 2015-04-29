import numpy as np
import matplotlib.pyplot as plt
import cvxopt as co

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, adjusted_rand_score
from sklearn.datasets import load_svmlight_file
import sklearn.cluster as cl

from latent_svr import LatentSVR
from latent_ridge_regression import LatentRidgeRegression
from multiclass_regression_model import MulticlassRegressionModel
from latent_cluster_regression import LatentClusterRegression


def load_svmlight_data(fname):
    (X, y) = load_svmlight_file(fname)
    print y.shape
    print X.shape
    return np.array(X.todense()), y.reshape(X.shape[0])


def get_1d_toy_data(plot_data=False):
    grid_x = 1600
    # generate 2D grid
    gx = np.linspace(0, 1, grid_x)

    # latent variable z
    z = np.zeros(grid_x, dtype=np.uint8)
    zx = np.where((gx > 0.2) & (gx < 0.6))[0]
    zx = np.where(gx > 0.5)[0]
    z[zx] = 1

    # inputs  x
    x = 4.0*np.sign(z-0.4)*gx + 2.0*np.sign(z-0.4) + 0.5*np.random.randn(grid_x)
    x = 4.0*np.sign(z-0.5)*gx + 0.5*(z+1.0)*np.random.randn(grid_x)
    # x = 4.0*np.sign(z-0.5)*gx + 1.0*np.sign(z-0.5) + 0.8*np.random.randn(grid_x)
    # x = 8.0*gx + 0.4*np.random.randn(grid_x)
    # x = 1.0*gx*gx + 0.1*np.random.randn(grid_x)

    # ..and corresponding target value y
    y = -20.*z + x*(6.*z+1.) + 0.01*np.random.randn(grid_x)
    y = -20.*z + x*(1.*z+1.) + 0.01*np.random.randn(grid_x)
    # y = -20.*z + x*(6.*z+1.) + 0.3*np.random.randn(grid_x)
    # y = 4.*z + x*(6.*z+1.) + 0.01*np.random.randn(grid_x)
    # y = -8*z + x*(6.*z) + 0.001*np.random.randn(grid_x)
    # y = -6.*z + 2.*x*(z-0.25) + 0.05*np.random.randn(grid_x)

    vecX = x.reshape(grid_x, 1)
    vecy = y.reshape(grid_x)
    vecz = z.reshape(grid_x)
    print vecX.shape
    print vecy.shape

    if plot_data:
        plt.figure(1)
        plt.plot(range(grid_x), vecy, 'or', alpha=0.5)
        plt.plot(range(grid_x), vecX, '.g', alpha=0.4)
        plt.plot(range(grid_x), 2.0*vecz-1.0, '-k', linewidth=2.0)

        plt.legend(['Labels', 'Inputs', 'Latent State'])
        plt.show()

    return vecX, vecy, vecz


def calc_error(truth, preds):
    # return root_mean_squared_error(truth, preds)
    return mean_absolute_error(truth, preds)
    # return r2_score(truth, preds)


def root_mean_squared_error(truth, preds):
    return np.sqrt(mean_squared_error(truth, preds))


def mean_abs_error(truth, preds):
    print '--------------'
    print preds.shape
    print '--------------'
    return np.sum(np.abs(truth-preds))


def single_run(vecX, vecy, vecz=None, states=2, plot=False):
    # generate training samples
    samples = vecX.shape[0]
    feats = vecX.shape[1]
    train_frac = 0.75
    inds = np.random.permutation(range(samples))
    train = inds[:np.floor(samples*train_frac)]
    test = inds[np.floor(samples*train_frac):]

    # OLS solution
    # vecX in (samples x dims)
    # vecy in (samples)
    # w in (dims)
    E = np.zeros((vecX.shape[1], vecX.shape[1]))
    np.fill_diagonal(E, 0.00001)
    XXt = vecX[train, :].T.dot(vecX[train, :]) + E
    XtY = (vecX[train, :].T.dot(vecy[train]))
    w = np.linalg.inv(XXt).dot(XtY.T)
    y_pred_rr = w.T.dot(vecX[test, :].T).T
    rr_abs = calc_error(vecy[test], y_pred_rr)
    rr_mse = root_mean_squared_error(vecy[test], y_pred_rr)

    # train ordinary support vector regression
    clf = SVR(C=1.0, epsilon=0.1, kernel='linear', shrinking=False)
    clf.fit(vecX[train, :], vecy[train])
    y_pred_svr = clf.predict(vecX[test, :])
    svr_abs = calc_error(vecy[test], y_pred_svr)
    svr_mse = root_mean_squared_error(vecy[test], y_pred_svr)

    train_mc = MulticlassRegressionModel(co.matrix(vecX[train, :].T), classes=states, y=co.matrix(vecy[train]))
    test_mc = MulticlassRegressionModel(co.matrix(vecX[test, :].T), classes=states)

    # train latent support vector regression
    lsvr = LatentRidgeRegression(train_mc, lam=0.00001, gam=1.0*float(len(train)))
    (_, train_lats) = lsvr.train_dc(max_iter=200)
    (y_pred_lrr, lats) = lsvr.apply(test_mc)
    y_pred_lrr = np.array(y_pred_lrr)[:, 0]
    lats = np.array(lats)
    lrr_abs = calc_error(vecy[test], y_pred_lrr)
    lrr_mse = root_mean_squared_error(vecy[test], y_pred_lrr)
    lrr_ars = 0.0
    if vecz is not None:
        lrr_ars = adjusted_rand_score(vecz[test], lats)

    # krr = LatentClusterRegression(cluster=states, l=0.00001, gamma=0.2*float(len(train)))
    # (_, _,) = krr.fit(co.matrix(vecX[train, :].T), co.matrix(vecy[train]), max_iter=200)
    # (y_pred_lrr, lats) = krr.predict(co.matrix(vecX[test, :].T))
    # y_pred_lrr = np.array(y_pred_lrr)[:, 0]
    # lats = np.array(lats)
    # lrr_mse = root_mean_squared_error(vecy[test], y_pred_lrr)
    # lrr_abs = calc_error(vecy[test], y_pred_lrr)
    # if vecz is not None:
    #     lrr_ars = adjusted_rand_score(vecz[test], lats)
    # lsvr = krr

    # KMEANS + RIDGE REGRESSION
    kmeans = cl.KMeans(n_clusters=states, init='random', n_init=10, max_iter=100, tol=0.0001)
    kmeans.fit(vecX[train, :])
    sol = np.zeros((states, feats))
    for i in range(states):
        inds = np.where(kmeans.labels_ == i)[0]
        ny = vecy[train[inds]].reshape(len(inds), 1)
        nX = vecX[train[inds], :].reshape(len(inds), feats)
        foo = lsvr.train_model(nX, ny)
        sol[i, :] = np.array(foo).reshape(1, feats)

    lbls = kmeans.predict(vecX[test, :])
    krr_ars = 0.0
    if vecz is not None:
        krr_ars = adjusted_rand_score(vecz[test], lbls)
    print lbls.shape
    print vecX[test, :].shape
    print sol[lbls, :].shape
    y_pred_krr = np.sum(sol[lbls, :] * vecX[test, :], axis=1)
    krr_abs = calc_error(vecy[test], y_pred_krr)
    krr_mse = root_mean_squared_error(vecy[test], y_pred_krr)

    print('------------------------------------------')
    print 'RR mse:        ', rr_mse
    print 'SVR mse:       ', svr_mse
    print 'KMeans mse:    ', krr_mse
    print 'LatentRR  mse: ', lrr_mse
    print('------------------------------------------')
    print 'LatentRR ars: ', lrr_ars
    print 'Kmeans   ars: ', krr_ars
    print('------------------------------------------')

    if plot:
        # sinds = np.argsort(test)
        plt.figure(1)
        # sinds = np.argsort(vecy[test])
        sinds = np.argsort(vecX[test, 0])
        truth = vecy[test[sinds]]

        # plt.subplot(1, 2, 1)
        # # plt.plot(range(test.size), lats[sinds]-2.0, '.k', linewidth=2.0)
        # plt.plot(range(test.size), vecX[test[sinds], 0], 'og', alpha=0.4, markersize=6.0)
        # plt.plot(range(test.size), truth, 'or', markersize=7.0, alpha=0.6)
        # plt.xlabel('Example')
        # plt.ylabel('Value')
        # plt.legend(['Inputs', 'Targets'], loc=4)

        # plt.subplot(1, 2, 1)
        # plt.plot(range(test.size), truth, 'or', markersize=10.0, alpha=0.6)
        # plt.plot(range(test.size), y_pred_rr[sinds], 'oc', alpha=0.6, markersize=6.0)
        # plt.plot(range(test.size), y_pred_svr[sinds], 'oy', alpha=0.6, markersize=6.0)
        # plt.plot(range(test.size), y_pred_krr[sinds], 'om', alpha=0.6, markersize=6.0)
        # plt.plot(range(test.size), y_pred_lrr[sinds], 'ob', alpha=0.6, markersize=6.0)
        # plt.xlabel('Examples', fontsize=20)
        # plt.ylabel('Targets', fontsize=20)
        # plt.legend(['Truth', 'RR', 'SVR', 'Kmeans+RR', 'LatentRR', 'True Labels'], loc=2, fontsize=18)

        # plt.subplot(1, 2, 2)
        plt.plot(vecX[test[sinds], 0], truth, 'or', alpha=0.6, markersize=10.0)
        plt.plot(vecX[test[sinds], 0], y_pred_rr[sinds], 'oc', alpha=0.6, markersize=6.0)
        plt.plot(vecX[test[sinds], 0], y_pred_svr[sinds], 'oy', alpha=0.6, markersize=6.0)
        plt.plot(vecX[test[sinds], 0], y_pred_krr[sinds], 'om', alpha=0.2, markersize=6.0)
        plt.plot(vecX[test[sinds], 0], y_pred_lrr[sinds], 'ob', alpha=0.2, markersize=6.0)
        # plt.plot(vecX[test[sinds], 0], lats[sinds], '.k', linewidth=2.0)
        plt.xlabel('Inputs', fontsize=20)
        plt.ylabel('Targets', fontsize=20)
        plt.legend(['Truth', 'RR', 'SVR', 'Kmeans+RR', 'LatentRR'], loc=1, fontsize=18)

        plt.show()

    return rr_abs, svr_abs, krr_abs, lrr_abs, rr_mse, svr_mse, krr_mse, lrr_mse, krr_ars, lrr_ars


if __name__ == '__main__':
    # (vecX, vecy) = load_svmlight_data('/home/nicococo/Data/housing_scale.dat')
    # (vecX, vecy) = load_svmlight_data('/home/nicococo/Data/space_ga_scale.dat')
    # (vecX, vecy) = load_svmlight_data('/home/nicococo/Data/mg_scale.dat')
    # (vecX, vecy) = load_svmlight_data('/home/nicococo/Data/mpg_scale.dat')
    # (vecX, vecy) = load_svmlight_data('/home/nicococo/Data/usps')
    # inds = np.where((vecy == 8.0) | (vecy == 1.0))[0]
    # inds = np.random.permutation(inds)
    # vecy = vecy[inds[:1000]]
    # vecX = vecX[inds[:1000], :]
    # (vecX, vecy) = load_svmlight_data('/home/nicococo/Data/YearPredictionMSD.t')
    # vecX = vecX[:8000, :]
    # vecy = vecy[:8000]
    (vecX, vecy, vecz) = get_1d_toy_data()

    # normalize data
    vecy = vecy-np.mean(vecy)
    vecX = vecX-np.mean(vecX)
    vecX /= np.max(vecX)
    vecy /= np.max(vecy)

    vecX = np.hstack((vecX, np.ones((vecX.shape[0], 1))))
    print '---------'
    print vecy.shape
    print vecX.shape
    print '---------'

    single_run(vecX, vecy, vecz, states=8, plot=True)
    single_run(vecX, vecy, vecz, states=2, plot=True)

    REPS = 20
    states = [1, 2, 4, 8, 12, 20, 30]
    # states = [1, 2, 4, 10]
    # states = [2]
    # states = [2, 10]
    mse = {}
    for s in range(len(states)):
        for n in range(REPS):
            if not mse.has_key(s):
                mse[s] = np.zeros((REPS, 10))
            cmse = mse[s]
            cmse[n, :] += single_run(vecX, vecy, vecz, states=states[s], plot=False)


    means = np.zeros((len(states), 10))
    stds = np.zeros((len(states), 10))
    print('RR, SVR, KRR, LRR, LRR-ARS, KRR-ARS')
    for key in mse.iterkeys():
        means[key, :] = np.mean(mse[key], axis=0)
        stds[key, :] = np.std(mse[key], axis=0)
        print states[key], ': ', np.mean(mse[key], axis=0)
        print '    ', np.std(mse[key], axis=0)


    plt.figure(1)
    # plt.subplot(1, 2, 1)
    plt.errorbar(states, means[:, 0], yerr=stds[:, 0], fmt='-c', elinewidth=1.0, linewidth=2.0, alpha=0.6)
    plt.errorbar(states, means[:, 1], yerr=stds[:, 1], fmt='-y', elinewidth=1.0, linewidth=2.0, alpha=0.6)
    plt.errorbar(states, means[:, 2], yerr=stds[:, 2], fmt='-m', elinewidth=1.0, linewidth=4.0, alpha=0.6)
    plt.errorbar(states, means[:, 3], yerr=stds[:, 3], fmt='-b', elinewidth=1.0, linewidth=4.0, alpha=0.6)
    plt.legend(['RR', 'SVR', 'Kmeans+RR', 'LatentRR'], loc=1, fontsize=18)
    plt.xlabel('Number of Latent States', fontsize=20)
    plt.ylabel('Mean Absolute Error', fontsize=20)

    # plt.subplot(1, 3, 2)
    # plt.errorbar(states, means[:, 4], yerr=stds[:, 4], fmt='-c', elinewidth=1.0, linewidth=2.0, alpha=0.6)
    # plt.errorbar(states, means[:, 5], yerr=stds[:, 5], fmt='-y', elinewidth=1.0, linewidth=2.0, alpha=0.6)
    # plt.errorbar(states, means[:, 6], yerr=stds[:, 6], fmt='-m', elinewidth=1.0, linewidth=4.0, alpha=0.6)
    # plt.errorbar(states, means[:, 7], yerr=stds[:, 7], fmt='-b', elinewidth=1.0, linewidth=4.0, alpha=0.6)
    # plt.legend(['RR', 'SVR', 'KRR', 'LRR'], loc=2)

    # plt.subplot(1, 2, 2)
    plt.figure(2)
    plt.errorbar(states, means[:, 8], yerr=stds[:, 8], fmt='-m', elinewidth=1.0, linewidth=4.0, alpha=0.6)
    plt.errorbar(states, means[:, 9], yerr=stds[:, 9], fmt='-b', elinewidth=1.0, linewidth=4.0, alpha=0.6)
    plt.legend(['Kmeans+RR', 'LatentRR'], loc=1, fontsize=18)
    plt.xlabel('Number of Latent States', fontsize=20)
    plt.ylabel('Adjusted Rand Index', fontsize=20)
    plt.show()

    # ..and stop
    print('Finish!')
