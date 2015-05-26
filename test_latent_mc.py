import numpy as np
import matplotlib.pyplot as plt
import cvxopt as co

from sklearn.svm import SVR
from sklearn.metrics import median_absolute_error, mean_squared_error, r2_score, mean_absolute_error, adjusted_rand_score
from sklearn.datasets import load_svmlight_file
import sklearn.cluster as cl

from latent_svr import LatentSVR
from latent_ridge_regression import LatentRidgeRegression, TransductiveLatentRidgeRegression
from multiclass_regression_model import MulticlassRegressionModel, TransductiveMulticlassRegressionModel
from latent_cluster_regression import LatentClusterRegression
from tcrf_regression import TransductiveCrfRegression
from tcrfr_indep_model import TCrfRIndepModel

def load_svmlight_data(fname):
    (X, y) = load_svmlight_file(fname)
    print y.shape
    print X.shape
    return np.array(X.todense()), y.reshape(X.shape[0])


def get_1d_toy_data(plot_data=False):
    grid_x = 1000
    # generate 2D grid
    gx = np.linspace(0, 1, grid_x)

    # latent variable z
    z = np.zeros(grid_x, dtype=np.uint8)
    # zx = np.where((gx > 0.2) & (gx < 0.6))[0]
    # z[zx] = 2
    zx = np.where(gx > 0.6)[0]
    z[zx] = 1

    # inputs  x
    # x = 4.0*np.sign(z-0.4)*gx + 2.0*np.sign(z-0.4) + 0.5*np.random.randn(grid_x)
    x = 4.0*np.sign(z-0.5)*gx + 0.5*(z+1.)*np.random.rand(grid_x)
    # x = 4.0*np.sign(z-0.5)*gx + 1.0*np.sign(z-0.5) + 0.8*np.random.randn(grid_x)
    # x = 8.0*gx + 0.4*np.random.randn(grid_xn)
    x = 1.0*gx*gx + 0.1*np.random.randn(grid_x)

    # ..and corresponding target value y
    # y = -20.*z + x*(6.*z+1.) + 0.01*np.random.randn(grid_x)
    y = -20.*z + x*(1.*z+1.) + 0.1*np.random.randn(grid_x)
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


def get_1d_toy_data_1(n=600, lats=4, dims=10, plot_data=True):
    nl = np.floor(n/lats)
    vecX = 0.5*np.random.rand(nl, dims)
    vecz = np.zeros(n)
    for l in range(lats-1):
        vecX = np.concatenate([vecX, 0.5*np.random.randn(nl, dims)+(l+1.0)*2.0])
        vecz[(l+1.0)*nl:(l+1.0)*nl+nl] = l + 1.0
    print vecX.shape
    print vecz.shape
    # ..and corresponding target value y
    vecy = -0.2*(vecz+1.0) + 0.1*np.random.randn(n)
    for d in range(dims):
        # vecy += vecX[:, d]*(4.0*np.random.randn()+(vecz+1.0)+np.random.randn())
        vecy += 0.1*vecX[:, d]*(vecz-0.5)*np.random.randn()

    print vecX.shape
    print vecy.shape

    if plot_data:
        plt.figure(1)
        plt.plot(vecX[:, 0], vecy, '.g', alpha=0.4)
        plt.show()

    return vecX, vecy, vecz


def measure_regression_performance(truth, preds):
    """ Measure regression performance
    :param truth: true values
    :param preds: predictions
    :return: list of error measures and corresponding names
    """
    names = []
    errs = []

    errs.append(mean_absolute_error(truth, preds))
    names.append('Mean Absolute Error')

    errs.append(mean_squared_error(truth, preds))
    names.append('Mean Squared Error')

    errs.append(np.sqrt(mean_squared_error(truth, preds)))
    names.append('Root Mean Squared Error')

    errs.append(median_absolute_error(truth, preds))
    names.append('Median Absolute Error')

    errs.append(r2_score(truth, preds))
    names.append('R2 Score')
    return errs, names


def method_transductive_regression(vecX, vecy, train, test, states=2, params=[0.0001, 0.8, 0.2]):
    warning('This method does change the inputs.')
    # OLS solution
    # vecX in (samples x dims)
    # vecy in (samples)
    # w in (dims)

    # Stage 1: locally estimate the labels of the test samples
    E = np.zeros((vecX.shape[1], vecX.shape[1]))
    np.fill_diagonal(E, params[0])
    XXt = vecX[train, :].T.dot(vecX[train, :]) + E
    XtY = (vecX[train, :].T.dot(vecy[train]))
    w = np.linalg.inv(XXt).dot(XtY.T)
    vecy[test] = w.T.dot(vecX[test, :].T)

    # Stage 2: perform global optimization with train + test samples
    C1 = params[1]
    C2 = params[2]
    I = np.identity(vecX.shape[1])
    XXt = I + C1*(vecX[train, :].T.dot(vecX[train, :])) + C2*(vecX[test, :].T.dot(vecX[test, :]))
    XtY = C1*(vecX[train, :].T.dot(vecy[train])) + C2*(vecX[test, :].T.dot(vecy[test]))
    w = np.linalg.inv(XXt).dot(XtY.T)
    return 'Transductive Regression', w.T.dot(vecX[test, :].T).T, np.ones(len(test))


def method_ridge_regression(vecX, vecy, train, test, states=2, params=[0.0001]):
    # OLS solution
    # vecX in (samples x dims)
    # vecy in (samples)
    # w in (dims)
    E = np.zeros((vecX.shape[1], vecX.shape[1]))
    np.fill_diagonal(E, params)
    XXt = vecX[train, :].T.dot(vecX[train, :]) + E
    XtY = (vecX[train, :].T.dot(vecy[train]))
    w = np.linalg.inv(XXt).dot(XtY.T)
    return 'Ridge Regression', w.T.dot(vecX[test, :].T).T, np.ones(len(test))


def method_svr(vecX, vecy, train, test, states=2, params=[1.0, 0.1, 'linear']):
    # train ordinary support vector regression
    clf = SVR(C=params[0], epsilon=params[1], kernel=params[2], shrinking=False)
    clf.fit(vecX[train, :], vecy[train])
    return 'Support Vector Regression', clf.predict(vecX[test, :]), np.ones(len(test))


def method_krr(vecX, vecy, train, test, states=2, params=[0.0001]):
    feats = vecX.shape[1]
    kmeans = cl.KMeans(n_clusters=states, init='random', n_init=10, max_iter=100, tol=0.0001)
    kmeans.fit(vecX[train, :])
    sol = np.zeros((states, feats))
    for i in range(states):
        inds = np.where(kmeans.labels_ == i)[0]
        ny = vecy[train[inds]].reshape(len(inds), 1)
        nX = vecX[train[inds], :].reshape(len(inds), feats)
        lrr = LatentRidgeRegression(1.0, params[0])
        foo = lrr.train_model(nX, ny)
        sol[i, :] = np.array(foo).reshape(1, feats)
    lbls = kmeans.predict(vecX[test, :])
    return 'k-means Ridge Regression', np.sum(sol[lbls, :] * vecX[test, :], axis=1), lbls


def method_tkrr(vecX, vecy, train, test, states=2, params=[0.0001]):
    feats = vecX.shape[1]
    kmeans = cl.KMeans(n_clusters=states, init='random', n_init=10, max_iter=100, tol=0.0001)
    kmeans.fit(vecX)
    sol = np.zeros((states, feats))
    for i in range(states):
        sinds = np.where(kmeans.labels_ == i)[0]
        inds = []
        for j in sinds:
            if j in train:
                inds.append(j)
        ny = vecy[inds].reshape(len(inds), 1)
        nX = vecX[inds, :].reshape(len(inds), feats)

        lrr = LatentRidgeRegression(1.0, params[0])
        foo = lrr.train_model(nX, ny)
        sol[i, :] = np.array(foo).reshape(1, feats)
    lbls = kmeans.labels_[test]
    return 'Transductive k-means Ridge Regression', np.sum(sol[lbls, :] * vecX[test, :], axis=1), lbls


def method_tlrr(vecX, vecy, train, test, states=2, params=[0.5, 0.00001, 0.5]):
    transductive_mc = TransductiveMulticlassRegressionModel(co.matrix(vecX.T), classes=states, y=co.matrix(vecy), lbl_idx=train, trans_idx=test)
    lsvr = TransductiveLatentRidgeRegression(theta=params[0], lam=params[1], gam=params[2]*float(len(train)+len(test)))
    (y_pred_lrr, lats) = lsvr.fit(transductive_mc, max_iter=50)
    return 'Transductive Latent Ridge Regression', np.array(y_pred_lrr)[:, 0], np.array(lats)[test]


def method_lrr(vecX, vecy, train, test, states=2, params=[0.5, 0.00001, 0.5]):
    train_mc = MulticlassRegressionModel(co.matrix(vecX[train, :].T), classes=states, y=co.matrix(vecy[train]))
    test_mc = MulticlassRegressionModel(co.matrix(vecX[test, :].T), classes=states)

    lsvr = LatentRidgeRegression(theta=params[0], lam=params[1], gam=params[2]*float(len(train)))
    (_, train_lats) = lsvr.fit(train_mc, max_iter=50)
    (y_pred_lrr, lats) = lsvr.predict(test_mc)
    return 'Latent Ridge Regression', np.array(y_pred_lrr)[:, 0], np.array(lats)


def method_tcrfr(vecX, vecy, train, test, states=2, params=[0.5, 0.00001, 1.0]):
    model = TCrfRIndepModel(data=vecX.T, labels=vecy[train], label_inds=train, unlabeled_inds=test, states=states)
    tcrfr = TransductiveCrfRegression(reg_theta=params[0], reg_lambda=params[1], reg_gamma=params[2]*float(len(train)+len(test)))
    # tcrfr = TransductiveCrfRegression(reg_theta=params[0], reg_lambda=params[1], reg_gamma=params[2])
    tcrfr.fit(model, max_iter=50)
    y_preds, lats = tcrfr.predict(model)
    print lats

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.plot(vecX[:, 0], vecy, '.g', alpha=0.1, markersize=10.0)
    plt.plot(vecX[test, 0], vecy[test], 'or', alpha=0.6, markersize=10.0)
    plt.plot(vecX[test, 0], y_preds, 'oc', alpha=0.6, markersize=6.0)
    plt.plot(vecX[test, 0], lats, 'ob', alpha=0.6, markersize=6.0)

    plt.subplot(1, 2, 2)
    plt.plot(vecX[train, 0], vecy[train], 'or', alpha=0.6, markersize=10.0)
    plt.plot(vecX[train, 0], model.latent[train], 'ob', alpha=0.6, markersize=6.0)
    ytrain = model.get_labeled_predictions(tcrfr.u)
    plt.plot(vecX[train, 0], ytrain, 'xg', alpha=0.8, markersize=10.0)

    plt.show()
    return 'Transductive CRF Regression', y_preds, lats


def method_flexmix(vecX, vecy, train, test, states=2, params=[]):
    # Use latent class regression FlexMix package from R
    import rpy2.robjects as robjects
    import pandas.rpy.common as com
    import pandas as pd
    r = robjects.r
    r.library("flexmix")

    feats = vecX.shape[1]-1
    train_data = np.hstack((vecX[train, 0:feats], vecy[train].reshape(-1, 1)))
    test_data = np.hstack((vecX[test, 0:feats], vecy[test].reshape(-1, 1)))
    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)
    colnames = []
    for i in range(feats):
        colnames.append(str(i))
    colnames.append('y')
    df_train.columns = colnames
    df_test.columns = colnames
    df_train_r = com.convert_to_r_dataframe(df_train)
    df_test_r = com.convert_to_r_dataframe(df_test)

    model = r.flexmix(robjects.Formula("y ~ ."), data=df_train_r, k=states)

    lats = np.zeros((vecy.shape[0], 1), dtype=int)
    lats[train] = np.array(r.clusters(model)).reshape(-1, 1)
    lats_pred = np.array(r.clusters(model, newdata=df_test_r)).reshape(-1, 1)
    lats[test] = lats_pred
    lats = np.concatenate(lats)

    pr = r.predict(model, newdata=df_test_r)
    df = com.convert_robj(pr)
    s = pd.Series(df)
    aux = s.values
    dim = aux.shape[0]
    y_pred = np.zeros((len(vecy[test]), dim))

    for i in range(dim):
        y_pred[:, i] = np.copy(aux[i]).reshape(1, -1)
    y_pred_flx = np.zeros(len(vecy[test]))
    for i in range(len(y_pred_flx)):
        y_pred_flx[i] = y_pred[i, lats_pred[i]-1]

    return 'FlexMix', np.array(y_pred_flx), np.array(lats[test])



def single_run(methods, vecX, vecy, vecz=None, train_frac=0.05, states=2, plot=False):
    # generate training samples
    samples = vecX.shape[0]
    inds = np.random.permutation(range(samples))
    train = inds[:np.floor(samples*train_frac)]
    test = inds[np.floor(samples*train_frac):]

    # normalize data
    vecy = vecy-np.mean(vecy[train])
    vecX = vecX-np.mean(vecX[train, :])
    vecX /= np.max(np.abs(vecX[train, :]))
    # vecX *= 4.
    vecy /= np.max(np.abs(vecy[train]))
    # vecy *= 10.

    # print '-----------'
    # print np.max(vecX[train, :])
    # print np.min(vecX[train, :])
    # print '-----------'

    vecX = np.hstack((vecX, np.ones((vecX.shape[0], 1))))

    names = []
    res = []
    for m in methods:
        (name, pred, lats) = m(vecX, vecy, train, test, states=states)
        names.append(name)
        print name
        res.append(measure_regression_performance(vecy[test], pred))

    print('------------------------------------------')
    print 'Total data           :', len(train)+len(test)
    print 'Labeled data (train) :', len(train)
    print 'Unlabeled data (test):', len(test)
    print 'Fraction             :', train_frac
    print 'Max States           :', states
    print('------------------------------------------')
    print ''.ljust(45), ': ', res[0][1]
    for m in range(len(names)):
        print names[m].ljust(45), ': ', res[m][0]
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

    return names, res


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


    methods = [method_ridge_regression, method_svr, method_krr, method_tkrr, method_flexmix,
               method_transductive_regression, method_lrr, method_tlrr]
    methods = [method_ridge_regression, method_svr, method_krr, method_tkrr,
               method_transductive_regression, method_lrr, method_tlrr, method_tcrfr]
    methods = [method_ridge_regression, method_svr, method_krr, method_tkrr,
               method_lrr, method_tlrr, method_tcrfr]
    methods = [method_ridge_regression, method_tcrfr]

    single_run(methods, vecX, vecy, vecz, train_frac=0.15, states=8, plot=False)
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
