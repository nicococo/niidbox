import matplotlib
matplotlib.use('QT4Agg')
# change to type 1 fonts!
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

import numpy as np
import cvxopt as co

from sklearn.svm import SVR
from sklearn.metrics import median_absolute_error, mean_squared_error, r2_score, mean_absolute_error, \
    adjusted_rand_score
import sklearn.cluster as cl

from tcrf_regression import TransductiveCrfRegression
from tcrfr_indep_model import TCrfRIndepModel
#from tcrfr_pair_model import TCrfRPairwisePotentialModel


def fit_ridge_regression(lam, vecX, vecy):
    # solve the ridge regression problem
    E = np.zeros((vecX.shape[1], vecX.shape[1]))
    np.fill_diagonal(E, lam)
    XXt = vecX.T.dot(vecX) + E
    XtY = (vecX.T.dot(vecy))
    if XXt.size > 1:
        w = np.linalg.inv(XXt).dot(XtY)
    else:
        w = 1.0/XXt * XtY
    return co.matrix(w)


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


def method_transductive_regression(vecX, vecy, train, test, states=2, params=[0.0001, 0.8, 0.2]):
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
        foo = fit_ridge_regression(params[0], nX, ny)
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
        foo = fit_ridge_regression(params[0], nX, ny)
        sol[i, :] = np.array(foo).reshape(1, feats)
    lbls = kmeans.labels_[test]
    return 'Transductive k-means Ridge Regression', np.sum(sol[lbls, :] * vecX[test, :], axis=1), lbls


def method_flexmix(vecX, vecy, train, test, states=2, params=[200, 0.001], plot=False):
    # Use latent class regression FlexMix package from R
    import rpy2.robjects as robjects
    import pandas.rpy.common as com
    import pandas as pd
    r = robjects.r
    r.library("flexmix")

    feats = vecX.shape[1]-1
    train_data = np.hstack((vecX[train, 0:feats], vecy[train].reshape(-1, 1)))
    df_train = pd.DataFrame(train_data)
    colnames = []
    for i in range(feats):
        colnames.append(str(i))
    colnames.append('y')
    df_train.columns = colnames
    df_train_r = com.convert_to_r_dataframe(df_train)

    r('''
        parms = list(iter=''' + str(params[0]) + ''', tol=''' + str(params[1]) + ''',class="CEM")
        as(parms, "FLXcontrol")
    ''')
    model = r.flexmix(robjects.Formula("y ~ ."), data=df_train_r, k=states)

#    test_data = np.hstack((vecX[test, 0:feats], 1000.*np.random.randn(len(test)).reshape(-1, 1)))
    test_data = np.hstack((vecX[test, 0:feats], np.zeros(len(test)).reshape(-1, 1)))
    df_test = pd.DataFrame(test_data)
    colnames = []
    for i in range(feats):
        colnames.append(str(i))
    colnames.append('y')
    df_test.columns = colnames
    df_test_r = com.convert_to_r_dataframe(df_test)

    pr = r.predict(model, newdata=df_test_r, aggregate=False)
    df = com.convert_robj(pr)
    s = pd.Series(df)
    aux = s.values
    dim = aux.shape[0]
    y_pred = np.zeros((len(vecy[test]), dim))

    lats = np.zeros((vecy.shape[0], 1), dtype=int)
    lats[train] = np.array(r.clusters(model)).reshape(-1, 1)
    lats_pred = np.array(r.clusters(model, newdata=df_test_r)).reshape(-1, 1)
    lats[test] = lats_pred

    for i in range(dim):
        y_pred[:, i] = np.copy(aux[i]).reshape(1, -1)
    y_pred_flx = np.zeros(len(vecy[test]))
    for i in range(len(y_pred_flx)):
        y_pred_flx[i] = y_pred[i, lats_pred[i]-1]

    if plot:
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.plot(vecX[:, 0], vecy, '.g', alpha=0.1, markersize=10.0)
        plt.plot(vecX[test, 0], vecy[test], 'or', alpha=0.6, markersize=10.0)
        plt.plot(vecX[test, 0], y_pred_flx, 'oc', alpha=0.6, markersize=6.0)
        plt.plot(vecX[test, 0], lats[test], 'ob', alpha=0.6, markersize=6.0)
        plt.show()
    return 'FlexMix', np.array(y_pred_flx), np.reshape(lats_pred, newshape=lats_pred.size), lats


def method_tcrfr_indep(vecX, vecy, train, test, A, states=2, params=[0.9, 0.00001, 0.4], plot=False):
    model = TCrfRIndepModel(data=vecX.T, labels=vecy[train], label_inds=train, unlabeled_inds=test, states=states, A=A)
    tcrfr = TransductiveCrfRegression(reg_theta=params[0], reg_lambda=params[1], reg_gamma=params[2]*float(len(train)+len(test)))
    tcrfr.fit(model, max_iter=40)
    y_preds, lats = tcrfr.predict(model)

    print lats

    if plot:
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
    return 'TCRFR (Indep)', y_preds, lats, model.latent


def evaluate(truth, preds, true_lats, lats):
    """ Measure regression performance
    :return: list of error measures and corresponding names
    """
    names = list()
    errs = list()
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
    errs.append(adjusted_rand_score(true_lats, lats))
    names.append('Adjusted Rand Score')
    return np.array(errs), names
