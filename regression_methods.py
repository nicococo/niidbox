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
from sklearn.datasets import load_svmlight_file
import sklearn.cluster as cl

from latent_svr import LatentSVR
from latent_ridge_regression import LatentRidgeRegression, TransductiveLatentRidgeRegression
from multiclass_regression_model import MulticlassRegressionModel, TransductiveMulticlassRegressionModel
from latent_cluster_regression import LatentClusterRegression
from latent_ridge_regression import LatentRidgeRegression
from tcrf_regression import TransductiveCrfRegression
from tcrfr_indep_model import TCrfRIndepModel
#from tcrfr_pair_model import TCrfRPairwisePotentialModel

#import argparse, sys
#from gridmap import Job, process_jobs

import rpy2.robjects as robjects
import pandas.rpy.common as com
import pandas as pd




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


def method_tlrr(vecX, vecy, train, test, states=2, params=[0.5, 0.0001, 1.0]):
    transductive_mc = TransductiveMulticlassRegressionModel(co.matrix(vecX.T), classes=states, y=co.matrix(vecy),
                                                            lbl_idx=train, trans_idx=test)
    lsvr = TransductiveLatentRidgeRegression(theta=params[0], lam=params[1],
                                             gam=params[2] * float(len(train) + len(test)))
    (y_pred_lrr, lats) = lsvr.fit(transductive_mc, max_iter=50)
    return 'Transductive Latent Ridge Regression', np.array(y_pred_lrr)[:, 0], np.array(lats)[test]


def method_lrr(vecX, vecy, train, test, states=2, params=[0.5, 0.0001, 1.0]):
    train_mc = MulticlassRegressionModel(co.matrix(vecX[train, :].T), classes=states, y=co.matrix(vecy[train]))
    test_mc = MulticlassRegressionModel(co.matrix(vecX[test, :].T), classes=states)

    lsvr = LatentRidgeRegression(theta=params[0], lam=params[1], gam=params[2] * float(len(train)))
    (_, train_lats) = lsvr.fit(train_mc, max_iter=50)
    (y_pred_lrr, lats) = lsvr.predict(test_mc)
    return 'Latent Ridge Regression', np.array(y_pred_lrr)[:, 0], np.array(lats)


def method_flexmix(vecX, vecy, train, test, states=2, params=[200, 0.001]):
    # Use latent class regression FlexMix package from R
    R = robjects.r
    R.library("flexmix")

    feats = vecX.shape[1] - 1
    trainData = np.hstack((vecX[train, 0:feats], vecy[train].reshape(-1, 1)))
    testData = np.hstack((vecX[test, 0:feats], vecy[test].reshape(-1, 1)))
    df_train = pd.DataFrame(trainData)
    df_test = pd.DataFrame(testData)
    colNames = []
    for i in range(feats):
        colNames.append(str(i))
    colNames.append('y')
    df_train.columns = colNames
    df_test.columns = colNames
    df_train_r = com.convert_to_r_dataframe(df_train)
    df_test_r = com.convert_to_r_dataframe(df_test)

    R('''
        parms = list(iter=''' + str(params[0]) + ''', tol=''' + str(params[1]) + ''',class="CEM")
        as(parms, "FLXcontrol")
    ''')
    model = R.flexmix(robjects.Formula("y ~ ."), data=df_train_r, k=states)

    lats = np.zeros((vecy.shape[0], 1), dtype=int)
    lats[train] = np.array(R.clusters(model)).reshape(-1, 1)
    lats_pred = np.array(R.clusters(model, newdata=df_test_r)).reshape(-1, 1)
    lats[test] = lats_pred
    lats = np.concatenate(lats)

    pr = R.predict(model, newdata=df_test_r, aggregate=False)
    df = com.convert_robj(pr)
    s = pd.Series(df)
    aux = s.values
    dim = aux.shape[0]
    y_pred = np.zeros((len(vecy[test]), dim))

    for i in range(dim):
        y_pred[:, i] = np.copy(aux[i]).reshape(1, -1)
    y_pred_flx = np.zeros(len(vecy[test]))
    for i in range(len(y_pred_flx)):
        y_pred_flx[i] = y_pred[i, lats_pred[i] - 1]
    return 'FlexMix', np.array(y_pred_flx), np.array(lats[test])

def method_tcrfr_indep(vecX, vecy, train, test, A, states=2, params=[0.9, 0.00001, 0.4], plot=False):
#    A = np.zeros((vecX.shape[0], vecX.shape[0]))
#    # A = sparse.lil_matrix((vecX.shape[0], vecX.shape[0]))
#    for i in range(vecX.shape[0]-4):
#        A[i, i+1] = 1
#        A[i+1, i] = 1
#        A[i, i+2] = 1
#        A[i+2, i] = 1
#        A[i, i+3] = 1
#        A[i+3, i] = 1
#        A[i, i+4] = 1
#        A[i+4, i] = 1

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



def single_run(methods, vecX, vecy, vecz=None, train_frac=0.05, states=2, plot=False):
    # generate training samples
    samples = vecX.shape[0]
    inds = np.random.permutation(range(samples))
    train = inds[:np.floor(samples * train_frac)]
    test = inds[np.floor(samples * train_frac):]

    # normalize data
    vecy = vecy - np.mean(vecy[train])
    vecX = vecX - np.mean(vecX[train, :])
    # vecX /= np.max(vecX)
    vecy /= np.max(vecy[train])
    vecX = np.hstack((vecX, np.ones((vecX.shape[0], 1))))

    names = []
    res = []
    for m in methods:
        (name, abbrv, pred, lats) = m(vecX, vecy, train, test, states=states)
        res.append(measure_regression_performance(vecy[test], pred))
        names.append(name)

    print('------------------------------------------')
    print 'Total data           :', len(train) + len(test)
    print 'Labeled data (train) :', len(train)
    print 'Unlabeled data (test):', len(test)
    print 'Fraction             :', train_frac
    print 'Max States           :', states
    print('------------------------------------------')
    print ''.ljust(45), ': ', res[0][1]
    for m in range(len(names)):
        print names[m].ljust(45), ': ', res[m][0]
    print('------------------------------------------')
