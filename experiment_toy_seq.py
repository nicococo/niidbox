import numpy as np
import cvxopt as co
import scipy.sparse as sparse

from sklearn.svm import SVR
from sklearn.metrics import median_absolute_error, \
    mean_squared_error, r2_score, mean_absolute_error, adjusted_rand_score
import sklearn.cluster as cl

from tcrfr_qp import TCRFR_QP
from tcrfr_fast import TCRFR_Fast

from tcrf_regression import TransductiveCrfRegression
from tcrfr_indep_model import TCrfRIndepModel
from tcrfr_pair_model import TCrfRPairwisePotentialModel


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


def get_1d_toy_data(num_exms=300, plot=False):
    grid_x = num_exms
    # generate 2D grid
    gx = np.linspace(0, 1, grid_x)
    # latent variable z
    z = np.zeros(grid_x, dtype=np.uint8)
    zx = np.where((gx > 0.3) & (gx < 0.75))[0]
    z[zx] = 2
    zx = np.where(gx >= 0.75)[0]
    z[zx] = 1

    # inputs  x
    x = 1.2*np.sign(z-0.5)*gx + 0.4*np.random.randn(grid_x)

    # ..and corresponding target value y
    y = 4.*z + x*(6.*z+1.) + 0.01*np.random.randn(grid_x)
    y = 4.*z + x*(6.*z+1.) + 0.1*np.random.randn(grid_x)

    vecX = x.reshape(grid_x, 1)
    vecy = y.reshape(grid_x)
    vecz = z.reshape(grid_x)
    print vecX.shape
    print vecy.shape
    if plot:
        means = np.ones(vecy.size)
        means[vecz==0] = np.mean(vecX[vecz==0], axis=0)
        means[vecz==1] = np.mean(vecX[vecz==1], axis=0)
        means[vecz==2] = np.mean(vecX[vecz==2], axis=0)

        samples = vecy.size
        inds = np.random.permutation(range(samples))
        train_frac = 0.2
        train_nums = np.floor(samples*train_frac)
        train = inds[:train_nums]

        plt.figure(1)
        plt.subplot(1, 2, 1)

        plt.plot(range(grid_x), vecX, '-ok', alpha=0.4)
        plt.plot(np.array(range(grid_x))[train], vecX[train, :], 'or', alpha=0.8)
        plt.plot(range(grid_x), means, '-k', linewidth=2.0)
        pos1 = np.where(vecz[1:]-vecz[:-1] > 0)[0]
        print pos1
        plt.fill_between(range(pos1[0], pos1[1]+2), 2.4*np.max(vecX), 2.2*np.min(vecX), alpha=0.2)
        plt.text(10., -2.5, 'State 1', fontsize=14)
        plt.text(45., -2.5, 'State 2', fontsize=14)
        plt.text(80., -2.5, 'State 3', fontsize=14)
        plt.legend(['Observations w/ Neighborhood Connection', 'Observations w/ Assigned Regression Target',
                    'Latent State Mean Input Values'], loc=2, fontsize=14, fancybox=True, framealpha=0.7)
        plt.ylim([-3.05, +4.55])
        plt.xlabel('Time', fontsize=18)
        plt.ylabel('Observations', fontsize=18)

        plt.subplot(1, 2, 2)
        for s in range(samples-1):
            if np.linalg.norm([vecy[s]-vecy[s+1]]) > 1.2:
                plt.plot([vecX[s, :], vecX[s+1,:]], [vecy[s], vecy[s+1]], '-k', alpha=0.1)
            else:
                plt.plot([vecX[s, :], vecX[s+1,:]], [vecy[s], vecy[s+1]], '-k', alpha=0.3)

        plt.plot(vecX, vecy, 'ok', alpha=0.4)
        plt.plot(vecX[train, :], vecy[train], 'or', alpha=0.8)
        # plt.ylim([-2.05, +4.05])
        plt.xlabel('Observations', fontsize=18)
        plt.ylabel('Regression Targets', fontsize=18)
        # plt.legend(['Input Data w/ Neighborhood Connection', 'Input Data w/ Assigned Regression Target', 'Latent State Mean Input Values'], loc=2, fontsize=14)
        # plt.legend(['Neighborhood Connection', 'Unlabeled Input Data', 'Labeled Data'], loc=2, fontsize=14)

        plt.show()
    return vecX, vecy, vecz


def evaluate(truth, preds, true_lats, lats, time):
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
    errs.append(time)
    names.append('Runtime')

    return np.array(errs), names


def method_transductive_regression(vecX, vecy, train, test, states=2, params=[0.0001, 0.7, 0.3], true_latent=None, plot=False):
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

    bak = vecy[test].copy()
    vecy[test] = w.T.dot(vecX[test, :].T)
    # Stage 2: perform global optimization with train + test samples
    C1 = params[1]
    C2 = params[2]
    I = np.identity(vecX.shape[1])
    XXt = I + C1*(vecX[train, :].T.dot(vecX[train, :])) + C2*(vecX[test, :].T.dot(vecX[test, :]))
    XtY = C1*(vecX[train, :].T.dot(vecy[train])) + C2*(vecX[test, :].T.dot(vecy[test]))
    w = np.linalg.inv(XXt).dot(XtY.T)
    vecy[test] = bak
    return 'Transductive Regression', w.T.dot(vecX[test, :].T).T, np.ones(len(test))


def method_tcrfr(vecX, vecy, train, test, states=2, params=[0.9, 0.00001, 0.5], true_latent=None, plot=False):
    # model = TCrfRIndepModel(data=vecX.T, labels=vecy[train], label_inds=train, unlabeled_inds=test, states=states)
    A = np.zeros((vecX.shape[0], vecX.shape[0]))
    for i in range(vecX.shape[0]-1):
        A[i, i+1] = 1
        A[i+1, i] = 1
    # for i in range(vecX.shape[0]-2):
    #     A[i, i+2] = 1
    #     A[i+2, i] = 1
    # for i in range(vecX.shape[0]-3):
    #     A[i, i+3] = 1
    #     A[i+3, i] = 1
    # for i in range(vecX.shape[0]-4):
    #     A[i, i+4] = 1
    #     A[i+4, i] = 1
    for i in range(vecX.shape[0]-4):
        if i in train:
            A[i, i+1] = 1
            A[i+1, i] = 1
            A[i, i+2] = 1
            A[i+2, i] = 1
            A[i, i+3] = 1
            A[i+3, i] = 1
            A[i, i+4] = 1
            A[i+4, i] = 1

        # A[i+1, i] = 1
    model = TCrfRPairwisePotentialModel(data=vecX.T, labels=vecy[train], label_inds=train, unlabeled_inds=test, states=states, A=A)
    # model.test_qp_param()

    tcrfr = TransductiveCrfRegression(reg_theta=params[0], reg_lambda=params[1], reg_gamma=params[2]*float(len(train)+len(test)))
    tcrfr.fit(model, max_iter=40, n_init=1, use_grads=False)
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

        print('Test performance: ')
        print evaluate(vecy[test], y_preds, true_latent[test], lats)
        print('Training performance: ')
        print evaluate(vecy[train], ytrain, true_latent[train], model.latent[train])

        plt.show()

    return 'TCRFR (Pairwise Potentials)', y_preds, lats


def method_tcrfr_qp(vecX, vecy, train, test, states=2, params=[0.9, 0.00001, 0.5, 10], true_latent=None, plot=False):
    # model = TCrfRIndepModel(data=vecX.T, labels=vecy[train], label_inds=train, unlabeled_inds=test, states=states)
    A = co.spmatrix(0.0, range(vecX.shape[0]), range(vecX.shape[0]))
    for i in range(vecX.shape[0]-1):
        A[i, i+1] = 1
        A[i+1, i] = 1
    for k in range(1,  params[3]):
        for i in range(vecX.shape[0]-k):
            if i in train or i+k in train:
                A[i, i+k] = 1
                A[i+k, i] = 1

    tcrfr = TCRFR_QP(data=vecX.T, labels=vecy[train], label_inds=train, unlabeled_inds=test, states=states, A=A,
                  reg_theta=params[0], reg_lambda=params[1], reg_gamma=params[2]*float(len(train)+len(test)),
                  trans_regs=[.05, 0.5], trans_sym=[0])

    tcrfr.fit(max_iter=20, use_grads=False)
    y_preds, lats = tcrfr.predict()
    print lats

    if plot:
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.plot(vecX[:, 0], vecy, '.g', alpha=0.1, markersize=10.0)
        plt.plot(vecX[test, 0], vecy[test], 'or', alpha=0.6, markersize=10.0)
        plt.plot(vecX[test, 0], y_preds[test], 'oc', alpha=0.6, markersize=6.0)
        plt.plot(vecX[test, 0], lats[test], 'ob', alpha=0.6, markersize=6.0)

        plt.subplot(1, 2, 2)
        plt.plot(vecX[train, 0], vecy[train], 'or', alpha=0.6, markersize=10.0)
        plt.plot(vecX[train, 0], lats[train], 'ob', alpha=0.6, markersize=6.0)
        plt.plot(vecX[train, 0], y_preds[train], 'xg', alpha=0.8, markersize=10.0)

        print('Test performance: ')
        print evaluate(vecy[test], y_preds[test], true_latent[test], lats[test])
        print('Training performance: ')
        print evaluate(vecy[train], y_preds[train], true_latent[train], lats[train])

        plt.show()

    return 'TCRFR-QP', y_preds[test], lats[test]


def method_tcrfr_pl(vecX, vecy, train, test, states=2, params=[0.9, 0.00001, 0.5, 10], true_latent=None, plot=False):
    A = co.spmatrix(0.0, range(vecX.shape[0]), range(vecX.shape[0]))
    for i in range(vecX.shape[0]-1):
        A[i, i+1] = 1
        A[i+1, i] = 1
    for k in range(1, params[3]):
        for i in range(vecX.shape[0]-k):
            if i in train or i+k in train:
                A[i, i+k] = 1
                A[i+k, i] = 1

    tcrfr = TCRFR_Fast(data=vecX.T, labels=vecy[train], label_inds=train, unlabeled_inds=test, states=states, A=A,
                  reg_theta=params[0], reg_lambda=params[1], reg_gamma=params[2]*float(len(train)+len(test)),
                  trans_regs=[.05, 0.5], trans_sym=[0], lbl_weight=1.0)

    tcrfr.fit(max_iter=20, use_grads=False)
    y_preds, lats = tcrfr.predict()
    print lats

    if plot:
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.plot(vecX[:, 0], vecy, '.g', alpha=0.1, markersize=10.0)
        plt.plot(vecX[test, 0], vecy[test], 'or', alpha=0.6, markersize=10.0)
        plt.plot(vecX[test, 0], y_preds[test], 'oc', alpha=0.6, markersize=6.0)
        plt.plot(vecX[test, 0], lats[test], 'ob', alpha=0.6, markersize=6.0)

        plt.subplot(1, 2, 2)
        plt.plot(vecX[train, 0], vecy[train], 'or', alpha=0.6, markersize=10.0)
        plt.plot(vecX[train, 0], lats[train], 'ob', alpha=0.6, markersize=6.0)
        plt.plot(vecX[train, 0], y_preds[train], 'xg', alpha=0.8, markersize=10.0)

        print('Test performance: ')
        print evaluate(vecy[test], y_preds[test], true_latent[test], lats[test])
        print('Training performance: ')
        print evaluate(vecy[train], y_preds[train], true_latent[train], lats[train])

        plt.show()

    return 'TCRFR-PL', y_preds[test], lats[test]


def method_rr(vecX, vecy, train, test, states=2, params=[0.0001], true_latent=None, plot=False):
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


def method_lb(vecX, vecy, train, test, states=2, params=[0.0001], true_latent=None, plot=False):
    # ridge regression lower bound by using ground truth latent state information

    preds = np.zeros(len(test))
    for s in range(states):
        train_inds = np.where(true_latent[train] == s)[0]
        test_inds = np.where(true_latent[test] == s)[0]
        if train_inds.size >= 2:
            E = np.zeros((vecX.shape[1], vecX.shape[1]))
            np.fill_diagonal(E, params)
            XXt = vecX[train[train_inds], :].T.dot(vecX[train[train_inds], :]) + E
            XtY = (vecX[train[train_inds], :].T.dot(vecy[train[train_inds]]))
            w = np.linalg.inv(XXt).dot(XtY.T)
            preds[test_inds] = w.T.dot(vecX[test[test_inds], :].T).T

    return 'Lower Bound', preds, np.ones(len(test))


def method_svr(vecX, vecy, train, test, states=2, params=[1.0, 0.1, 'linear'], true_latent=None, plot=False):
    # train ordinary support vector regression
    if len(params) == 3:
        clf = SVR(C=params[0], epsilon=params[1], kernel=params[2], shrinking=False)
    else:
        clf = SVR(C=params[0], epsilon=params[1], kernel='linear', shrinking=False)
    clf.fit(vecX[train, :], vecy[train])
    return 'Support Vector Regression', clf.predict(vecX[test, :]), np.ones(len(test))


def method_krr(vecX, vecy, train, test, states=2, params=[0.0001], true_latent=None, plot=False):
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
    return 'K-means + Ridge Regression', np.sum(sol[lbls, :] * vecX[test, :], axis=1), lbls


def method_tcrfr_indep(vecX, vecy, train, test, states=2, params=[0.9, 0.00001, 0.4, 100.], true_latent=None, plot=False):
    # A = np.zeros((vecX.shape[0], vecX.shape[0]))
    A = sparse.lil_matrix((vecX.shape[0], vecX.shape[0]))
    for i in range(vecX.shape[0]-4):
        A[i, i+1] = 1
        A[i+1, i] = 1
        A[i, i+2] = 1
        A[i+2, i] = 1
        A[i, i+3] = 1
        A[i+3, i] = 1
        A[i, i+4] = 1
        A[i+4, i] = 1
    print params
    model = TCrfRIndepModel(data=vecX.T, labels=vecy[train], label_inds=train,
                            unlabeled_inds=test, states=states, A=A, lbl_neighbor_gain=params[3])
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

        print('Test performance: ')
        print evaluate(vecy[test], y_preds, true_latent[test], lats)
        print('Training performance: ')
        print evaluate(vecy[train], ytrain, true_latent[train], model.latent[train])

        plt.show()
    return 'TCRFR (Indep)', y_preds, lats


def method_flexmix(vecX, vecy, train, test, states=2, params=[200, 0.001], true_latent=None, plot=False):
    my_const = 20.0
    new_vecy = vecy.copy() / my_const

    if len(train)<8*states:
        return 'FlexMix', np.zeros(len(test)), np.zeros(len(test))

    # Use the mixture of regressions FlexMix package from R
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    import pandas as pd

    r = robjects.r
    r.library("flexmix")

    feats = vecX.shape[1]-1
    train_data = np.hstack((vecX[train, 0:feats], new_vecy[train].reshape(-1, 1)))
    df_train = pd.DataFrame(train_data)
    colnames = []
    for i in range(feats):
        colnames.append(str(i))
    colnames.append('y')
    df_train.columns = colnames

    df_test = pd.DataFrame(vecX[test, 0:feats])
    colnames = []
    for i in range(feats):
        colnames.append(str(i))
    df_test.columns = colnames

    # Fit the model
    r('''
        rparms = list(iter=''' + str(params[0]) + ''', tol=''' + str(params[1]) + ''',class="CEM")
        as(rparms, "FLXcontrol")
    ''')

    model = r.flexmix(robjects.Formula("y ~ ."), data=df_train, k=states)

    # Predict
    pr = r.predict(model, newdata=df_test, aggregate=True)
    y_pred = r.unlist(pr)
    y_pred_flx = np.array(y_pred).reshape(-1,1)

    lats = np.zeros((new_vecy.shape[0], 1), dtype=int)
    lats[train] = np.array(r.clusters(model)).reshape(-1, 1)
    lats_pred = np.zeros(len(new_vecy[test])).reshape(-1, 1)
    # sys.stdout.flush()
    lats[test] = lats_pred

    if plot:
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.plot(vecX[:, 0], new_vecy, '.g', alpha=0.1, markersize=10.0)
        plt.plot(vecX[test, 0], new_vecy[test], 'or', alpha=0.6, markersize=10.0)
        plt.plot(vecX[test, 0], y_pred_flx, 'oc', alpha=0.6, markersize=6.0)
        plt.plot(vecX[test, 0], lats[test], 'ob', alpha=0.6, markersize=6.0)
        plt.show()

    return 'FlexMix', y_pred_flx*my_const, np.reshape(lats_pred, newshape=lats_pred.size)


def main_run(methods, params, vecX, vecy, vecz, train_frac, val_frac, states, plot):
    import numpy as np
    import time

    from sklearn.svm import SVR
    from sklearn.metrics import median_absolute_error, mean_squared_error, r2_score, mean_absolute_error, adjusted_rand_score
    from sklearn.datasets import load_svmlight_file
    import sklearn.cluster as cl

    from tcrf_regression import TransductiveCrfRegression
    from tcrfr_indep_model import TCrfRIndepModel
    from tcrfr_pair_model import TCrfRPairwisePotentialModel

    # generate training samples
    samples = vecX.shape[0]
    inds = np.random.permutation(range(samples))
    val_nums = np.floor(samples*train_frac*val_frac)
    train_nums = np.floor(samples*train_frac-val_nums)
    train = inds[:train_nums]
    test = inds[train_nums:]

    # normalize data
    vecy = vecy-np.mean(vecy[train])
    vecX = vecX-np.mean(vecX[train, :])
    vecX /= np.max(np.abs(vecX[train, :]))
    # vecX *= 4.
    vecy /= np.max(np.abs(vecy[train]))
    vecy *= 20.
    vecX = np.hstack((vecX, np.ones((vecX.shape[0], 1))))

    names = []
    times = []
    res = []
    if plot:
        plt.figure(1)
        plt.plot(vecX[test, 0], vecy[test], 'or', color=[0.3, 0.3, 0.3],  alpha=0.4, markersize=18.0)
    fmts = ['8c', '1m', '2g', '*y', '4k', 'ob', '.r']
    for m in range(len(methods)):
        val_error = 1e14
        best_param = None
        # only use train-validate step if there actually more than 1 parameter settings
        print params[m]
        if len(params[m]) > 1:
            for p in params[m]:
                # 1. only training examples are labeled and test performance only on validation data
                print('Train-validate parameters {0} for method {1}'.format(p, methods[m]))
                (name, _pred, _lats) = methods[m](np.array(vecX, copy=True), np.array(vecy, copy=True),
                                                np.array(train, copy=True), np.array(test, copy=True),
                                                states=states, params=p, true_latent=vecz, plot=False)
                eval_val, _ = evaluate(vecy[test[:val_nums]], _pred[:val_nums], vecz[test[:val_nums]], _lats[:val_nums], 0)
                if eval_val[1] < val_error:
                    best_param = p
        else:
            print("No Train-validation step for method {0}.".format(methods[m]))
            best_param = params[m][0]

        # 2. Using the best parameter to train on training+validation data but test only on test data
        print('Test parameters {0} for method {1}'.format(best_param, methods[m]))
        tst_train_nums = np.floor(samples*train_frac)
        tst_train = inds[:tst_train_nums]
        tst_test = inds[tst_train_nums:]
        starttime = time.time()
        (name, pred, lats) = methods[m](np.array(vecX, copy=True), np.array(vecy, copy=True),
                                        np.array(tst_train, copy=True), np.array(tst_test, copy=True),
                                        states=states, params=best_param, true_latent=vecz, plot=False)
        stoptime = time.time() - starttime
        times.append(stoptime)
        res.append(evaluate(vecy[tst_test], pred, vecz[tst_test], lats, stoptime))
        names.append(name)
        if plot:
            plt.figure(1)
            plt.plot(vecX[tst_test, 0], pred, fmts[m], alpha=0.8, markersize=10.0)

    if plot:
        plt_names = ['Datapoints']
        plt_names.extend(names)
        plt.legend(plt_names, fontsize=18)
        plt.xlabel('Inputs', fontsize=20)
        plt.ylabel('Targets', fontsize=20)
        plt.show()

    print('------------------------------------------')
    print 'Total data           :', len(train)+len(test)
    print 'Labeled data (train) :', len(train)
    print 'Unlabeled data (val) :', val_nums
    print 'Unlabeled data (test):', len(test)-val_nums
    print 'Fraction train       :', train_frac
    print 'Fraction val         :', val_frac
    print 'Max States           :', states
    print('------------------------------------------')
    print ''.ljust(44), '', res[0][1]
    for m in range(len(names)):
        ll = res[m][0].tolist()
        name = names[m].ljust(45)
        for i in range(len(ll)):
            name += '    {0:+3.4f}'.format(ll[i]).ljust(24)
        print name
    print('------------------------------------------')
    return names, res


def generate_param_set(set_name = 'full'):
    param_flx = [[1000, 0.001], [1000, 0.0001]]
    param_rr = [[0.1], [0.01], [0.001], [0.0001], [0.00001], [0.000001]]
    param_svr = [[0.1, 0.01], [0.1, 0.1], [1.0, .01], [1.0, 0.1], [10., .01], [10., .1], [100., .1], [100., .01]]
    param_krr = param_rr
    param_tr = list()
    for i in range(len(param_rr)):
        for j in range(len(param_rr)):
            for k in range(len(param_rr)):
                param_tr.append([param_rr[i][0], 100.*param_rr[j][0], 100.*param_rr[k][0]])

    param_tcrfr_indep = list()
    param_tcrfr = list()
    param_tcrfr_qp = list()
    param_tcrfr_pl = list()

    tcrfr_theta = [0.85, 0.95]
    tcrfr_lambda = [0.000001]
    tcrfr_gamma = [100.0]
    tcrfr_k1 = [8, 20, 30]

    tcrfr_k1 = [30] # 5%
    tcrfr_k1 = [30] # 10%
    tcrfr_k1 = [15, 30]
    tcrfr_k1 = [30, 20] # 20%

    tcrfr_k2 = [4]  # 20%
    tcrfr_k2 = [18] # 5%
    tcrfr_k2 = [4] # 5%
    tcrfr_k2 = [4, 12] # 20%

    tcrfr_neighb = [10.]

    for i in range(len(tcrfr_theta)):
        for j in range(len(tcrfr_lambda)):
            for k in range(len(tcrfr_gamma)):
                param_tcrfr.append([tcrfr_theta[i], tcrfr_lambda[j], tcrfr_gamma[k]])
                for l in range(len(tcrfr_neighb)):
                    param_tcrfr_indep.append([tcrfr_theta[i], tcrfr_lambda[j], tcrfr_gamma[k], tcrfr_neighb[l]])
                for l in range(len(tcrfr_k1)):
                    param_tcrfr_pl.append([tcrfr_theta[i], tcrfr_lambda[j], tcrfr_gamma[k], tcrfr_k1[l]])
                for l in range(len(tcrfr_k2)):
                    param_tcrfr_qp.append([tcrfr_theta[i], tcrfr_lambda[j], tcrfr_gamma[k], tcrfr_k2[l]])

    params = []
    methods = []
    if set_name == 'full':
        params = [param_rr, param_svr, param_krr, param_tr, param_flx, param_tcrfr_indep, param_tcrfr]
        methods = [method_rr, method_svr, method_krr,
                   method_transductive_regression, method_flexmix,
                   method_tcrfr_indep, method_tcrfr]
    if 'tcrfr_qp' in set_name:
        methods.append(method_tcrfr_qp)
        params.append(param_tcrfr_qp)
    if 'tcrfr_pl' in set_name:
        methods.append(method_tcrfr_pl)
        params.append(param_tcrfr_pl)
    if 'tcrfr_indep' in set_name:
        methods.append(method_tcrfr_indep)
        params.append(param_tcrfr_indep)
    if 'rr' in set_name:
        methods.append(method_rr)
        params.append(param_rr)
    if 'lb' in set_name:
        methods.append(method_lb)
        params.append(param_rr)
    if 'svr' in set_name:
        methods.append(method_svr)
        params.append(param_svr)
    if 'krr' in set_name:
        methods.append(method_krr)
        params.append(param_krr)
    if 'tr' in set_name:
        methods.append(method_transductive_regression)
        params.append(param_tr)
    if 'flexmix' in set_name:
        methods.append(method_flexmix)
        params.append(param_flx)
    return methods, params


def plot_results(name):
    import matplotlib.pyplot as plt

    f = np.load(name)
    means = f['means']
    stds = f['stds']
    states = f['states']
    methods = f['methods']
    names = f['names']
    measures = f['MEASURES']

    plt.figure(1)
    cnt = 0
    fmts = ['--xm', '--xy', '--xc', ':xg', ':xm', '-ob', '-or']
    lws = [2., 2., 2., 2., 2., 2., 2.]
    for i in range(measures+1):
        plt.subplot(2, 4, i+1)
        if i < measures:
            for m in range(len(methods)):
                plt.errorbar(states, means[:, cnt], yerr=stds[:, cnt], fmt=fmts[m],
                             elinewidth=1.0, linewidth=lws[m], alpha=0.6)
                cnt += 1
            plt.xlabel('Number of Latent States', fontsize=20)
            plt.ylabel(f['measure_names'][i], fontsize=20)
            plt.xlim([1, states[-1]])
        if i == measures:
           plt.legend(names, loc=1, fontsize=18)
    plt.show()
    print "DONE"


def plot_frac_results(name):
    import matplotlib.pyplot as plt

    f = np.load(name)
    means = f['means']
    stds = f['stds']
    train_fracs = f['train_fracs']
    methods = f['methods']
    names = f['names']
    measures = f['MEASURES']
    print measures

    print names

    plt.figure(1)
    cnt = 0
    fmts = ['--oy', '-or', '--oc', '--og', '--om', '--ob', '--or']
    lws = [2., 2., 2., 2., 2., 2., 2.]
    for i in range(measures+1):
        if i < measures:
            plt.subplot(2, 4, i+1)
            for m in range(len(methods)):
                plt.errorbar(train_fracs[::-1], means[:, cnt], yerr=stds[:, cnt], fmt=fmts[m],
                             elinewidth=1.0, linewidth=lws[m], alpha=0.6)
                cnt += 1
            plt.xlabel('Fraction of labeled examples', fontsize=20)
            plt.ylabel(f['measure_names'][i], fontsize=20)
            plt.xlim([0.0, train_fracs[-1]])
        if i == measures-1:
           plt.legend(names, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
           # plt.legend(names, loc=1, fontsize=18)
    plt.show()
    print "DONE"
