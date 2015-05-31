import matplotlib
matplotlib.use('QT4Agg')
# change to type 1 fonts!
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt

import numpy as np

from sklearn.svm import SVR
from sklearn.metrics import median_absolute_error, mean_squared_error, r2_score, mean_absolute_error, adjusted_rand_score
from sklearn.datasets import load_svmlight_file
import sklearn.cluster as cl

from gridmap import Job, process_jobs
import argparse, sys

from latent_ridge_regression import LatentRidgeRegression
from tcrf_regression import TransductiveCrfRegression
from tcrfr_indep_model import TCrfRIndepModel
from tcrfr_pair_model import TCrfRPairwisePotentialModel


def load_svmlight_data(fname):
    (X, y) = load_svmlight_file(fname)
    print y.shape
    print X.shape
    return np.array(X.todense()), y.reshape(X.shape[0])


def get_1d_toy_data(num_exms=300, plot_data=False):
    grid_x = num_exms
    # generate 2D grid
    gx = np.linspace(0, 1, grid_x)
    # latent variable z
    z = np.zeros(grid_x, dtype=np.uint8)
    zx = np.where((gx > 0.3) & (gx < 0.6))[0]
    z[zx] = 2
    zx = np.where(gx > 0.6)[0]
    z[zx] = 1

    # inputs  x
    # x = 4.0*np.sign(z-0.4)*gx + 2.0*np.sign(z-0.4) + 0.5*np.random.randn(grid_x)
    x = 4.0*np.sign(z-0.5)*gx + 0.6*(z+1.)*np.random.rand(grid_x)
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


def method_transductive_regression(vecX, vecy, train, test, states=2, params=[0.0001, 0.9, 0.1], plot=False):
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

    bak = vecy[test]
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


def method_tcrfr(vecX, vecy, train, test, states=2, params=[0.9, 0.00001, 0.5], plot=False):
    # model = TCrfRIndepModel(data=vecX.T, labels=vecy[train], label_inds=train, unlabeled_inds=test, states=states)
    A = np.zeros((vecX.shape[0], vecX.shape[0]))
    for i in range(vecX.shape[0]-1):
        A[i, i+1] = 1
        A[i+1, i] = 1

        # A[i+1, i] = 1
    model = TCrfRPairwisePotentialModel(data=vecX.T, labels=vecy[train], label_inds=train, unlabeled_inds=test, states=states, A=A)
    # model.test_qp_param()

    tcrfr = TransductiveCrfRegression(reg_theta=params[0], reg_lambda=params[1], reg_gamma=params[2]*float(len(train)+len(test)))
    tcrfr.fit(model, max_iter=40, use_grads=False)
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
    return 'TCRFR (Pairwise Potentials)', y_preds, lats


def method_ridge_regression(vecX, vecy, train, test, states=2, params=[0.0001], plot=False):
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


def method_svr(vecX, vecy, train, test, states=2, params=[1.0, 0.1, 'linear'], plot=False):
    # train ordinary support vector regression
    clf = SVR(C=params[0], epsilon=params[1], kernel=params[2], shrinking=False)
    clf.fit(vecX[train, :], vecy[train])
    return 'Support Vector Regression', clf.predict(vecX[test, :]), np.ones(len(test))


def method_krr(vecX, vecy, train, test, states=2, params=[0.0001], plot=False):
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
    return 'K-means + Ridge Regression', np.sum(sol[lbls, :] * vecX[test, :], axis=1), lbls


def method_tcrfr_indep(vecX, vecy, train, test, states=2, params=[0.9, 0.00001, 0.4], plot=False):
    A = np.zeros((vecX.shape[0], vecX.shape[0]))
    for i in range(vecX.shape[0]-4):
        A[i, i+1] = 1
        A[i+1, i] = 1
        A[i, i+2] = 1
        A[i+2, i] = 1
        A[i, i+3] = 1
        A[i+3, i] = 1
        A[i, i+4] = 1
        A[i+4, i] = 1

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
    return 'TCRFR (Indep)', y_preds, lats


def method_flexmix(vecX, vecy, train, test, states=2, params=[], plot=False):
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

    model = r.flexmix(robjects.Formula("y ~ ."), data=df_train_r, k=states)

    test_data = np.hstack((vecX[test, 0:feats], np.zeros(len(test)).reshape(-1, 1)))
    df_test = pd.DataFrame(test_data)
    colnames = []
    for i in range(feats):
        colnames.append(str(i))
    colnames.append('y')
    df_test.columns = colnames
    df_test_r = com.convert_to_r_dataframe(df_test)

    pr = r.predict(model, newdata=df_test_r)
    df = com.convert_robj(pr)
    s = pd.Series(df)
    aux = s.values
    dim = aux.shape[0]
    y_pred = np.zeros((len(vecy[test]), dim))

    lats = np.zeros((vecy.shape[0], 1), dtype=int)
    lats_pred = np.array(r.clusters(model, newdata=df_test_r)).reshape(-1, 1)

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
    return 'FlexMix', np.array(y_pred_flx), np.reshape(lats_pred, newshape=lats_pred.size)


def single_run(methods, vecX, vecy, vecz, train_frac, states, plot):
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
    vecy *= 4.
    vecX = np.hstack((vecX, np.ones((vecX.shape[0], 1))))

    names = []
    res = []
    for m in methods:
        (name, pred, lats) = m(np.array(vecX, copy=True), np.array(vecy, copy=True),
                               np.array(train, copy=True), np.array(test, copy=True), states=states, plot=plot)
        names.append(name)
        print name
        res.append(evaluate(vecy[test], pred, vecz[test], lats))

    print('------------------------------------------')
    print 'Total data           :', len(train)+len(test)
    print 'Labeled data (train) :', len(train)
    print 'Unlabeled data (test):', len(test)
    print 'Fraction             :', train_frac
    print 'Max States           :', states
    print('------------------------------------------')
    print ''.ljust(45), ': ', res[0][1]
    for m in range(len(names)):
        print names[m].ljust(45), ': ', res[m][0].tolist()
    print('------------------------------------------')
    return names, res


def plot_results(name):
    f = np.load(name)
    means = f['means']
    stds = f['stds']
    states = f['states']
    methods = f['methods']
    names = f['names']
    MEASURES = f['MEASURES']

    plt.figure(1)
    cnt = 0
    fmts = ['--xm', '--xy', '--xc', ':xg', ':xm', '-ob', '-or']
    lws = [2., 2., 2., 2., 2., 2., 2.]
    for i in range(MEASURES):
        plt.subplot(2, 3, i+1)
        for m in range(len(methods)):
            plt.errorbar(states, means[:, cnt], yerr=stds[:, cnt], fmt=fmts[m],
                         elinewidth=1.0, linewidth=lws[m], alpha=0.6)
            cnt += 1
        plt.xlabel('Number of Latent States', fontsize=20)
        plt.ylabel(f['measure_names'][i], fontsize=20)
        plt.xlim([1, states[-1]])
        if i == MEASURES-1:
            plt.legend(names, loc=4, fontsize=18)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--max_states", help="Max state for testing (default=3).", default="3", type=int)
    parser.add_argument("-f", "--train_frac", help="Fraction of training exms (default=0.75)", default=0.75, type=float)
    parser.add_argument("-d", "--datapoints", help="Amount of data points (default=300)", default=300, type=int)
    parser.add_argument("-r", "--reps", help="Number of repetitions (default 10)", default=10, type=int)
    parser.add_argument("-p", "--processes", help="Number of processes (default 4)", default=4, type=int)
    parser.add_argument("-l", "--local", help="Run local or distribute? (default True)", default=True, type=bool)
    parser.add_argument("-s", "--set", help="Select active methods set. (default 'full')", default='full', type=str)
    arguments = parser.parse_args(sys.argv[1:])
    print arguments

    # plot_results('res_toy_{0}.npz'.format(arguments.max_states))
    (vecX, vecy, vecz) = get_1d_toy_data(num_exms=arguments.datapoints)

    # full stack of methods
    methods = [method_ridge_regression, method_tcrfr]
    if arguments.set == 'full':
        methods = [method_ridge_regression, method_svr, method_krr,
                   method_transductive_regression, method_flexmix,
                   method_tcrfr_indep, method_tcrfr]
    # methods = [method_flexmix, method_tcrfr_indep]
    # methods = [method_tcrfr_indep, method_tcrfr]
    # methods = [method_ridge_regression, method_tcrfr_indep]
    # single_run(methods, vecX, vecy, vecz, train_frac=0.75, states=3, plot=True)

    jobs = []
    MEASURES = 6
    REPS = arguments.reps
    states = range(1, arguments.max_states+1)
    mse = {}
    sn_map = {}
    cnt = 0
    for s in range(len(states)):
        if s not in mse:
            mse[s] = np.zeros((REPS, MEASURES*len(methods)))
        for n in range(REPS):
            job = Job(single_run, [methods, vecX, vecy, vecz, arguments.train_frac, states[s], False],
                      mem_max='8G', mem_free='16G', name='TCRFR it({0}) state({1})'.format(n, states[s]))
            jobs.append(job)
            sn_map[cnt] = (s, n)
            cnt += 1

    print '---------------'
    print mse

    processedJobs = process_jobs(jobs, max_processes=arguments.processes, local=arguments.local)
    results = []
    print "ret fields AFTER execution on local machine"
    for (i, result) in enumerate(processedJobs):
        print "Job #", i
        (names, res) = result
        (s, n) = sn_map[i]
        perf = mse[s]
        cnt = 0
        for p in range(MEASURES):
            for m in range(len(methods)):
                perf[n, cnt] = res[m][0][p]
                cnt += 1

    measure_names = res[0][1]
    means = np.zeros((len(states), MEASURES*len(methods)))
    stds = np.zeros((len(states), MEASURES*len(methods)))
    print names
    print res[0][1]
    for key in mse.iterkeys():
        means[key, :] = np.mean(mse[key], axis=0)
        stds[key, :] = np.std(mse[key], axis=0)
        print states[key], ': ', np.mean(mse[key], axis=0).tolist()
        print 'STD ', np.std(mse[key], axis=0).tolist()

    # save results
    np.savez('res_toy_{0}.npz'.format(states), MEASURES=MEASURES, methods=methods, means=means, stds=stds, states=states,
             measure_names=measure_names, names=names)
    # ..and stop
    print('Finish!')
