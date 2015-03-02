import numpy as np
import matplotlib.pyplot as plt
import cvxopt as co

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_svmlight_file
import sklearn.cluster as cl

from latent_svr import LatentSvr
from latent_ridge import LatentRidgeRegression, LatentRidgeRegression2, LatentRidgeRegression3
from so_multiclass import SOMultiClass


def get_2D_toy_data():
    GRID_X = 40
    GRID_Y = 40
    # generate 2D grid
    gx = np.linspace(0, 1, GRID_X)
    gy = np.linspace(0, 1, GRID_Y)
    (gx, gy) = np.meshgrid(gx, gy)

    # latent variable z
    z = np.zeros((GRID_X, GRID_Y), dtype=np.uint8)
    (zx, zy) = np.where((gx>0.3))
    z[zx,zy] = 1
    #(zx, zy) = np.where((gx>0.6))
    #z[zx,zy] = 2

    # inputs  x
    x = 4.0*np.sign(z-0.5)*gy + 1.0*np.sign(z-0.5) +  0.5*np.random.randn(GRID_X, GRID_Y)
    #x = 1.0*(z-0.5)*(gx)
    #x = 1.0*(gx)

    # ..and corresponding target value y
    y = (np.sign(z-0.5)*10.1)*x + 0.01*np.random.randn(GRID_X, GRID_Y)
    #y = ((z-0.3)*4.)*x
    #y = (2.)*gx
    #print z

    # scikit learns samples x features format
    #vecX = np.hstack( (x.reshape(GRID_X*GRID_Y, 1), np.ones((GRID_X*GRID_Y,1)) ) )
    vecX = x.reshape(GRID_X*GRID_Y, 1)
    vecy = y.reshape(GRID_X*GRID_Y)
    vecz = z.reshape(GRID_X*GRID_Y)
    print vecX.shape
    print vecy.shape
    return vecX, vecy


def single_run(vecX, vecy, states=2, plot=False):
    # generate training samples
    SAMPLES = vecX.shape[0]
    FEATS = vecX.shape[1]
    TRAIN_FRAC = 0.75
    inds = np.random.permutation(range(SAMPLES))
    train = inds[:np.floor(SAMPLES*TRAIN_FRAC)]
    test = inds[np.floor(SAMPLES*TRAIN_FRAC):]

    # OLS solution
    # vecX in (samples x dims)
    # vecy in (samples)
    # w in (dims)
    E = np.zeros((vecX.shape[1],vecX.shape[1]))
    np.fill_diagonal(E, 0.00001)
    XXt = vecX[train,:].T.dot(vecX[train,:]) + E
    XtY = (vecX[train,:].T.dot(vecy[train]))
    w = np.linalg.inv(XXt).dot(XtY.T)
    print w.shape
    y_pred = w.T.dot(vecX[test,:].T).T
    rr_mse = mean_squared_error(vecy[test], y_pred)

    # train ordinary support vector regression
    #clf = SVR(C=1.0, epsilon=0.0, kernel='linear', shrinking=False)
    #clf.fit(vecX[train,:], vecy[train]) 
    #print clf.intercept_
    #y_pred = clf.predict(vecX[test,:])

    trainMC = SOMultiClass(co.matrix(vecX[train, :].T), classes=states, y=co.matrix(vecy[train]))
    testMC = SOMultiClass(co.matrix(vecX[test, :].T), classes=states)

    # train latent support vector regression
    lsvr =LatentRidgeRegression3(trainMC, l=0.00001/states)
    (foo, lats) = lsvr.train_dc(max_iter=100)
    (y_pred2, lats) = lsvr.apply(testMC)
    y_pred2 = np.array(y_pred2)
    lrr_mse = mean_squared_error(vecy[test], y_pred2)

    print('------------------------------------------')
    print 'RR mse:        ', rr_mse
    print 'LatentRR  mse: ', lrr_mse
    print('------------------------------------------')

    if plot:
        sinds = np.argsort(vecy[test])
        truth = vecy[test[sinds]]
        plt.plot(range(test.size),vecX[test[sinds],0],'og',alpha=0.4, markersize=6.0)
        plt.plot(range(test.size),y_pred[sinds],'oc',alpha=0.4, markersize=6.0)
        plt.plot(range(test.size),y_pred2[sinds],'ob',alpha=0.6, markersize=10.0)
        plt.plot(range(test.size),truth,'or', markersize=7.0, alpha=0.6)
        plt.legend(['Inputs', 'RR', 'LatentRR', 'True Labels'], loc=2)
        plt.show()

    return rr_mse, lrr_mse



if __name__=='__main__':
    (vecX, vecy) = get_2D_toy_data()

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

    single_run(vecX, vecy, states=6, plot=True)

    # REPS = 4
    # states = [1, 2, 8, 12, 20]
    # #states = [1,2,4]
    # mse = {}
    # for s in range(len(states)):
    #     for n in range(REPS):
    #         if not mse.has_key(s):
    #             mse[s] = np.zeros((REPS, 4))
    #         cmse = mse[s]
    #         cmse[n, :] += single_run(vecX, vecy, states=states[s], plot=False)
    #
    #
    # means = np.zeros((len(states), 4))
    # stds = np.zeros((len(states), 4))
    # print '     RR       -    LatentRR'
    # for key in mse.iterkeys():
    #     means[key, :] = np.mean(mse[key], axis=0)
    #     stds[key, :] = np.std(mse[key], axis=0)
    #     print states[key], ': ', np.mean(mse[key], axis=0)
    #     print '    ', np.std(mse[key], axis=0)

    # plt.figure(1)
    # plt.errorbar(states, means[:, 0], yerr=stds[:, 0], fmt='-b', elinewidth=1.0, linewidth=2.0, alpha=0.6)
    # plt.errorbar(states, means[:, 1], yerr=stds[:, 1], fmt='-.r', elinewidth=1.0, linewidth=2.0, alpha=0.6)
    # plt.errorbar(states, means[:, 2], yerr=stds[:, 2], fmt='-g', elinewidth=1.0, linewidth=4.0, alpha=0.6)
    # plt.errorbar(states, means[:, 3], yerr=stds[:, 3], fmt='.-m', elinewidth=1.0, linewidth=2.0, alpha=0.6)
    # plt.legend(['RR','Kmeans','LRR','LRR2'], loc=2)
    # plt.show()

    # ..and stop
    print('Finish!')
