import numpy as np
import matplotlib.pyplot as plt
import cvxopt as co

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_svmlight_file

from latent_svr import LatentSvr
from latent_ridge import LatentRidgeRegression
from so_multiclass import SOMultiClass


def load_svmlight_data(fname):
    (X, y) = load_svmlight_file(fname)
    print y.shape
    print X.shape
    #print X
    return (X.todense(), y.reshape(X.shape[0]))


def get_toy_data():
    GRID_X = 20
    GRID_Y = 20
    # generate 2D grid
    gx = np.linspace(0, 1, GRID_X)
    gy = np.linspace(0, 1, GRID_Y)
    (gx, gy) = np.meshgrid(gx, gy)

    # latent variable z
    z = np.zeros((GRID_X, GRID_Y), dtype=np.uint8)
    (zx, zy) = np.where((gx>0.5))
    z[zx,zy] = 1

    # inputs  x
    x = 4.0*np.sign(z-0.3)*gy + 1.0*np.sign(z-0.5) +  0.5*np.random.randn(GRID_X, GRID_Y)
    #x = 1.0*(z-0.5)*(gx)
    #x = 1.0*(gx)

    # ..and corresponding target value y
    y = (np.sign(z-0.5)*10.1)*x + 0.1*np.random.randn(GRID_X, GRID_Y)
    #y = ((z-0.3)*4.)*x
    #y = (2.)*gx
    #print z

    # scikit learns samples x features format
    vecX = np.hstack( (x.reshape(GRID_X*GRID_Y, 1), np.ones((GRID_X*GRID_Y,1)) ) )
    vecy = y.reshape(GRID_X*GRID_Y)
    vecz = z.reshape(GRID_X*GRID_Y)
    print vecX.shape
    print vecy.shape
    return (vecX, vecy)



if __name__=='__main__':

    (vecX, vecy) = load_svmlight_data('/home/nicococo/Data/space_ga_scale.dat')
    #(vecX, vecy) = get_toy_data()

    vecy = vecy-np.mean(vecy)
    vecX = vecX-np.mean(vecX)

    # generate training samples
    SAMPLES = vecX.shape[0]
    TRAIN_FRAC = 0.5
    inds = np.random.permutation(range(SAMPLES))
    train = inds[:np.floor(SAMPLES*TRAIN_FRAC)]
    test = inds[np.floor(SAMPLES*TRAIN_FRAC):]

    # OLS soution
    # vecX in (samples x dims)
    # vecy in (samples)
    # w in (dims)
    E = np.zeros((vecX.shape[1],vecX.shape[1]))
    np.fill_diagonal(E, 0.001)
    XXt = vecX[train,:].T.dot(vecX[train,:]) + E
    XtY = (vecX[train,:].T.dot(vecy[train]))
    print XXt.shape
    print XtY.shape
    w = np.linalg.inv(XXt).dot(XtY.T).T
    y_pred = w.dot(vecX[test,:].T)

    # train ordinary support vector regression
    #clf = SVR(C=1.0, epsilon=0.0, kernel='linear', shrinking=False)
    #clf.fit(vecX[train,:], vecy[train]) 
    #print clf.intercept_
    #y_pred = clf.predict(vecX[test,:])

    print vecX[train,:].T
    trainMC = SOMultiClass(co.matrix(vecX[train,:].T), classes=2, y=co.matrix(vecy[train]))
    testMC = SOMultiClass(co.matrix(vecX[test,:].T), classes=2)

    # train latent support vector regression
    lsvr =LatentRidgeRegression(trainMC, l=0.001)
    #lsvr =LatentSvr(trainMC, C=10.0, epsilon=0.01)
    (foo, lats) = lsvr.train_dc(max_iter=20)
    (y_pred2, lats) = lsvr.apply(testMC)
    y_pred2 = np.array(y_pred2)
    lats = np.array(lats)

    print('------------------------------------------')
    print 'OLS mse:       ',mean_squared_error(vecy[test], y_pred)
    print 'LatentOLS mse: ',mean_squared_error(vecy[test], y_pred2)
    print('------------------------------------------')


    # show plots
    # plt.figure(1)

    # plt.subplot(1,3,1)
    # plt.scatter(gx, gy+1.2, s=20, c=x, marker='s')
    # plt.scatter(gx, gy, s=20, c=y)

    # plt.xlim((-0.02, 1.02))
    # plt.ylim((-0.05, 2.25))
    # plt.xticks([])
    # plt.yticks([])

    # plt.subplot(1,3,2)
    # sinds = np.argsort(vecy[test])
    # truth = vecy[test[sinds]]
    # plt.plot(range(test.size),y_pred[sinds],'ob',alpha=0.4)
    # plt.plot(range(test.size),y_pred2[sinds],'og',alpha=0.4)
    # plt.plot(range(test.size),truth,'.r',alpha=0.7)


    # plt.subplot(1,3,3)
    # plt.plot(range(test.size),lats[sinds],'oy',markersize=8)
    # plt.plot(range(test.size),vecz[test[sinds]],'.k',markersize=8)
    # plt.ylim((-0.25, 1.25))

    sinds = np.argsort(vecy[test])
    truth = vecy[test[sinds]]
    plt.plot(range(test.size),y_pred[sinds],'ob',alpha=0.4)
    plt.plot(range(test.size),y_pred2[sinds],'og',alpha=0.4)
    plt.plot(range(test.size),truth,'.r',alpha=0.7)

    plt.show()

    # ..and stop
    print('Finish!')

