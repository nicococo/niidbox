from cvxopt import matrix,spmatrix,sparse,uniform,normal,setseed
from cvxopt.blas import dot,dotu
from cvxopt.solvers import qp
from cvxopt.lapack import syev
import numpy as np
import math as math

from sklearn.svm import SVR

from latent_ridge import LatentRidgeRegression


class LatentSvr(LatentRidgeRegression):
    """ Latent Variable Support Vector Regression.
        Written by Nico Goernitz, TU Berlin, 2015
    """
    epsilon = 0.001
    svr = None

    def __init__(self, sobj, C=1.0, epsilon=0.001):
        LatentRidgeRegression.__init__(self, sobj, l=C)
        self.epsilon = epsilon
        self.svr = SVR(C=self.reg, epsilon=self.epsilon, kernel='linear', shrinking=False)

    # override 
    def train_model(self, vecX, vecy):
        print vecX.shape
        print vecy.shape
        self.svr.fit(vecX, vecy)
        self.intercept = self.svr.intercept_
        return matrix(self.svr.coef_)
