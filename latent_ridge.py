from cvxopt import matrix,spmatrix,sparse,uniform,normal,setseed
from cvxopt.blas import dot,dotu
from cvxopt.solvers import qp
from cvxopt.lapack import syev
import numpy as np
import math as math

from sklearn.svm import SVR



class LatentRidgeRegression:
    """ Latent Variable Ridge Regression.
        Written by Nico Goernitz, TU Berlin, 2015
    """
    reg = 0.001 # (scalar) the regularization constant > 0
    sobj = None 
    sol = None 
    intercept = 0.0

    def __init__(self, sobj, l=0.001):
        self.reg = l
        self.sobj = sobj

    def train_dc(self, max_iter=50, hotstart=matrix([])):
        N = self.sobj.get_num_samples()
        DIMS = self.sobj.get_num_dims()
        # intermediate solutions
        # latent variables
        latent = [0.0]*N
        self.sol = self.sobj.get_hotstart_sol()
        if hotstart.size==(DIMS,1):
            print('New hotstart position defined.')
            self.sol = hotstart
        psi = matrix(0.0, (DIMS,N)) # (dim x exm)
        old_psi = matrix(0.0, (DIMS,N)) # (dim x exm)

        iter = 0 
        # terminate if objective function value doesn't change much
        while iter<max_iter and (iter<3 or sum(sum(abs(np.array(psi-old_psi))))>=0.001):
            print('Starting iteration {0}.'.format(iter))
            print(sum(sum(abs(np.array(psi-old_psi)))))
            iter += 1
            old_psi = matrix(psi)
            old_sol = self.sol
            # 1. linearize
            # for the current solution compute the 
            # most likely latent variable configuration
            for i in range(N):
                (foo, latent[i], psi[:,i]) = self.sobj.argmax(self.sol, i, add_prior=True)
            # 2. Solve the intermediate optimization problem
            vecy = np.array(matrix(self.sobj.y))[:,0]
            vecX = np.array(psi.trans())
            self.sol = self.train_model(vecX, vecy)
        return (self.sol, latent)

    def train_model(self, vecX, vecy):
        # solve the ridge regression problem 
        E = np.zeros((vecX.shape[1],vecX.shape[1]))
        np.fill_diagonal(E, self.reg)
        XXt = vecX.T.dot(vecX) + E
        print XXt
        XtY = (vecX.T.dot(vecy))
        if XXt.size>1:
            w = np.linalg.inv(XXt).dot(XtY)
        else: 
            w = 1.0/XXt * (XtY)
        return matrix(w)

    def apply(self, pred_sobj):
        """ Application of the Latent Ridge Regression:

            score = max_z <sol*,\Psi(x,z)> 
            latent_state = argmax_z <sol*,\Psi(x,z)> 
        """
        N = pred_sobj.get_num_samples()
        vals = matrix(0.0, (N,1))
        structs = []
        for i in range(N):
            (vals[i], struct, psi) = pred_sobj.argmax(self.sol, i, add_prior=False)
            vals[i] += self.intercept
            structs.append(struct)
        return (vals, structs)
