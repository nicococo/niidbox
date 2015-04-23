from cvxopt import matrix, normal, mul
import numpy as np


class KmeansRidgeRegression(object):
    """ Kmeans Ridge Regression.
        Written by Nico Goernitz, TU Berlin, 2015

        min lambda/2 ||u||^2 + sum_i |y_i - <u_zi,x_i>|^2 +
             gamma/2 ||v||^2 + sum_i ||v_zi - x_i||^2
    """
    lam = 0.001  # (scalar) the regularization constant > 0
    gam = 1.0  # density estimation regularizer

    u = None  # regression
    v = None  # clustering

    X = None  # input data
    y = None  # target data

    cluster = 2  # number of clusters
    latent = None

    def __init__(self, cluster=2, l=0.001, gamma=1.0):
        self.lam = l
        self.gam = gamma
        self.cluster = cluster

    def argmax(self, u, v, X, y):
        dims, n = self.X.size
        if u is not None:
            v = matrix(np.array(v).reshape((dims, self.cluster), order='F'))
            f_density = matrix(0.0, (self.cluster, 1))
            for c in range(self.cluster):
                f_density[c, 0] = (v[:, c] - X).trans() * (v[:, c] - X)

            u = matrix(np.array(u).reshape((dims, self.cluster), order='F'))
            f_squares = y - u.trans()*X
            f_squares = mul(f_squares, f_squares)

            foo = - (f_density + f_squares)

            # highest value first
            inds = np.argsort(-foo, axis=0)[0]
            cls = inds[0]
            val = foo[cls]
        else:
            active = np.array(range(self.cluster))
            if self.latent is not None:
                active = np.unique(self.latent)

            v = matrix(np.array(v).reshape((dims, self.cluster), order='F'))
            f_density = matrix(0.0, (self.cluster, 1))
            for c in range(self.cluster):
                f_density[c, 0] = (v[:, c] - X).trans() * (v[:, c] - X)
            foo = np.array(-f_density)
            # highest value first
            inds = np.argsort(-foo[active], axis=0)[0]
            cls = active[inds[0]]
            val = foo[cls]

        psi = matrix(0.0, (dims*self.cluster, 1))
        psi[dims*cls:dims*(cls+1)] = X
        return val, cls, psi

    def fit(self, X, y, runs=10, max_iter=50):
        self.X = X
        self.y = y
        obj = 1e14
        best_u = 0
        best_v = 0
        best_lats = None
        for i in range(runs):
            (u, v, n_lat, n_obj, is_converged) = self.train_dc_single(max_iter=max_iter)
            # if is_converged and np.single(obj) > np.single(n_obj):
            if np.single(obj) > np.single(n_obj):
                best_u = u
                best_v = v
                best_lats = n_lat
                obj = n_obj
        self.u = best_u
        self.v = best_v
        self.latent = best_lats
        return u, v

    def train_dc_single(self, max_iter=50):
        dims, n = self.X.size

        u = normal(dims*self.cluster, 1)
        v = normal(dims*self.cluster, 1)

        latent = [0.0]*n
        psi = matrix(0.0, (dims*self.cluster, n))  # (dims x exm)
        obj = 1e09
        old_obj = 1e10
        rel = 1
        iter = 0
        is_converged = False

        # terminate if objective function value doesn't change much
        while iter < max_iter and not is_converged:
            # 1. linearize
            # for the current solution compute the most likely latent variable configuration
            for i in range(n):
                if iter > 0:
                    (foo, latent[i], psi[:, i]) = self.argmax(u, v, self.X[:, i], self.y[i, 0])
                else:
                    (foo, latent[i], psi[:, i]) = self.argmax(None, v, self.X[:, i], None)

            # v = matrix(1.0/(float(n)*(self.gam+1.0)) * np.sum(psi, axis=1))
            for c in range(self.cluster):
                inds = np.where(latent == c)[0]
                if inds.size > 0:
                    v[c*dims:c*dims+dims] = matrix(1.0/(float(inds.size)+self.gam/2.0) * np.sum(psi[:, inds], axis=1))

            # calc density objective function:
            obj_density = self.gam/2.0 * v.trans()*v + np.sum(v.trans()*v - 2.0*v.trans()*psi + np.diag(psi.trans()*psi))

            # Solve the regression problem
            vecy = np.array(matrix(self.y))[:, 0]
            vecX = np.array(psi.trans())
            u = self.train_model(vecX, vecy)

            # calc square objective function:
            w = u  # (dims x 1)
            l = self.lam  # scalar
            b = self.y  # (exms x 1)
            phi = psi  # (dims x exms)
            old_obj = obj
            obj_regression = self.lam/2.0*w.trans()*w + b.trans()*b - 2.0*w.trans()*phi*b + w.trans()*phi*phi.trans()*w
            obj = obj_regression + obj_density
            rel = np.abs((old_obj - obj)/obj)
            print('Iter={0}  least squares objective={1:4.2f}  rel={2:2.4f}'.format(iter, obj[0, 0], rel[0, 0]))
            print np.unique(latent)

            if iter > 3 and rel < 0.0001:
                is_converged = True
            iter += 1

        self.u = u
        self.v = v

        return self.u, self.v, latent, obj, is_converged

    def train_model(self, vecX, vecy):
        # solve the ridge regression problem
        E = np.zeros((vecX.shape[1], vecX.shape[1]))
        np.fill_diagonal(E, self.lam)
        XXt = vecX.T.dot(vecX) + E
        XtY = (vecX.T.dot(vecy))
        if XXt.size > 1:
            w = np.linalg.inv(XXt).dot(XtY)
        else:
            w = 1.0/XXt * XtY
        return matrix(w)

    def predict(self, Xpred):
        dims, samples = Xpred.size
        vals = matrix(0.0, (samples, 1))
        structs = []
        for i in range(samples):
            (foo, struct, psi) = self.argmax(None, self.v, Xpred[:, i], None)
            vals[i] = self.u.trans() * psi
            structs.append(struct)

        print '---apply---'
        print np.unique(self.latent)
        print np.unique(structs)
        print '---apply---'
        return vals, structs
