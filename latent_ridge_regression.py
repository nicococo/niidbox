from cvxopt import matrix
import numpy as np


class AbstractLatentRegression(object):
    """ Abstract Latent Regression Model.
    """
    lam = 0.001  # (scalar) the regularization constant > 0
    gamma = 1.0  # density estimation regularizer
    theta = 0.5  # 0<= thata <= 1: trade-off between density estimation (0.0) and regression (1.0)

    sobj = None # structured object
    sol = None  # p(pi|x) map model
    cls = None  # p(y|pi,x) regression models

    intercept = 0.0  # possible global intercept

    def __init__(self, theta=0.5, lam=0.001, gam=1.0):
        self.cls = []
        self.lam = lam
        self.gamma = gam
        self.theta = theta

    def fit(self, sobj, max_iter=50, n_init=5):
        self.sobj = sobj
        obj = 1e14
        best_cls = 0
        best_sol = 0
        best_lats = []
        for i in range(n_init):
            (sol, cls, n_lat, n_obj, is_converged) = self.train_dc_single(max_iter=max_iter)
            # if is_converged and np.single(obj) > np.single(n_obj):
            if np.single(obj) > np.single(n_obj):
                best_cls = cls
                best_sol = sol
                best_lats = n_lat
                obj = n_obj
            if not is_converged and i == 0:
                best_cls = cls
                best_sol = sol
                best_lats = n_lat
        self.sol = best_sol
        self.cls = best_cls
        print self.theta
        return best_sol, best_lats

    def train_dc_single(self, max_iter=50, hotstart=None):
        pass

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

    def predict(self, pred_sobj):
        """ Application of the Latent Ridge Regression:
            latent state: pi = argmax_z <sol*,Psi(x,z)>
            score =  <sol,Psi(x,pi)>
        """
        samples = pred_sobj.get_num_samples()
        vals = matrix(0.0, (samples, 1))
        structs = []
        pred_sobj.update_solution(self.sol)
        for i in range(samples):
            (foo, struct, psi) = pred_sobj.map(i, add_prior=False)
            vals[i] = self.cls.trans() * psi + self.intercept
            structs.append(struct)
        print np.unique(structs)
        return vals, structs


class LatentRidgeRegression(AbstractLatentRegression):

    def __init__(self, theta=0.5, lam=0.001, gam=1.0):
        AbstractLatentRegression.__init__(self, theta, lam, gam)

    def get_density_objective(self, w, psi):
        (d, n) = psi.size
        b = matrix(1.0, (n, 1))  # (exms x 1)
        return np.single(self.gamma*w.trans()*w - w.trans()*psi*b)

    def get_regression_objective(self, w, psi, b):
        return self.lam*w.trans()*w + b.trans()*b - 2.0*w.trans()*psi*b + w.trans()*psi*psi.trans()*w

    def train_dc_single(self, max_iter=50, hotstart=None):
        n = self.sobj.get_num_samples()
        dims = self.sobj.get_num_dims()
        self.sol = self.sobj.get_hotstart_sol()
        self.cls = self.sobj.get_hotstart_sol()
        if hotstart is not None and hotstart.size == (dims, 1):
            print('New hotstart position defined.')
            self.sol = hotstart
        latent = [0.0]*n
        psi = matrix(0.0, (dims, n))  # (dims x exm)
        obj = 1e09
        old_obj = 1e10
        rel = 1
        iter = 0
        is_converged = False

        best_obj = 1e14
        best_cls = 0
        best_sol = 0
        best_lats = []

        # terminate if objective function value doesn't change much
        while iter < max_iter and not is_converged:
            # 1. linearize
            # for the current solution compute the most likely latent variable configuration
            self.sobj.update_solution([self.sol, self.cls])
            for i in range(n):
                (foo, latent[i], psi[:, i]) = self.sobj.map(i, add_prior=True, add_loss=True, theta=self.theta)

            self.sol = matrix(1.0/(2.0*self.gamma) * np.sum(psi, axis=1))
            # calc objective function:
            # w = self.sol  # (dims x 1)
            # b = matrix(1.0, (n, 1))  # (exms x 1)
            # phi = psi  # (dims x exms)
            # obj_density = np.single(self.gamma*w.trans()*w - w.trans()*phi*b)
            obj_density = self.get_density_objective(self.sol, psi)

            # Solve the regression problem
            vecy = np.array(matrix(self.sobj.y))[:, 0]
            vecX = np.array(psi.trans())
            self.cls = self.train_model(vecX, vecy)
            # calc objective function:
            # w = self.cls  # (dims x 1)
            # l = self.lam  # scalar
            # b = self.sobj.y  # (exms x 1)
            # phi = psi  # (dims x exms)
            # obj_regression = l*w.trans()*w + b.trans()*b - 2.0*w.trans()*phi*b + w.trans()*phi*phi.trans()*w
            obj_regression = self.get_regression_objective(self.cls, psi, self.sobj.y)

            old_obj = obj
            obj = self.theta * obj_regression + (1.0-self.theta) * obj_density
            rel = np.abs((old_obj - obj)/obj)
            print('Iter={0} combined objective={1:4.2f} rel={2:2.4f} used_lats={3}'.format(iter, obj[0, 0], rel[0, 0], np.unique(latent).size))

            if np.single(best_obj) > np.single(obj):
                best_cls = self.cls
                best_sol = self.sol
                best_lats = latent
                best_obj = obj

            if iter > 3 and rel < 0.0001:
                is_converged = True

            iter += 1

        self.cls = best_cls
        self.sol = best_sol
        latent = best_lats
        obj = best_obj
        return self.sol, self.cls, latent, obj, is_converged


class TransductiveLatentRidgeRegression(LatentRidgeRegression):

    def __init__(self, theta=0.5, lam=0.001, gam=1.0):
        LatentRidgeRegression.__init__(self, theta, lam, gam)

    def train_dc_single(self, max_iter=50, hotstart=None):
        n = self.sobj.get_num_samples()
        n_lbl = self.sobj.get_num_labeled_samples()

        dims = self.sobj.get_num_dims()
        self.sol = self.sobj.get_hotstart_sol()
        self.cls = self.sobj.get_hotstart_sol()
        if hotstart is not None and hotstart.size == (dims, 1):
            print('New hotstart position defined.')
            self.sol = hotstart

        latent = [0.0]*n
        psi = matrix(0.0, (dims, n))  # (dims x exm)
        psi_lbl = matrix(0.0, (dims, n_lbl))  # (dims x exm)
        obj = 1e09
        old_obj = 1e10
        rel = 1
        iter = 0
        is_converged = False

        best_obj = 1e14
        best_cls = 0
        best_sol = 0
        best_lats = []

        # terminate if objective function value doesn't change much
        while iter < max_iter and not is_converged:
            # 1. linearize
            # for the current solution compute the most likely latent variable configuration
            self.sobj.update_solution([self.sol, self.cls])

            print n
            for i in range(n):
                (foo, latent[i], psi[:, i]) = self.sobj.map(i, add_prior=True, add_loss=True, theta=self.theta)

            ind = 0
            print len(self.sobj.lbl_idx)
            for idx in self.sobj.lbl_idx:
                psi_lbl[:, ind] = self.sobj.get_labeled_joint_feature_map(idx, y=latent[idx])
                ind += 1

            self.sol = matrix(1.0/(2.0*self.gamma) * np.sum(psi, axis=1))
            obj_density = self.get_density_objective(self.sol, psi)

            # Solve the regression problem
            vecy = np.array(matrix(self.sobj.y))[self.sobj.lbl_idx, 0]
            vecX = np.array(psi_lbl.trans())
            self.cls = self.train_model(vecX, vecy)
            obj_regression = self.get_regression_objective(self.cls, psi_lbl, self.sobj.y[matrix(self.sobj.lbl_idx)])

            old_obj = obj
            obj = self.theta * obj_regression + (1.0-self.theta) * obj_density
            rel = np.abs((old_obj - obj)/obj)
            print('Iter={0} combined objective={1:4.2f} rel={2:2.4f} used_lats={3}'.format(iter, obj[0, 0], rel[0, 0], np.unique(latent).size))

            if np.single(best_obj) > np.single(obj):
                best_cls = self.cls
                best_sol = self.sol
                best_lats = latent
                best_obj = obj

            if iter > 3 and rel < 0.0001:
                is_converged = True

            iter += 1

        self.cls = best_cls
        self.sol = best_sol
        latent = best_lats
        obj = best_obj

        ind = 0
        vals = matrix(0.0, (n - n_lbl, 1))
        structs = list()
        for idx in self.sobj.trans_idx:
            psi = self.sobj.get_labeled_joint_feature_map(idx, y=latent[idx])
            vals[ind] = self.cls.trans() * psi + self.intercept
            structs.append(latent[idx])
            ind += 1

        print np.unique(latent)
        print np.unique(structs)
        return vals, self.cls, latent, obj, is_converged

