from cvxopt import matrix
import numpy as np


class LatentRidgeRegression(object):
    """ Latent Variable Ridge Regression.
        Written by Nico Goernitz, TU Berlin, 2015
    """
    reg = 0.001  # (scalar) the regularization constant > 0
    gamma = 1.0  # density estimation regularizer
    sobj = None
    sol = None
    cls = None  # regression
    intercept = 0.0

    def __init__(self, sobj, l=0.001, gamma=1.0):
        self.cls = []
        self.reg = l
        self.gamma = gamma
        self.sobj = sobj

    def train_dc(self, max_iter=50):
        runs = 10
        obj = 1e14
        best_cls = 0
        best_sol = 0
        best_lats = []
        for i in range(runs):
            (sol, cls, n_lat, n_obj, is_converged) = self.train_dc_single(max_iter=max_iter)
            if is_converged and np.single(obj) > np.single(n_obj):
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
        return best_sol, best_lats

    def train_dc_single(self, max_iter=50, hotstart=None):
        n = self.sobj.get_num_samples()
        dims = self.sobj.get_num_dims()
        self.sol = self.sobj.get_hotstart_sol()
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
        # terminate if objective function value doesn't change much
        while iter < max_iter and not is_converged:
            # 1. linearize
            # for the current solution compute the most likely latent variable configuration
            for i in range(n):
                (foo, latent[i], psi[:, i]) = self.sobj.argmax(self.sol, i, add_prior=True, add_loss=True)
            print np.unique(latent)

            self.sol = matrix(1.0/(float(n)*self.gamma) * np.sum(psi, axis=1))
            # calc objective function:
            w = self.sol  # (dims x 1)
            b = matrix(1.0, (n, 1))  # (exms x 1)
            phi = psi  # (dims x exms)
            old_obj = obj
            obj = np.single(self.gamma/2.0*w.trans()*w - 1.0/float(n)*w.trans()*phi*b)
            rel = np.abs((old_obj - obj)/obj)
            print('Iter {0} objective={1:4.2f}  rel={2:2.4f}'.format(iter, obj[0, 0], rel[0, 0]))
            print np.unique(latent)
            if iter > 3 and rel < 0.0001:
                is_converged = True
            iter += 1

        # Solve the regression problem
        vecy = np.array(matrix(self.sobj.y))[:, 0]
        vecX = np.array(psi.trans())
        self.cls = self.train_model(vecX, vecy)
        # calc objective function:
        w = self.cls  # (dims x 1)
        l = self.reg  # scalar
        b = self.sobj.y  # (exms x 1)
        phi = psi  # (dims x exms)
        obj = l*w.trans()*w + b.trans()*b - 2.0*w.trans()*phi*b + w.trans()*phi*phi.trans()*w
        print('Overall objective={1:4.2f}  rel={2:2.4f}'.format(iter, obj[0, 0], rel[0, 0]))
        print np.unique(latent)
        return self.sol, self.cls, latent, obj, is_converged

    def train_model(self, vecX, vecy):
        # solve the ridge regression problem
        E = np.zeros((vecX.shape[1], vecX.shape[1]))
        np.fill_diagonal(E, self.reg)
        XXt = vecX.T.dot(vecX) + E
        XtY = (vecX.T.dot(vecy))
        if XXt.size > 1:
            w = np.linalg.inv(XXt).dot(XtY)
        else:
            w = 1.0/XXt * XtY
        return matrix(w)

    def apply(self, pred_sobj):
        """ Application of the Latent Ridge Regression:
            score = max_z <sol*,\Psi(x,z)>
            latent_state = argmax_z <sol*,\Psi(x,z)>
        """
        N = pred_sobj.get_num_samples()
        vals = matrix(0.0, (N, 1))
        structs = []
        for i in range(N):
            (foo, struct, psi) = pred_sobj.argmax(self.sol, i, add_prior=False)
            vals[i] = self.cls.trans() * psi + self.intercept
            structs.append(struct)
        print np.unique(structs)
        return vals, structs


class LatentRidgeRegressionExperimental(LatentRidgeRegression):
    """ Latent Variable Ridge Regression.
        Written by Nico Goernitz, TU Berlin, 2015
    """
    def __init__(self, sobj, l=0.001):
        LatentRidgeRegression.__init__(self, sobj, l=l)

    def train_dc(self, max_iter=50, hotstart=matrix([])):
        runs = 10
        obj = 1e14

        best_sol = 0
        best_lats = []
        for i in range(runs):
            (sol, n_lat, n_obj, is_converged) = self.train_dc_single(max_iter=max_iter)

            if is_converged and np.single(obj) > np.single(n_obj):
                best_sol = sol
                best_lats = n_lat
                obj = n_obj

            if not is_converged and i == 0:
                best_sol = sol
                best_lats = n_lat

        self.sol = best_sol
        return best_sol, best_lats


    def train_dc_single(self, max_iter=50, hotstart=matrix([])):
        N = self.sobj.get_num_samples()
        DIMS = self.sobj.get_num_dims()

        # intermediate solutions
        # latent variables
        latent = [0.0]*N
        self.sol = self.sobj.get_hotstart_sol()
        if hotstart.size == (DIMS, 1):
            print('New hotstart position defined.')
            self.sol = hotstart
        psi = matrix(0.0, (DIMS, N))  # (dim x exm)

        obj = 1e09
        old_obj = 1e10
        rel = 1.0
        iter = 0
        is_converged = False
        # terminate if objective function value doesn't change much
        while iter < max_iter and not is_converged:
            # 1. linearized
            # for the current solution compute the 
            # most likely latent variable configuration
            num_changed = 0
            for i in range(N):
                if iter > -1:
                    (foo, lat, psi[:, i]) = self.sobj.argmax(self.sol, i, add_prior=True)
                else:
                    lat = np.int(self.sobj.y[i, 0]-1.0)
                    psi[:, i] = self.sobj.get_joint_feature_map(i, y=lat)
                if not latent[i] == lat:
                    num_changed += 1
                latent[i] = lat
            #print('{0}/{1} examples changed latent var.'.format(num_changed,N))

            # 2. Solve the intermediate optimization problem
            vecy = np.array(matrix(self.sobj.y))[:, 0]
            vecX = np.array(psi.trans())
            self.sol = self.train_model(vecX, vecy)

            # calc objective function:
            w = self.sol  # (dims x 1)
            l = self.reg  # scalar
            b = self.sobj.y  # (exms x 1)
            phi = psi  # (dims x exms)
            old_obj = obj
            obj = l*w.trans()*w + b.trans()*b - 2.0*w.trans()*phi*b + w.trans()*phi*phi.trans()*w
            rel = np.abs((old_obj - obj)/obj)
            print('Iter {0} objective={1:4.2f}  rel={2:2.4f}'.format(iter, obj[0, 0], rel[0, 0]))

            if iter > 3 and rel < 0.0001:
                is_converged = True
            iter += 1

        print np.unique(latent)
        return self.sol, latent, obj, is_converged

    def apply(self, pred_sobj):
        """ Application of the Latent Ridge Regression:
            score = max_z <sol*,\Psi(x,z)>
            latent_state = argmax_z <sol*,\Psi(x,z)> 
        """
        N = pred_sobj.get_num_samples()
        vals = matrix(0.0, (N, 1))
        structs = []
        for i in range(N):
            (vals[i], struct, psi) = pred_sobj.argmax(self.sol, i, add_prior=False)
            vals[i] += self.intercept
            structs.append(struct)
        return vals, structs


