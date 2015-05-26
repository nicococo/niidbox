import numpy as np
import scipy.optimize as op


class TransductiveCrfRegression(object):
    """ Transductive Conditional Random Fields Regression.
    """
    reg_lambda = 0.001  # (scalar) the regularization constant > 0
    reg_gamma = 1.0  # crf regularizer
    reg_theta = 0.5  # 0<= thata <= 1: trade-off between density estimation (0.0) and regression (1.0)

    v = None
    u = None
    obj = None

    crf_grad_based_opt = False

    def __init__(self, reg_theta=0.5, reg_lambda=0.001, reg_gamma=1.0, crf_grad_based_opt=True):
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.reg_theta = reg_theta
        self.crf_grad_based_opt = crf_grad_based_opt

    def crf_obj(self, x, model, psi):
        return self.reg_gamma/2.0*x.T.dot(x) - x.T.dot(psi) + model.log_partition(x)

    def crf_grad(self, x, model, psi):
        return self.reg_gamma*x - psi + model.log_partition_derivative(x)

    def estimate_crf_parameters(self, v, psi, model):
        # solve the crf
        # hot start
        vstar = 1.0 / self.reg_gamma * psi
        # res = op.minimize(self.crf_obj, x0=vstar, args=(model, psi), method='L-BFGS-B')
        # print res.nfev, ' - ', res.nit, ' - ', res.fun
        if not self.crf_grad_based_opt:
            res = op.minimize(self.crf_obj, jac=self.crf_grad, x0=vstar, args=(model, psi), method='L-BFGS-B')
            # print res.nfev, ' - ', res.nit, ' - ', res.fun
            return res.fun, res.x
        else:
            # avoid objective function value calls:
            # gradient based optimization
            grad = vstar
            step_len = np.linalg.norm(vstar)/100.0  # 1%
            cnt_iter = 0
            max_iter = 100
            while (cnt_iter < 3 or np.linalg.norm(grad)/(float(grad.size)**2) >= 1e-3) and cnt_iter <= max_iter:
                grad = self.crf_grad(vstar, model, psi)
                vstar -= step_len*grad/np.linalg.norm(grad)
                # print cnt_iter, '-', self.crf_obj(vstar, model, psi), ' - ', np.linalg.norm(grad)/(float(grad.size)**2)
                cnt_iter += 1
            return 0.0, vstar
        # print self.crf_obj(vstar, model, psi)
        # print cnt_iter, '-', self.crf_obj(vstar, model, psi), ' - ', np.linalg.norm(grad)/(float(grad.size)**2)
        # print '-----------'
        # print op.check_grad(self.crf_obj, self.crf_grad, vstar, model, psi)
        # print '-----------'

    def estimate_regression_parameters(self, X, y):
        # solve the ridge regression problem
        E = np.zeros((X.shape[1], X.shape[1]))
        np.fill_diagonal(E, self.reg_lambda)
        XXt = X.T.dot(X) + E
        XtY = (X.T.dot(y))
        if XXt.size > 1:
            u = np.linalg.inv(XXt).dot(XtY)
        else:
            u = 1.0 / XXt * XtY
        obj = self.reg_lambda/2.0*u.dot(u) + y.dot(y)/2.0 - u.dot(X.T.dot(y)) + u.dot(X.T.dot(X.dot(u)))/2.0
        return obj, u

    def predict(self, model):
        """ Assume the model as used for training.  """
        structs, phis = model.get_joint_feature_maps(predict=True)
        vals = self.u.dot(phis)
        print np.unique(structs)
        return vals, structs

    def fit(self, model, max_iter=50, n_init=4):
        best_sol = [1e14, None, None, None]
        for i in range(n_init):
            self.fit_single_run(model, max_iter=max_iter)
            if self.obj < best_sol[0] or i == 0:
                best_sol = [self.obj, self.u, self.v, model.latent]
        self.obj, self.u, self.v, model.latent = best_sol
        print self.obj

    def fit_single_run(self, model, max_iter=50, hotstart=None):
        u, v = model.get_hotstart()
        if hotstart is not None:
            print('Manual hotstart position defined.')
            u, v = hotstart

        obj = 1e09
        cnt_iter = 0
        is_converged = False

        # best objective, u and v
        best_sol = [1e14, None, None]

        # terminate if objective function value doesn't change much
        while cnt_iter < max_iter and not is_converged:
            # 1. infer the latent states given the current intermediate solutions u and v
            phis, psi = model.maps([self.reg_theta, u, v])

            # 2. solve the crf parameter estimation problem
            obj_crf, v = self.estimate_crf_parameters(v, psi, model)

            # 3. estimate new regression parameters
            obj_regression, u = self.estimate_regression_parameters(phis.T, model.labels)

            # 4.a. check termination based on objective function progress
            old_obj = obj
            obj = self.reg_theta * obj_regression + (1.0 - self.reg_theta) * obj_crf
            rel = np.abs((old_obj - obj) / obj)
            print('Iter={0} objective={1:4.2f} rel={2:2.4f} lats={3}'.format(
                cnt_iter, obj, rel, np.unique(model.latent).size))
            if best_sol[0] > obj:
                best_sol = [obj, u, v]
            if cnt_iter > 3 and rel < 0.0001:
                is_converged = True

            # 4.b. check termination based on latent states changes
            if model.get_latent_diff == 0:
                is_converged = True

            cnt_iter += 1
        self.obj, self.u, self.v = best_sol
        return is_converged