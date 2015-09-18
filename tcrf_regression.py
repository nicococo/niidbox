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

    def __init__(self, reg_theta=0.5, reg_lambda=0.001, reg_gamma=1.0):
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.reg_theta = reg_theta

    def crf_obj(self, x, model, psi):
        # foo = 0.1*np.ones((3, 3))
        # for s in range(3):
        #     foo[s, s] = 0.8
        # x[:3*3] = foo.reshape(3*3)
        xn = model.unpack_param(x)
        dims = xn.size
        Q = np.diag(1000.0*self.reg_gamma*np.ones(dims))
        for i in range(model.S*model.S):
            Q[i, i] *= 0.001

        #print 1.0/2.0*xn.T.dot(Q.dot(xn))
        #print self.reg_gamma/2.0*xn.T.dot(xn)
        #return self.reg_gamma/2.0*xn.T.dot(xn) - xn.T.dot(psi) + model.log_partition(xn)
        return 1.0/2.0*xn.T.dot(Q.dot(xn)) - xn.T.dot(psi) + model.log_partition(xn)

    def crf_grad(self, x, model, psi):
        xn = model.unpack_param(x)
        start = self.reg_gamma*xn
        grad_log_part = model.log_partition_derivative(xn)
        print grad_log_part
        print grad_log_part.size
        return start - psi + grad_log_part

    def estimate_crf_parameters(self, v, psi, model, use_grads=True):
        vstar = v
        if use_grads:
            res = op.minimize(self.crf_obj, jac=self.crf_grad, x0=vstar, args=(model, psi), method='L-BFGS-B')
        else:
            res = op.minimize(self.crf_obj, x0=vstar, args=(model, psi), method='L-BFGS-B')
        print res.nfev, ' - ', res.nit, ' - ', res.fun

        # foo = 0.1*np.ones((3, 3))
        # for s in range(3):
        #     foo[s, s] = 0.8
        # res.x[:3*3] = foo.reshape(3*3)

        print model.unpack_param(res.x)
        return res.fun, res.x
        # else:
        #     # avoid objective function value calls:
        #     # gradient based optimization
        #     grad = vstar
        #     step_len = np.linalg.norm(vstar)/10.0  # 1%
        #     cnt_iter = 0
        #     max_iter = 10
        #     while (cnt_iter < 3 or np.linalg.norm(grad)/(float(grad.size)**2) >= 1e-3) and cnt_iter <= max_iter:
        #         grad = self.crf_grad(vstar, model, psi)
        #         vstar -= step_len*grad/np.linalg.norm(grad)
        #         print 'GD: ',  cnt_iter, '-', np.linalg.norm(grad)/(float(grad.size)**2)
        #         cnt_iter += 1
        #     return 0.0, vstar
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

    def fit(self, model, max_iter=50, n_init=5, use_grads=True):
        best_sol = [1e14, None, None, None]
        for i in range(n_init):
            self.fit_single_run(model, max_iter=max_iter, use_grads=use_grads)
            if self.obj < best_sol[0] or i == 0:
                best_sol = [self.obj, self.u, self.v, model.latent]
        self.obj, self.u, self.v, model.latent = best_sol
        print self.obj

    def fit_single_run(self, model, max_iter=50, hotstart=None, use_grads=True):
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
            vn = model.unpack_param(v)
            phis, psi = model.maps([self.reg_theta, u, vn])

            # 2. solve the crf parameter estimation problem
            obj_crf, v = self.estimate_crf_parameters(v, psi, model, use_grads=use_grads)

            # 3. estimate new regression parameters
            obj_regression, u = self.estimate_regression_parameters(phis.T, model.labels)

            # 4.a. check termination based on objective function progress
            old_obj = obj
            obj = self.reg_theta * obj_regression + (1.0 - self.reg_theta) * obj_crf
            rel = np.abs((old_obj - obj) / obj)
            print('Iter={0} regr={1:4.2f} crf={2:4.2f}; objective={3:4.2f} rel={4:2.4f} lats={5}'.format(
                cnt_iter, obj_regression, obj_crf, obj, rel, np.unique(model.latent).size))
            if best_sol[0] > obj:
                best_sol = [obj, u, v]
            if cnt_iter > 3 and rel < 0.0001:
                is_converged = True
            if best_sol[0] > obj:
                best_sol = [obj, u, v]
            if np.isinf(obj) or np.isnan(obj):
                return False

            cnt_iter += 1
        self.obj, self.u, self.v = best_sol

        print self.v
        return is_converged