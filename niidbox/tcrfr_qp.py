__author__ = 'nicococo'
import numpy as np
from cvxopt import matrix, spmatrix, sparse
import cvxopt.solvers as solver
# import mosek as msk  # can not be imported from anaconda environment

from numba import autojit

from abstract_tcrfr import AbstractTCRFR
from utils import profile

class TCRFR_QP(AbstractTCRFR):
    """ Pairwise Conditional Random Field for transductive regression.
    """

    # pre-computed linear program approximation constraints
    qp_eq_A = None
    qp_eq_b = None
    qp_ineq_G = None
    qp_ineq_h = None

    psi = None  # copy of the current joint feature map, corresponding to self.latent
    phis = None  # copy of the current joint feature map, corresponding to self.latent

    def __init__(self, data, labels, label_inds, states, A,
                 reg_theta=0.5, reg_lambda=0.001, reg_gamma=1.0, trans_regs=[1.0, 1.0], trans_sym=[1], verbosity_level=1):
        AbstractTCRFR.__init__(self, data, labels, label_inds, states, A,
                 reg_theta, reg_lambda, reg_gamma, trans_regs, trans_sym, verbosity_level=verbosity_level)
        # pre-compute qp relaxation constraints
        self.qp_relax_init()

    @profile
    def map_inference(self, u, v):
        if self.latent is not None:
            self.latent_prev = self.latent
        self.latent = self.qp_relax_max(u, v, self.reg_theta)
        return self.get_joint_feature_maps()

    @profile
    def qp_relax_init(self):
        """ Pre-calulate the equality and inequality constants for the
            linear program relaxation.
                min_x   <c,x>
                s.t.    Gx+s=h
                        Ax = b
                        s >= 0  (element-wise)
        """
        states = self.S
        edges = len(self.E)
        dims = edges*states*states + len(self.V)*states
        offset = edges*states*states
        if self.verbosity_level>=1:
            print('Init constraint matrices for relaxed QP with {0} marginal and {1} vertex constraints.'.format(2*edges*states, len(self.V)))
        num_constr = 2*edges*states + len(self.V)
        A = spmatrix([0.0],[0],[0],(num_constr, dims))
        b = matrix(0.0, (num_constr, 1))
        row = 0
        cnt = 0
        # pair-wise potentials
        for e in self.E:
            ind_i = e[0]
            ind_j = e[1]
            for s in range(states):
                for k in range(states):
                    A[row, cnt + s*states + k] = 1.0
                A[row, offset + ind_i*states + s] = -1.0
                row += 1
            for k in range(states):
                for s in range(states):
                    A[row, cnt + k + s*states] = 1.0
                A[row, offset + ind_j*states + k] = -1.0
                row += 1
            cnt += states*states
        num_margs = row
        # emissions
        for v in self.V:
            # emission constraints: sum_s x_is = 1
            A[row, offset:offset + states] = 1.0
            b[row] = 1.0
            row += 1
            offset += states
        self.qp_eq_A = A
        self.qp_eq_b = b
        # lower bounds
        self.qp_ineq_G = spmatrix(-1.0, range(dims), range(dims))
        self.qp_ineq_h = matrix(0.0, (dims, 1))
        if self.verbosity_level>=1:
            print('There are {0} marginal contraints and {1} vertex constraints.'.format(num_margs, len(b)-num_margs))

    @profile
    def qp_relax_max(self, u, v, theta):
        """ Estimate the MAP by relaxing the integer quadratic program
            to a quadratic program (QP):
                \mbox{minimize}   (1/2) xT P x + qT x \\
                \mbox{subject to} Gx <= h
                                  Ax  = b
        """
        # set optimization parameters
        G = self.qp_ineq_G
        h = self.qp_ineq_h
        A = self.qp_eq_A
        b = self.qp_eq_b
        # convert u to Q
        P, q = self.get_qp_params(u, v, theta)

        # solver.options['MOSEK'] = {msk.iparam.log: 0}   # cannot be called from anaconda environment
        solution = solver.qp(P, q, G, h, A, b, solver='mosek')
        res = solution['x']

        # print solution['primal objective']
        # obj = matrix(solution['primal objective'])
        # convert into state sequence
        edges = len(self.E)
        states = self.S
        vertices = len(self.V)
        offset = edges*states*states
        max_states = np.zeros(vertices, dtype='i')

        # error check
        if self.verbosity_level>=2:
            print solution['status']
        if res is None:
            print('QP optimization did not finish (status):')
            print 'max P - ', max(P), ' - min P - ', min(P)
            # print 'Non-numbers in P - ', any(np.isnan(P)), any(np.isinf(P))
            print 'max q - ', max(q), ' - min q - ', min(q)
            # print 'Non-numbers in q - ', any(np.isnan(q)), any(np.isinf(q))
            if self.latent is None:
                return max_states
            else:
                return self.latent

        for v in range(vertices):
            max_states[v] = np.argmax(res[offset + v*states:offset + v*states + states])
        return max_states

    def test_qp_param(self):
        # generate random latent states
        test_lats = np.random.randint(0, 1, self.data.shape[1])
        # generate random solutions
        u, v = self.get_hotstart()
        theta = np.random.rand()

        psi = self.get_crf_joint_feature_map(test_lats)
        y_preds = self.get_labeled_predictions(u, test_lats)

        # get the qp-transformed parameters
        cP, cq = self.get_qp_params(u, v, theta)
        P = np.array(matrix(cP))
        q = np.array(cq)

        states = self.S
        edges = len(self.E)
        vertices = len(self.V)
        offset = edges*states*states
        dims = edges*states*states + vertices*states

        x = np.zeros(dims)
        cnt = 0
        for e in self.E:
            ind_i = e[0]
            ind_j = e[1]
            for s1 in range(states):
                for s2 in range(states):
                    if s1 == test_lats[ind_i] and s2 == test_lats[ind_j]:
                        x[cnt] = 1.0
                    cnt += 1
        cnt = offset
        for i in range(len(self.V)):
            for s in range(states):
                if test_lats[i] == s:
                    x[cnt] = 1.0
                cnt += 1

        # our true objective value
        obj = theta/2.0 * np.sum((y_preds-self.labels)**2.0) - (1.0-theta)*v.T.dot(psi)
        # the linear part only
        y2 = 0.5*theta*np.sum(self.labels*self.labels)
        obj_int_lp = -theta * np.sum(y_preds*self.labels) - (1.0-theta)*v.T.dot(psi)
        obj_int_qp = 0.5*theta * np.sum(y_preds*y_preds) + y2
        # the qp transformed objective
        obj_transf = 0.5*x.T.dot(P.dot(x)) + q.T.dot(x) + y2

        print 'QP: ', obj, ' - ', obj_transf
        print 'Linear part only: ', obj_int_lp, ' - ', q.T.dot(x)
        print 'Quadratic part only: ', obj_int_qp - y2, ' - ', 0.5*x.T.dot(P.dot(x))
        print obj_int_lp + obj_int_qp

    @profile
    def get_qp_params(self, param_u, param_v, theta):
        print param_v.size
        states = self.S
        edges = len(self.E)
        vertices = len(self.V)
        feats = self.get_num_feats()
        dims = edges*states*states + vertices*states
        # build weighting term
        d = matrix(0.0, (dims, 1))
        c = matrix(0.0, (dims, 1))

        # pair-wise potentials
        cnt = 0
        for e in self.E:
            etype = self.A[e[0], e[1]]
            etype_offset = (etype-1)*self.trans_d_full
            d[cnt:cnt+states*states] = param_v[etype_offset:etype_offset+states*states]
            cnt += states*states

        # emissions
        offset = edges*states*states
        cnt = offset
        for v in self.V:
            sol_cnt = self.trans_n*states*states
            for s in range(states):
                d[cnt] = param_v[sol_cnt:sol_cnt + feats].T.dot(self.data[:, v])
                sol_cnt += feats
                cnt += 1

        P = spmatrix(0.0, range(dims), range(dims))

        # ridge regression part
        cnt = offset
        idx = 0
        u = param_u.reshape((self.feats, self.S), order='F')
        for ind in self.label_inds:
            i_start = cnt + ind*self.S
            i_end = cnt + (ind+1)*self.S
            c[i_start:i_end] = self.labels[idx]*u.T.dot(self.data[:, ind])

            for s1 in range(states):
                for s2 in range(states):
                    P[i_start+s1, i_start+s2] = 2.0*u[:, s2].T.dot(self.data[:, ind]) * u[:, s1].T.dot(self.data[:, ind])
            # next label
            idx += 1

        return theta/2.0*P, -theta*c-(1.0-theta)*d
