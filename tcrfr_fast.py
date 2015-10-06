__author__ = 'nicococo'
import numpy as np
from cvxopt import matrix, spmatrix, sparse
import cvxopt.solvers as solver
import mosek as msk

import scipy.sparse as sparse

from abstract_tcrfr import AbstractTCRFR

class TCRFR_Fast(AbstractTCRFR):
    """ Pairwise Conditional Random Field for transductive regression.
    """
    psi = None  # copy of the current joint feature map, corresponding to self.latent
    phis = None  # copy of the current joint feature map, corresponding to self.latent
    sol_dot_psi = None

    def __init__(self, data, labels, label_inds, unlabeled_inds, states, A,
                 reg_theta=0.5, reg_lambda=0.001, reg_gamma=1.0, trans_regs=[1.0, 1.0], trans_sym=[1], lbl_weight=1.0):
        AbstractTCRFR.__init__(self, data, labels, label_inds, unlabeled_inds, states, A,
                 reg_theta, reg_lambda, reg_gamma, trans_regs, trans_sym)

        # labeled examples get an extra weight (parameter)
        for ind in self.V:
            for i in range(self.N[ind, :].size):
                if self.N[ind, i] in self.label_inds and self.N_weights[ind, i]>0.00001:
                    self.N_weights[ind, i] = lbl_weight


    def map_inference(self, u, vn):
        theta = self.reg_theta
        u = u.reshape((self.feats, self.S), order='F')
        v = vn[self.trans_d_full*self.trans_n:]
        v = v.reshape((self.feats, self.S), order='F')
        map_objs = np.zeros((self.S, self.samples))

        for s in range(self.S):
            map_objs[s, :] = (1.0 - theta)*v[:, s].dot(self.data)
            f_squares = self.labels - u[:, s].dot(self.data[:, self.label_inds])
            map_objs[s, self.label_inds] -= theta/2. * f_squares*f_squares

        # highest value first
        if self.latent is not None:
            self.latent_prev = self.latent.copy()
        self.latent = np.argmax(map_objs, axis=0)

        lats = self.latent.copy()
        map_objs_bak = map_objs.copy()
        for i in range(1):

            yn = lats[self.N]
            for s in range(self.S):
                add = 0.0
                for s2 in range(self.S):
                    n_cnts = np.sum(np.array((yn[:,:] == s2), dtype='d')*self.N_weights[:,:], axis=1)
                    add += vn[self.trans_mtx2vec_full[s, s2]]*n_cnts
                map_objs[s, :] += add

            lats_b = np.argmax(map_objs[:, self.unlabeled_inds], axis=0)
            print np.sum(lats!=lats_b)/float(lats.size)
            lats[self.unlabeled_inds] = lats_b
            map_objs = map_objs_bak.copy()

        # highest value first
        if self.latent is not None:
            self.latent_prev = self.latent.copy()
        self.latent = lats

        phis, psi = self.get_joint_feature_maps()
        self.psi = psi
        self.phis = phis
        self.sol_dot_psi = vn.T.dot(psi)
        # print np.unique(self.latent)
        return phis, psi

    def log_partition(self, v):
        # pseudolikelihood approximation = fix the neighbors
        yn = self.latent[self.N]
        v_em = v[self.trans_n*self.trans_d_full:].reshape((self.feats, self.S), order='F')
        f_inner = np.zeros((self.S, self.samples))
        for s in range(self.S):
            foo = np.zeros(self.samples)
            for s2 in range(self.S):
                n_cnts = np.sum(np.array((yn[:,:] == s2), dtype='d')*self.N_weights[:,:], axis=1)
                foo += v[self.trans_mtx2vec_full[s, s2]]*n_cnts
            f_inner[s, :] = v_em[:, s].dot(self.data) + foo
        max_score = np.max(f_inner)
        f_inner = np.sum(np.exp(f_inner - max_score), axis=0)
        foo = np.sum(np.log(f_inner) + max_score)
        if np.isnan(foo) or np.isinf(foo):
            print 'TCRFR Pairwise Potential Model: the log_partition is NAN or INF!!'
        return foo

    def log_partition_derivative(self, v):
        v_trans = v[:self.S*self.S].reshape((self.S, self.S), order='C')
        v_em = v[self.trans_n*self.trans_d_full:].reshape((self.feats, self.S), order='C')

        # (A)
        f = np.zeros((self.S, self.samples))
        for s in range(self.S):
            w = v_trans[s, self.latent]
            foo = np.zeros(self.samples)
            #for n in range(len(self.N)):
            #    foo[n] = np.sum(w[self.N[n]])
            f[s, :] = np.exp(v_em[:, s].dot(self.data) + foo)
        sum_f = np.sum(f, axis=0)

        # (B)
        for s in range(self.S):
            f[s, :] /= sum_f

        # (C)
        phis = np.zeros((self.get_num_compressed_dims(), self.samples))
        for s in range(self.S):
            foo = np.zeros(self.samples)
            for n in range(len(self.N)):
                foo[n] = np.sum(self.N[n]==s)
            phis[s, :] = foo * f[s, :]

        idx = self.trans_n*self.trans_d_full
        for s in range(self.S):
            for feat in range(self.feats):
                phis[idx, :] = self.data[feat, :] * f[s, :]
                idx += 1
        return np.sum(phis, axis=1)

