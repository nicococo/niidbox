__author__ = 'nicococo'

import numpy as np
import scipy.sparse as sparse

from structured_object import TransductiveStructuredModel


class TCrfRIndepModel(TransductiveStructuredModel):
    """ [T]ransductive [C]onditional [R]andom [F]ield [R]egression for [Indep]endent examples [Model].
        Number of latent states must be set in advance. Target values 'labels' are
        continuous regression targets for a subset of data entries.
    """
    states = -1  # (scalar) number of hidden states

    N = None  # Labeled neighbors for each unlabeled datapoint based on sparse Adjacency matrix
    N_weights = None
    A = None

    psi = None  # copy of the current joint feature map, corresponding to self.latent
    phis = None  # copy of the current joint feature map, corresponding to self.latent
    sol_dot_psi = None

    def __init__(self, data, labels, label_inds, unlabeled_inds, states, A):
        TransductiveStructuredModel.__init__(self, data, labels, label_inds, unlabeled_inds)
        self.states = states
        # A should be a sparse lil_matrix
        # 1. find the max length
        max_len = 0
        for ind in self.unlabeled_inds:
            foo = np.intersect1d(self.label_inds, A.rows[ind])
            max_len = np.max([len(foo), max_len])
        print max_len
        # 2. fill the data with max length:
        self.N = np.zeros((len(self.unlabeled_inds), max_len), dtype='i')
        self.N_weights = np.ones((len(self.unlabeled_inds), max_len), dtype='d')
        idx = 0
        for ind in self.unlabeled_inds:
            foo = np.intersect1d(self.label_inds, A.rows[ind])
            lens = len(foo)
            self.N[idx, :lens] = foo
            if lens < max_len:
                if lens > 0:
                    self.N[idx, lens:] = foo[0]
                    weight = 1./float(max_len - lens + 1)
                    self.N_weights[idx, 0] = weight
                    self.N_weights[idx, lens:] = weight
                else:
                    self.N[idx, :] = 0
                    self.N_weights[idx, :] = 0.0
            idx += 1
        # self.A = np.array(A.todense())

    def get_num_dims(self):
        return self.get_num_feats()*self.states

    def get_labeled_predictions(self, sol):
        # for debugging only
        phis = np.zeros((self.get_num_dims(), self.samples))
        for s in range(self.states):
            inds = np.where(self.latent == s)[0]
            phis[s*self.feats:(s+1)*self.feats, inds] = self.data[:, inds]
        return sol.dot(phis[:, self.label_inds])

    def get_joint_feature_maps(self, predict=False):
        phis = np.zeros((self.get_num_dims(), self.samples))
        for s in range(self.states):
            inds = np.where(self.latent == s)[0]
            phis[s*self.feats:(s+1)*self.feats, inds] = self.data[:, inds]
        if predict:
            return self.latent[self.unlabeled_inds], phis[:, self.unlabeled_inds]
        else:
            return phis[:, self.label_inds], np.sum(phis, axis=1)

    def get_hotstart(self):
        return [np.random.randn(self.get_num_dims()), np.random.randn(self.get_num_dims())]

    def log_partition_derivative(self, sol):
        v = sol.reshape((self.feats, self.states), order='F')
        # (A)
        f = np.zeros((self.states, self.samples))
        for s in range(self.states):
            f[s, :] = np.exp(v[:, s].dot(self.data))
        sum_f = np.sum(f, axis=0)
        # (B)
        for s in range(self.states):
            f[s, :] /= sum_f
        # (C)
        phis = np.zeros((self.get_num_dims(), self.samples))
        idx = 0
        for s in range(self.states):
            for feat in range(self.feats):
                phis[idx, :] = self.data[feat, :] * f[s, :]
                idx += 1
        return np.sum(phis, axis=1)

    def log_partition(self, sol):
        v = sol.reshape((self.feats, self.states), order='F')
        # log sum_Z exp(<sol, Psi(X,Z)>)
        # = log sum_Z exp(<sol, sum_i Phi(x_i,z_i)>)
        # = log sum_z1..zN prod_i exp(<sol, Phi(x_i,z_i)>)
        # = log sum_zN..z2 sum_z1 exp(<sol, Phi(x_1,z_1)>)*prod_{i/1}exp(<sol, Phi(x_i,z_i)>)
        # = log sum_zN..z2 prod_{i/1}exp(<sol, Phi(x_i,z_i)>) * (sum_z1 exp(<sol, Phi(x_1,z_1)>))
        f_inner = np.zeros((self.states, self.samples))
        for s in range(self.states):
            f_inner[s, :] = v[:, s].dot(self.data)
        max_score = np.max(f_inner)
        f_inner = np.sum(np.exp(f_inner - max_score), axis=0)
        foo = np.sum(np.log(f_inner) + max_score)
        if np.isnan(foo) or np.isinf(foo):
            print 'TCRFR Indep Model: the log_partition is NAN or INF!!'
        return foo

    def maps(self, sol):
        theta = sol[0]
        u = sol[1].reshape((self.feats, self.states), order='F')
        v = sol[2].reshape((self.feats, self.states), order='F')
        map_objs = np.zeros((self.states, self.samples))

        for s in range(self.states):
            map_objs[s, :] = (1.0 - theta)*v[:, s].dot(self.data)
            f_squares = self.labels - u[:, s].dot(self.data[:, self.label_inds])
            map_objs[s, self.label_inds] -= theta/2. * f_squares*f_squares

        # highest value first
        if self.latent is not None:
            self.latent_prev = self.latent.copy()
        self.latent = np.argmax(map_objs, axis=0)

        # number of labeled neighbors that are in this state
        # y = self.latent[self.label_inds]
        # B = self.A[self.unlabeled_inds, :]
        yn = self.latent[self.N]
        for s in range(self.states):
            # inds = np.where(s == y)[0]
            # cnts = np.sum(B[:, self.label_inds[inds]], axis=1)
            n_cnts = np.sum(np.array((yn == s), dtype='d')*self.N_weights, axis=1)
            # if any(np.abs(n_cnts-cnts) > 0.1):
            #     print n_cnts[n_cnts != cnts]
            #     print cnts[n_cnts != cnts]
            #     print '**************************'
            #     print n_cnts
            #     print '--------------------------'
            #     print cnts
            #     print '=========================='
            #     print n_cnts == cnts
            #     print '++++++++++++++++++++++++++'
            map_objs[s, self.unlabeled_inds] += 100.*n_cnts

        # highest value first
        if self.latent is not None:
            self.latent_prev = self.latent.copy()
        self.latent = np.argmax(map_objs, axis=0)

        phis, psi = self.get_joint_feature_maps()
        self.psi = psi
        self.phis = phis
        self.sol_dot_psi = sol[2].T.dot(psi)
        # print np.unique(self.latent)
        return phis, psi
