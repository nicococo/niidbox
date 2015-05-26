__author__ = 'nicococo'

import numpy as np

from structured_object import TransductiveStructuredModel


class TCrfRIndepModel(TransductiveStructuredModel):
    """ [T]ransductive [C]onditional [R]andom [F]ield [R]egression for [Indep]endent examples [Model].
        Number of latent states must be set in advance. Target values 'labels' are
        continuous regression targets for a subset of data entries.
    """
    states = -1  # (scalar) number of hidden states

    def __init__(self, data, labels, label_inds, unlabeled_inds, states):
        TransductiveStructuredModel.__init__(self, data, labels, label_inds, unlabeled_inds)
        self.states = states

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

    def log_partition_derivatives(self, sol):
        super(TCrfRIndepModel, self).log_partition_derivatives(sol)

    def log_partition(self, sol):
        v = sol.reshape((self.feats, self.states), order='F')
        # log sum_Z exp(<sol, Psi(X,Z)>)
        # = log sum_Z exp(<sol, sum_i Phi(x_i,z_i)>)
        # = log sum_z1..zN prod_i exp(<sol, Phi(x_i,z_i)>)
        # = log sum_zN..z2 sum_z1 exp(<sol, Phi(x_1,z_1)>)*prod_{i/1}exp(<sol, Phi(x_i,z_i)>)
        # = log sum_zN..z2 prod_{i/1}exp(<sol, Phi(x_i,z_i)>) * (sum_z1 exp(<sol, Phi(x_1,z_1)>))
        f_inner = np.zeros((self.states, self.samples))
        for s in range(self.states):
            f_inner[s, :] = np.exp(v[:, s].dot(self.data))
        f_inner = np.sum(f_inner, axis=0)
        return np.log(np.prod(f_inner))

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
        self.latent = np.argmax(map_objs, axis=0)
        # print np.unique(self.latent)
        return self.get_joint_feature_maps()
