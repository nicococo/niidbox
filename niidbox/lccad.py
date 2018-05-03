__author__ = 'Nico Goernitz, TU Berlin, 2015'
import numpy as np

from numba import jit

from abstract_tcrfr import AbstractTCRFR, _extern_get_crf_joint_feature_map
from utils import profile


class LCCAD(AbstractTCRFR):
    """ Latent-class Contextual Anomaly Detector.
    """

    def __init__(self, data, states, A, reg_theta=0.5, reg_gamma=1.0, verbosity_level=1):
        AbstractTCRFR.__init__(self, data, np.zeros(data.shape[1]), np.arange(data.shape[1]), states, A,
                 reg_theta, 1.0, reg_gamma, [[10.0, 1.0]], [1], verbosity_level)

    @profile
    def map_inference(self, u, v):
        if self.latent is not None:
            self.latent_prev = self.latent

        # choose MAP inference scheme
        self.latent = _extern_map_lbp_svdd(self.data, self.N, self.N_inv, self.N_weights, u, v,
                                      self.reg_theta, self.feats, self.samples, self.S, self.trans_d_full, self.trans_n, \
                                      self.trans_mtx2vec_full, self.verbosity_level)

        return self.get_joint_feature_maps()

    @profile
    def get_joint_feature_maps(self, latent=None):
        if latent is None:
            latent = self.latent
        # Svdd Feature Map
        # (self.feats, self.samples))
        phis = self.data
        # return phis, self.get_crf_joint_feature_map(latent)
        return phis, _extern_get_crf_joint_feature_map(self.data, latent, self.E, np.int32(self.V), \
            self.trans_mtx2vec_full, self.get_num_dims(), self.get_num_feats(), \
            self.trans_d_full, self.trans_n)

    def em_estimate_u(self, X):
        u = np.zeros((self.feats, self.S))
        obj = 0.
        for s in range(self.S):
            inds = np.where(self.latent == s)[0]
            num = inds.size
            if num > 0:
                u[:, s] = 1./np.float(num) * np.sum(X[inds, :].T, axis=1)

                foo = np.tile(u[:, s][:, np.newaxis], (1, num)) - X[inds, :].T
                obj += 1./np.float(num) * np.sum(np.sum(foo*foo, axis=0))
        return obj, u

    def predict(self, lats=None):
        if lats is None:
            lats = self.latent
        # for debugging only
        scores = np.zeros(self.samples)
        for s in range(self.S):
            inds = np.where(lats == s)[0]
            foo = np.tile(self.u[:, s][:, np.newaxis], (1, inds.size)) - self.data[:, inds]
            scores[inds] = np.sum(foo*foo, axis=0)
        return scores, lats


@jit(nopython=True)
def _extern_map_lbp_svdd(data, N, N_inv, N_weights, u, v, theta, feats, samples, states, trans_d_full, trans_n, trans_mtx2vec_full, verbosity):

    unary = np.zeros((states, samples))
    latent = -np.ones(samples, dtype=np.int8)
    msgs = np.zeros((N.shape[0], N.shape[1], states))  # samples x neighbors x states
    psis = np.zeros((N.shape[0], N.shape[1], states), dtype=np.int8)  # samples x neighbors x states

    # Fast initialization: assume independent variables (no interactions)
    # Hence, latent states depend only on the unary terms (+regression terms)
    for s in range(states):
        offset = trans_d_full*trans_n
        offset += s*feats
        # crf obj for all samples
        for i in range(samples):
            for f in range(feats):
                unary[s, i] += (1.0 - theta)*v[offset+f]*data[f, i]

        # svdd part
        # offset = s*feats
        for i in range(samples):
            f_norm = 0.
            for f in range(feats):
                f_norm += (u[f, s] - data[f, i])*(u[f, s] - data[f, i])
            unary[s, i] -= theta * f_norm

    # LOOPY BELIEF: single run through all nodes
    iter = 0
    change = 1e16
    foo = np.zeros(states)
    foo_full = np.zeros(states)
    while change > 1e-5 and iter < 50:
        change = 0.0
        for i in range(samples):
            # i = np.random.randint(0, samples)
            # get neighbors and weights
            num_neighs = np.sum(N_weights[i, :])
            neighs = N[i, :num_neighs]

            for j in range(num_neighs):
                bak = msgs[i, j, :].copy()
                max_msg = -1e12
                for s in range(states):
                    for t in range(states):
                        sum_msg = 0.
                        msg_j = 0.
                        for n1 in range(num_neighs):
                            if n1 == j:
                                msg_j = msgs[neighs[n1], N_inv[i, n1], t]
                            sum_msg += msgs[neighs[n1], N_inv[i, n1], t]
                        foo[t] = unary[t, i] + (1.0 - theta)*(v[trans_mtx2vec_full[s, t]]) + sum_msg - msg_j
                        # foo_full[t] = unary[t, i] + (1.0 - theta)*(v[trans_mtx2vec_full[s, t]]+sum_msg)  # ERR! 1: msgs include transition term already
                        # foo_full[t] = unary[t, i] + (1.0 - theta)*sum_msg  # ERR! 2: msgs are (1-theta)-normalized already
                        foo_full[t] = unary[t, i] + sum_msg
                    msgs[i, j, s] = np.max(foo)
                    psis[i, j, s] = np.argmax(foo_full)
                    if msgs[i, j, s] > max_msg:
                        max_msg = msgs[i, j, s]

                # msgs[i, j, :] -= np.max((msgs[i, j, :]))  # normalization of the new message from i->j
                for m in range(states):
                    msgs[i, j, m] -= max_msg   # normalization of the new message from i->j
                change += np.sum(np.abs(msgs[i, j, :]-bak))
        iter += 1
        if verbosity >= 2:
            print change

    # BACKTRACKING INIT: choose maximizing state for the last variable that was optimized
    #i = samples-1
    num_neighs = np.sum(N_weights[i, :])
    neighs = N[i, :num_neighs]
    foo = np.zeros(states, dtype=np.float32)
    for s in range(states):
        sum_msg = 0.
        for n1 in range(num_neighs):
            sum_msg += msgs[neighs[n1], N_inv[i, n1], s]
        foo[s] = unary[s, i] + sum_msg
        #foo[s] = unary[s, i] + (1. - theta)*sum_msg  # ERR!: msgs are (1-theta)-normalized already

    # backtracking step
    # debug: return backtracking(i, latent, np.argmax(foo), psis, N, N_inv, N_weights)
    idxs = np.zeros(samples, dtype=np.int32)
    latent[i] = np.argmax(foo)
    idxs[0] = i
    cnt = 1
    for s in range(samples):
        i = idxs[s]
        for j in range(np.sum(N_weights[i, :])):
            if latent[N[i, j]] < 0:
                latent[N[i, j]] = psis[N[i, j], N_inv[i, j], latent[i]]
                idxs[cnt] = N[i, j]
                cnt += 1
    return latent
