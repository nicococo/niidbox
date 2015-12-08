__author__ = 'nicococo'
import numpy as np
from abstract_tcrfr import AbstractTCRFR

class TCRFR_lbpa(AbstractTCRFR):
    """ Pairwise Conditional Random Field for transductive regression.
    """
    psi = None  # copy of the current joint feature map, corresponding to self.latent
    phis = None  # copy of the current joint feature map, corresponding to self.latent

    fix_lbl_map = False  # fix the labeled data in the inference (only infer once after calling map_inference)?

    def __init__(self, data, labels, label_inds, unlabeled_inds, states, A,
                 reg_theta=0.5, reg_lambda=0.001, reg_gamma=1.0, trans_regs=[1.0, 1.0],
                 trans_sym=[1], lbl_weight=1.0, verbosity_level=1):
        AbstractTCRFR.__init__(self, data, labels, label_inds, unlabeled_inds, states, A,
                 reg_theta, reg_lambda, reg_gamma, trans_regs, trans_sym, verbosity_level=verbosity_level)

        # labeled examples get an extra weight (parameter)
        for ind in self.V:
            for i in range(self.N[ind, :].size):
                if self.N[ind, i] in self.label_inds and self.N_weights[ind, i]>0.00001:
                    self.N_weights[ind, i] = lbl_weight

    def log_partition(self, v):
        # This function calculates/estimates the log-partition function by
        # pseudolikelihood approximation. Therefore, we assume the states for
        # the neighbors fixed (e.g. from previous map inference).
        #
        # log Z = log \sum_z exp( <v,\Psi(X,z)> )    # intractable even for small z
        #
        # Hence, for a node i in state s given the neighbors j with fixed states n_j:
        #       f(i, s) = f_em(i, s) + sum_j f_trans(i=s, j=n_j)
        #

        # self.N is a (Nodes x max_connection_count) Matrix containing the indices for each neighbor
        # of each node (indices are 0 for non-neighbors, therefore N_weights is need to multiply this
        # unvalid value with 0.
        yn = self.latent[self.N] # (nodex x max_connection_count) latent state matrix
        v_em = v[self.trans_n*self.trans_d_full:].reshape((self.feats, self.S), order='F')
        f_inner = np.zeros((self.S, self.samples))
        for s1 in range(self.S):
            foo = np.zeros(self.samples)
            for s2 in range(self.S):
                n_cnts = np.sum(np.array((yn == s2), dtype='d')*self.N_weights, axis=1)
                foo += v[self.trans_mtx2vec_full[s1, s2]]*n_cnts
            f_inner[s1, :] = v_em[:, s1].dot(self.data) + foo
        # exp-trick (to prevent NAN because of large numbers): log[sum_i exp(x_i-a)]+a = log[sum_i exp(x_i)]
        max_score = np.max(f_inner)
        f_inner = np.sum(np.exp(f_inner - max_score), axis=0)
        foo = np.sum(np.log(f_inner) + max_score)
        if np.isnan(foo) or np.isinf(foo):
            print('TCRFR Pairwise Potential Model: the log_partition is NAN or INF!!')
        return foo

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

        iter = 0
        change = 1.0
        max_iter = 10
        lats = self.latent.copy()
        map_objs_bak = map_objs.copy()
        while change>0.001 and iter<max_iter:

            yn = lats[self.N]
            for s in range(self.S):
                add = 0.0
                for s2 in range(self.S):
                    n_cnts = np.sum(np.array((yn[:,:] == s2), dtype='d')*self.N_weights[:,:], axis=1)
                    add += vn[self.trans_mtx2vec_full[s, s2]]*n_cnts
                map_objs[s, :] += add

            # if this is true, then only change the latent states for unlabeled examples
            if self.fix_lbl_map:
                lats_b = np.argmax(map_objs[:, self.unlabeled_inds], axis=0)
                change = np.sum(lats!=lats_b)/float(lats.size)
                lats[self.unlabeled_inds] = lats_b
            else:
                lats_b = np.argmax(map_objs, axis=0)
                change = np.sum(lats!=lats_b)/float(lats.size)
                lats = lats_b

            iter += 1
            if self.verbosity_level>=2:
                print "(", iter, "): ", change
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
