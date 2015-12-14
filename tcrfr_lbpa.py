__author__ = 'nicococo'
import numpy as np

from numba import autojit

from abstract_tcrfr import AbstractTCRFR
from tools import profile

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

    @profile
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

    @profile
    def map_inference(self, u, v):
        if self.latent is not None:
            self.latent_prev = self.latent
        # self.latent = self.map_lbpa(u, v)
        self.latent = _extern_map_lbp(self.data, self.labels, self.label_inds, self.unlabeled_inds, \
                                      self.N, self.N_inv, self.N_weights, u, v, self.reg_theta, self.feats, \
                                      self.samples, self.S, self.trans_d_full, self.trans_n, \
                                      self.trans_mtx2vec_full, self.fix_lbl_map)

        return self.get_joint_feature_maps()

    @profile
    def map_indep(self, u, vn):
        theta = self.reg_theta
        u = u.reshape((self.feats, self.S), order='F')
        v = vn[self.trans_d_full*self.trans_n:]
        v = v.reshape((self.feats, self.S), order='F')
        map_objs = np.zeros((self.S, self.samples))

        for s in range(self.S):
            map_objs[s, :] = (1.0 - theta)*v[:, s].dot(self.data)
            f_squares = self.labels - u[:, s].dot(self.data[:, self.label_inds])
            map_objs[s, self.label_inds] -= theta/2. * f_squares*f_squares

        return np.argmax(map_objs, axis=0)

    @profile
    def map_lbpa(self, u, vn):
        theta = self.reg_theta
        u = u.reshape((self.feats, self.S), order='F')
        v = vn[self.trans_d_full*self.trans_n:]
        v = v.reshape((self.feats, self.S), order='F')
        map_objs = np.zeros((self.S, self.samples))

        for s in range(self.S):
            map_objs[s, :] = (1.0 - theta)*v[:, s].dot(self.data)
            f_squares = self.labels - u[:, s].dot(self.data[:, self.label_inds])
            map_objs[s, self.label_inds] -= theta/2. * f_squares*f_squares

        latent = np.argmax(map_objs, axis=0)

        iter = 0
        change = 1.0
        max_iter = 10
        lats = latent.copy()
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
        return lats


@autojit(nopython=True)
def _extern_map_lbp(data, labels, label_inds, unlabeled_inds, N, N_inv, N_weights, \
             u, v, theta, feats, samples, states, trans_d_full, trans_n, trans_mtx2vec_full, fix_lbl_map):

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
        # regression term for labeled examples only
        offset = s*feats
        for i in range(label_inds.size):
            f_squares = labels[i]
            for f in range(feats):
                f_squares -= u[offset+f]*data[f, label_inds[i]]
            unary[s, label_inds[i]] -= theta/2. * f_squares*f_squares

    # LOOPY BELIEF: single run through all nodes
    iter = 0
    change = 1e16
    foo = np.zeros(states)
    while change>1e-3 and iter<50:
        change = 0.0
        for i in range(samples):
            # get neighbors and weights
            num_neighs = np.sum(N_weights[i, :])
            neighs = N[i, :num_neighs]

            for j in range(num_neighs):
                bak = msgs[i, j, :].copy()
                max_msg = -1e12
                for s in range(states):
                    for t in range(states):

                        sum_msg = 0.
                        for n1 in range(num_neighs):
                            if not n1==j:
                                sum_msg += msgs[neighs[n1], N_inv[i, n1], t]

                        foo[t] = unary[t, i] + (1.0 - theta)*(v[trans_mtx2vec_full[s, t]] + sum_msg)
                    msgs[i, j, s] = np.max(foo)
                    psis[i, j, s] = np.argmax(foo)
                    if msgs[i, j, s] > max_msg:
                        max_msg = msgs[i, j, s]

                # msgs[i, j, :] -= np.max((msgs[i, j, :]))  # normalization of the new message from i->j
                for m in range(states):
                    msgs[i, j, m] -= max_msg   # normalization of the new message from i->j
                change += np.sum(np.abs(msgs[i, j, :]-bak))

        iter += 1
        print change

    # BACKTRACKING INIT: choose maximizing state for the last variable that was optimized
    i = np.int(samples-1)
    num_neighs = np.sum(N_weights[i, :])
    neighs = N[i, :num_neighs]
    foo = np.zeros(states)
    for t in range(states):
        sum_msg = 0.
        for n1 in range(num_neighs):
            sum_msg += msgs[neighs[n1], N_inv[i, n1], t]
        foo[t] = unary[t, i] + (1.0 - theta)*sum_msg
    # recursive backtracking

    # lats = np.ones(samples, dtype=np.int8)
    # lats = backtracking(i, lats, np.argmax(foo), psis, N, N_inv, N_weights)
    # print lats
    # return backtracking(i, latent, np.argmax(foo), psis, N, N_inv, N_weights)

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


@autojit(nopython=False)
def backtracking(i, latent, fixed, psis, N, N_inv, N_weights):
    if latent[i]>-1:
        return latent
    latent[i] = fixed
    for j in range(np.sum(N_weights[i, :])):
        j_fixed = np.int(psis[N[i,j], N_inv[i, j], fixed])
        latent = backtracking(N[i, j], latent, j_fixed, psis, N, N_inv, N_weights)
    return latent
