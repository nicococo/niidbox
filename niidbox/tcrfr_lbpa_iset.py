__author__ = 'Nico Goernitz, TU Berlin, 2015'
import numpy as np

from numba import autojit

from abstract_tcrfr import AbstractTCRFR
from tcrfr_lbpa import TCRFR_lbpa
from utils import profile


class TCRFR_lbpa_iset(TCRFR_lbpa):
    """ Pairwise Conditional Random Field for transductive regression.

        The graphical model is assumed to split into multiple disjunct sets
        without any connection. This will speedup calculations for large datasets tremendously.
        Various map inference schemes are provided.
    """

    MAP_ISET_FULL = 0   # map inference on the full set of variables (fallback)
    MAP_ISET_MEAN = 1   # full map inference only on cluster with multiple labels
                        # otherwise, map inference only on the mean of unlabeled clusters
    MAP_ISET_INDEP1 = 2 # full map inference only on cluster with multiple labels
                        # otherwise, map inference assuming independent examples within type 0,1 clusters
    MAP_ISET_INDEP2 = 3 # full map inference on cluster with multiple AND single labels
                        # otherwise, map inference assuming independent examples within type 0 clusters
    MAP_ISET_MEAN_LBL = 4   # full map inference only on cluster with multiple labels
                            # for unlabeled clusters, map inference only on the mean of unlabeled clusters,
                            # for single label clusters, map inference based solely on the labeled example

    map_iset_inference_scheme = MAP_ISET_MEAN_LBL

    num_isets = -1          # number of clusters

    isets = None            # list of np.array indices of cluster-memberships
    iset_type = None        # 3 types (per cluster): 0-no lbld exms, 1-one lbld exm, 2-multiple lbld exms
    iset_lbl_inds = None    # for each cluster a (np.array-)list of labeled examples

    iset_edges = None       # number of edges in each cluster
    iset_vertices = None    # number of vertices in each cluster

    iset_type1 = None       # indices of clusters of type 1
    iset_type1_lbl = None   # indices of the label of clusters of type 1 (in label_inds)

    data_means = None       # the overall mean of the input instances for each cluster
    data_iset = None        # for each niidbox-data point its corresponding cluster

    def __init__(self, cluster, data, labels, label_inds, states, A,
                 reg_theta=0.5, reg_lambda=0.001, reg_gamma=1.0, trans_regs=[[1.0, 1.0]],
                 trans_sym=[1], lbl_weight=1.0, verbosity_level=1):

        self.isets = cluster
        self.num_isets = len(cluster)
        AbstractTCRFR.__init__(self, data, labels, label_inds, states, A, \
                 reg_theta, reg_lambda, reg_gamma, trans_regs, trans_sym, verbosity_level=verbosity_level)

        # add the iset log partition function
        self.set_log_partition(self.LOGZ_CUSTOM, custom_fct=self.log_partition_pl_mean_iset)

        # ASSUMPTION I: cluster do NOT share any edges (completely independent of each other)
        # ASSUMPTION II: niidbox-data is clustered, hence, input instances within each cluster look similar
        # calculate the means of the clusters
        type1_inds = []
        type1_lbl_inds = []
        self.iset_lbl_inds = []
        self.iset_type = np.zeros(self.num_isets, dtype=np.int8)
        self.iset_vertices = np.zeros(self.num_isets, dtype=np.float)
        self.data_means = np.zeros( (self.feats, self.num_isets), dtype=np.float64)
        self.data_iset = np.zeros(self.samples, dtype=np.int32)

        for i in range(self.num_isets):
            # calculate cluster means
            self.iset_vertices[i] = self.isets[i].size
            self.data_means[:, i] = np.mean(self.data[:, self.isets[i]], axis=1)
            inds = np.intersect1d(self.label_inds, self.isets[i])
            self.iset_lbl_inds.append(inds)
            self.data_iset[self.isets[i]] = i
            if inds.size == 1:
                self.iset_type[i] = 1
                type1_inds.append(i)
                ind = np.where(self.label_inds == inds[0])[0]
                type1_lbl_inds.append(ind[0])
            elif inds.size > 1:
                self.iset_type[i] = 2

        self.iset_type1 = np.array(type1_inds, dtype=np.int)
        self.iset_type1_lbl = np.array(type1_lbl_inds, dtype=np.int)

        cnt_wrong_edges = 0
        self.iset_edges = np.zeros(len(self.isets), dtype=np.float)
        for e in self.E:
            self.iset_edges[self.data_iset[e[0]]] += 2
            if self.data_iset[e[0]] != self.data_iset[e[1]]:
                cnt_wrong_edges += 1
        print('Number of edges across clusters (is assumed to be 0): {0}'.format(cnt_wrong_edges))
        assert cnt_wrong_edges == 0

    def print_stats(self):
        AbstractTCRFR.print_stats(self)
        # plot some statistics
        stats = np.zeros(6, dtype=np.int)
        stats[0] = 1e10
        stats[4] = 1e10
        for cset in self.isets:
            stats[0] = min(stats[0], cset.size)
            stats[1] = max(stats[1], cset.size)
            stats[2] += cset.size
            foo = np.intersect1d(cset, self.label_inds)
            stats[3] = max(stats[3], foo.size)
            if foo.size > 0:
                stats[4] = min(stats[4], foo.size)
                stats[5] += 1
        means = stats[2]/len(self.isets)
        print('Cluster properties')
        print('===============================')
        print('There are {0} disjunct clusters.'.format(len(self.isets)))
        print('-------------------------------')
        print('Stats samples (total={0}):'.format(self.samples))
        print('- min #samples within cluster  : {0}'.format(stats[0]))
        print('- max #samples within cluster  : {0}'.format(stats[1]))
        print('- mean #samples within cluster : {0:1.2f}'.format(means))
        print('-------------------------------')
        print('Stats labels (total={0}):'.format(self.labels.size))
        print('- #cluster with labeled examples : {0}'.format(stats[5]))
        print('- max #labels within clusters    : {0}'.format(stats[3]))
        print('- min #labels in cluster that have labels : {0}'.format(stats[4]))
        print('-------------------------------')
        print('- Map inference scheme : {0}'.format(self.map_iset_inference_scheme))
        print('===============================')

    @profile
    def map_inference(self, u, v):
        # if full inference is requested, then just
        # let the base-class do the job.
        if self.map_iset_inference_scheme == self.MAP_ISET_FULL:
            return super(TCRFR_lbpa_iset, self).map_inference(u, v)

        if self.latent is not None:
            self.latent_prev = self.latent

        # CASE 0 & 1: no or one lbld example in this cluster
        iu = u.reshape((self.feats, self.S), order='F')
        iv = v[self.trans_d_full*self.trans_n:].reshape((self.feats, self.S), order='F')

        # infer latent states for the mean of each cluster and then apply it for the whole cluster
        if self.map_iset_inference_scheme == self.MAP_ISET_MEAN:
            map_objs = np.zeros((self.S, len(self.isets)))
            for s in range(self.S):
                map_objs[s, :] = (1.0 - self.reg_theta)*iv[:, s].dot(self.data_means)
                f_squares = self.labels[self.iset_type1_lbl] - iu[:, s].dot(self.data_means[:, self.iset_type1])
                map_objs[s, self.iset_type1] -= self.reg_theta/2. * f_squares*f_squares
            # to expand
            latent_means = np.argmax(map_objs, axis=0)
            self.latent = latent_means[self.data_iset]

        # infer latent states based on the means, except for single label cluster, then replace the mean
        # by the single label
        if self.map_iset_inference_scheme == self.MAP_ISET_MEAN_LBL:
            type1_inds = self.label_inds[self.iset_type1_lbl]
            map_objs = np.zeros((self.S, len(self.isets)))
            for s in range(self.S):
                map_objs[s, :] = (1.0 - self.reg_theta)*iv[:, s].dot(self.data_means)
                map_objs[s, self.iset_type1] = (1.0 - self.reg_theta)*iv[:, s].dot(self.data[:, type1_inds])
                f_squares = self.labels[self.iset_type1_lbl] - iu[:, s].dot(self.data[:, type1_inds])
                map_objs[s, self.iset_type1] -= self.reg_theta/2. * f_squares*f_squares
            # to expand
            latent_means = np.argmax(map_objs, axis=0)
            self.latent = latent_means[self.data_iset]

        # infer latent states based on inputs only (no connnections considered here)
        if self.map_iset_inference_scheme == self.MAP_ISET_INDEP1 \
            or self.map_iset_inference_scheme == self.MAP_ISET_INDEP2:
            map_objs = np.zeros((self.S, self.samples))
            for s in range(self.S):
                map_objs[s, :] = (1.0 - self.reg_theta)*iv[:, s].dot(self.data)
                f_squares = self.labels - iu[:, s].dot(self.data[:, self.label_inds])
                map_objs[s, self.label_inds] -= self.reg_theta/2. * f_squares*f_squares
                self.latent = np.argmax(map_objs, axis=0)

        # fix the latent states
        self.latent[self.latent_fixed_inds] = self.latent_fixed[self.latent_fixed_inds]

        # do full belief propagation sweep for some cluster
        for i in range(self.num_isets):
            if self.iset_type[i] == 2 or \
                    (self.iset_type[i] == 1 and self.map_iset_inference_scheme == self.MAP_ISET_INDEP2):
                # CASE 2: multiple lbld examples in this cluster
                lats = _extern_map_partial_lbp(self.data, self.latent_fixed, self.labels, self.label_inds, \
                              self.N, self.N_inv, self.N_weights, u, v, self.reg_theta, self.feats, \
                              self.isets[i], self.S, self.trans_d_full, self.trans_n, \
                              self.trans_mtx2vec_full, False, self.verbosity_level)
                self.latent[self.isets[i]] = lats[self.isets[i]]

        return self.get_joint_feature_maps()

    @profile
    def log_partition_pl_mean_iset(self, v):
        # self.N is a (Nodes x max_connection_count) Matrix containing the indices for each neighbor
        # of each node (indices are 0 for non-neighbors, therefore N_weights is need to multiply this
        # unvalid value with 0.
        v_em = v[self.trans_n*self.trans_d_full:].reshape((self.feats, self.S), order='F')
        f_inner = np.zeros((self.S, self.num_isets), dtype=np.float)

        for s1 in range(self.S):
            f_trans = np.zeros(self.num_isets)
            for s2 in range(self.S):
                f_trans += v[self.trans_mtx2vec_full[s1, s2]]*self.iset_edges
            f_inner[s1, :] = v_em[:, s1].dot(self.data_means)*self.iset_vertices + f_trans

        # exp-trick (to prevent NAN because of large numbers): log[sum_i exp(x_i-a)]+a = log[sum_i exp(x_i)]
        # max_score = np.max(f_inner)
        # f_inner = np.sum(np.exp(f_inner - max_score))
        # foo = np.log(f_inner) + max_score

        max_score = np.max(f_inner, axis=0).reshape((1, self.num_isets))  # max-score for each sample
        max_score = np.repeat(max_score, self.S, axis=0)
        f_inner = np.sum(np.exp(f_inner - max_score), axis=0)
        foo = np.sum(np.log(f_inner) + max_score)

        if np.isnan(foo) or np.isinf(foo):
            print('TCRFR Pairwise Potential Model: the log_partition is NAN or INF!!')
        return foo


@autojit(nopython=True)
def _extern_map_partial_lbp(data, latent_fixed, labels, label_inds, N, N_inv, N_weights, \
             u, v, theta, feats, sample_inds, states, trans_d_full, trans_n, trans_mtx2vec_full, fix_lbl_map, verbosity):

    MIN_INF = 1e-10
    samples = N.shape[0]

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
        for i in sample_inds:
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
    foo_full = np.zeros(states)
    while change>1e-3 and iter<50:
        change = 0.0
        for i in sample_inds:
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
                    # if latent_fixed[i] >= 0:
                    #     msgs[i, j, s] = foo[latent_fixed[i]]
                    #     psis[i, j, s] = latent_fixed[i]

                    if msgs[i, j, s] > max_msg:
                        max_msg = msgs[i, j, s]

                if latent_fixed[i] >= 0:
                    max_msg += 10000.
                    msgs[i, j, latent_fixed[i]] = max_msg
                    psis[i, j, latent_fixed[i]] = latent_fixed[i]
                    # msgs[i, j, latent_fixed[i]] = 1e20

                for m in range(states):
                    msgs[i, j, m] -= max_msg   # normalization of the new message from i->j
                change += np.sum(np.abs(msgs[i, j, :]-bak))
        iter += 1
        if verbosity >= 2:
            print change

    # BACKTRACKING INIT: choose maximizing state for the last variable that was optimized
    i = sample_inds[-1]
    num_neighs = np.sum(N_weights[i, :])
    neighs = N[i, :num_neighs]
    foo = np.zeros(states)
    for s in range(states):
        sum_msg = 0.
        for n1 in range(num_neighs):
            sum_msg += msgs[neighs[n1], N_inv[i, n1], s]
        foo[s] = unary[s, i] + sum_msg
        #foo[s] = unary[s, i] + (1. - theta)*sum_msg  # ERR!: msgs are (1-theta)-normalized already

    # backtracking step
    idxs = np.zeros(sample_inds.size, dtype=np.int32)
    if latent_fixed[i] >= 0:
        latent[i] = latent_fixed[i]
    else:
        latent[i] = np.argmax(foo)
    idxs[0] = i
    cnt = 1
    for s in range(sample_inds.size):
        i = idxs[s]
        for j in range(np.sum(N_weights[i, :])):
            if latent[N[i, j]] < 0:
                latent[N[i, j]] = psis[N[i, j], N_inv[i, j], latent[i]]
                # if latent_fixed[N[i, j]] >= 0:
                #     latent[N[i, j]] = latent_fixed[N[i, j]]
                idxs[cnt] = N[i, j]
                cnt += 1
    return latent
