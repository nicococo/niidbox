__author__ = 'Nico Goernitz, TU Berlin, 2015'
import numpy as np

from numba import autojit

from abstract_tcrfr import AbstractTCRFR
from tcrfr_lbpa import TCRFR_lbpa, _extern_map_lbp
from tools import profile


class TCRFR_lbpa_iset(AbstractTCRFR):
    """ Pairwise Conditional Random Field for transductive regression.

        The graphical model is assumed to split into multiple disjunct sets
        without any connection. This will speedup calculations for large datasets
        tremendously.
    """
    isets = None  # list of np.array indices of cluster-memberships
    data_means = None

    def __init__(self, cluster, data, labels, label_inds, unlabeled_inds, states, A,
                 reg_theta=0.5, reg_lambda=0.001, reg_gamma=1.0, trans_regs=[1.0, 1.0],
                 trans_sym=[1], lbl_weight=1.0, verbosity_level=1):
        self.isets = cluster
        AbstractTCRFR.__init__(self, data, labels, label_inds, unlabeled_inds, states, A, \
                 reg_theta, reg_lambda, reg_gamma, trans_regs, trans_sym, verbosity_level=verbosity_level)

        # ASSUMPTION: data is clustered, hence, input instances within each cluster look similar
        # calculate the means of the clusters
        data_means = np.zeros( (self.feats, len(self.isets)), dtype=np.float64)
        for i in range(len(self.isets)):
            data_means[:, i] = np.mean(self.data[:, self.isets[i]], axis=1)
        print data_means

    def print_stats(self):
        AbstractTCRFR.print_stats(self)
        # plot some statistics
        stats = np.zeros(6, dtype=np.int)
        stats[0] = 1e10
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
        print('There are {0} disjunct clusters.'.format(len(self.isets)))
        print('Stats samples:\n  min/cluster={0}, max/cluster={1}, mean/cluster={2:1.2f}'.format(stats[0], stats[1], means))
        print('Stats labels: \n  num-lbld-cluster={0}, max-num/cluster={1}, min-num/cluster={2}'.format(stats[5], stats[3], stats[4]))
        print('===============================')

    @profile
    def map_inference(self, u, v):
        if self.latent is not None:
            self.latent_prev = self.latent

        # choose loopy belief propagation for cluster with multiple labeled examples
        # TODO
        self.latent = _extern_map_lbp(self.data, self.labels, self.label_inds, self.unlabeled_inds, \
                                      self.N, self.N_inv, self.N_weights, u, v, self.reg_theta, self.feats, \
                                      self.samples, self.S, self.trans_d_full, self.trans_n, \
                                      self.trans_mtx2vec_full, False, self.verbosity_level)

        # simply infer the latent state of the mean of unlabeled cluster
        # TODO

        return self.get_joint_feature_maps()

