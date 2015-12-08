import numpy as np
from abstract_tcrfr import AbstractTCRFR


class TCRFR_BF(AbstractTCRFR):
    """ Pairwise Conditional Random Field for transductive regression.
     Calculation of map inference and log partition function as well as
     the gradient of the log partition function, are done brute force.

     This class is usefull only for debugging and testing.

     DO NOT USE THIS CLASS FOR REAL PROBLEMS.

     Author: Nico Goernitz, 2015
    """
    def __init__(self, data, labels, label_inds, unlabeled_inds, states, A,
                 reg_theta=0.5, reg_lambda=0.001, reg_gamma=1.0, trans_regs=[1.0, 1.0],
                 trans_sym=[1], verbosity_level=1):
        AbstractTCRFR.__init__(self, data, labels, label_inds, unlabeled_inds, states, A,
                 reg_theta, reg_lambda, reg_gamma, trans_regs, trans_sym, verbosity_level=verbosity_level)

    def _direct_computation(self, u, v):
        # keep track of the best parameters
        max_states = np.zeros(self.samples)
        # ...and values
        max_obj = -1e16

        num_combs = self.S**len(self.V)  # this is the number of combinations
        # print('There are {0} combinations for {1} samples and {2} states.'.format(num_combs, self.samples, self.S))

        # partition function values
        part_value = 0.0  # partition function Z value
        part_grad = np.zeros(v.size)  # partition function Z gradient

        y = self.labels.reshape((self.labels.size, 1))
        u = u.reshape((u.size, 1))
        v = v.reshape((v.size, 1))

        # test every single combination
        for i in range(num_combs):
            comb = np.base_repr(i, base=self.S)  # convert current id of combinations to string of states
            states = np.zeros(self.samples, dtype=np.int8)
            for s in range(len(comb)):
                states[s] = np.int(comb[len(comb)-s-1])

            phis, psi = self.get_joint_feature_maps(latent=states)  # get the corresponding crf feature map
            # map inference and objective functions
            obj_crf = -v.T.dot(psi)
            obj_rr = y - u.T.dot(phis[:, self.label_inds]).T
            obj_rr = 0.5 * np.sum(obj_rr*obj_rr)
            obj = -(self.reg_theta*obj_rr + (1.-self.reg_theta)*obj_crf)

            # partition function and gradient
            part_value += np.exp(-obj_crf)
            part_grad += psi*np.exp(-obj_crf)

            if obj >= max_obj:
                max_obj = obj
                max_states = states
                max_psi = psi
        return max_obj, max_states, max_psi, np.float64(np.log(part_value)), part_grad.reshape((part_grad.size))

    def map_inference(self, u, v):
        if self.latent is not None:
            self.latent_prev = self.latent
        _, self.latent, _, _, _ = self._direct_computation(self.u, v)
        return self.get_joint_feature_maps()

    def log_partition(self, v):
        _, _, _, logZ, _ = self._direct_computation(self.u, v)
        return logZ

    def log_partition_derivative(self, v):
        _, _, _, _, dZ = self._direct_computation(self.u, v)
        return dZ




