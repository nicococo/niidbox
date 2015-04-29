import numpy as np
from cvxopt import matrix, exp, uniform
from structured_object import StructuredObject


class WellModel(StructuredObject):
    """ Hidden Markov based model for prediction of rock porosity."""

    ninf = -10.0**15
    start_p = None  # (vector) start probabilities
    states = -1     # (scalar) number transition states
    hotstart_tradeoff = 0.1  # (scalar) this tradeoff is used for hotstart
                             # > 1.0: transition have more weight
                             # < 1.0: emission have more weight

    state_dims_map = None
    state_dims_jfm_inds = None
    state_dims_entries = 0

    transitions = -1

    def __init__(self, X, y=None, num_states=3, hotstart_tradeoff=0.1, state_dims_map=None):
        StructuredObject.__init__(self, X, y)
        self.states = num_states
        self.start_p = matrix(1.0, (self.states, 1))
        self.hotstart_tradeoff = hotstart_tradeoff
        self.transitions = num_states*num_states

        if state_dims_map is None:
            print 'Create default state-dimension-map.'
            # generate a default state-dimension map (every state uses all avail dims)
            self.state_dims_map = []
            for s in range(self.states):
                self.state_dims_map.append(range(self.feats))
        else:
            self.state_dims_map = state_dims_map

        print 'Calculating jfm dimension indices..'
        cnt = self.transitions
        self.state_dims_jfm_inds = []
        for s in range(self.states):
            foo = matrix(0, (1, self.feats))
            for d in self.state_dims_map[s]:
                foo[d] = cnt
                cnt += 1
            self.state_dims_jfm_inds.append(foo)
        self.state_dims_entries = cnt - self.transitions

        for s in range(self.states):
            print self.state_dims_jfm_inds[s]

        print self.state_dims_entries
        print self.state_dims_map

    def get_hotstart_sol(self):
        sol = uniform(self.get_num_dims(), 1, a=-1, b=+1)
        print('Hotstart position uniformly random with transition tradeoff {0}.'.format(self.hotstart_tradeoff))
        return sol

    def calc_emission_matrix(self, sol, idx, augment_loss=False, augment_prior=False):
        T = len(self.X[idx][0,:])
        N = self.states
        F = self.feats

        em = matrix(0.0, (N, T))
        for t in xrange(T):
            for s in xrange(N):
                for f in self.state_dims_map[s]:
                    em[s, t] += sol[self.state_dims_jfm_inds[s][f]] * self.X[idx][f, t]

        # augment with loss 
        if augment_loss:
            loss = matrix(1.0, (N, T))
            loss[0, :] = 1.0
            for t in xrange(T):
                loss[self.y[idx][t], t] = 0.0
                if np.uint(self.y[idx][t]) == 0:
                    loss[:, t] = 5.0
                    loss[self.y[idx][t], t] = 0.0
            em += loss
        
        if augment_prior:
            prior = matrix(0.0, (N, T))
            em += prior

        return em

    def get_loss(self, idx, y):
        loss = matrix(0.0, (1, len(y)))
        for i in xrange(len(y)):
            if np.uint(self.y[idx][i]) != np.uint(y[i]):
                loss[i] = 1.0
                if np.uint(self.y[idx][i]) == 0:
                    loss[i] = 5.0
                if np.uint(y[i]) == 0:
                    loss[i] = 1.0
        return sum(loss)

    def get_transition_matrix(self, sol):
        N = self.states
        # transition matrix
        A = matrix(0.0, (N, N))
        for i in xrange(N):
            for j in xrange(N):
                A[i, j] = sol[i*N+j]
        return A

    def map(self, idx=-1, add_loss=False, add_prior=False):
        # if labels are present, then argmax will solve
        # the loss augmented programm
        T = len(self.X[idx][0, :])
        N = self.states
        F = self.feats

        # get transition matrix from current solution
        A = self.get_transition_matrix(self.sol)
        # calc emission matrix from current solution, data points and
        # augment with loss if requested
        em = self.calc_emission_matrix(self.sol, idx, augment_loss=add_loss, augment_prior=add_prior)

        delta = matrix(0.0, (N, T))
        psi = matrix(0, (N, T))
        # initialization
        for i in xrange(N):
            delta[i, 0] = self.start_p[i] + em[i, 0]
            
        # recursion
        for t in xrange(1, T):
            for i in xrange(N):
                (delta[i, t], psi[i, t]) = max([(delta[j, t-1] + A[j, i] + em[i, t], j) for j in xrange(N)])
        
        states = matrix(0, (1, T))
        (prob, states[T-1]) = max([delta[i, T-1], i] for i in xrange(N))
            
        for t in reversed(xrange(1, T)):
            states[t-1] = psi[states[t], t]
        
        psi_idx = self.get_joint_feature_map(idx, states)

        val = self.sol.trans()*psi_idx
        return val, states, psi_idx

    def get_jfm_norm2(self, idx, y=None):
        y = np.array(y)
        if y.size is None:
            y = np.array(self.y[idx])
        jfm = self.get_joint_feature_map(idx, y)
        return jfm.trans()*jfm

    def get_scores(self, sol, idx, y=None):
        y = np.array(y)
        if y.size is None:
            y = np.array(self.y[idx])

        foo, T = y.shape
        N = self.states
        F = self.feats
        scores = matrix(0.0, (1, T))

        # this is the score of the complete example
        anom_score = sol.trans()*self.get_joint_feature_map(idx)

        # transition matrix
        A = self.get_transition_matrix(sol)
        # emission matrix without loss
        em = self.calc_emission_matrix(sol, idx, augment_loss=False, augment_prior=False)
        
        # store scores for each position of the sequence        
        scores[0] = self.start_p[int(y[0, 0])] + em[int(y[0, 0]), 0]
        for t in range(1, T):
            scores[t] = A[int(y[0, t-1]), int(y[0, t])] + em[int(y[0, t]), t]

        # transform for better interpretability
        if max(abs(scores)) > 10.0**(-15):
            scores = exp(-abs(4.0*scores/max(abs(scores))))
        else:
            scores = matrix(0.0, (1, T))

        return float(np.single(anom_score)), scores

    def get_joint_feature_map(self, idx, y=None):
        y = np.array(y)
        if y.size is None:
            y = self.y[idx]

        T = y.size
        y = y.reshape(1, T)

        N = self.states
        F = self.feats
        jfm = matrix(0.0, (self.get_num_dims(), 1))
        
        # transition part
        for i in range(N):
            (foo, inds) = np.where([y[0, 1:T] == i])
            for j in range(N):
                (foo, indsj) = np.where([y[0, inds] == j])
                jfm[j*N+i] = len(indsj)

        # emission parts
        for t in range(T):
            state = int(y[0, t])
            for f in self.state_dims_map[state]:
                jfm[self.state_dims_jfm_inds[state][f]] += self.X[idx][f, t]
        return jfm

    def get_num_dims(self):
        return self.state_dims_entries + self.transitions

    def evaluate(self, pred): 
        N = self.samples
        
        # assume 'pred' to be corresponding to 'y'
        if len(pred) != N:
            print len(pred)
            raise Exception('Wrong number of examples!')

        lens = TP = TN = FP = FN = 0
        for i in xrange(N):
            lens = len(pred[i])
            TP += float(sum([np.uint(self.y[i][t]) == 0 and np.uint(pred[i][t]) == 0 for t in range(lens)]))
            FP += float(sum([not np.uint(self.y[i][t]) == 0 and np.uint(pred[i][t]) == 0 for t in range(lens)]))
            TN += float(sum([not np.uint(self.y[i][t]) == 0 and not np.uint(pred[i][t]) == 0 for t in range(lens)]))
            FN += float(sum([np.uint(self.y[i][t]) == 0 and not np.uint(pred[i][t]) == 0 for t in range(lens)]))
        
        err = dict()
        err['sensitivity'] = TP / (TP+FN)
        err['specificity'] = TN / (FP+TN)
        err['accuracy'] = (TP+TN) / ((TP+FN) + (FP+TN))
        err['f1-score'] = 2.0*TP / (2.0*TP + FP + FN)
        err['length'] = lens

        return err, dict()




