__author__ = 'nicococo'
import numpy as np
from cvxopt import matrix, spmatrix, sparse
import cvxopt.solvers as solver
import mosek as msk
from structured_object import TransductiveStructuredModel


class TCrfRPairwisePotentialModel(TransductiveStructuredModel):
    """ Homogenous Conditional Random Field with pairwise potentials
        for transductive regression.
    """
    ninf = -10.0**15

    A = None  # (#V x #V) MRF structure, binary transition matrix
    S = -1  # number of discrete states for each node {0,..,S-1}

    V = None  # list of vertices in the graph (according to network structure matrix A)
    E = None  # list of tupels of transitions from state i to state j
    N = None  # list of neighbors for each vertex

    # pre-computed linear program approximation constraints
    qp_eq_A = None
    qp_eq_b = None
    qp_ineq_G = None
    qp_ineq_h = None

    psi = None  # copy of the current joint feature map, corresponding to self.latent
    phis = None  # copy of the current joint feature map, corresponding to self.latent

    local_pot_inds = None

    sigma = 10.  # labeled edge weights

    def __init__(self, data, labels, label_inds, unlabeled_inds, states, A):
        TransductiveStructuredModel.__init__(self, data, labels, label_inds, unlabeled_inds)
        self.A = A
        self.S = states
        (verts, foo) = A.shape
        self.latent = np.zeros(verts, dtype='i')
        # some inits
        self.local_pot_inds = self.get_local_potential_indices()
        # construct edges-, vertices- and neighbors-set
        self.E = []
        self.V = []
        self.N = []
        for s in range(verts):
            self.V.append(s)
            nl = []
            for n in range(verts):
                if A[s, n] == 1:
                    nl.append(n)
                    if not [n, s] in self.E:
                        self.E.append([s, n])  # add an edge between node s and n
            self.N.append(nl)
        # pre-compute qp relaxation constraints
        self.qp_relax_init()

    def get_hotstart(self):
        n_trans = np.round(self.S*(self.S-1.)/2.+self.S)
        return [np.random.randn(self.S*self.feats), np.random.randn(n_trans + self.S*self.get_num_feats())]

    def get_num_dims(self):
        # number of values that need to be stored for a symmetric matrix
        n_trans = np.round(self.S*(self.S-1.)/2.+self.S)
        # transition part: (states*(states-1)+states) since transitions should be symmetric
        # emission part: states*features
        return self.S*self.S + self.S*self.get_num_feats()

    def unpack_param(self, param_v):
        states = self.S
        v = np.zeros(states*states + self.S*self.get_num_feats())
        trans = np.zeros((states, states))
        cnt = 0
        for s1 in range(states):
            for s2 in range(s1, states):
                trans[s1, s2] = param_v[cnt]
                trans[s2, s1] = param_v[cnt]
                cnt += 1
        v[:states*states] = trans.reshape(states*states)
        v[states*states:] = param_v[cnt:]
        return v

    def get_labeled_predictions(self, sol, sample=None):
        lats = self.latent
        if sample is not None:
            lats = sample
        # for debugging only
        phis = np.zeros((self.S*self.feats, self.samples))
        for s in range(self.S):
            inds = np.where(lats == s)[0]
            phis[s*self.feats:(s+1)*self.feats, inds] = self.data[:, inds]
        return sol.dot(phis[:, self.label_inds])

    def log_partition_derivative(self, sol):
        return self.get_log_partition_derivative(sol)

    def maps(self, sol):
        theta = sol[0]
        u = sol[1]
        v = sol[2]
        # highest value first
        if self.latent is not None:
            self.latent_prev = self.latent
        self.latent = self.qp_relax_max(u, v, theta)
        self.phis, self.psi = self.get_joint_feature_maps()
        return self.phis, self.psi

    def get_joint_feature_maps(self, predict=False):
        phis = np.zeros((self.S*self.feats, self.samples))
        for s in range(self.S):
            inds = np.where(self.latent == s)[0]
            phis[s*self.feats:(s+1)*self.feats, inds] = self.data[:, inds]
        if predict:
            return self.latent[self.unlabeled_inds], phis[:, self.unlabeled_inds]
        return phis[:, self.label_inds], self.get_crf_joint_feature_map()

    def get_crf_joint_feature_map(self, sample=None):
        if sample is not None:
            y = sample
        else:
            y = self.latent
        psi = np.zeros(self.get_num_dims())
        # transitions
        for e in self.E:
            yi = y[e[0]]
            yj = y[e[1]]
            trans_idx = self.local_pot_inds[yi, yj]
            psi[trans_idx] += 1.0
            # if e[0] in self.label_inds or e[1] in self.label_inds:
            #    if yi == yj:
            #        psi[trans_idx] += -1.0 + 10.0
        # emissions
        feats = self.get_num_feats()
        cnt = self.S*self.S
        for v in self.V:
            psi[cnt+y[v]*feats:cnt+y[v]*feats+feats] += self.data[:, v]
            # for s in range(self.S):
            #     if s == y[v]:
            #         psi[cnt:cnt+feats] += self.data[:, v]
            #     cnt += feats
        return psi

    def get_local_potential_indices(self):
        pmap = np.zeros((self.S, self.S))
        cnt = 0
        for i in range(self.S):
            for j in range(self.S):
                pmap[i, j] = cnt
                cnt += 1
        return pmap

    def log_partition(self, sol):
        # pseudolikelihood approximation = fix the neighbors
        v_trans = sol[:self.S*self.S].reshape((self.S, self.S), order='F')
        v_em = sol[self.S*self.S:].reshape((self.feats, self.S), order='F')
        f_inner = np.zeros((self.S, self.samples))
        for s in range(self.S):
            w = v_trans[s, self.latent]
            foo = np.zeros(self.samples)
            for n in range(len(self.N)):
                foo[n] = np.sum(w[self.N[n]])
            f_inner[s, :] = v_em[:, s].dot(self.data) + foo
        max_score = np.max(f_inner)
        f_inner = np.sum(np.exp(f_inner - max_score), axis=0)
        foo = np.sum(np.log(f_inner) + max_score)
        if np.isnan(foo) or np.isinf(foo):
            print 'TCRFR Pairwise Potential Model: the log_partition is NAN or INF!!'
        return foo

    def get_log_partition_derivative(self, sol):
        v_trans = sol[:self.S*self.S].reshape((self.S, self.S), order='F')
        v_em = sol[self.S*self.S:].reshape((self.feats, self.S), order='F')
        # (A)
        f = np.zeros((self.S, self.samples))
        for s in range(self.S):
            w = v_trans[s, self.latent]
            foo = np.zeros(self.samples)
            for n in range(len(self.N)):
                foo[n] = np.sum(w[self.N[n]])
            f[s, :] = np.exp(v_em[:, s].dot(self.data) + foo)
        sum_f = np.sum(f, axis=0)
        # (B)
        for s in range(self.S):
            f[s, :] /= sum_f
        # (C)
        phis = np.zeros((self.get_num_dims(), self.samples))
        for s in range(self.S):
            foo = np.zeros(self.samples)
            for n in range(len(self.N)):
                foo[n] = np.sum(self.N[n]==s)
            phis[s, :] = foo * f[s, :]

        idx = self.S*self.S
        for s in range(self.S):
            for feat in range(self.feats):
                phis[idx, :] = self.data[feat, :] * f[s, :]
                idx += 1
        return np.sum(phis, axis=1)

    def get_gibbs_partition_derivative(self, sol, max_iter=5):
        """ Gibbs sampler for the expectation of psi-feature map
            (used for the derivative of the partition function).
            returns expectation E_z[psi(X,z)]
        """
        # HINT: avoiding overflow
        # log sum_z exp(z) = a + log sum_z exp(z-a)

        # 1. get good starting point (MAP - solution)
        psi = self.psi
        sample = self.latent
        max_score = sol.T.dot(psi)
        psi_cache = np.zeros((self.get_num_dims(), self.S))

        all_samples = np.zeros((self.get_num_dims(), max_iter))
        all_scores = np.zeros(max_iter)

        num_iter = 1
        all_samples[:, 0] = psi
        all_scores[0] = max_score
        while max_iter > num_iter:
            # 2. (sampling) inner loop: sample states
            for v in range(len(self.V)):
                for s in range(self.S):
                    sample[v] = s
                    psi_cache[:, s] = self.get_crf_joint_feature_map(sample)
                # get scores
                scores = np.exp(sol.T.dot(psi_cache) - max_score)
                # normalize
                prop_scores = scores / np.sum(scores)
                # choose uniform sample
                thres = np.random.rand()
                ind = -1
                add = 0.0
                for i in range(self.S):
                    if add <= thres <= prop_scores[i] + add:
                        ind = i
                        break
                    else:
                        add += prop_scores[i]
                # final sample
                sample[v] = ind
            # 3. update expectation
            all_samples[:, num_iter] = psi_cache[:, ind]
            all_scores[num_iter] = scores[ind]
            print('{0} - score={1}'.format(num_iter, scores[ind]))
            num_iter += 1

        all_scores /= np.sum(all_scores)
        grad = np.zeros(self.get_num_dims())
        for i in range(max_iter):
            grad += all_scores[i]*all_samples[:, i]
        return grad

    def qp_relax_init(self):
        """ Pre-calulate the equality and inequality constants for the
            linear program relaxation.
                min_x   <c,x>
                s.t.    Gx+s=h
                        Ax = b
                        s >= 0  (element-wise)
        """
        states = self.S
        edges = len(self.E)
        dims = edges*states*states + len(self.V)*states
        offset = edges*states*states
        A = []
        b = []
        cnt = 0
        # pair-wise potentials
        for e in self.E:
            ind_i = e[0]
            ind_j = e[1]
            for s in range(states):
                constr = sparse(matrix(0.0, (1, dims)))
                for k in range(states):
                    constr[cnt + s*states + k] = 1.0
                constr[offset + ind_i*states + s] = -1.0
                A.append(constr)
                b.append(matrix([0.0]))
            for k in range(states):
                constr = sparse(matrix(0.0, (1, dims)))
                for s in range(states):
                    constr[cnt + k + s*states] = 1.0
                constr[offset + ind_j*states + k] = -1.0
                A.append(constr)
                b.append(matrix([0.0]))
            cnt += states*states
        num_margs = len(b)
        # emissions
        for v in self.V:
            # emission constraints: sum_s x_is = 1
            constr = sparse(matrix(0.0, (1, dims)))
            constr[offset:offset + states] = 1.0
            A.append(constr)
            b.append(matrix([1.0]))
            offset += states
        self.qp_eq_A = sparse(A)
        self.qp_eq_b = matrix(b)
        # lower bounds
        self.qp_ineq_G = spmatrix(-1.0, range(dims), range(dims))
        self.qp_ineq_h = matrix(0.0, (dims, 1))
        print('There are {0} marginal contraints and {1} vertex constraints.'.format(num_margs, len(b)-num_margs))


    def qp_relax_max(self, u, v, theta):
        """ Estimate the MAP by relaxing the integer quadratic program
            to a quadratic program (QP):
                \mbox{minimize}   (1/2) xT P x + qT x \\
                \mbox{subject to} Gx <= h
                                  Ax  = b
        """
        # set optimization parameters
        G = self.qp_ineq_G
        h = self.qp_ineq_h
        A = self.qp_eq_A
        b = self.qp_eq_b
        # convert u to Q
        P, q = self.get_qp_params(u, v, theta)

        solver.options['MOSEK'] = {msk.iparam.log: 0}
        solution = solver.qp(P, q, G, h, A, b, solver='mosek')
        res = solution['x']

        # print solution['primal objective']
        # obj = matrix(solution['primal objective'])
        # convert into state sequence
        edges = len(self.E)
        states = self.S
        vertices = len(self.V)
        offset = edges*states*states
        max_states = np.zeros(vertices, dtype='i')

        # error check
        print solution['status']
        if res is None:
            print('QP optimization did not finish (status):')
            print 'max P - ', max(P), ' - min P - ', min(P)
            # print 'Non-numbers in P - ', any(np.isnan(P)), any(np.isinf(P))
            print 'max q - ', max(q), ' - min q - ', min(q)
            # print 'Non-numbers in q - ', any(np.isnan(q)), any(np.isinf(q))
            if self.latent is None:
                return max_states
            else:
                return self.latent

        for v in range(vertices):
            max_states[v] = np.argmax(res[offset + v*states:offset + v*states + states])
        return max_states

    def test_qp_param(self):
        # generate random latent states
        test_lats = np.random.randint(0, 1, self.data.shape[1])
        # generate random solutions
        u, v = self.get_hotstart()
        theta = np.random.rand()

        psi = self.get_crf_joint_feature_map(test_lats)
        y_preds = self.get_labeled_predictions(u, test_lats)

        # get the qp-transformed parameters
        cP, cq = self.get_qp_params(u, v, theta)
        P = np.array(matrix(cP))
        q = np.array(cq)

        states = self.S
        edges = len(self.E)
        vertices = len(self.V)
        offset = edges*states*states
        dims = edges*states*states + vertices*states

        x = np.zeros(dims)
        cnt = 0
        for e in self.E:
            ind_i = e[0]
            ind_j = e[1]
            for s1 in range(states):
                for s2 in range(states):
                    if s1 == test_lats[ind_i] and s2 == test_lats[ind_j]:
                        x[cnt] = 1.0
                    cnt += 1
        cnt = offset
        for i in range(len(self.V)):
            for s in range(states):
                if test_lats[i] == s:
                    x[cnt] = 1.0
                cnt += 1

        # our true objective value
        obj = theta/2.0 * np.sum((y_preds-self.labels)**2.0) - (1.0-theta)*v.T.dot(psi)
        # the linear part only
        y2 = 0.5*theta*np.sum(self.labels*self.labels)
        obj_int_lp = -theta * np.sum(y_preds*self.labels) - (1.0-theta)*v.T.dot(psi)
        obj_int_qp = 0.5*theta * np.sum(y_preds*y_preds) + y2
        # the qp transformed objective
        obj_transf = 0.5*x.T.dot(P.dot(x)) + q.T.dot(x) + y2

        print 'QP: ', obj, ' - ', obj_transf
        print 'Linear part only: ', obj_int_lp, ' - ', q.T.dot(x)
        print 'Quadratic part only: ', obj_int_qp - y2, ' - ', 0.5*x.T.dot(P.dot(x))
        print obj_int_lp + obj_int_qp



    def get_qp_params(self, param_u, param_v, theta):
        print param_v.size
        states = self.S
        edges = len(self.E)
        vertices = len(self.V)
        feats = self.get_num_feats()
        dims = edges*states*states + vertices*states
        # build weighting term
        d = matrix(0.0, (dims, 1))
        c = matrix(0.0, (dims, 1))
        offset = edges*states*states

        # pair-wise potentials
        cnt = 0
        for e in self.E:
            d[cnt:cnt+states*states] = param_v[:states*states]
            if e[0] in self.label_inds or e[1] in self.label_inds:
                for i in range(states):
                   ind = int(self.local_pot_inds[i, i])
                   d[cnt+ind] += 0.
            cnt += states*states

        # emissions
        cnt = offset
        for v in self.V:
            sol_cnt = states*states
            for s in range(states):
                d[cnt] = param_v[sol_cnt:sol_cnt + feats].T.dot(self.data[:, v])
                sol_cnt += feats
                cnt += 1

        P2 = spmatrix(0.0, range(dims), range(dims))
        cnt = 0
        for e in self.E:
            for s1 in range(states):
                for s2 in range(states):
                    if s1 == s2:
                        if e[0] in self.label_inds or e[1] in self.label_inds:
                            P2[cnt, cnt] = 0.000001
                        else:
                            P2[cnt, cnt] = 0.1
                    else:
                        P2[cnt, cnt] = 0.1
                    cnt += 1

        P2 = spmatrix(0.0, range(dims), range(dims))
        P = spmatrix(0.0, range(dims), range(dims))

        # ridge regression part
        cnt = offset
        idx = 0
        u = param_u.reshape((self.feats, self.S), order='F')
        for ind in self.label_inds:
            i_start = cnt + ind*self.S
            i_end = cnt + (ind+1)*self.S
            c[i_start:i_end] = self.labels[idx]*u.T.dot(self.data[:, ind])

            for s1 in range(states):
                for s2 in range(states):
                    P[i_start+s2, i_start+s1] = 2.0*u[:, s2].T.dot(self.data[:, ind]) * u[:, s1].T.dot(self.data[:, ind])
            # next label
            idx += 1

        return theta/2.0*P + (1.0-theta)/2.0*P2, -theta*c-(1.0-theta)*d
