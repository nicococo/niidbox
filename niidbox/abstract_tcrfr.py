from cvxopt.base import matrix
import numpy as np

from scipy import optimize as op
import sklearn.cluster as cl
import time

from abc import ABCMeta, abstractmethod

from numba import autojit

from tools import profile

__author__ = 'nicococo'


class AbstractTCRFR(object):
    """ Basic functions for the Transductive Conditional Random Field Regression.

        Written by Nico Goernitz, TU Berlin, 2015
    """
    __metaclass__ = ABCMeta  # This is an abstract base class (ABC)

    verbosity_level = 0     # (0: no prints, 1:print some globally interesting stuff (iterations etc), 2:print debugs

    data = None             # (either matrix or list) data
    labels = None           # (list or matrix or array) labels
    label_inds = None       # index of corresponding data object for each label
    unlabeled_inds = None   # indices for unlabeled examples

    latent_prev = None   # (#V in {0,...,S-1}) previous latent states
    latent = None        # (#V in {0,...,S-1}) latent states (1-to-1 correspondence to data/labels object)
    latent_fixed = None  # (#V int) '1':corresponding state in 'latent' is fixed

    samples = -1  # (scalar) number of training data samples
    feats = -1    # (scalar) number of features != get_num_dims() !!!

    reg_lambda = 0.001  # (scalar) the regularization constant > 0
    reg_gamma = 1.0     # (scalar) crf regularizer
    reg_theta = 0.5     # (scalar) 0<= theta <= 1: trade-off between density estimation (0.0) and regression (1.0)

    v = None  # parameter vector of the crf (consisting of transition matrices and emission matrices)
    u = None  # parameter vector of the regression part

    A = None  # (Nodes x Nodes) = (#V x #V) sparse connectivity matrix (use cvxopt spmatrix) (symmetric with diag = 0!)
    S = -1    # number of discrete states for each node {0,..,S-1}

    V = None  # list of vertices in the graph (according to network structure matrix A)
    E = None  # list of tupels of transitions from edge i to edge j and transition matrix type

    # edge related neighbor indices
    N = None                # (samples x max_conn) matrix of neighbors for each vertex
    N_inv = None            # (samples x max_conn) N[j, N_inv[i,j]] = i
    N_weights = None        # symmetric (samples x max_conn) {0,1} 1:corresponding N[i,j] is a valid neighbor
    N_edge_weights = None   # unsymmetric (samples x max_conn) {0,1} 1:corresponding N[i,j] is a valid neighbor

    Q = None  # (dims x dims) Crf regularization matrix

    trans_sym = None    # (trans_types {0,1} vector) '1':Transition matrix is symmetric,
                        # i.e. learn only S(S-1)/2 parameters instead of S*S, related to unpack_v
    trans_n = 0         # (scalar) number of transition matrices used (= max(A))

    trans_mtx2vec_full = None   # (S x S -> {0,...,S*S-1}) helper for converting full 2d transition
                                # matrices into 1d vectors (e.g. (1, 2) -> 3)
    trans_mtx2vec_sym = None    # (S x S) helper for converting symmetric 2d transition
                                # matrices into 1d vectors (e.g. (1, 2) -> 3)
    trans_vec2vec_mtx = None    # Linear transformation (matrix) converting a packed symmetric
                                # transition vector into an unpacked transition vector
                                # (\in {0,1}^(states*states x np.round(states*(states-1)/2 +states))
    trans_total_dims = -1         # (scalar) start of the emission scores in the final weight vector

    trans_regs = None   # (vector) \in R^trans_num_types, regularizer for transition matrices

    trans_d_sym = 0   # (scalar) number of values that need to be stored for a symmetric transition matrix
    trans_d_full = 0  # (scalar) number of values that need to be stored for a full transition matrix

    @profile
    def __init__(self, data, labels, label_inds, states, A,
                 reg_theta=0.5, reg_lambda=0.001, reg_gamma=1.0, trans_regs=[1.0], trans_sym=[1], verbosity_level=1):
        # set verbosity
        self.verbosity_level = verbosity_level

        # sparse connectivity matrix (numbers indicate the type of connection = id of transition matrix)
        self.A = A
        (verts, foo) = A.size

        # transition types
        self.trans_n = np.int(max(A))

        # number of states that is used for all transition/emission matrices
        self.S = states
        self.latent = np.zeros(verts, dtype=np.int8)
        self.latent_prev = np.zeros(verts, dtype=np.int8)
        self.latent_fixed = np.zeros(verts, dtype=np.int8)

        # some transition inits
        self.trans_d_sym = np.round(self.S * (self.S - 1.) / 2. + self.S)
        self.trans_d_full = np.round(self.S * self.S)
        # mark transition matrices as symmetric
        if len(trans_sym) == 1:
            self.trans_sym = trans_sym[0]*np.ones(self.trans_n, dtype='i')
        else:
            self.trans_sym = trans_sym
        # transition matrix regularization
        if len(trans_regs) == 1:
            self.trans_regs = trans_regs[0]*np.ones(self.trans_n, dtype='i')
        else:
            self.trans_regs = trans_regs
        self.trans_mtx2vec_full, self.trans_mtx2vec_sym, self.trans_vec2vec_mtx = self.get_trans_converters()

        n_sym_mtx = np.sum(self.trans_sym)
        self.trans_total_dims = np.int(n_sym_mtx * self.trans_d_sym + (self.trans_n - n_sym_mtx) * self.trans_d_full)

        self.V = range(verts)
        # neighbor array (and weights {0,1} for each vertex
        max_conn = np.int(max(self.A*matrix(1, (self.A.size[0], 1), tc='i')))
        print max_conn
        self.N = np.zeros((len(self.V), max_conn), dtype=np.int32)
        self.N_inv = np.zeros((len(self.V), max_conn), dtype=np.int32)
        self.N_weights = np.zeros((len(self.V), max_conn), dtype=np.int8)
        self.N_edge_weights = np.zeros((len(self.V), max_conn), dtype=np.int8)
        N_idx = np.zeros(len(self.V), dtype='i')
        N_edge_idx = np.zeros(len(self.V), dtype='i')

        # construct edge matrix
        t = time.time()
        AI = A.I
        AJ = A.J
        AV = np.array(A.V, dtype=np.int8)
        num_entries = np.int(np.sum(AV > 0))
        num_edges = np.int(num_entries / 2)
        self.E = np.zeros((num_edges, 3), dtype=np.int64)
        print num_entries, num_edges
        assert 2*num_edges == num_entries  # is assumed to be twice the number of edges!
        cnt = 0
        for idx in range(AV.size):
            s = AI[idx]
            n = AJ[idx]
            if s < n and AV[idx] >= 1:
                self.E[cnt, :] = (s, n, AV[idx])
                self.N_edge_weights[s, N_edge_idx[s]] = n
                N_edge_idx[s] += 1
                cnt += 1
                # update neighbors
                self.N_inv[s, N_idx[s]] = N_idx[n]     # N[j, N_inv[i,j]] = i
                self.N_inv[n, N_idx[n]] = N_idx[s]     

                self.N[s, N_idx[s]] = n
                self.N[n, N_idx[n]] = s

                self.N_weights[s, N_idx[s]] = 1
                self.N_weights[n, N_idx[n]] = 1

                N_idx[s] += 1
                N_idx[n] += 1
        print time.time()-t
        # print self.E

        # regularization constants
        self.reg_lambda = reg_lambda
        self.reg_gamma = reg_gamma
        self.reg_theta = reg_theta

        # check the data
        self.data = data
        self.labels = np.array(labels)
        self.label_inds = np.array(label_inds, dtype=np.int)
        self.unlabeled_inds = np.setdiff1d(np.arange(self.samples), self.label_inds)
        # assume either co.matrix or list-of-objects
        if isinstance(data, matrix):
            self.feats, self.samples = data.size
            self.isListOfObjects = False
        elif isinstance(data, np.ndarray):
            self.feats, self.samples = data.shape
            self.isListOfObjects = False
        else:
            raise Exception("Could not recognize input data format.")

        # init crf-regularization matrix
        self.init_Q()

        self.u = np.zeros(self.get_num_feats()*self.S)
        self.v = np.zeros(self.get_num_compressed_dims())

        # print some stats
        if self.verbosity_level >= 1:
            self.print_stats()

    @profile
    def print_stats(self):
        # output some stats
        n_sym_mtx = np.sum(self.trans_sym)
        n_trans = np.int(n_sym_mtx * self.trans_d_sym + (self.trans_n - n_sym_mtx) * self.trans_d_full)
        n_em = self.S * self.get_num_feats()
        print('')
        print('===============================')
        print('TCRFR Properties:')
        print('===============================')
        print('- Samples       : {0}'.format(self.samples))
        print('- Labeled       : {0}'.format(self.label_inds.size))
        print('- Unlabeled     : {0}'.format(self.unlabeled_inds.size))
        print('- Features      : {0}'.format(self.feats))
        print('- CRF Dims      : {0} = {1}+{2}'.format(self.get_num_compressed_dims(), n_trans, n_em))
        print('-------------------------------')
        print('- Lambda        : {0}'.format(self.reg_lambda))
        print('- Gamma         : {0}'.format(self.reg_gamma))
        print('- Theta         : {0}'.format(self.reg_theta))
        print('- Q_regs        : {0}'.format(self.trans_regs))
        print('-------------------------------')
        print('- Edges         : {0}'.format(len(self.E)))
        print('- States        : {0}'.format(self.S))
        print('- Trans-types   : {0}'.format(self.trans_n))
        print('- Trans-Sym     : {0}'.format(self.trans_sym))
        # print('-------------------------------')
        # print('- Trans-Sym V2V : \n{0}'.format(self.trans_vec2vec_mtx))
        print('===============================')
        print('')

    @profile
    def init_Q(self):
        # build the crf regularization matrix
        dims = self.trans_n*self.trans_d_full + self.S*self.feats
        foo = np.ones(dims)
        cnt = 0
        for i in range(self.trans_n):
            foo[cnt:cnt+self.trans_d_full] = 1.0
            for s in range(self.S):
                idx = self.trans_mtx2vec_full[s, s]
                foo[cnt:cnt+idx] = self.trans_regs[i]
            cnt += self.trans_d_full
        self.Q = np.diag(self.reg_gamma * foo)

    def get_trans_converters(self):
        # P: states x states -> states*states
        P = np.zeros((self.S, self.S), dtype=np.int32)
        cnt = 0
        for s1 in range(self.S):
            for s2 in range(self.S):
                P[s1, s2] = cnt
                cnt += 1
        # R: states x states -> np.round(states*(states-1)/2 +states)
        R = np.zeros((self.S, self.S), dtype=np.int32)
        cnt = 0
        for s1 in range(self.S):
            for s2 in range(s1, self.S):
                R[s1, s2] = cnt
                R[s2, s1] = cnt
                cnt += 1
        # vector of symmetric transitions to unpacked vector of transitions
        # M: np.round(states*(states-1)/2 +states) -> states*states
        # M \in {0,1}^(states*states x np.round(states*(states-1)/2 +states))
        N_sym = np.int(self.S*(self.S-1.)/2. + self.S)
        M = np.zeros((self.S*self.S, N_sym), dtype=np.int32)
        row = 0
        for s1 in range(self.S):
            for s2 in range(self.S):
                M[row, R[s1, s2]] = 1
                row += 1
        return P, R, M

    def em_estimate_v_obj_callback(self, v, psi, boolean):
        vn = self.unpack_v(v)
        return .5 * vn.T.dot(self.Q.dot(vn)) - vn.T.dot(psi) + self.log_partition(vn)

    def em_estimate_v_grad_callback(self, v, psi, boolean):
        vn = self.unpack_v(v)
        return .5 * vn.T.dot(self.Q) - psi + self.log_partition_derivative(vn)

    def em_estimate_v(self, v, psi, use_grads=True):
        if use_grads and np.sum(self.trans_sym)>0:
            print('Warning! Cannot compute gradient logZ, if symmetric (compressed) feature vector is used!')
        vstar = v
        if use_grads and np.sum(self.trans_sym)==0:
            res = op.minimize(self.em_estimate_v_obj_callback, jac=self.em_estimate_v_grad_callback,
                              x0=vstar, args=(psi, True), method='L-BFGS-B')
        else:
            res = op.minimize(self.em_estimate_v_obj_callback, x0=vstar, args=(psi, True), method='L-BFGS-B')
        # print res.nfev, ' - ', res.nit, ' - ', res.fun
        # print self.unpack_v(res.x)
        print res.nfev, ' - ', res.nit
        return res.fun, res.x

    def em_estimate_u(self, X):
        y = self.labels
        # solve the ridge regression problem
        E = np.zeros((X.shape[1], X.shape[1]))
        np.fill_diagonal(E, self.reg_lambda)
        XXt = X.T.dot(X) + E
        XtY = (X.T.dot(y))
        if XXt.size > 1:
            u = np.linalg.inv(XXt).dot(XtY)
        else:
            u = 1.0 / XXt * XtY
        obj = self.reg_lambda / 2.0 * u.dot(u) + y.dot(y) / 2.0 - u.dot(X.T.dot(y)) + u.dot(X.T.dot(X.dot(u))) / 2.0
        return obj, u

    def fit(self, max_iter=50, hotstart=None, use_grads=True, auto_adjust=True):
        u, v = self.get_hotstart()
        if hotstart is not None:
            print('Manual hotstart position defined.')
            u, v = hotstart

        obj = 1e50
        cnt_iter = 0
        is_converged = False

        # best objective, u and v
        best_sol = [0, 1e14, None, None, None]

        if auto_adjust:
            # adjust reg_gamma
            self.reg_gamma = np.linalg.norm(v)
            v /= np.linalg.norm(v)
            self.init_Q()
            print('Auto-adjust reg_gamma = {0} norm_v={1}'.format(self.reg_gamma, np.linalg.norm(v)))

        # terminate if objective function value doesn't change much
        while cnt_iter < max_iter and not is_converged:
            # 1. infer the latent states given the current intermediate solutions u and v
            phis, psi = self.map_inference(u, self.unpack_v(v))

            if self.verbosity_level >= 2:
                lats = ''
                for i in range(self.latent.size):
                    lats += '{0}'.format(self.latent[i])
                    if i in self.label_inds:
                        lats += '.'
                    else:
                        lats += ' '
                    lats += ' '
                    if i % 50 == 0:
                        lats += '\n'
                print lats

            # 2. solve the crf parameter estimation problem
            obj_crf, v = self.em_estimate_v(v, psi, use_grads=use_grads)
            # 3. estimate new regression parameters
            obj_regression, u = self.em_estimate_u(phis[:, self.label_inds].T)
            # 4.a. check termination based on objective function progress
            old_obj = obj
            obj = self.reg_theta * obj_regression + (1.0 - self.reg_theta) * obj_crf
            rel = np.abs((old_obj - obj) / obj)
            if self.verbosity_level >= 1:
                print('Iter={0} regr={1:4.2f} crf={2:4.2f}; objective={3:4.2f} rel={4:2.4f} lats={5}'.format(
                    cnt_iter, obj_regression, obj_crf, obj, rel, np.unique(self.latent).size))
                print('  norm_v={0}'.format(np.linalg.norm(v)))

            if best_sol[1] >= obj:
                best_sol = [cnt_iter, obj, u, v, self.latent]
                print('*')

            if cnt_iter > 3 and rel < 0.0001:
                is_converged = True

            if np.isinf(obj) or np.isnan(obj):
                return False
            cnt_iter += 1
        iter, _, self.u, self.v, self.latent = best_sol
        if self.verbosity_level >= 1:
            print('Take best solution from iteration {0}/{1}.'.format(iter, cnt_iter-1))
        if self.verbosity_level >= 2:
            vup = self.unpack_v(self.v)
            cnt = 0
            for i in range(self.trans_n):
                print i
                print vup[cnt:cnt+self.S*self.S].reshape((self.S, self.S), order='C')
                cnt += self.trans_d_full
            print 'Emissions:'
            print vup[cnt:]
        return is_converged

    def predict(self, lats=None):
        if lats is None:
            lats = self.latent
        # for debugging only
        phis = np.zeros((self.S*self.feats, self.samples))
        for s in range(self.S):
            inds = np.where(lats == s)[0]
            phis[s*self.feats:(s+1)*self.feats, inds] = self.data[:, inds]
        return self.u.dot(phis), lats

    @profile
    def get_joint_feature_maps(self, latent=None):
        if latent is None:
            latent = self.latent
        # Regression Joint Feature Map
        phis = np.zeros((self.S*self.feats, self.samples))
        for s in range(self.S):
            inds = np.where(latent == s)[0]
            phis[s*self.feats:(s+1)*self.feats, inds] = self.data[:, inds]

        # return phis, self.get_crf_joint_feature_map(latent)
        return phis, _extern_get_crf_joint_feature_map(self.data, latent, self.E, np.int32(self.V), \
            self.trans_mtx2vec_full, self.get_num_dims(), self.get_num_feats(), \
            self.trans_d_full, self.trans_n)

    def get_latent_diff(self):
        if self.latent is None:
            return -1
        if self.latent_prev is None:
            return 1e10
        return np.sum(np.abs(self.latent - self.latent_prev))

    @profile
    def unpack_v(self, v):
        upv = np.zeros(self.trans_n*self.trans_d_full + self.S * self.get_num_feats())
        # transitions include various transition matrices, each either symmetric or full
        cnt = 0
        cnt_full = 0
        for i in range(self.trans_n):
            if self.trans_sym[i] == 1:
                # print ".................................................."
                # print v[cnt:cnt+self.trans_d_sym]
                # print self.trans_vec2vec_mtx.dot(v[cnt:cnt+self.trans_d_sym]).reshape((self.S, self.S), order='C')
                # print ".................................................."
                upv[cnt_full:cnt_full+self.trans_d_full] = self.trans_vec2vec_mtx.dot(v[cnt:cnt+self.trans_d_sym])
                cnt += self.trans_d_sym
            else:
                upv[cnt_full:cnt_full+self.trans_d_full] = v[cnt:cnt+self.trans_d_full]
                cnt += self.trans_d_full
            cnt_full += self.trans_d_full
        # emissions
        upv[cnt_full:] = v[cnt:]
        return upv

    solution_latent = None

    def get_hotstart(self):
        # initialize all non-fixed latent variables with random states
        kmeans = cl.KMeans(n_clusters=self.S, init='random', n_init=4, max_iter=100, tol=0.0001)
        kmeans.fit(self.data.T)
        self.latent = kmeans.labels_
        phis, psi = self.get_joint_feature_maps()
        # point in the direction of psi (unpacked)
        _, v = self.em_estimate_v(np.zeros(self.get_num_compressed_dims()), psi, use_grads=False)
        # estimate regression parameters
        _, u = self.em_estimate_u(phis[:, self.label_inds].T)
        return u, v

    def get_num_compressed_dims(self):
        # number of symmetric transition matrices
        n_sym_mtx = np.sum(self.trans_sym)
        return np.int(n_sym_mtx * self.trans_d_sym + (self.trans_n - n_sym_mtx) * self.trans_d_full + self.S * self.get_num_feats())

    def get_num_dims(self):
        # number of unpacked dimensions
        return self.trans_n*self.trans_d_full + self.S * self.get_num_feats()

    def get_num_labeled(self):
        return len(self.labels)

    def get_num_unlabeled(self):
        return len(self.unlabeled_inds)

    def get_num_samples(self):
        return self.samples

    def get_num_feats(self):
        return self.feats

    @abstractmethod
    def map_inference(self, u, v):
        raise NotImplementedError

    def log_partition(self, v):
        return self.log_partition_pl(v)

    def log_partition_derivative(self, v):
        """
        This method is not particularly needed. If provided, it can speed up the
        l-bfgs method in em_estimate_v otherwise auto-gradients are used (could be noisy due
        to approximations and therefore, lead to more iterations).
        """
        raise NotImplementedError

    @profile
    def log_partition_pl(self, v):
        # This function calculates/estimates the log-partition function by
        # pseudolikelihood approximation. Therefore, we assume the states for
        # the neighbors fixed (e.g. from previous map inference).
        #
        # log Z = log \sum_z exp( <v,\Psi(X,z)> )    # intractable even for small z
        #       = log \sum_z exp( \sum_ij f_trans(i=z_i,j=z_j) + \sum_v f_em(v=z_v) )
        #       ~ log \sum_v \sum_z_v exp( f_pl(v, z_v) )
        #
        # Hence, for a node i in state s given the neighbors j with fixed states n_j:
        #       f_pl(i, s) = f_em(i, s) + sum_j sum_t f_trans(i=s, j=t)+f_em(j, t)
        #

        # self.N is a (Nodes x max_connection_count) Matrix containing the indices for each neighbor
        # of each node (indices are 0 for non-neighbors, therefore N_weights is need to multiply this
        # unvalid value with 0.
        v_em = v[self.trans_n*self.trans_d_full:].reshape((self.feats, self.S), order='F')
        f_inner = np.zeros((self.S, self.samples))

        cnt_neighs = np.sum(self.N_weights, axis=1)
        for s1 in range(self.S):
            f_trans = np.zeros(self.samples)
            for s2 in range(self.S):
                f_trans += v[self.trans_mtx2vec_full[s1, s2]]*cnt_neighs
            f_inner[s1, :] = v_em[:, s1].dot(self.data) + f_trans

        # exp-trick (to prevent NAN because of large numbers): log[sum_i exp(x_i-a)]+a = log[sum_i exp(x_i)]
        max_score = np.max(f_inner)
        f_inner = np.sum(np.exp(f_inner - max_score))
        foo = np.log(f_inner) + max_score
        # max_score = np.max(f_inner, axis=0).reshape((1, self.samples))  # max-score for each sample
        # max_score = np.repeat(max_score, self.S, axis=0)
        # f_inner = np.sum(np.exp(f_inner - max_score), axis=0)
        # foo = np.sum(np.log(f_inner) + max_score)
        if np.isnan(foo) or np.isinf(foo):
            print('TCRFR Pairwise Potential Model: the log_partition is NAN or INF!!')
        return foo

    @profile
    def log_partition_unary(self, v):
        # self.N is a (Nodes x max_connection_count) Matrix containing the indices for each neighbor
        # of each node (indices are 0 for non-neighbors, therefore N_weights is need to multiply this
        # unvalid value with 0.
        v_em = v[self.trans_n*self.trans_d_full:].reshape((self.feats, self.S), order='F')
        f_inner = np.zeros((self.S, self.samples))
        for s1 in range(self.S):
            f_inner[s1, :] = v_em[:, s1].dot(self.data)
        # exp-trick (to prevent NAN because of large numbers): log[sum_i exp(x_i-a)]+a = log[sum_i exp(x_i)]
        max_score = np.max(f_inner)
        f_inner = np.sum(np.exp(f_inner - max_score))
        foo = np.log(f_inner) + max_score
        if np.isnan(foo) or np.isinf(foo):
            print('TCRFR Pairwise Potential Model: the log_partition is NAN or INF!!')
        return foo

    def log_partition_derivative_indep_experimental(self, v):
        v_trans = v[:self.S*self.S].reshape((self.S, self.S), order='F')
        v_em = v[self.trans_n*self.trans_d_full:].reshape((self.feats, self.S), order='F')

        # (A)
        f = np.zeros((self.S, self.samples))
        for s in range(self.S):
            w = v_trans[s, self.latent]
            foo = np.zeros(self.samples)
            #for n in range(len(self.N)):
            #    foo[n] = np.sum(w[self.N[n]])
            f[s, :] = np.exp(v_em[:, s].dot(self.data) + foo)
        sum_f = np.sum(f, axis=0)

        # (B)
        for s in range(self.S):
            f[s, :] /= sum_f

        # (C)
        phis = np.zeros((self.get_num_compressed_dims(), self.samples))
        for s in range(self.S):
            foo = np.zeros(self.samples)
            for n in range(len(self.N)):
                foo[n] = np.sum(self.N[n]==s)
            phis[s, :] = foo * f[s, :]

        idx = self.trans_n*self.trans_d_full
        for s in range(self.S):
            for feat in range(self.feats):
                phis[idx, :] = self.data[feat, :] * f[s, :]
                idx += 1
        return np.sum(phis, axis=1)

    def log_partition_derivative_gibbs_experimental(self, v, max_iter=5):
        """ Gibbs sampler for the expectation of psi-feature map
            (used for the derivative of the partition function).
            returns expectation E_z[psi(X,z)]
        """
        # HINT: avoiding overflow
        # log sum_z exp(z) = a + log sum_z exp(z-a)

        # 1. get good starting point (MAP - solution)
        psi = self.psi
        sample = self.latent
        max_score = v.T.dot(psi)
        psi_cache = np.zeros((self.get_num_compressed_dims(), self.S))

        all_samples = np.zeros((self.get_num_compressed_dims(), max_iter))
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
                scores = np.exp(v.T.dot(psi_cache) - max_score)
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
            if self.verbosity_level>=2:
                print('{0} - score={1}'.format(num_iter, scores[ind]))
            num_iter += 1

        all_scores /= np.sum(all_scores)
        grad = np.zeros(self.get_num_compressed_dims())
        for i in range(max_iter):
            grad += all_scores[i]*all_samples[:, i]
        return grad


@profile
@autojit(nopython=True)
def _extern_get_crf_joint_feature_map(data, y, E, V, mtx2vec, dims, feats, trans_d_full, trans_n):
    psi = np.zeros(dims, dtype=np.float64)
    # Transitions
    for e_idx in range(E.shape[0]):
        yi = y[E[e_idx, 0]]
        yj = y[E[e_idx, 1]]
        etype_offset = np.int32((E[e_idx, 2]-1)*trans_d_full)
        trans_idx = np.int32(mtx2vec[yi, yj])
        psi[trans_idx + etype_offset] += 1.0
    # Emissions
    cnt = trans_n*trans_d_full
    for v in V:
        for f in range(feats):
            psi[cnt+y[v]*feats + f] += data[f, v]
    return psi
