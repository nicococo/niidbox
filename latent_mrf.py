import numpy as np
from cvxopt import matrix, spmatrix, normal, uniform, sparse, exp, log
import cvxopt.solvers as solver

from so_interface import SOInterface


class LatentMrf(SOInterface):
    """ Homogenous and pairwise Markov random field (MRF)
        that resembles a splice graph.
    """
    ninf = -10.0**15

    A = None  # (#V x #V) MRF structure, binary transition matrix
    S = -1  # number of discrete states for each node {0,..,S-1}

    V = None  # list of vertices in the graph (according to network structure matrix A)
    E = None  # list of tupels of transitions from state i to state j
    N = None  # list of neighbors for each vertex

    # pre-computed linear program approximation constraints
    lp_eq_A = None
    lp_eq_b = None
    lp_ineq_G = None
    lp_ineq_h = None

    def __init__(self, X, A, num_states=2, y=None):
        SOInterface.__init__(self, X, y)
        self.A = A
        self.S = num_states
        (verts, foo) = A.size
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
        # pre-compute lp relaxation constraints
        self.lp_relax_init()

    def get_hotstart_sol(self):
        print('Hot start: setting transition weights to zero and Gaussian emission scores.')
        sol_init = 1.0*normal(self.get_num_dims(), 1)
        return sol_init

    def simple_max(self, v, u=None):
        """ Find the MAP and the log-partition function for 2-state models by exhaustive search. This
            method serves as a sanity check. Do not use this method for real applications!
        """
        if not self.S == 2:
            raise NotImplementedError('Exhaustive search only handles binary state problems.')
        # keep track of the best values
        max_states = matrix(0, (1, len(self.V)))
        max_obj = self.ninf
        max_psi = matrix(0.0, (self.get_num_dims(), 1))
        min_obj = -self.ninf
        combinations = self.S**len(self.V)
        # test every single combination
        for i in range(combinations):
            comb = bin(i)[2:].zfill(len(self.V))  # skip the leading '0b' substring
            # convert to states sequence
            states = matrix(0, (1, len(self.V)))
            for s in range(len(comb)):
                if comb[s] == '1':
                    states[s] = 1
            # check objective function
            psi = self.get_joint_feature_map(states)
            obj = np.single(v.trans()*psi)
            if u is not None:
                obj -= np.single(u.trans()*psi*psi.trans()*u)
            # print 'Iter(', i, ') has obj=', obj

            if obj > max_obj:
                max_obj = obj
                max_states = states
                max_psi = psi

            if obj < min_obj:
                min_obj = obj
        return max_obj, max_states, max_psi

    def lp_relax_init(self):
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
                constr = matrix(0.0, (1, dims))
                for k in range(states):
                    constr[cnt + s*states + k] = 1.0
                constr[offset + ind_i*states + s] = -1.0
                A.append(constr)
                b.append(matrix([0.0]))
            for k in range(states):
                constr = matrix(0.0, (1, dims))
                for s in range(states):
                    constr[cnt + k + s*states] = 1.0
                constr[offset + ind_j*states + k] = -1.0
                A.append(constr)
                b.append(matrix([0.0]))
            cnt += states*states
        # emissions
        for v in self.V:
            # emission constraints: sum_s x_is = 1
            constr = matrix(0.0, (1, dims))
            constr[offset:offset + states] = 1.0
            A.append(constr)
            b.append(matrix([1.0]))
            offset += states
        self.lp_eq_A = sparse(matrix(A))
        self.lp_eq_b = matrix(b)
        # lower bounds
        self.lp_ineq_G = spmatrix(-1.0, range(dims), range(dims))
        self.lp_ineq_h = matrix(0.0, (dims, 1))

    def lp_relax_max(self, sol, Fa=None, Fb=None):
        """ Estimate the MAP by relaxing the integer linear program
            to a linear program (LP):
                min_x   <c,x>
                s.t.    Gx+s=h
                        Ax = b
                        s >= 0  (element-wise)
        """
        states = self.S
        edges = len(self.E)
        vertices = len(self.V)
        feats = self.get_num_feats()
        dims = edges*states*states + vertices*states
        # build weighting term
        c = matrix(0.0, (dims, 1))
        offset = edges*states*states
        # pair-wise potentials
        cnt = 0
        for e in self.E:
            c[cnt:cnt+states*states] = sol[:states*states]
            cnt += states*states
        # emissions
        cnt = offset
        for v in self.V:
            sol_cnt = states*states
            for s in range(states):
                c[cnt] = sol[sol_cnt:sol_cnt + feats].trans()*self.X[:, v]
                sol_cnt += feats
                cnt += 1
        # solution = solver.lp(-c, self.lp_ineq_G, self.lp_ineq_h, self.lp_eq_A, self.lp_eq_b, solver='glpk')
        G = self.lp_ineq_G
        h = self.lp_ineq_h
        A = self.lp_eq_A
        b = self.lp_eq_b
        if Fa is not None:
            print('Extend constraint list by {0} constraints.'.format(Fa.size[0]))
            c = matrix([c, -1.0])  # add slack variable
            # Ax = b
            foo = matrix(0.0, (1, A.size[0]))
            A = matrix([A.trans(), foo]).trans()
            # Gx=<h
            foo = matrix(0.0, (1, G.size[0]))
            G = matrix([G.trans(), foo]).trans()
            # add constraints
            foo = matrix(-1.0, (1, Fa.size[0]))
            Fa = matrix([Fa.trans(), foo]).trans()
            h = matrix([h, -Fb])
            G = matrix([G, Fa])
            print A.size
            print b.size
            print G.size
            print h.size

        solution = solver.lp(-c, G, h, A, b, solver='glpk')
        res = solution['x']
        if Fa is not None:
            res = res[:-1]
            c = c[:-1]
        # print -solution['primal objective']
        # obj = -res.trans()*c
        obj = matrix(solution['primal objective'])

        # convert into state sequence
        max_states = matrix(0, (1, vertices))
        for v in range(vertices):
            max_states[v] = int(np.argmax(res[offset + v*states:offset + v*states + states]))
        psi = self.get_joint_feature_map(max_states)
        # obj = np.single(sol.trans()*psi)
        return obj, max_states, psi, res

    def lp_get_scores(self, w):
        states = self.S
        edges = len(self.E)
        vertices = len(self.V)
        feats = self.get_num_feats()
        dims = edges*states*states + vertices*states
        # build weighting term
        c = matrix(0.0, (dims, 1))
        # pair-wise potentials
        cnt = 0
        for e in self.E:
            c[cnt:cnt+states*states] = w[:states*states]
            cnt += states*states
        # emissions
        cnt = edges*states*states
        for v in self.V:
            sol_cnt = states*states
            for s in range(states):
                c[cnt] = w[sol_cnt:sol_cnt + feats].trans()*self.X[:, v]
                sol_cnt += feats
                cnt += 1
        return c

    def lp_get_objective_at(self, x, w):
        states = self.S
        edges = len(self.E)
        vertices = len(self.V)
        feats = self.get_num_feats()
        dims = edges*states*states + vertices*states
        # build weighting term
        c = matrix(0.0, (dims, 1))
        # pair-wise potentials
        cnt = 0
        for e in self.E:
            c[cnt:cnt+states*states] = w[:states*states]
            cnt += states*states
        # emissions
        cnt = edges*states*states
        for v in self.V:
            sol_cnt = states*states
            for s in range(states):
                c[cnt] = w[sol_cnt:sol_cnt + feats].trans()*self.X[:, v]
                sol_cnt += feats
                cnt += 1
        return c.trans()*x, c

    def qp_relax_max(self, v, u):
        """ Solving the following problem by linear approximation:
                min_x xT*u*uT*x - xT*v
        """
        Fa = None
        Fb = None

        rel = 1.0
        num_iter = 0
        while num_iter < 400 and rel > 1e-6:
            (obj, states, psi, x) = self.lp_relax_max(v, Fa, Fb)
            (xTv, _) = self.lp_get_objective_at(x, v)
            (xTq, q) = self.lp_get_objective_at(x, u)
            obj_quad = x.trans()*q*q.trans()*x
            obj_real = obj_quad - xTv

            Fai = 2.0*q*q.trans()*x  # (dims x 1)
            Fbi = obj_quad - Fai.trans()*x  # scalar

            if Fa is None:
                Fa = Fai.trans()  # a single row
                Fb = Fbi
            else:
                Fa = matrix([Fa, Fai.trans()])  # append row
                Fb = matrix([Fb, Fbi])
            # next iteration
            rel = np.abs((obj_real-obj)/obj_real)[0, 0]
            print('Iter({0}) objectives: real={1} lb={2} rel={3}'.format(num_iter, obj_real[0, 0], obj[0, 0], rel))
            num_iter += 1

        obj = np.single(u.trans()*psi*psi.trans()*u - v.trans()*psi)
        return obj, states, psi

    def argmax(self, idx=-1, add_loss=False, add_prior=False, opt_type='linear'):
        max_obj, max_states, max_psi = self.simple_max(self.sol)
        return float(max_obj), max_states, max_psi

    def get_local_potential_indices(self):
        pmap = matrix(0, (self.S, self.S))
        cnt = 0
        for i in range(self.S):
            for j in range(self.S):
                pmap[i, j] = cnt
                cnt += 1
        return pmap

    def get_joint_feature_map(self, y=None):
        y = np.array(y)
        pot_inds = self.get_local_potential_indices()
        psi = matrix(0.0, (self.get_num_dims(), 1))

        # transitions
        for e in self.E:
            yi = y[0, int(e[0])]
            yj = y[0, int(e[1])]
            trans_idx = pot_inds[int(yi), int(yj)]
            psi[trans_idx] += 1.0

        # emissions
        feats = self.get_num_feats()
        for v in self.V:
            cnt = self.S*self.S
            for s in range(self.S):
                if s == y[0, v]:
                    for f in range(feats):
                        psi[cnt+f] += self.X[f, v]
                cnt += feats
        return psi

    def get_num_dims(self):
        # transition part: edges x (states*states)
        # emission part: nodes x states x features
        return self.S*self.S + self.S*self.get_num_feats()

    def evaluate(self, pred):
        pass
