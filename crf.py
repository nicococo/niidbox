import numpy as np
from cvxopt import matrix, spmatrix, normal, uniform, sparse, exp, log
import cvxopt.solvers as solver

from structured_object import StructuredObject


class CRF(StructuredObject):
    """ (Inhomogenous and pairwise) Markov random field (MRF) 
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
        StructuredObject.__init__(self, X, y)
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

    def get_Z_gibbs(self, sol, idx, max_iter=100):
        """ Gibbs sampler for the expectation and partition function.
            Returns (a) (log partition function) log sum_i score_i over unique samples i (typically under-estimator)
                    and corresponding gradient (w.r.t. to 'sol')
                    (b) (expectation) E_z[exp(sol*psi_z)] and gradient (w.r.t to 'sol')
        """
        # HINT: avoiding overflow
        # log sum_z exp(z) = a + log sum_z exp(z-a)

        # 1. get good starting point (MAP - solution)
        (max_score, sample, psi) = self.lp_relax_max(sol, idx)
        current_score = float(np.single(np.exp(max_score - max_score)))
        sample_txt = np.array_str(np.array(sample)[0, :])[1:-1]
        flags = dict()
        flags[sample_txt] = 1

        # 2. outer loop
        Z_score = current_score
        Z_grad = psi*current_score
        E_score = current_score
        E_grad = psi*current_score
        num_iter = 0
        while max_iter > num_iter:

            # 2.1. (sampling) inner loop: sample states
            for v in range(len(self.V)):
                prop_scores = matrix(0.0, (1, self.S))
                for s in range(self.S):
                    sample[v] = s
                    psi = self.get_joint_feature_map(idx, sample)
                    prop_scores[s] = np.exp(sol.trans()*psi)
                # normalize
                prop_scores /= sum(prop_scores)
                # choose uniform sample
                thres = np.single(uniform(1))
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
            
            # 2.2. update expectation
            psi = self.get_joint_feature_map(idx, sample)
            current_score = float(np.single(np.exp(sol.trans()*psi - max_score)))
            E_score += current_score
            E_grad += psi*current_score*(sol.trans()*psi - max_score)

            # 2.3. (partition function and partition gradient) add previously unseen samples
            sample_txt = np.array_str(np.array(sample)[0, :])[1:-1]
            if not flags.has_key(sample_txt):
                Z_score += current_score
                Z_grad += psi*current_score*(sol.trans()*psi - max_score)
                flags[sample_txt] = 1

            num_iter += 1
        # return estimates of scores and gradients
        print len(flags.keys())
        return float(np.single(max_score + np.log(Z_score))), Z_grad/Z_score, float(np.single(max_score + np.log(E_score/float(num_iter)))), (E_grad/float(num_iter))/(E_score/float(num_iter))

    def simple_max(self, sol, idx):
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
        # partition function values
        part_value = 0.0  # partition function Z value
        part_grad = matrix(0.0, (len(sol), 1))  # partition function Z gradient
        # test every single combination 
        for i in range(combinations):
            comb = bin(i)[2:].zfill(len(self.V))  # skip the leading '0b' substring
            # convert to states sequence
            states = matrix(0, (1, len(self.V)))
            for s in range(len(comb)):
                if comb[s] == '1':
                    states[s] = 1
            # check objective function 
            psi = self.get_joint_feature_map(idx, states)
            obj = np.single(sol.trans()*psi)

            part_value += exp(float(obj))
            part_grad += psi*exp(float(obj))

            if obj > max_obj:
                max_obj = obj
                max_states = states
                max_psi = psi

            if obj < min_obj:
                min_obj = obj

        return max_obj, max_states, max_psi, part_value, part_grad, min_obj

    def log_loopy_bp(self, sol, idx, max_iter=250):
        feats = self.get_num_feats()
        verts = len(self.V)
        states = self.S

        # init messages
        msgs = [] 
        max_msgs = [] 
        psis = []
        for v in self.V:
            msgs.append(matrix(0.0, (len(self.N[v]), states)))
            max_msgs.append(matrix(0.0, (len(self.N[v]), states)))
            psis.append(matrix(0.0, (len(self.N[v]), states*states)))
        
        # pre-calc transitions 
        offset = 0
        for e in self.E:
            i = int(e[0])
            j = int(e[1]) 
            # from i to j
            ind = -1
            for n in range(len(self.N[i])):
                if self.N[i][n] == j:
                    ind = n

            psis[i][ind, :] = sol[offset:offset + states*states].trans()
            # from j to i
            ind = -1
            for n in range(len(self.N[j])):
                if self.N[j][n] == i:
                    ind = n

            psis[j][ind, :] = matrix(np.array(sol[offset:offset + states*states]).reshape(states, states).T.reshape(1, states*states))
            offset += states*states

        # pre-calc emission scores
        offset = len(self.E)*states*states
        em = matrix(0.0, (states, verts))
        for v in self.V:
            for s in range(states):
                em[s, v] = sol[offset + s*feats: offset + s*feats + feats].trans() * self.X[idx][:, v]
            offset += states*feats

        print em
        is_converged = False
        num_iter = 0
        while num_iter<max_iter and not is_converged:
            print('BP-iter {0}'.format(num_iter))
            for v in self.V:  # scheduling
                Nv = self.N[v]  # list of neighbor vertices for v
                mv = msgs[v]    # collected messages of v
                max_mv = max_msgs[v]  # collected messages of v
                
                # update messages for all neighbors
                for n in range(len(Nv)):
                    # calculate message to neighbor n 
                    m = matrix(np.sum(mv, 0) - mv[n, :]) # neighbors

                    # calculate message to neighbor n 
                    psi = matrix(np.array(psis[v][n, :]).reshape(states, states))
                    foo = matrix(np.sum(max_mv, 0) - max_mv[n, :]) + em[:, v].trans()
                    
                    max_m = matrix(0.0, (1, states))
                    for s in range(states):
                        ind = int(np.argmax(psi[:, s] + foo[s]))
                        max_m[s] = psi[ind, s] + foo[s]
                    # max_m -= min(max_m)
                    # mn = matrix(np.sum(np.exp(max_m)))

                    # normalize message
                    # m -= min(m)
                    # m /= sum(abs(m))
                    # m = matrix(np.exp(m))
                    # m /= sum(m)
                    # m = matrix(np.log(m))
                    # send
                    ind = -1
                    for nn in range(len(self.N[Nv[n]])):
                        if self.N[Nv[n]][nn] == v:
                            ind = nn
                    msgs[Nv[n]][ind, :] = m
                    max_msgs[Nv[n]][ind, :] = max_m
            num_iter += 1

        # calculate beliefs
        b = []
        for v in self.V:
            belief = np.exp(np.sum(msgs[v], 0)+em[:, v].trans())
            belief /= sum(matrix(belief))
            b.append(belief)
            print belief

        print b
        print '....'
        max_states = []
        for v in self.V:
            state_v = np.argmax(matrix(np.sum(max_msgs[v], 0)).trans() + em[:, v].trans())
            print matrix(np.sum(max_msgs[v], 0)).trans() + em[:, v].trans()
            max_states.append(state_v)

        print max_states
        print '....'

        # calc objective function values
        states = matrix(0, (1, len(self.V)))
        psi = self.get_joint_feature_map(idx, states)
        obj = np.single(sol.trans()*psi)
        return b

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

    def lp_relax_max(self, sol, idx, add_loss=False):
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
        # get loss term
        loss = matrix(0.0, (states, vertices))
        if add_loss:
            loss = matrix(1.0, (states, vertices))
            for i in range(vertices):
                loss[int(self.y[idx][i]), i] = 0.0
        # build weighting term
        c = matrix(0.0, (dims, 1))
        offset = edges*states*states
        # pair-wise potentials
        c[0:offset] = sol[0:offset]
        # emissions
        cnt = offset
        sol_cnt = offset
        for v in self.V:
            for s in range(states):
                c[cnt] = sol[sol_cnt:sol_cnt + feats].trans()*self.X[idx][:, v] + loss[s, v]
                sol_cnt += feats
                cnt += 1
        res = solver.lp(-c, self.lp_ineq_G, self.lp_ineq_h, self.lp_eq_A, self.lp_eq_b, solver='glpk')['x'].trans()
        # convert into state sequence
        max_states = matrix(0, (1, vertices))
        for v in range(vertices):
            max_states[v] = int(np.argmax(res[offset + v*states:offset + v*states + states]))
        psi = self.get_joint_feature_map(idx, max_states)
        obj = np.single(sol.trans()*psi)
        return obj, max_states, psi

    def map(self, sol, idx=-1, add_loss=False, add_prior=False, opt_type='linear'):
        max_obj, max_states, max_psi = self.lp_relax_max(sol, idx, add_loss=add_loss)
        return float(max_obj), max_states, max_psi

    def log_partition(self, sol, idx, add_loss=False, add_prior=False, opt_type='linear'):
        max_obj, max_states, max_psi, Z, Zgrad, min_obj = self.simple_max(sol, idx)
        logE = float(log(Z))
        logE_grad = Zgrad/Z
        return max_obj, max_psi, logE, logE_grad

    def get_loss(self, idx, y):
        return 1.0*sum(abs(self.y[idx].trans()-y))

    def get_local_potential_indices(self):
        pmap = matrix(0, (self.S, self.S))
        cnt = 0
        for i in range(self.S):
            for j in range(self.S):
                pmap[i, j] = cnt
                cnt += 1
        return pmap

    def get_regularizer(self, lambda_pair, lamda_unary, eta_pair, eta_unary, gamma_sym):
        dims = self.get_num_dims()
        offset = len(self.E)*self.S*self.S
        pot_inds = self.get_local_potential_indices()

        # start with a Gaussian prior: diagonal regularization matrix
        prior = spmatrix(lambda_pair, range(dims), range(dims))
        for i in range(offset, dims):
            prior[i, i] = lamda_unary

        # pairwise potentials that include the same vertex should be similar
        start_dim1 = 0
        for e in self.E:
            start_dim2 = 0
            for ee in self.E:
                if not ee == e and (ee[0] in e or ee[1] in e):
                    # bind both pairwise potentials
                    for s in range(self.S*self.S):
                        prior[start_dim1+s, start_dim1+s] += 0.5*eta_pair
                        prior[start_dim2+s, start_dim2+s] += 0.5*eta_pair
                        prior[start_dim1+s, start_dim2+s] -= 1.0*eta_pair  # this will happen twice, hence 2*eta_pair
                start_dim2 += self.S*self.S
            start_dim1 += self.S*self.S

        # neighboring unary potentials shall have similar feature weightings for each state
        start = 0
        for e in self.E:
            vi = e[0]
            vj = e[1]

            start_dim_i = offset + vi*self.S*self.get_num_feats()
            start_dim_j = offset + vj*self.S*self.get_num_feats()

            for f in range(self.get_num_feats()):
                prior[start_dim_i+f, start_dim_i+f] += eta_unary
                prior[start_dim_j+f, start_dim_j+f] += eta_unary
                prior[start_dim_i+f, start_dim_j+f] -= 1.0*eta_unary
                prior[start_dim_j+f, start_dim_i+f] -= 1.0*eta_unary

            # enforce symmetric pairwise potentials
            for s1 in range(self.S):
                for s2 in range(self.S):
                    if not s1 == s2:
                        ind1 = pot_inds[s1, s2]
                        ind2 = pot_inds[s2, s1]
                        prior[start+ind1, start+ind1] += gamma_sym
                        prior[start+ind2, start+ind2] += gamma_sym
                        prior[start+ind1, start+ind2] -= 1.0*gamma_sym
                        prior[start+ind2, start+ind1] -= 1.0*gamma_sym
            start += self.S*self.S
        print prior
        return prior

    def get_joint_feature_map(self, idx, y=None):
        y = np.array(y)
        if np.any(y) is None:
            y = np.array(self.y[idx].trans())
        pot_inds = self.get_local_potential_indices()
        psi = matrix(0.0, (self.get_num_dims(), 1))
        # transitions
        cnt = 0
        for e in self.E:
            yi = y[0, int(e[0])]
            yj = y[0, int(e[1])]
            add = pot_inds[int(yi), int(yj)]
            psi[cnt+add] = 1.0
            cnt += self.S*self.S
        # emissions
        feats = self.get_num_feats()
        for v in self.V:
            for s in range(self.S):
                if s == y[0, v]:
                    for f in range(feats):
                        psi[cnt+f] = self.X[idx][f, v]
                cnt += feats
        return psi

    def get_num_dims(self): 
        # transition part: edges x (states*states)
        # emission part: nodes x states x features
        return len(self.E)*self.S*self.S + len(self.V)*self.S*self.get_num_feats() 

    def evaluate(self, pred): 
        pass
