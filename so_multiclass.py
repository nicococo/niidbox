import numpy as np
from cvxopt import matrix, normal

from so_interface import SOInterface


class SOMultiClass(SOInterface):
    """ Multi class structured object. """
    num_classes = -1  # (scalar) number of classes

    def __init__(self, X, classes, y=None):
        """ The number of classes directly translate into individual indices
            therefore ensure that y >= 0.
        """
        SOInterface.__init__(self, X, y)
        self.num_classes = classes      

    def get_hotstart_sol(self): 
        print('Generate a random solution vector for hot start.')
        return 1.0*normal(self.get_num_dims(), 1)

    def argmax(self, sol, idx, add_loss=False, add_prior=False, opt_type='linear'):
        # opt_type = 'quadratic':
        # the argmax is equal to the argmax of the linear function
        # foo = -normSol + 2*foo - normPsi
        # since ||\Psi(x_i,z)|| = ||\phi(x_i)|| = y \forall z
        # and normSol is also constant
        w = matrix(np.array(sol).reshape(self.feats, self.num_classes))
        foo = w.trans()*self.X[:, idx]

        # highest value first
        inds = np.argsort(-foo, axis=0)[0]
        cls = inds[0]
        val = foo[cls]

        psi_idx = self.get_joint_feature_map(idx, cls)
        return val, cls, psi_idx

    def calc_loss(self, idx, y):
        return self.y[idx] != y

    def get_joint_feature_map(self, idx, y=None):
        if y is None:
            y = self.y[idx]
        nd = self.feats
        mc = self.num_classes
        psi = matrix(0.0, (nd*mc, 1))
        psi[nd*y:nd*(y+1)] = self.X[:, idx]
        return psi

    def get_num_dims(self):
        return self.feats*self.num_classes