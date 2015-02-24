from cvxopt import matrix, normal
import numpy as np
from so_interface import SOInterface


class SOMultiClass(SOInterface):
    """ Multi class structured object."""

    num_classes = -1 # (scalar) number of classes 


    def __init__(self, X, classes, y=[]):
        # the class also acts as indices therefore:
        # y >= 0!
        SOInterface.__init__(self, X, y)
        self.num_classes = classes      

    def get_hotstart_sol(self): 
        print('Generate a random solution vector for hot start.')
        return 1.0*normal(self.get_num_dims(), 1)

    def argmax(self, sol, idx, add_prior=False, add_loss=False):
        nd = self.feats
        d = 0  # start of dimension in sol
        val = -10.0**10
        cls = -1 # best class

        for c in range(self.num_classes):
            foo = sol[d:d+nd].trans()*self.X[:,idx]
            # the argmax of the above function
            # is equal to the argmax of the quadratic function
            # foo = -normSol + 2*foo - normPsi
            # since ||\Psi(x_i,z)|| = ||\phi(x_i)|| = y \forall z   
            # and normSol is also constant
            d += nd
            if (np.single(foo)>np.single(val)):
                val = foo
                cls = c

        psi_idx = self.get_joint_feature_map(idx, cls)
        return (val, cls, psi_idx)

    def calc_loss(self, idx, y):
        return self.y[idx] != y

    def get_joint_feature_map(self, idx, y=-1):
        if y == -1:
            y = self.y[idx]

        nd = self.feats
        mc = self.num_classes
        psi = matrix(0.0, (nd*mc, 1))
        psi[nd*y:nd*(y+1)] = self.X[:, idx]
        return psi

    def get_num_dims(self):
        return self.feats*self.num_classes