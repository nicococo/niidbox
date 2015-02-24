from cvxopt import matrix
import numpy as np
import sklearn.cluster as cl


class LatentRidgeRegression:
    """ Latent Variable Ridge Regression.
        Written by Nico Goernitz, TU Berlin, 2015
    """
    reg = 0.001  # (scalar) the regularization constant > 0
    sobj = None 
    sol = None 
    intercept = 0.0

    def __init__(self, sobj, l=0.001):
        self.reg = l
        self.sobj = sobj

    def train_dc(self, max_iter=50, hotstart=matrix([])):
        runs = 10
        obj = 1e14

        best_sol = 0
        best_lats = []
        for i in range(runs):
            (sol, n_lat, n_obj, is_converged) = self.train_dc_single(max_iter=max_iter)

            if is_converged and np.single(obj) > np.single(n_obj):
                best_sol = sol
                best_lats = n_lat
                obj = n_obj

            if not is_converged and i == 0:
                best_sol = sol
                best_lats = n_lat

        self.sol = best_sol
        return best_sol, best_lats


    def train_dc_single(self, max_iter=50, hotstart=matrix([])):
        N = self.sobj.get_num_samples()
        DIMS = self.sobj.get_num_dims()

        # intermediate solutions
        # latent variables
        latent = [0.0]*N
        self.sol = self.sobj.get_hotstart_sol()
        if hotstart.size == (DIMS, 1):
            print('New hotstart position defined.')
            self.sol = hotstart
        psi = matrix(0.0, (DIMS, N))  # (dim x exm)

        obj = 1e09
        old_obj = 1e10
        rel = 1.0
        iter = 0
        is_converged = False
        # terminate if objective function value doesn't change much
        while iter < max_iter and not is_converged:
            # 1. linearized
            # for the current solution compute the 
            # most likely latent variable configuration
            num_changed = 0
            for i in range(N):
                if iter > -1:
                    (foo, lat, psi[:, i]) = self.sobj.argmax(self.sol, i, add_prior=True)
                else:
                    lat = np.int(self.sobj.y[i, 0]-1.0)
                    psi[:, i] = self.sobj.get_joint_feature_map(i, y=lat)
                if not latent[i] == lat:
                    num_changed += 1
                latent[i] = lat
            #print('{0}/{1} examples changed latent var.'.format(num_changed,N))

            # 2. Solve the intermediate optimization problem
            vecy = np.array(matrix(self.sobj.y))[:, 0]
            vecX = np.array(psi.trans())
            self.sol = self.train_model(vecX, vecy)

            # calc objective function:
            w = self.sol  # (dims x 1)
            l = self.reg  # scalar
            b = self.sobj.y  # (exms x 1)
            phi = psi  # (dims x exms)
            old_obj = obj
            obj = l*w.trans()*w + b.trans()*b - 2.0*w.trans()*phi*b + w.trans()*phi*phi.trans()*w
            rel = np.abs((old_obj - obj)/obj)
            print('Iter {0} objective={1:4.2f}  rel={2:2.4f}'.format(iter, obj[0, 0], rel[0, 0]))

            if iter > 3 and rel < 0.0001:
                is_converged = True
            iter += 1

        print np.unique(latent)
        return self.sol, latent, obj, is_converged

    def train_model(self, vecX, vecy):
        # solve the ridge regression problem 
        E = np.zeros((vecX.shape[1], vecX.shape[1]))
        np.fill_diagonal(E, self.reg)
        XXt = vecX.T.dot(vecX) + E
        #print XXt.shape
        #print XXt
        XtY = (vecX.T.dot(vecy))
        if XXt.size>1:
            w = np.linalg.pinv(XXt).dot(XtY)
        else: 
            w = 1.0/XXt * XtY
        return matrix(w)

    def apply(self, pred_sobj):
        """ Application of the Latent Ridge Regression:
            score = max_z <sol*,\Psi(x,z)>
            latent_state = argmax_z <sol*,\Psi(x,z)> 
        """
        N = pred_sobj.get_num_samples()
        vals = matrix(0.0, (N, 1))
        structs = []
        for i in range(N):
            (vals[i], struct, psi) = pred_sobj.argmax(self.sol, i, add_prior=False)
            vals[i] += self.intercept
            structs.append(struct)
        return vals, structs





class LatentRidgeRegression2(LatentRidgeRegression):
    """ Latent Variable Ridge Regression.
        Written by Nico Goernitz, TU Berlin, 2015
    """
    cls = None

    def __init__(self, sobj, l=0.001):
        LatentRidgeRegression.__init__(self, sobj, l)
        self.cls = []

    def train_dc_single(self, max_iter=50, hotstart=matrix([])):
        N = self.sobj.get_num_samples()
        DIMS = self.sobj.get_num_dims()
        # intermediate solutions
        # latent variables
        latent = [0.0]*N
        self.sol = self.sobj.get_hotstart_sol()
        if hotstart.size==(DIMS,1):
            print('New hotstart position defined.')
            self.sol = hotstart
        psi = matrix(0.0, (DIMS,N)) # (dim x exm)
        old_psi = matrix(0.0, (DIMS,N)) # (dim x exm)

        # kmeans = cl.KMeans(n_clusters=self.sobj.num_classes, init='k-means++', n_init=10, max_iter=300, tol=0.0001)
        # kmeans.fit(self.sobj.X.trans())
        # print kmeans.cluster_centers_.shape
        # print self.sobj.feats
        # print kmeans.labels_
        # self.sol = matrix(kmeans.cluster_centers_.reshape(self.sobj.num_classes*self.sobj.feats, 1))
        #
        # cnt = 0
        # self.cls = []
        # for i in range(self.sobj.num_classes):
        #     inds = np.where(kmeans.labels_==i)[0]
        #     vecy = np.array(matrix(self.sobj.y))[inds,0]
        #     vecX = np.array(self.sobj.X.trans())[inds,:]
        #     self.sol[cnt:cnt+self.sobj.feats] = self.train_model(vecX, vecy)
        #     self.cls.append(self.sol[cnt:cnt+self.sobj.feats])
        #     cnt += self.sobj.feats
        for i in range(self.sobj.num_classes):
            self.cls.append(0)

        obj = 1e09
        old_obj = 1e10
        rel = 1
        iter = 0 
        is_converged = False
        # terminate if objective function value doesn't change much
        while iter < max_iter and not is_converged:
            # 1. linearize
            # for the current solution compute the 
            # most likely latent variable configuration
            num_changed = 0
            for i in range(N):
                if iter>-1:
                    (foo, lat, psi[:,i]) = self.sobj.argmax(self.sol, i, add_prior=True)
                else:
                    lat = np.int(self.sobj.y[i,0]-1.0)
                    psi[:,i] = self.sobj.get_joint_feature_map(i, y=lat)
                if not latent[i]==lat:
                    num_changed += 1
                latent[i] = lat
            print('{0}/{1} examples changed latent var.'.format(num_changed,N))

            # 2. Solve the intermediate optimization problem
            vecy = np.array(matrix(self.sobj.y))[:,0]
            vecX = np.array(psi.trans())
            self.sol = self.train_model(vecX, vecy)

            for i in range(self.sobj.num_classes):
                inds = np.where(np.uint(latent) == i)[0]
                vecy = np.array(matrix(self.sobj.y))[inds, 0]
                vecX = np.array(self.sobj.X.trans())[inds, :]
                self.cls[i] = self.train_model(vecX, vecy)

            # calc objective function:
            w = self.sol  # (dims x 1)
            l = self.reg  # scalar
            b = self.sobj.y  # (exms x 1)
            phi = psi  # (dims x exms)
            old_obj = obj
            obj = l*w.trans()*w + b.trans()*b - 2.0*w.trans()*phi*b + w.trans()*phi*phi.trans()*w
            rel = np.abs((old_obj - obj)/obj)
            print('Iter {0} objective={1:4.2f}  rel={2:2.4f}'.format(iter, obj[0, 0], rel[0, 0]))

            if iter > 3 and rel < 0.0001:
                is_converged = True
            iter += 1

        print np.unique(latent)
        return self.sol, latent, obj, is_converged

    def apply(self, pred_sobj):
        """ Application of the Latent Ridge Regression:

            score = max_z <sol*,\Psi(x,z)> 
            latent_state = argmax_z <sol*,\Psi(x,z)> 
        """
        N = pred_sobj.get_num_samples()
        vals = matrix(0.0, (N, 1))
        structs = []
        for i in range(N):
            (vals[i], struct, psi) = pred_sobj.argmax(self.sol, i, add_prior=False)

            vecX = pred_sobj.X[:, i]
            vals[i] = self.cls[struct].trans() * vecX

            vals[i] += self.intercept
            structs.append(struct)
        return vals, structs
