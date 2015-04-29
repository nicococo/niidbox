from cvxopt import matrix, normal


class StructuredObject(object):
    """ Structured Object Interface """

    X = None  # (either matrix or list) data
    y = None  # (list of vectors) state sequences (if present)

    samples = -1  # (scalar) number of training data samples
    feats = -1    # (scalar) number of features != get_num_dims() !!!

    sol = None  # (vector) solution vector

    isListOfObjects = True  # X is either list-of-objects or co.matrix

    def __init__(self, X, y=None):
        self.X = X
        self.y = y

        # assume either co.matrix or list-of-objects
        if isinstance(X, matrix):
            (self.feats, self.samples) = X.size
            self.isListOfObjects = False
        else:
            self.samples = len(X)
            (self.feats, foo) = X[0].shape
        print('Creating structured object with #{0} examples and #{1} features.'.format(self.samples, self.feats))

    def get_hotstart_sol(self): 
        print('Generate a random solution vector for hot start.')
        return normal(self.get_num_dims(), 1)

    def update_solution(self, sol):
        self.sol = sol

    def get_num_samples(self):
        return self.samples

    def get_num_feats(self):
        return self.feats

    def get_num_dims(self):
        pass

    def get_loss(self, idx, y):
        pass

    def get_joint_feature_map(self, idx, y=None):
        pass

    def map(self, idx=-1, add_loss=False, add_prior=False):
        """
        :param idx: index of example or -1 for all examples (default)
        :param add_loss: add a structured loss
        :param add_prior: add prior knowledge
        :return: value(s), structure(s), joint feature map(s)
        """
        pass
        
    def log_partition(self, idx, add_loss=False, add_prior=False):
        pass

    def evaluate(self, pred):
        pass
