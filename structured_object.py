from cvxopt import matrix, normal

from numpy import array
import numpy as np


class StructuredObject(object):
    """ Structured Object Interface """

    X = None  # (either matrix or list) data
    y = None  # (list of vectors) state sequences (if present)

    samples = -1  # (scalar) number of training data samples
    feats = -1    # (scalar) number of features != get_num_dims() !!!

    sol = None  # (matrix or array) solution vector(s)

    isListOfObjects = True  # X is either list-of-objects or (cvxopt matrix or numpy array)

    def get_hotstart_sol(self):
        print('Generate a random solution vector for hot start.')
        return 10.0*normal(self.get_num_dims(), 1)

    def __init__(self, X, y=None):
        self.X = X
        self.y = y
        # assume either co.matrix or list-of-objects
        if isinstance(X, matrix):
            (self.feats, self.samples) = X.size
            self.isListOfObjects = False
        elif isinstance(X, np.array):
            (self.feats, self.samples) = X.shape
            self.isListOfObjects = False
        else:
            self.samples = len(X)
            (self.feats, foo) = X[0].shape
        print('Creating structured object with #{0} examples and #{1} features.'.format(self.samples, self.feats))

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
        
    def log_partition(self, idx):
        pass

    def log_partition_derivative(self, idx):
        pass

    def evaluate(self, pred):
        pass


class TransductiveStructuredModel(object):
    """ Transductive Structured Object Interface
    """
    data = None  # (either matrix or list) data
    labels = None  # (list or matrix or array) labels
    label_inds = None  # index of corresponding data object for each label
    unlabeled_inds = None  # indices for unlabeled examples

    latent = None  # target

    samples = -1  # (scalar) number of training data samples
    feats = -1    # (scalar) number of features != get_num_dims() !!!

    isListOfObjects = True  # X is either list-of-objects or (cvxopt matrix or numpy array)

    def __init__(self, data, labels, label_inds, unlabeled_inds):
        self.data = data
        self.labels = np.array(labels)
        self.label_inds = np.array(label_inds)
        self.unlabeled_inds = np.array(unlabeled_inds)
        # assume either co.matrix or list-of-objects
        if isinstance(data, matrix):
            self.feats, self.samples = data.size
            self.isListOfObjects = False
        elif isinstance(data, np.ndarray):
            self.feats, self.samples = data.shape
            self.isListOfObjects = False
        else:
            self.samples = len(data)
            self.feats, foo = data[0].shape
        # self.unlabeled_inds = np.array(range(self.samples))
        # self.unlabeled_inds = np.setxor1d(self.unlabeled_inds, self.label_inds)
        print('Creating structured object with #{0} examples and #{1} features.'.format(self.samples, self.feats))
        print('Dims:')
        print('- Data          : {0}'.format(self.data.shape))
        print('- Labels        : {0}'.format(self.labels.shape))
        print('- Label Inds    : {0}'.format(self.label_inds.shape))
        print('- Unlabeled Inds: {0}'.format(self.unlabeled_inds.shape))


    def get_hotstart(self):
        pass

    def get_num_labeled(self):
        return len(self.labels)

    def get_num_unlabeled(self):
        return len(self.unlabeled_inds)

    def get_num_samples(self):
        return self.samples

    def get_num_feats(self):
        return self.feats

    def get_num_dims(self):
        pass

    def get_joint_feature_maps(self):
        pass

    def maps(self, sol):
        pass

    def log_partition(self, sol):
        pass

    def log_partition_derivatives(self, sol):
        pass

    def evaluate(self, true_labels):
        pass
