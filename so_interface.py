import numpy as np
import math as math
from cvxopt import matrix, normal



class SOInterface(object):
    """ Structured Object Interface"""

    X = [] # (either matrix or list) data 
    y = [] # (list of vectors) state sequences (if present)

    samples = -1 # (scalar) number of training data samples
    feats = -1 # (scalar) number of features != get_num_dims() !!!

    isListOfObjects = True # X is either list-of-objects or co.matrix

    def __init__(self, X, y=[]):
        self.X = X
        self.y = y

        # assume either co.matrix or list-of-objects
        if isinstance(X, matrix):
            (self.feats, self.samples) = X.size
            isListOfObjects = False
        else: #list
            self.samples = len(X)
            (self.feats, foo) = X[0].shape
        print('Creating structured object with #{0} training examples, each consisting of #{1} features.'.format(self.samples,self.feats))

    def get_hotstart_sol(self): 
        print('Generate a random solution vector for hot start.')
        return  normal(self.get_num_dims(), 1)

    def get_num_samples(self):
        return self.samples

    def get_num_feats(self):
        return self.feats

    def argmax(self, sol, idx, add_loss=False, add_prior=False, opt_type='linear'): 
        pass
        
    def logsumexp(self, sol, idx, add_loss=False, add_prior=False, opt_type='linear'): 
        pass

    def calc_loss(self, idx, y): 
        pass

    def get_joint_feature_map(self, idx, y=[]): 
        pass

    def get_num_dims(self): 
        pass

    def evaluate(self, pred): 
        pass
