from __future__ import absolute_import
import theano
import theano.tensor as T
import numpy as np
from six.moves import range

epsilon = 1.0e-15

def mean_squared_error(y_true, y_pred):
    return T.sqr(y_pred - y_true).mean()

def mean_absolute_error(y_true, y_pred):
    return T.abs_(y_pred - y_true).mean()

def squared_hinge(y_true, y_pred):
    return T.sqr(T.maximum(1. - y_true * y_pred, 0.)).mean()

def hinge(y_true, y_pred):
    return T.maximum(1. - y_true * y_pred, 0.).mean()

def categorical_crossentropy(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes
    '''
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    # scale preds so that the class probas of each sample sum to 1
    y_pred /= y_pred.sum(axis=1, keepdims=True) 
    return T.nnet.categorical_crossentropy(y_pred, y_true).mean()

def binary_crossentropy(y_true, y_pred):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    return T.nnet.binary_crossentropy(y_pred, y_true).mean()

#from conv_3d code
cse_eps = .000001
def cross_entropy_error(y_true, y_pred):
    y_true = y_true.flatten(2)
    y_pred = T.clip(y_pred, cse_eps, 1.0 - cse_eps)
    L = - T.sum(y_true * T.log(y_pred) + (1 - y_true) * T.log(1 - y_pred), axis=1)
    cost = T.mean(L)
    return cost

# aliases
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error

from .utils.generic_utils import get_from_module
def get(identifier):
    return get_from_module(identifier, globals(), 'objective')

def to_categorical(y):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy
    '''
    nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def jaccard_similarity(a, b):
    '''
    Returns the number of pixels of the intersection of two voxel grids divided by the number of pixels in the union.
    A return value of 1 means that the two binary grids are identical.
    The inputs are expected to be theano tensors where we flatten all dimensions except for the first, and we average the simmilarity accross the 1st dimension.
    '''
    a = a.flatten(ndim=2)
    b = b.flatten(ndim=2)
    return T.mean( T.sum(a*b,       axis=1) \
                    /T.sum((a+b)-a*b, axis=1) )