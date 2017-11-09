import numpy as np
from entropy import *
import tensorflow as tf

def entropy_estimator(x = None, y = None):
    return get_mi(x,y)

def get_h(x=None, y = None):
        # returns entropy
    current_var = tf.exp(0.2)
    return kde_entropy_category(x, y, current_var)

def get_mi(x=None, y = None):
    mi = get_h(x,y) 
    return mi
