import numpy as np
import scipy
#import keras.backend as K
import tensorflow as tf

def np_entropy(p):
    cp = np.log(p)
    cp[np.isclose(p,0.)]=0.
    return -p.dot(cp)

def logsumexp(mx, axis):
    cmax = tf.reduce_max(mx, axis=axis)
    cmax2 = tf.expand_dims(cmax, 1)
    mx2 = mx - cmax2
    return cmax + tf.log(tf.reduce_sum(tf.exp(mx2), axis=1))

def kde_entropy(output, var):
    # Kernel density estimate of entropy, in nats

    dims = tf.cast(tf.shape(output)[1], tf.float32) 
    N    = tf.cast(tf.shape(output)[0], tf.float32)
    
    normconst = (dims/2.0)*tf.log(2*np.pi*var)
            
    # get dists matrix
    x2 = tf.expand_dims(tf.reduce_sum(tf.square(output), axis=1), 1)
    dists = x2 + tf.transpose(x2) - 2*tf.matmul(output, tf.transpose(output))
    dists = dists / (2*var)
    
    lprobs = logsumexp(-dists, axis=1) - tf.log(N) - normconst
    h = -tf.mean(lprobs)
    
    return h

def kde_entropy_category(output, label, var):
    # Kernel density estimate of entropy, in nats

    dims = tf.cast(tf.shape(output)[1], tf.float32) 
    N    = tf.cast(tf.shape(output)[0], tf.float32)
    
    normconst = (dims/2.0)*tf.log(2*np.pi*var)
     
    # get dists matrix
    x2 = tf.expand_dims(tf.reduce_sum(tf.square(output), axis=1), 1)
    label_matrix = tf.tile(tf.expand_dims(tf.argmax(label,axis = 1),1),[1,tf.shape(output)[0]])
    #print(label.eval())
    #print(label_matrix.eval())
    #break
    similarity_matrix = tf.cast((tf.equal(label_matrix,tf.transpose(label_matrix))),tf.float32)
    #similar_num = tf.reduce_sum(tf.reduce_sum(similarity_matrix))
    dists = x2 + tf.transpose(x2) - 2*tf.matmul(output, tf.transpose(output))
    dists = dists / (2*var)
    category_dists = tf.multiply(similarity_matrix,dists)
    
    #lprobs = logsumexp(-category_dists, axis=1) - tf.log(N) - normconst
    #h = -tf.reduce_mean(lprobs)
    h = tf.reduce_mean(tf.reduce_mean(category_dists))*10
    return h, similarity_matrix

def kde_condentropy(output, var):
    # Return entropy of a multivariate Gaussian, in nats

    dims = tf.cast(tf.shape(output)[1], tf.float32)
    normconst = (dims/2.0)*tf.log(2*np.pi*var)
    return normconst

def kde_entropy_from_dists_loo(dists, N, dims, var):
    # Given a distance matrix dists, return leave-one-out kernel density estimate of entropy
    # Dists should have very large values on diagonal (to make those contributions drop out)
    dists2 = dists / (2*var)
    normconst = (dims/2.0)*tf.log(2*np.pi*var)
    lprobs = logsumexp(-dists2, axis=1) - np.log(N-1) - normconst
    h = -tf.mean(lprobs)
    return h

