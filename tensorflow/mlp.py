""" Multilayer Perceptron.

A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library. This example is using the MNIST database of handwritten
digits (http://yann.lecun.com/exdb/mnist/).

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

# ------------------------------------------------------------------
#
# THIS EXAMPLE HAS BEEN RENAMED 'neural_network.py', FOR SIMPLICITY.
#
# ------------------------------------------------------------------


from __future__ import print_function

# Import MNIST data
import mnist as mnist_dataset
mnist = mnist_dataset.read_data_sets("../data/", one_hot=True)

import tensorflow as tf
import os
import numpy as np

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import plotly.plotly as py

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Parameters
learning_rate = 0.001
training_epochs = 20
batch_size = 128
display_step = 1

BETA_VAL = 0.0

# Network Parameters
n_hidden_1 = 800 # 1st layer number of neurons
n_hidden_2 = 800 # 2nd layer number of neurons
n_hidden_3 = 2 # 2nd layer number of neurons
n_hidden_4 = 800 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
#weights = {
#    'h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1])),
#    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
#    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
#    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
#    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_classes]))
#}
weights = {
    'h1': tf.get_variable("W1", shape=[n_input, n_hidden_1],
        initializer=tf.contrib.layers.xavier_initializer()),
    'h2': tf.get_variable("W2", shape=[n_hidden_1, n_hidden_2],
        initializer=tf.contrib.layers.xavier_initializer()),
    'h3': tf.get_variable("W3", shape=[n_hidden_2, n_hidden_3],
        initializer=tf.contrib.layers.xavier_initializer()),
    'h4': tf.get_variable("W4", shape=[n_hidden_3, n_hidden_4],
        initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable("W5", shape=[n_hidden_4, n_classes],
        initializer=tf.contrib.layers.xavier_initializer())
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'b3': tf.Variable(tf.zeros([n_hidden_3])),
    'b4': tf.Variable(tf.zeros([n_hidden_4])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4']))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer, layer_3

# Construct model
logits, bottleneck_layer = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
#loss_op = tf.contrib.keras.losses.categorical_crossentropy(Y, logits)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        ## Display logs per epoch step
        offset = 0
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            
            bn_layer  = np.zeros([mnist.train.num_examples,n_hidden_3],)
            labels = np.zeros(mnist.train.num_examples,)
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                fetch = {
                    "bn_layer": bottleneck_layer,
                }
                result = sess.run(fetch, feed_dict={X: batch_x,\
                        Y: batch_y})
                bn_layer[offset:offset + batch_size] = result['bn_layer']
                #print(np.argmax(batch_y, 1))
                labels[offset:offset + batch_size] = np.argmax(batch_y, 1)
                offset = offset+batch_size
            try: 
                os.stat('../tensorflow_distribution_map/beta_{:1.1f}/'.format(BETA_VAL))
            except:
                os.makedirs('../tensorflow_distribution_map/beta_{:1.1f}/'.format(BETA_VAL))
            
            for i in range(10):
                point_by_number = bn_layer[labels==i,:]
                plt.scatter(point_by_number[:,0],point_by_number[:,1], color='C{}'.format(i), label=str(i), alpha=0.05)
            
            plt.legend(loc=4)
            plt.savefig('../tensorflow_distribution_map/beta_{:1.1f}/point_distribution_epoch{:03d}.png'.format(BETA_VAL, epoch), bbox_inches='tight')
            plt.clf()

            # Test model
            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
