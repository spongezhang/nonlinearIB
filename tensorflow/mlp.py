from __future__ import print_function

# Import MNIST data
import mnist as mnist_dataset
mnist = mnist_dataset.read_data_sets("../data/", one_hot=True)

import tensorflow as tf
import os
import numpy as np

from Loggers import Logger, FileLogger

import layers

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import plotly.plotly as py
from mpl_toolkits.mplot3d import axes3d, Axes3D

import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--log_dir', default='../tensorflow_logs/', help='folder to output log')
parser.add_argument('--beta' , type=float, default=100, help='beta hyperparameter value')
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--nb_epoch', type=int, default=20, help='Number of epochs')
parser.add_argument("--draw", action="store_true",
                    help="Draw scatter map")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

suffix = 'beta{:1.1f}'.format(args.beta)

LOG_DIR = args.log_dir
if not os.path.isdir(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_DIR = args.log_dir + suffix
logger = Logger(LOG_DIR)

# Parameters
learning_rate = 0.001
training_epochs = args.nb_epoch
batch_size = 128
display_step = 1

# Network Parameters
n_hidden_1 = 800 # 1st layer number of neurons
n_hidden_2 = 800 # 2nd layer number of neurons
n_hidden_3 = 3 # 2nd layer number of neurons
n_hidden_4 = 800 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.get_variable("W1", shape=[n_input, n_hidden_1],
        initializer=tf.contrib.layers.xavier_initializer()),
    'h2': tf.get_variable("W2", shape=[n_hidden_1, n_hidden_2],
        initializer=tf.contrib.layers.xavier_initializer()),
    'h3': tf.get_variable("W3", shape=[n_hidden_2, n_hidden_3],
        initializer=tf.contrib.layers.xavier_initializer()),
    #'h4': tf.get_variable("W4", shape=[n_hidden_3, n_hidden_4],
    #    initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable("W5", shape=[n_hidden_3, n_classes],
        initializer=tf.contrib.layers.xavier_initializer())
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'b3': tf.Variable(tf.zeros([n_hidden_3])),
    'b4': tf.Variable(tf.zeros([n_hidden_4])),
    'out': tf.Variable(tf.zeros([n_classes]))
}

#d1 = tf.Variable(1.0, tf.float32)
# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    norm_w3 = tf.nn.l2_normalize(weights['h3'],dim = 0)#*d1
    corr_mat = tf.matmul(tf.transpose(norm_w3),norm_w3)

    layer_3 = tf.nn.l2_normalize(tf.add(tf.matmul(layer_2, norm_w3), biases['b3']),dim=1)
    #layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, weights['h4']), biases['b4']))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
    return out_layer, layer_3, corr_mat# Construct model

logits, bottleneck_layer, corr_mat = multilayer_perceptron(X)

mi,label_matrix = layers.entropy_estimator(bottleneck_layer, Y)

corr = tf.reduce_mean(tf.reduce_mean(corr_mat))
# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y)) + args.beta*corr

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        avg_corr = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, l, c = sess.run([train_op, loss_op, corr], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += l / total_batch
            avg_corr += c / total_batch
        ## Display logs per epoch step
        offset = 0
        logger.log_value('loss', avg_cost, step = epoch)
        logger.log_value('corr', avg_corr, step = epoch)
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.4f}, corr={:.4f}".format(avg_cost,avg_corr))
            if args.draw:
                if n_hidden_3 == 3:
                    fig = plt.figure()
                    ax = Axes3D(fig)
                bn_layer  = np.zeros([mnist.train.num_examples,n_hidden_3],)
                labels = np.zeros(mnist.train.num_examples,)
                for i in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    fetch = {
                        "bn_layer": bottleneck_layer,
                        "label_matrix": label_matrix,
                    }
                    result = sess.run(fetch, feed_dict={X: batch_x,\
                            Y: batch_y})
                    bn_layer[offset:offset + batch_size] = result['bn_layer']
                    #print(np.argmax(batch_y, 1))
                    labels[offset:offset + batch_size] = np.argmax(batch_y, 1)
                    offset = offset+batch_size
                    #print(batch_y) 
                    #print(result['label_matrix'])
                    #break
                #break

                try: 
                    os.stat('../tensorflow_distribution_map/beta_{:1.1f}/'.format(args.beta))
                except:
                    os.makedirs('../tensorflow_distribution_map/beta_{:1.1f}/'.format(args.beta))
                
                
                for i in range(10):
                    point_by_number = bn_layer[labels==i,:]
                    if n_hidden_3==2:
                        plt.scatter(point_by_number[:,0],point_by_number[:,1],\
                            color='C{}'.format(i), label=str(i), alpha=0.05)
                    elif n_hidden_3 == 3:
                            ax.scatter(point_by_number[:,0],point_by_number[:,1],point_by_number[:,2],\
                            color='C{}'.format(i), label=str(i), alpha=0.05)
                
                plt.legend(loc=4)
                plt.savefig('../tensorflow_distribution_map/beta_{:1.1f}/point_distribution_epoch{:03d}.png'.format(args.beta, epoch), bbox_inches='tight')
                plt.close()

            # Test model
            pred = tf.nn.softmax(logits)  # Apply softmax to logits
            correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            test_acc  = accuracy.eval({X: mnist.test.images, Y: mnist.test.labels})
            print("Accuracy:", test_acc)
            logger.log_value('Test Accuracy', test_acc, step = epoch)

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
