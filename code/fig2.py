# Example that demonstrates how to generate a figure similar to Fig.2 in Artemy Kolchinsky, Brendan D. Tracey, David H. Wolpert, "Nonlinear Information Bottleneck", https://arxiv.org/abs/1705.02436

# Minimal example of how to create a model with nonlinearIB layers
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense
import keras
import buildmodel, layers, training, reporting
import os
import scipy.io as sio
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import plotly.plotly as py

from Loggers import Logger, FileLogger

import keras.backend as K
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--beta' , type=float, default=0.4, help='beta hyperparameter value')
parser.add_argument('--gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--nb_epoch', type=int, default=100, help='Number of epochs')
parser.add_argument('--log_dir', default='../fig_logs/', help='folder to output log')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

suffix = 'beta{:1.1f}'.format(args.beta)
    
BETA_VAL = args.beta

trn, tst = buildmodel.get_mnist()
input_layer = Input((trn.X.shape[1],))

# hidden_layers_to_add should include a list of all layers that will get added before the nonlinearIB layers
hidden_layers_to_add = [Dense(800, activation='relu'),
                        Dense(800, activation='relu'),
                        Dense(2, activation='linear'), ]

# *** The following creates the layers and callbacks necessary to run nonlinearIB ***
micalculator = layers.MICalculator(BETA_VAL, model_layers=hidden_layers_to_add,
        data=trn.X, label = trn.y, miN=1000)
noiselayer = layers.NoiseLayer(logvar_trainable=True, test_phase_noise=False)
micalculator.set_noiselayer(noiselayer)

#    Start hooking up the layers together
cur_hidden_layer = input_layer
for l in hidden_layers_to_add:
    cur_hidden_layer = l(cur_hidden_layer)

noise_input_layer = layers.IdentityMap(activity_regularizer=micalculator)(cur_hidden_layer)
nonlinearIB_output_layer = noiselayer(noise_input_layer)

nonlinearIB_callback = training.KDETrain(mi_calculator=micalculator)
# *** Done setting up nonlinearIB stuff ***

decoder = Dense(800, activation='relu')(nonlinearIB_output_layer)

outputs = Dense(trn.nb_classes, activation='softmax')(decoder)

model = Model(inputs=input_layer, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

f = K.function([model.layers[0].input, K.learning_phase()], [noiselayer.output,])
class draw_scatter(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        hiddenlayer_activations = f([trn.X,0])[0]
        try: 
            os.stat('../distribution_map/beta_{:1.1f}/'.format(BETA_VAL))
        except:
            os.makedirs('../distribution_map/beta_{:1.1f}/'.format(BETA_VAL))
        
        for i in range(10):
            point_by_number = hiddenlayer_activations[trn.y==i,:]
            plt.scatter(point_by_number[:,0],point_by_number[:,1], color='C{}'.format(i), label=str(i), alpha=0.05)
        
        plt.legend(loc=4)
        plt.savefig('../distribution_map/beta_{:1.1f}/point_distribution_epoch{:03d}.png'.format(BETA_VAL, epoch), bbox_inches='tight')
        plt.clf()
        return

draw_scatter_callback = draw_scatter()
init_lr = 0.001
lr_decay = 0.5
lr_decaysteps = 15
import keras.callbacks
def lrscheduler(epoch):
    lr = init_lr * lr_decay**np.floor(epoch / lr_decaysteps)
    #lr = max(lr, 1e-5)
    print('Learning rate: %.7f' % lr)
    return lr
lr_callback = keras.callbacks.LearningRateScheduler(lrscheduler)

model.fit(x=trn.X, y=trn.Y, verbose=2, batch_size=128, epochs=args.nb_epoch, 
          validation_data=(tst.X, tst.Y), 
          callbacks=[nonlinearIB_callback, lr_callback, draw_scatter_callback])


