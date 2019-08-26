########## Machine Learning in Condensed Matter Physics ###############################
###                     Roger Melko
###   with dataset and code by 
####  Giacomo Torlai, Juan Carrasquilla, Estelle Inack and Lauren Hayward Sierens
###
### This code will train a Restricted Boltzmann Machine (RBM) to learn the distribution of 
### spin configurations of the two-dimensional Ising model at a given temperature.
#####################################################################################


from __future__ import print_function
import tensorflow as tf
import itertools as it
from rbm import RBM
import matplotlib.pyplot as plt
import numpy as np
import math
import os

#Specify font sizes for plots:
plt.rcParams['axes.labelsize']  = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

plt.ion() # turn on interactive mode (for plotting)

# Input parameters:
L                   = 4       #linear size of the system
T                   = 2.269   #a temperature for which there are MC configurations stored in Data_ising2d/MC_results
num_visible         = L*L     #number of visible nodes
num_hidden          = 4       #number of hidden nodes
nsteps              = 20000   #number of training steps (iterations over the mini-batches)
learning_rate_start = 1e-3    #the learning rate will start at this value and decay exponentially
bsize               = 100     #batch size
num_gibbs           = 10      #number of Gibbs iterations (steps of contrastive divergence)
num_samples         = 10      #number of chains in PCD

### Function to save weights and biases to a parameter file ###
def save_parameters(sess, rbm):
    weights, visible_bias, hidden_bias = sess.run([rbm.weights, rbm.visible_bias, rbm.hidden_bias])
    
    parameter_dir = 'Data_ising2d/RBM_parameters'
    if not(os.path.isdir(parameter_dir)):
      os.mkdir(parameter_dir)
    parameter_file_path =  '%s/parameters_nH%d_L%d' %(parameter_dir,num_hidden,L)
    parameter_file_path += '_T' + str(T)
    np.savez_compressed(parameter_file_path, weights=weights, visible_bias=visible_bias, hidden_bias=hidden_bias)

class Placeholders(object):
    pass

class Ops(object):
    pass

weights      = None  #weights
visible_bias = None  #visible bias
hidden_bias  = None  #hidden bias

# Load the MC configuration training data:
trainFileName = 'Data_ising2d/MC_results/ising2d_L'+str(L)+'_T'+str(T)+'_train.txt'
xtrain        = np.loadtxt(trainFileName)
testFileName  = 'Data_ising2d/MC_results/ising2d_L'+str(L)+'_T'+str(T)+'_test.txt'
xtest         = np.loadtxt(testFileName)

xtrain_randomized = np.random.permutation(xtrain) # random permutation of training data
xtest_randomized  = np.random.permutation(xtest) # random permutation of test data
iterations_per_epoch = xtrain.shape[0] / bsize  

# Initialize the RBM class
rbm = RBM(num_hidden=num_hidden, num_visible=num_visible, weights=weights, visible_bias=visible_bias,hidden_bias=hidden_bias, num_samples=num_samples) 

# Initialize operations and placeholders classes
ops          = Ops()
placeholders = Placeholders()
placeholders.visible_samples = tf.placeholder(tf.float32, shape=(None, num_visible), name='v') # placeholder for training data

total_iterations = 0 # starts at zero 
ops.global_step  = tf.Variable(total_iterations, name='global_step_count', trainable=False)
learning_rate    = tf.train.exponential_decay(
    learning_rate_start,
    ops.global_step,
    100 * xtrain.shape[0]/bsize,
    1.0 # decay rate = 1 means no decay
)
  
cost      = rbm.neg_log_likelihood_forGrad(placeholders.visible_samples, num_gibbs=num_gibbs)
optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-2)
ops.lr    = learning_rate
ops.train = optimizer.minimize(cost, global_step=ops.global_step)
#ops.init  = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables()) #depricated
ops.init  = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# Define the negative log-likelihood
# We can use this to plot the RBM's training progress.
# This calculation is intractable for large networks so let's only do it for small num_hidden
logZ = rbm.exact_log_partition_function()
placeholders.logZ = tf.placeholder(tf.float32)
NLL = rbm.neg_log_likelihood(placeholders.visible_samples,placeholders.logZ)

sess = tf.Session()
sess.run(ops.init)
  
bcount         = 0  #counter
epochs_done    = 0  #epochs counter
nll_test_list  = [] #negative log-likelihood for each epoch
nll_train_list = [] #negative log-likelihood for each epoch
for ii in range(nsteps):
    if bcount*bsize+ bsize>=xtrain.shape[0]:
        bcount = 0
        xtrain_randomized = np.random.permutation(xtrain)

    batch     =  xtrain_randomized[ bcount*bsize: bcount*bsize+ bsize,:]
    bcount    += 1
    feed_dict =  {placeholders.visible_samples: batch}

    _, num_steps = sess.run([ops.train, ops.global_step], feed_dict=feed_dict)

    if num_steps % iterations_per_epoch == 0:
        lz = sess.run(logZ)
        nll_test = sess.run(NLL,feed_dict={placeholders.visible_samples: xtest_randomized, placeholders.logZ: lz})
        nll_test_list.append(nll_test)
    
        print ('Epoch = %d, Nsteps = %d, NLL on test data = %.6f' %(epochs_done,num_steps,nll_test))
        save_parameters(sess, rbm)
        epochs_done += 1

        # Update the plot:
        plt.figure(1)
        plt.clf()
        plt.plot( np.arange(epochs_done), nll_test_list, 'o-')
        plt.xlabel('Epoch')
        plt.ylabel('NLL')
        plt.pause(0.1)

plt.savefig('NLL_vs_epoch_T%s.pdf' %(str(T)))
