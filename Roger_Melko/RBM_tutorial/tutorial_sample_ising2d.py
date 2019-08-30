########## Machine Learning in Condensed Matter Physics ###############################
###                     Roger Melko
###   with dataset and code by 
####  Giacomo Torlai, Juan Carrasquilla, Estelle Inack and Lauren Hayward Sierens
###
### This code will sample from a Restricted Boltzmann Machine (RBM) and interpret the
### resulting samples as spin configurations corresponding to the two-dimensional
### Ising model. It will then save to file each sample's energy, magnetization
### specific heat and susceptibility.
#####################################################################################


from __future__ import print_function
import tensorflow as tf
from rbm import RBM
import numpy as np
import os

#Input parameters:
L           = 4    #linear size of the system
num_visible = L*L  #number of visible nodes
num_hidden  = 4    #number of hidden nodes

#Temperature list for which there are trained RBM parameters stored in Data_ising2d/RBM_parameters
#T_list = [1.0,1.254,1.508,1.762,2.016,2.269,2.524,2.778,3.032,3.286,3.540]
T_list = [2.269]

#Sampling parameters:
num_samples  = 500  # how many independent chains will be sampled
gibb_updates = 2    # how many gibbs updates per call to the gibbs sampler
nbins        = 100  # number of calls to the RBM sampler

#Specify where the sampled configurations will be stored:
samples_dir = 'Data_ising2d/RBM_samples'
if not(os.path.isdir(samples_dir)):
  os.mkdir(samples_dir)
samples_filePaths = [] #file paths where samples for each T will be stored

#Initialize the RBM for each temperature in T_list:
rbms           = []
rbm_samples    = []
for i in range(len(T_list)):
  T = T_list[i]
  
  samples_filePath =  '%s/samples_nH%d_L%d' %(samples_dir,num_hidden,L)
  samples_filePath += '_T' + str(T) + '.txt'
  samples_filePaths.append(samples_filePath)
  fout = open(samples_filePath,'w')
  fout.close()
  
  #Read in the trained RBM parameters:
  path_to_params =  'Data_ising2d/RBM_parameters/parameters_nH%d_L%d' %(num_hidden,L)
  path_to_params += '_T'+str(T)+'.npz'
  params         =  np.load(path_to_params)
  weights        =  params['weights']
  visible_bias   =  params['visible_bias']
  hidden_bias    =  params['hidden_bias']
  hidden_bias    =  np.reshape(hidden_bias,(hidden_bias.shape[0],1))
  visible_bias   =  np.reshape(visible_bias,(visible_bias.shape[0],1))
  
  # Initialize RBM class
  rbms.append(RBM(
    num_hidden=num_hidden, num_visible=num_visible,
    weights=weights, visible_bias=visible_bias,hidden_bias=hidden_bias,
    num_samples=num_samples
  ))
  rbm_samples.append(rbms[i].stochastic_maximum_likelihood(gibb_updates))
#end of loop over temperatures

# Initialize tensorflow
init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

# Sample thermodynamic observables:
N = num_visible
with tf.Session() as sess:
  sess.run(init)
  
  for i in range(nbins):
    print ('bin %d' %i)

    for t in range(len(T_list)):
      fout = open(samples_filePaths[t],'a')
      
      _,samples=sess.run(rbm_samples[t])
      spins = np.asarray((2*samples-1)) #convert from 0,1 variables to -1,+1 variables
      for k in range(num_samples):
        for i in range(N):
         fout.write('%d ' %int(spins[k,i]))
        fout.write('\n')
      fout.close()
