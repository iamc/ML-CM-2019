########## Machine Learning in Condensed Matter Physics ###############################
###                     Roger Melko
###   with dataset and code by 
####  Giacomo Torlai, Juan Carrasquilla, Estelle Inack and Lauren Hayward Sierens
###
### This code defines a Restricted Boltzmann Machine (RBM) object.
#####################################################################################


import tensorflow as tf
import itertools as it
import numpy as np

class RBM(object):
  
  ### Constructor ###
  def __init__(self, num_hidden, num_visible, num_samples=128, weights=None, visible_bias=None, hidden_bias=None):
    self.num_hidden = num_hidden   #number of hidden units
    self.num_visible = num_visible #number of visible units

    #visible bias:
    default = tf.zeros(shape=(self.num_visible, 1))
    self.visible_bias = self._create_parameter_variable(visible_bias, default)
    
    #hidden bias:
    default = tf.zeros(shape=(self.num_hidden, 1))
    self.hidden_bias = self._create_parameter_variable(hidden_bias, default)
    
    #pairwise weights:
    default = tf.random_normal(shape=(self.num_visible, self.num_hidden), mean=0, stddev=0.05)
    self.weights = self._create_parameter_variable(weights, default)

    #variables for sampling (num_samples is the number of samples to return):
    self.hidden_samples = tf.Variable(
      self.sample_binary_tensor(tf.constant(0.5), num_samples, self.num_hidden),
      trainable=False, name='hidden_samples'
    )
  
    self._all_hidden_states = None
    self.max_feasible_for_log_pf = 24
  #end of constructor
  
  ### Build array with all possible configuration of the hidden layer: ###
  @property
  def all_hidden_states(self):
    if self._all_hidden_states is None:
        assert self.num_hidden <= self.max_feasible_for_log_pf, \
            'cannot generate all hidden states for num_hidden > {}'.format(self.max_feasible_for_log_pf)
        self._all_hidden_states = np.array(list(it.product([0, 1], repeat=self.num_hidden)), dtype=np.float32)
    return self._all_hidden_states

  ### Method to initialize variables: ###
  @staticmethod
  def _create_parameter_variable(initial_value=None, default=None):
    if initial_value is None:
      initial_value = default
    return tf.Variable(initial_value)

  ### Method to calculate the conditional probability of the hidden layer given a visible state: ###
  def p_of_h_given(self, v):
    # type: (tf.Tensor) -> tf.Tensor
    return tf.nn.sigmoid(tf.matmul(v, self.weights) + tf.transpose(self.hidden_bias))

  ### Method to calculate the conditional probability of the visible layer given a hidden state: ###
  def p_of_v_given(self, h):
    # type: (tf.Tensor) -> tf.Tensor
    return tf.nn.sigmoid(tf.matmul(h, self.weights, transpose_b=True) + tf.transpose(self.visible_bias))

  ### Method to sample the hidden nodes given a visible state: ###
  def sample_h_given(self, v):
    # type: (tf.Tensor) -> (tf.Tensor, tf.Tensor)
    b = tf.shape(v)[0]  # number of samples
    m = self.num_hidden
    prob_h = self.p_of_h_given(v)
    samples = self.sample_binary_tensor(prob_h, b, m)
    return samples

  ### Method to sample the visible nodes given a hidden state: ###
  def sample_v_given(self, h):
    # type: (tf.Tensor) -> (tf.Tensor, tf.Tensor)
    b = tf.shape(h)[0]  # number rof samples
    n = self.num_visible
    prob_v = self.p_of_v_given(h)
    samples = self.sample_binary_tensor(prob_v, b, n)
    return samples

  ###
  # Method for persistent contrastive divergence (CD_k):
  # Stores the results of `num_iterations` of contrastive divergence in class variables.
  #
  # :param int num_iterations: The 'k' in CD_k.
  ###
  def stochastic_maximum_likelihood(self, num_iterations):
    # type: (int) -> (tf.Tensor, tf.Tensor, tf.Tensor)
    h_samples = self.hidden_samples
    v_samples = None
    for i in range(num_iterations):
      v_samples = self.sample_v_given(h_samples)
      h_samples = self.sample_h_given(v_samples)

    self.hidden_samples = self.hidden_samples.assign(h_samples)
    return self.hidden_samples, v_samples
 
  ###
  # Method to compute the energy E = - aT*v - bT*h - vT*W*h
  # Note that since we want to support larger batch sizes, we do element-wise multiplication between
  # vT*W and h, and sum along the columns to get a Tensor of shape batch_size by 1
  #
  # :param hidden_samples:  Tensor of shape batch_size by num_hidden
  # :param visible_samples: Tensor of shape batch_size by num_visible
  ###
  def energy(self, hidden_samples, visible_samples):
      # type: (tf.Tensor, tf.Tensor) -> tf.Tensor
      return (-tf.matmul(hidden_samples, self.hidden_bias)  # b x m * m x 1
              - tf.matmul(visible_samples, self.visible_bias)  # b x n * n x 1
              - tf.reduce_sum(tf.matmul(visible_samples, self.weights) * hidden_samples, 1))
  
  ### Method used to calculate the gradient of the negative log-likelihood ###
  def neg_log_likelihood_forGrad(self, visible_samples, num_gibbs=2):
    # type: (tf.Tensor, tf.Tensor, int) -> tf.Tensor
    hidden_samples = self.sample_h_given(visible_samples)
    expectation_from_data = tf.reduce_mean(self.energy(hidden_samples, visible_samples))
    
    model_hidden, model_visible = self.stochastic_maximum_likelihood(num_gibbs)
    expectation_from_model = tf.reduce_mean(self.energy(model_hidden, model_visible))

    return expectation_from_data - expectation_from_model
  
  ###
  # Method to ompute the average negative log likelihood over a batch of visible samples:
  # NLL = - <log(p(v))> = - <F> + log(Z)
  ###
  def neg_log_likelihood(self, visible_samples, log_Z):
        free_energy = (tf.matmul(visible_samples, self.visible_bias)
                       + tf.reduce_sum(tf.nn.softplus(tf.matmul(visible_samples, self.weights)
                                                      + tf.transpose(self.hidden_bias)), 1))
        return -tf.reduce_mean(free_energy - log_Z)

  ### Method to evaluate the partition function by exact enumerations: ###
  ### (Intractable for large sizes) ###
  def exact_log_partition_function(self):
        with tf.name_scope('exact_log_Z'):
            # Define the exponent: H*b + sum(softplus(1 + exp(a + w*H.T)))
            first_term = tf.matmul(self.all_hidden_states, self.hidden_bias, name='first_term')
            with tf.name_scope('second_term'):
                second_term = tf.matmul(self.weights, self.all_hidden_states, transpose_b=True)
                second_term = tf.nn.softplus(tf.add(self.visible_bias, second_term))
                second_term = tf.transpose(tf.reduce_sum(second_term, reduction_indices=[0], keepdims=True))
            exponent = tf.cast(first_term + second_term, dtype=tf.float64, name='exponent')
            #exponent_mean = tf.reduce_mean(exponent)
            exponent_mean = tf.reduce_max(exponent)

            return tf.log(tf.reduce_sum(tf.exp(exponent - exponent_mean))) + exponent_mean

  ###
  # Convenience method for generating a binary Tensor using a given probability
  #
  # :param prob: Tensor of shape (m, n)
  # :param m: number of rows in result.
  # :param n: number of columns in result.
  ###
  @staticmethod
  def sample_binary_tensor(prob, m, n):
    # type: (tf.Tensor, int, int) -> tf.Tensor
    return tf.where(
      tf.less(tf.random_uniform(shape=(m, n)), prob),
      tf.ones(shape=(m, n)),
      tf.zeros(shape=(m, n))
    )

#end of RBM class
