import tensorflow as tf
import numpy as np

import image_processing

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
  'train_dir', '/Users/dgu/Desktop/lstm_train',
  "Directory where to write event logs and checkpoint."
)
tf.app.flags.DEFINE_string(
  'model', 'small',
  "Model configuration. Possible options are: small, medium, large."
)

class SmallConfig(object):
  """Small config."""
  # Parameters
  learning_rate = 0.1
  training_iters = 100000
  batch_size = 20
  display_step = 10
  # Network parameters
  num_input = 28
  num_steps = 28
  num_hidden = 200 # hidden layer number of features
  num_classes = 24 # total action class

def get_config():
  if FLAGS.model == "small":
    return SmallConfig()
  else
    raise ValueError("Invalid model: %s", FLAGS.model)

def train(dataset):
  # get the configuration settings
  config = get_config()

  # tf Graph inputs
  x = tf.placeholder("float", [None, config.num_steps, config.num_input])
  y = tf.placeholder("float", [None, n_classes])

  # Define weights
  weights = {
    'out': tf.Variable(tf.random_normal([2*config.num_hidden, 
                                        config.num_classes]))
  }
  biases = {
    'out': tf.Variable(tf.random_normal([config.num_classes]))
  }

def BiRNN(x, weights, biases):
  """Bidrection recurrent neural network"""
  # Prepare data shape to match `bidirectional_rnn` function requirements
  # Current data input shape: (batch_size, n_steps, n_input)
  # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
  
  # Permuting batch_size and n_steps
  x = tf.transpose(x, [1, 0, 2])
  # Reshape to (n_steps*batch_size, n_input)
  x = tf.reshape(x, [-1, n_input])
  # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
  x = tf.split(0, n_steps, x)

  # Define lstm cells with tensorflow
  # Forward direction cell
  lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0)
  # Backward direction cell
  lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0)

  # Get lstm cell output
  try:
    outputs, _, _ = tf.nn.rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                          dtype=tf.float32)
  except Exception: # Old TensorFlow version only returns outputs not states
    outputs = tf.nn.rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                    dtype=tf.float32)

  # Linear activation, using rnn inner loop last output
  return tf.matmul(outputs[-1], weights['out']) + biases['out']
