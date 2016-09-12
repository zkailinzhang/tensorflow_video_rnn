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

def BiRNN(x, weights, biases):
  """Bidrection recurrent neural network"""
  # Prepare data shape to match `bidirectional_rnn` function requirements
  # Current data input shape: (n_step, n_row, n_column, n_channel)
  # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
  # in this case (batch_size, n_input) is equal to (n_row*n_column, n_channel)

  # Reshape to (n_step, n_row*n_column, n_channel), and get the step number
  x = tf.reshape(x, [-1, 299*299, 3])
  n_step = tf.shape(x)[0]
  # Reshape to (n_step*n_row*n_column, n_channel)
  x = tf.reshape(x, [-1, 3])
  # Split to get a list of 'n_step' tensors of shape 
  # (n_row*n_column, n_channel)
  x = tf.split(0, n_step, x)

  # Define lstm cells with tensorflow
  # Forward direction cell
  lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0)
  # Backward direction cell
  lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(n_hidden, forget_bias=1.0)

  # Get lstm cell output
  try:
    outputs, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                          dtype=tf.float32)
  except Exception: # Old TensorFlow version only returns outputs not states
    outputs = tf.nn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                    dtype=tf.float32)

  # Linear activation, using rnn inner loop last output
  return tf.matmul(outputs[-1], weights['out']) + biases['out']

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

  pred = BiRNN(x, weights, biases)

  # Define loss and optimizer
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

  # Evaluate model
  correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  # Initializing the variables
  init = tf.initialize_all_variables()

  # Launch the graph
  with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
      batch_x, batch_y = mnist.train.next_batch(batch_size)
      # Reshape data to get 28 seq of 28 elements
      batch_x = batch_x.reshape((batch_size, n_steps, n_input))
      # Run optimization op (backprop)
      sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
      if step % display_step == 0:
        # Calculate batch accuracy
        acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
        # Calculate batch loss
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
          "{:.6f}".format(loss) + ", Training Accuracy= " + \
          "{:.5f}".format(acc))
      step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
      sess.run(accuracy, feed_dict={x: test_data, y: test_label}))