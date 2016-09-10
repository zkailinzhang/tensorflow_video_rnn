import copy
import matplotlib.pyplot as plt
import tensorflow as tf
from lca_data import LCAData
from image_processing import inputs


FLAGS = tf.app.flags.FLAGS

def main(_):
  with tf.Graph().as_default(), tf.device('/cpu:0'), tf.Session() as sess:
    # coordinator for controlling queue threads
    coord = tf.train.Coordinator()
    # Build an initialization op eration to run below.
    init = tf.initialize_all_variables()
    sess.run(init)
    # print out the images
    train_dataset = LCAData('train')
    images_op, labels_op, filename_op = inputs(train_dataset)
    # merge all the summary and then write then out to the writer
    summary_op = tf.merge_all_summaries()
    # writer for tensorboard visualization
    writer = tf.train.SummaryWriter(
      "summary/train",
      #graph=sess.graph.as_graph_def(add_shapes=True))
      graph=sess.graph)
    # start all the queue thread
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # run the ops for two iteration
    for _ in xrange(10):
      summary_result, images, labels, filenames = sess.run([summary_op, 
                                                            images_op, 
                                                            labels_op, 
                                                            filename_op])
      print(filenames)
      # write the summary result to the writer
      writer.add_summary(summary_result)
    # request to stop the input queue
    coord.request_stop()
    # Wait for threads to finish.
    coord.join(threads)

    # close the writer
    writer.close()

if __name__ == '__main__':
  tf.app.run()
