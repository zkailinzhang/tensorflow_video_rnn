import tensorflow as tf

from lca_data import LCAData
from image_processing import inputs

FLAGS = tf.app.flags.FLAGS

def main(_):
  with tf.device('/cpu:0'), tf.Session() as sess:
    coord = tf.train.Coordinator()

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()
    sess.run(init)

    try:
      # print out the images
      dataset = LCAData('train')
      images, labels = inputs(dataset)
      # merge all the summary and then write then out to the 
      summary_op = tf.merge_all_summaries()
      writer = tf.train.SummaryWriter("summary" + '/train',
                                      sess.graph)
      # start all the queue thread
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      # run the ops
      image, label = sess.run([images, labels])
      print(image)
    except Exception as e:
      print(e)
    finally:
      coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)

if __name__ == '__main__':
  tf.app.run()
