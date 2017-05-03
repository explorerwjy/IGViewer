from datetime import datetime
import math
import time
import os
import numpy as np
import tensorflow as tf
from Input import *
import Models


GPUs = [5]
available_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([ available_devices[x] for x in GPUs])

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', './test',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 320,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

EPOCHS = 1000000

def eval_once(saver, summary_writer, labels, logits, top_k_op, summary_op):
    """Run Eval once.
    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """

    global_step = tf.Variable(0, trainable=False, name='global_step')
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            print "Check Point is:",ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path)
            #print (sess.run(global_step))
        else:
            print('No checkpoint file found')
            return
        # Start the queue runners.
        sess.run(init)
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            print ckpt.model_checkpoint_path
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "CKPT starts with step",(sess.run(global_step))
            while step < num_iter and not coord.should_stop():
                print (sess.run(global_step))
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1

            # Compute precision @ 1.
            print "Predicted Right:{}\t\tTotal:{}".format(true_count, total_sample_count)
            precision = float(true_count) / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate_2(DataFile):
    InputData = INPUT(DataFile)
    with tf.Graph().as_default() as g:
                # Get images and labels for CIFAR-10.

        images, labels = InputData.PipeLine(FLAGS.batch_size)
        # Build a Graph that computes the logits predictions from the
        # inference model.
        model = Models.ConvNets()
        logits = model.Inference(images)

        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        # Restore the moving average version of the learned variables for eval.
        #variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        #variables_to_restore = variable_averages.variables_to_restore()
        #saver = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, labels, logits, top_k_op, summary_op)
            break
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)

class Evaluate():
    def __init__(self, batch_size, epochs, model, DataFile):
        self.batch_size = batch_size
        self.epochs = epochs
        self.InputData = INPUT(self.DataFile)
        self.model = model

    def run(self):
        with tf.Graph().as_default():
            global_step = tf.Variable(0, trainable=False, name='global_step')
            images, labels = self.InputData.PipeLine(self.batch_size, self.epochs)
            logits = self.model.Inference(images)
            loss = self.model.loss(logits, labels)
            top_k_op = tf.nn.in_top_k(logits, labels, 1)
            summary_op = tf.summary.merge_all()
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=FLAGS.log_device_placement))
            
            saver.restore(sess, self.getCheckPoint)
            v_step = sess.run(global_step)
            print sess.run(global_step)
            print "Start with step", v_step
            sess.run(init)
            # Start the queue runners.
            tf.train.start_queue_runners(sess=sess)
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
            coord = tf.train.Coordinator()
            try:    
                num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
                true_count = 0  # Counts the number of correct predictions.
                total_sample_count = num_iter * FLAGS.batch_size
                step = 0
                print ckpt.model_checkpoint_path
                saver.restore(sess, self.getCheckPoint)
                print "CKPT starts with step",(sess.run(global_step))
                while step < num_iter and not coord.should_stop():
                    print (sess.run(global_step))
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    step += 1

                # Compute precision @ 1.
                print "Predicted Right:{}\t\tTotal:{}".format(true_count, total_sample_count)
                precision = float(true_count) / total_sample_count
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Precision @ 1', simple_value=precision)
                summary_writer.add_summary(summary, step)
            except Exception, e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join()

    def getCheckPoint(self):
        ckptfile = FLAGS.checkpoint_dir + '/checkpoint'
        f = open(ckptfile, 'rb')
        ckpt = f.readline().split(':')[1].strip().strip('"')
        f.close()
        prefix = os.path.abspath(FLAGS.train_dir)
        ckpt = prefix + '/' + ckpt
        return ckpt



def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    TrainingDataFile = "/home/yufengshen/IGViewer/Data/TrainingData.txt"
    TestingDataFile = "/home/yufengshen/IGViewer/Data/TrainingData.txt"
    model = Models.ConvNets()
    evaluate = Evaluate(FLAGS.batch_size, EPOCHS, model, TrainingDataFile)
    evaluate.run()


if __name__ == '__main__':
    tf.app.run()
