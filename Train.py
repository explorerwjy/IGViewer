#!/home/yufengshen/anaconda2/bin/python
# Author: jywang explorerwjy@gmail.com

#=========================================================================
# Training The ConvNet for View IGV
#=========================================================================

import argparse
from datetime import datetime
import time
import os
from threading import Thread
import numpy as np
import tensorflow as tf
import Models
from Input import *
import sys
sys.stdout = sys.stderr

GPUs = [0,1]
available_devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([ available_devices[x] for x in GPUs])

EPOCHS = 1000000

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './train_3',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

BATCH_SIZE = FLAGS.batch_size

class Train():
    def __init__(self, batch_size, epochs, model, TrainingDataFile, TestingDataFile):
        self.TrainingDataFile = TrainingDataFile
        self.TestingDataFile = TestingDataFile
        self.batch_size = batch_size
        self.epochs = epochs
        self.InputData = INPUT(self.TrainingDataFile)
        self.model = model

    def tower_loss(self, scope):
        """Calculate the total loss on a single tower running the CIFAR model.
        Args:
        scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
        Returns:
        Tensor of shape [] containing the total loss for a batch of data
        """
        # Get images and labels for CIFAR-10.
        images, labels = self.InputData.PipeLine(self.batch_size, self.epochs)

        # Build inference Graph.
        logits = self.model.inference(images)

        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.
        _ = self.model.loss(logits, labels)

        # Assemble all of the losses for the current tower only.
        losses = tf.get_collection('losses', scope)

        # Calculate the total loss for the current tower.
        total_loss = tf.add_n(losses, name='total_loss')

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
            # session. This helps the clarity of presentation on tensorboard.
            loss_name = re.sub('%s_[0-9]*/' % Models.TOWER_NAME, '', l.op.name)
            tf.summary.scalar(loss_name, l)

        return total_loss

    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    # Training Model on multiple GPU
    def run_multiGPU(self, continueModel=False):
        with tf.Graph().as_default(), tf.device('/cpu:0'):

            global_step = tf.Variable(0, trainable=False, name='global_step')

            lr = tf.constant(1e-2)
            # Create an optimizer that performs gradient descent.
            opt = tf.train.RMSPropOptimizer(lr)
            # Calculate the gradients for each model tower.
            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in xrange(FLAGS.num_gpus):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
                            # Calculate the loss for one tower of the model. This function
                            # constructs the entire model but shares the variables across
                            # all towers.
                            loss = self.tower_loss(scope)
                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()
                            # Retain the summaries from the final tower.
                            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                            # Calculate the gradients for the batch of data on this CIFAR tower.
                            grads = opt.compute_gradients(loss)
                            # Keep track of the gradients across all towers.
                            tower_grads.append(grads)
            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grads = self.average_gradients(tower_grads)
                
            # Add a summary to track the learning rate.
            summaries.append(tf.summary.scalar('learning_rate', lr))

            # Add histograms for gradients.
            for grad, var in grads:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
            # Apply the gradients to adjust the shared variables.
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))

            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(
                        cifar10.MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())

            # Group all updates to into a single train op.
            train_op = tf.group(apply_gradient_op, variables_averages_op)

            # Create a saver.
            saver = tf.train.Saver(tf.global_variables())

            # Build the summary operation from the last tower summaries.
            summary_op = tf.summary.merge(summaries)

            # Build an initialization operation to run below.
            init = tf.global_variables_initializer()

            # Start running operations on the Graph. allow_soft_placement must be set to
            # True to build towers on GPU, as some of the ops do not have GPU
            # implementations.
            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=FLAGS.log_device_placement))
            
            # Continue to train from a checkpoint
            if continueModel != None:
                saver.restore(sess, continueModel)

            sess.run(init)
            # Start the queue runners.
            tf.train.start_queue_runners(sess=sess)
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

            min_loss = 100
            for step in xrange(FLAGS.max_steps):
                start_time = time.time()
                _, loss_value, v_step = sess.run([train_op, loss, global_step])
                duration = time.time() - start_time

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if v_step % 10 == 0:
                    num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / FLAGS.num_gpus
                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
                    print (format_str % (datetime.now(), v_step, loss_value,
                             examples_per_sec, sec_per_batch))
                
                if v_step % 100 == 0:
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, v_step)

                # Save the model checkpoint periodically.
                if v_step % 1000 == 0 or (v_step + 1) == FLAGS.max_steps:
                    #self.EvalWhileTraining()
                    if loss_value < min_loss:
                        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=v_step)
                        min_loss = loss_value
                        print "Write A CheckPoint at %d" % (v_step)
    
    def run(self, continueModel=None):
        with tf.Graph().as_default():
            global_step = tf.Variable(0, trainable=False, name='global_step')
            images, labels = self.InputData.PipeLine(self.batch_size, self.epochs)
            logits = self.model.Inference(images)
            loss = self.model.loss(logits, labels)
            train_op = self.model.train(loss, global_step)
            summary_op = tf.summary.merge_all()
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            sess = tf.Session(config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=FLAGS.log_device_placement))
            
            # Continue to train from a checkpoint
            if continueModel != None:
                saver.restore(sess, continueModel)
            v_step = sess.run(global_step)
            print sess.run(global_step)
            print "Start with step", v_step
            sess.run(init)
            # Start the queue runners.
            tf.train.start_queue_runners(sess=sess)
            summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
            coord = tf.train.Coordinator()
            min_loss = 500
            try:    
                
                saver.restore(sess, continueModel)
                for step in xrange(FLAGS.max_steps):
                    print 'GlobalStep',sess.run(global_step)
                    if coord.should_stop():
                        break
                    start_time = time.time()
                    _, loss_value, v_step = sess.run([train_op, loss, global_step])
                    duration = time.time() - start_time

                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                    if v_step % 10 == 0:
                        num_examples_per_step = FLAGS.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = duration / FLAGS.num_gpus
                        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
                        print (format_str % (datetime.now(), v_step, loss_value,
                                 examples_per_sec, sec_per_batch))
                    
                    if v_step % 100 == 0:
                        summary_str = sess.run(summary_op)
                        summary_writer.add_summary(summary_str, v_step)

                    # Save the model checkpoint periodically.
                    if v_step % 1000 == 0 or (v_step + 1) == FLAGS.max_steps:
                        #self.EvalWhileTraining()
                        if loss_value < min_loss:
                            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                            saver.save(sess, checkpoint_path, global_step=v_step)
                            min_loss = loss_value
                            print "Write A CheckPoint at %d" % (v_step)
            except Exception, e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join()

    def getCheckPoint(self):
        ckptfile = FLAGS.train_dir + '/checkpoint'
        f = open(ckptfile, 'rb')
        ckpt = f.readline().split(':')[1].strip().strip('"')
        f.close()
        prefix = os.path.abspath(FLAGS.train_dir)
        ckpt = prefix + '/' + ckpt
        return ckpt

def GetOptions():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--Continue", action='store_true', default=False,
        help="continue training from a checkpoint")
    args = parser.parse_args()
    return args.Continue

def main(argv=None):  # pylint: disable=unused-argument
    Continue = GetOptions()
    model = Models.ConvNets()
    TrainingDataFile = "/home/yufengshen/IGViewer/Data/TrainingData.txt"
    TestingDataFile = "/home/yufengshen/IGViewer/Data/TrainingData.txt"
    train = Train(BATCH_SIZE, EPOCHS, model, TrainingDataFile, TestingDataFile)
    if Continue:
        ckpt = train.getCheckPoint()
        print "Train From a Check Point:", ckpt
        train.run(continueModel=ckpt)
    else:
        train.run()

if __name__ == '__main__':
    tf.app.run()
