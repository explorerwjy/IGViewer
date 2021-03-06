#!/home/local/users/jw/anaconda2/bin/python
# Author: jywang	explorerwjy@gmail.com

#=========================================================================
# Models would like to try on Tensor Variant Caller
#=========================================================================

from optparse import OptionParser
import tensorflow as tf
#from Window2Tensor import *
import re
from Input import *

TOWER_NAME = 'tower'
MOVING_AVERAGE_DECAY = 0.9999


class ConvNets():
    def __init__(self):
        pass

    def Inference(self, RawTensor):
        print RawTensor
        InputTensor = tf.reshape(RawTensor, [-1, WIDTH, HEIGHT, 3])
        print InputTensor
        # ==========================================================================================
        # conv1 3-64
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 3, 64], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                InputTensor, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv1)
            print conv1
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
        # ==========================================================================================
        # ==========================================================================================
        # conv3 3-128
        with tf.variable_scope('conv3') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 64, 64], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                pool1, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv3)
            print conv3
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool2 = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
        # ==========================================================================================
        # ==========================================================================================
        # conv5 3-256
        with tf.variable_scope('conv5') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                pool2, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [128], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv5 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv5)
            print conv5
        # ==========================================================================================
        # ==========================================================================================
        # conv6 3-256
        with tf.variable_scope('conv6') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv5, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [128], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv6 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv6)
            print conv6
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool3 = tf.nn.max_pool(conv6, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool3')
        # ==========================================================================================
        # ==========================================================================================
        # conv9 3-512
        with tf.variable_scope('conv9') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 128, 256], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                pool3, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv9 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv9)
            print conv9
        # ==========================================================================================
        # ==========================================================================================
        # conv10 3-512
        with tf.variable_scope('conv10') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 256, 256], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv9, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv10 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv10)
            print conv10
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool4 = tf.nn.max_pool(conv10, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool4')
        # ==========================================================================================
        # ==========================================================================================
        # conv13 3-512
        with tf.variable_scope('conv13') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 256, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                pool4, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv13 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv13)
            print conv13
        # ==========================================================================================
        # ==========================================================================================
        # conv14 3-512
        with tf.variable_scope('conv14') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv13, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv14 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv14)
            print conv14
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool5 = tf.nn.max_pool(conv14, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool5')
        # ==========================================================================================
        # ==========================================================================================
        # local1
        with tf.variable_scope('local1') as scope:
            reshape = tf.reshape(pool5, [FLAGS.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = _variable_with_weight_decay(
                'weights', shape=[dim, 4096], stddev=0.04, wd=0.004)
            biases = _variable_on_cpu(
                'biases', [4096], tf.constant_initializer(0.1))
            local1 = tf.nn.relu(
                tf.matmul(
                    reshape,
                    weights) + biases,
                name=scope.name)
            _activation_summary(local1)
            #local1_drop = tf.nn.dropout(local1, 0.9)
            #_activation_summary(local6_drop)
        print local1
        # ==========================================================================================
        # ==========================================================================================
        # local2
        with tf.variable_scope('local2') as scope:
            weights = _variable_with_weight_decay(
                'weights', shape=[4096, 4096], stddev=0.04, wd=0.004)
            biases = _variable_on_cpu(
                'biases', [4096], tf.constant_initializer(0.1))
            local2 = tf.nn.relu(
                tf.matmul(
                    local1,
                    weights) + biases,
                name=scope.name)
            #local7_drop = tf.nn.dropout(local2, 0.9)
            _activation_summary(local2)
        print local2
        # ==========================================================================================
        # ==========================================================================================
        # local3
        with tf.variable_scope('local3') as scope:
            weights = _variable_with_weight_decay(
                'weights', shape=[4096, 1000], stddev=0.04, wd=0.004)
            biases = _variable_on_cpu(
                'biases', [1000], tf.constant_initializer(0.1))
            local3 = tf.nn.relu(
                tf.matmul(
                    local2,
                    weights) + biases,
                name=scope.name)
            #local7_drop = tf.nn.dropout(local3, 0.9)
            _activation_summary(local3)
        print local3
        # ==========================================================================================
        # ==========================================================================================
        # linear layer (WX + b)
        with tf.variable_scope('softmax') as scope:
            weights = _variable_with_weight_decay(
                'weights', [1000, NUM_CLASSES], stddev=1 / 1000.0, wd=0.0)
            biases = _variable_on_cpu(
                'biases', [NUM_CLASSES], tf.constant_initializer(0.0))
            softmax_linear = tf.add(
                tf.matmul(
                    local3,
                    weights),
                biases,
                name=scope.name)
            _activation_summary(softmax_linear)
            #softmax = tf.nn.softmax(softmax_linear, dim=-1, name=None)
        print softmax_linear
        # ==========================================================================================
        return softmax_linear

    def Inference_2(self, RawTensor):
        print RawTensor
        InputTensor = tf.reshape(RawTensor, [-1, WIDTH, HEIGHT + 1, 3])
        print InputTensor
        # ==========================================================================================
        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 3, 64], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                InputTensor, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv1)
            print conv1
        # ==========================================================================================
        # ==========================================================================================
        # conv2
        with tf.variable_scope('conv2') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 64, 64], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv1, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [64], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv2)
            print conv2
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool1 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool1')
        # ==========================================================================================
        # ==========================================================================================
        # conv3
        with tf.variable_scope('conv3') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                pool1, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [128], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv3)
            print conv3
        # ==========================================================================================
        # ==========================================================================================
        # conv4
        with tf.variable_scope('conv4') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv3, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [128], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv4)
            print conv4
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool2 = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool2')
        # ==========================================================================================
        # ==========================================================================================
        # conv5
        with tf.variable_scope('conv5') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 128, 256], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                pool2, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv5 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv5)
            print conv5
        # ==========================================================================================
        # ==========================================================================================
        # conv6
        with tf.variable_scope('conv6') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 256, 256], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv5, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv6 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv6)
            print conv6
        # ==========================================================================================
        # ==========================================================================================
        # conv7
        with tf.variable_scope('conv7') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 256, 256], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv6, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv7 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv7)
            print conv7
        # ==========================================================================================
        # ==========================================================================================
        # conv8
        with tf.variable_scope('conv8') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 256, 256], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv7, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [256], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv8 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv8)
            print conv8
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool3 = tf.nn.max_pool(conv8, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool3')
        # ==========================================================================================
        # ==========================================================================================
        # conv9
        with tf.variable_scope('conv9') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 256, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                pool3, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv9 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv9)
            print conv9
        # ==========================================================================================
        # ==========================================================================================
        # conv10
        with tf.variable_scope('conv10') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv9, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv10 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv10)
            print conv10
        # ==========================================================================================
        # ==========================================================================================
        # conv11
        with tf.variable_scope('conv11') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv10, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv11 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv11)
            print conv11
        # ==========================================================================================
        # ==========================================================================================
        # conv12
        with tf.variable_scope('conv12') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv11, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv12 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv12)
            print conv12
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool4 = tf.nn.max_pool(conv12, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool4')
        # ==========================================================================================
        # ==========================================================================================
        # conv13
        with tf.variable_scope('conv13') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                pool4, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv13 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv13)
            print conv13
        # ==========================================================================================
        # ==========================================================================================
        # conv14
        with tf.variable_scope('conv14') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv13, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv14 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv14)
            print conv14
        # ==========================================================================================
        # ==========================================================================================
        # conv15
        with tf.variable_scope('conv15') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv14, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv15 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv15)
            print conv15
        # ==========================================================================================
        # ==========================================================================================
        # conv16
        with tf.variable_scope('conv16') as scope:
            kernel = _variable_with_weight_decay(
                'weights', shape=[3, 3, 512, 512], stddev=5e-2, wd=0.0)
            conv = tf.nn.conv2d(
                conv15, kernel, [
                    1, 1, 1, 1], padding='SAME')
            biases = _variable_on_cpu(
                'biases', [512], tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv16 = tf.nn.relu(pre_activation, name=scope.name)
            _activation_summary(conv16)
            print conv16
        # ==========================================================================================
        # ==========================================================================================
        # MaxPooling
        pool5 = tf.nn.max_pool(conv16, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='pool5')
        # ==========================================================================================
        # ==========================================================================================
        # local1
        with tf.variable_scope('local1') as scope:
            reshape = tf.reshape(pool5, [FLAGS.batch_size, -1])
            dim = reshape.get_shape()[1].value
            weights = _variable_with_weight_decay(
                'weights', shape=[dim, 4096], stddev=0.04, wd=0.004)
            biases = _variable_on_cpu(
                'biases', [4096], tf.constant_initializer(0.1))
            local1 = tf.nn.relu(
                tf.matmul(
                    reshape,
                    weights) + biases,
                name=scope.name)
            _activation_summary(local1)
            #local1_drop = tf.nn.dropout(local1, 0.9)
            #_activation_summary(local6_drop)
        print local1
        # ==========================================================================================
        # ==========================================================================================
        # local2
        with tf.variable_scope('local2') as scope:
            weights = _variable_with_weight_decay(
                'weights', shape=[4096, 4096], stddev=0.04, wd=0.004)
            biases = _variable_on_cpu(
                'biases', [4096], tf.constant_initializer(0.1))
            local2 = tf.nn.relu(
                tf.matmul(
                    local1,
                    weights) + biases,
                name=scope.name)
            #local7_drop = tf.nn.dropout(local2, 0.9)
            _activation_summary(local2)
        print local2
        # ==========================================================================================
        # ==========================================================================================
        # local3
        with tf.variable_scope('local3') as scope:
            weights = _variable_with_weight_decay(
                'weights', shape=[4096, 1000], stddev=0.04, wd=0.004)
            biases = _variable_on_cpu(
                'biases', [1000], tf.constant_initializer(0.1))
            local3 = tf.nn.relu(
                tf.matmul(
                    local2,
                    weights) + biases,
                name=scope.name)
            #local7_drop = tf.nn.dropout(local3, 0.9)
            _activation_summary(local3)
        print local3
        # ==========================================================================================
        # ==========================================================================================
        # linear layer (WX + b)
        with tf.variable_scope('softmax') as scope:
            weights = _variable_with_weight_decay(
                'weights', [1000, NUM_CLASSES], stddev=1 / 1000.0, wd=0.0)
            biases = _variable_on_cpu(
                'biases', [NUM_CLASSES], tf.constant_initializer(0.0))
            softmax_linear = tf.add(
                tf.matmul(
                    local3,
                    weights),
                biases,
                name=scope.name)
            _activation_summary(softmax_linear)
            #softmax = tf.nn.softmax(softmax_linear, dim=-1, name=None)
        print softmax_linear
        # ==========================================================================================
        return softmax_linear

    def loss(self, logits, labels):
        #labels = tf.cast(labels, tf.int64)
        print 'logits', logits
        print 'labels', labels
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def add_loss_summaries(self, total_loss):
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        for l in losses + [total_loss]:
            tf.summary.scalar(l.op.name + ' (raw) ', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))
        return loss_averages_op

    def train(self, total_loss, global_step):
        lr = tf.constant(1e-3)
        tf.summary.scalar('learning_rate', lr)
        loss_averages_op = self.add_loss_summaries(total_loss)

        with tf.control_dependencies([loss_averages_op]):
            #opt = tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.8, epsilon=1e-10, centered=False)
            opt = tf.train.AdamOptimizer(lr)
            grads = opt.compute_gradients(total_loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variable_averages_op = variable_averages.apply(
            tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
            train_op = tf.no_op(name='train')
        return train_op


def _variable_on_cpu(name, shape, initializer):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(
            name,
            shape,
            initializer=initializer,
            dtype=dtype)
    return var


def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name, shape, tf.truncated_normal_initializer(
            stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def GetOptions():
    parser = OptionParser()
    parser.add_option('-', '--', dest='', metavar='', help='')
    (options, args) = parser.parse_args()

    return


def main():
    return


if __name__ == '__main__':
    main()
