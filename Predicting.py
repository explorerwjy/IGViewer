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

tf.app.flags.DEFINE_string('DataFile', '/home/yufengshen/IGViewer/Predict/ToPredict_0503.txt',
                           """Data File to Predict.""")
tf.app.flags.DEFINE_string('eval_dir', './test',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', './train_3',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('num_examples', 5543,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

EPOCHS = 1000000

class INPUTPIPE:
    def __init__(self, DataFile):
        self.DataFile = DataFile
    def PipeLine(self, batch_size, num_epochs=None):
        image_list = self.read_image_with_fname_list()
        images = tf.convert_to_tensor(image_list, dtype=tf.string)
        input_queue = tf.train.slice_input_producer(images,
                                            #num_epochs=num_epochs,
                                            shuffle=False, name="PredictionQueue")
        fname, image = self.read_images_from_disk(input_queue)
        image = self.preprocess_image(image)
        image_batch = tf.train.batch([fname, image],batch_size=batch_size)
        return image_batch

    def read_image_with_fname_list(self, Limit=None):
        fin = open(self.DataFile, 'rb')
        filenames = []
        if Limit != None:
            count = 0
        for l in fin:
            if Limit != None and count >= Limit:
                break
            filename = l.strip().split('\t')
            filenames.append(filename)
            if Limit != None:
                count += 1
        return filenames

    def read_images_from_disk(self, input_queue):
        file_contents = tf.read_file(input_queue)
        example = tf.image.decode_image(file_contents, channels=DEPTH)
        return input_queue, example

    def preprocess_image(self, image):
        resized_image = tf.image.resize_image_with_crop_or_pad(image,
                                                         HEIGHT, WIDTH)
        float_image = tf.image.per_image_standardization(resized_image)
        float_image.set_shape([HEIGHT, WIDTH, DEPTH])
        return float_image


class Predict():
    def __init__(self, batch_size, epochs, model, DataFile):
        self.batch_size = batch_size
        self.epochs = epochs
        print "=" * 50
        print "InputData is:", DataFile
        print "=" * 50
        self.InputData = INPUTPIPE(DataFile)
        self.model = model
        self.fout = open('Predicted.txt','wb')
        self.fout.write('FileName\tPrediction\tProbability\n')

    def run(self):
        with tf.Graph().as_default():
            global_step = tf.Variable(0, trainable=False, name='global_step')
            fnames, images = self.InputData.PipeLine(self.batch_size, self.epochs)
            logits = self.model.Inference(images)
            normed_logits = tf.nn.softmax(logits, dim=-1, name=None)
            
            predict = tf.argmax(normed_logits, 1)
            #predict = tf.nn.in_top_k(normed_logits, labels, 1)
            #top_k_op = tf.nn.in_top_k(logits, labels, 1)
            summary_op = tf.summary.merge_all()
            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            sess = tf.Session()
            sess.run(init)
            # Start the queue runners.
            tf.train.start_queue_runners(sess=sess)
            summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, sess.graph)
            coord = tf.train.Coordinator()
            try:    
                num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
                true_count = 0  # Counts the number of correct predictions.
                total_sample_count = num_iter * FLAGS.batch_size
                step = 0
                print self.getCheckPoint()
                saver.restore(sess, self.getCheckPoint())
                print "CKPT starts with step",(sess.run(global_step))
                while step < num_iter and not coord.should_stop():
                    _fnames, _logits, _predictions = sess.run([fnames, normed_logits, predict])
                    for _fname, _logit, _predict in zip(_fnames, _logits, _predictions):
                        print _fname, _logit, _predict
                        #self.fout.write('{}\t{}\t{}\n'.format(_fname, _predict, _logit))
                    step += 1
            except Exception, e:
                coord.request_stop(e)
            finally:
                self.fout.close()
                coord.request_stop()
                coord.join()

    def getCheckPoint(self):
        ckptfile = FLAGS.checkpoint_dir + '/checkpoint'
        f = open(ckptfile, 'rb')
        ckpt = f.readline().split(':')[1].strip().strip('"')
        f.close()
        prefix = os.path.abspath(FLAGS.checkpoint_dir)
        ckpt = prefix + '/' + ckpt
        return ckpt



def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    DataFile = FLAGS.DataFile
    model = Models.ConvNets()
    evaluate = Predict(FLAGS.batch_size, EPOCHS, model, DataFile)
    evaluate.run()
