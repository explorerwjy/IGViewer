#!/home/local/users/jw/anaconda2/bin/python
# Author: jywang explorerwjy@gmail.com

#=========================================================================
# Prepare Input Data For Training
#=========================================================================

from optparse import OptionParser
import re
import os
import time
import gzip
import numpy as np
import tensorflow as tf

IMAGE_SIZE = 224
HEIGHT = 224
WIDTH = 224
DEPTH = 3
NUM_CLASSES = 2 #(0:Not Variant   1:Inherited Variant)
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 100

FLAGS = tf.app.flags.FLAGS

# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir', './Data',
                           """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")

class INPUT:
    def __init__(self, TrainingDataFile, TestingDataFile):
        self.TrainingDataFile = TrainingDataFile
        self.TestingDataFile = TestingDataFile

    def PipeLine(self, batch_size, num_epochs):
        image_list, label_list = self.read_labeled_image_list()
        images = tf.convert_to_tensor(image_list, dtype=tf.string)
        labels = tf.convert_to_tensor(label_list, dtype=tf.int32)
        # Makes an input queue
        input_queue = tf.train.slice_input_producer([images, labels],
                                            #num_epochs=num_epochs,
                                            shuffle=True, name="TrainingDataQueue")
        image, label = self.read_images_from_disk(input_queue)
        # Optional Preprocessing or Data Augmentation
        # tf.image implements most of the standard image augmentation
        image = self.preprocess_image(image)
        label = self.preprocess_label(label)

        # Optional Image and Label Batching
        image_batch, label_batch = tf.train.batch([image, label],
                                                  batch_size=batch_size)
        return image_batch, label_batch
      
    def read_labeled_image_list(self, Eval=False, Limit=None):
        """Reads a .txt file containing pathes and labeles
        Args:
           DataFile: a .txt file with one "/path/to/image\tlabel" per line
           label: optionally, if set label will be pasted after each line
        Returns:
           List with all filenames in file DataFile
        """
        if Eval:
            fin = open(self.TrainingDataFile, 'rb')
        else:
            fin = open(self.TestingDataFile, 'rb')
        filenames, labels = [], []
        if Limit != None:
            count = 0
        fin.readline()
        for l in fin:
            if Limit != None and count >= Limit:
                break
            filename, label = l.strip().split('\t')
            print filename, label
            filenames.append(filename)
            labels.append(int(label)) 
            if Limit != None:
                count += 1
        return filenames, labels

    def read_images_from_disk(self, input_queue):
        """Consumes a single filename and label as a ' '-delimited string.
        Args:
          filename_and_label_tensor: A scalar string tensor.
        Returns:
          Two tensors: the decoded image, and the string label.
        """
        label = input_queue[1]
        file_contents = tf.read_file(input_queue[0])
        example = tf.image.decode_png(file_contents, channels=DEPTH)
        return example, label

    def preprocess_image(self, image):
        resized_image = tf.image.resize_image_with_crop_or_pad(image,
                                                         HEIGHT, WIDTH)
        float_image = tf.image.per_image_standardization(resized_image)
        float_image.set_shape([HEIGHT, WIDTH, DEPTH])
        return float_image 
    def preprocess_label(self, label):
        #label.set_shape([1])
        return label

def main():
    return


if __name__ == '__main__':
    main()
