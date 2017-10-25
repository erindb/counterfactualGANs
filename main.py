#! /usr/bin/env python

import numpy as np
import tensorflow as tf

from model import *
from ops import *
from utils import *

DIM = 100

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", True, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

def restore_model():    
  with tf.Session(config=run_config) as sess:
    dcgan = DCGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        dataset_name=FLAGS.dataset,
        input_fname_pattern=FLAGS.input_fname_pattern,
        crop=FLAGS.crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir)
    
    dcgan.load(checkpoint_dir)  # by here, we should have our recovered model
    return dcgan, sess

#img_vector = network.generator(z)
# use imsave() and save_images() to save into a path (from utils.py)
  
def generate_image(z, dcgan, session, visualize_option):
  with session as sess:
    image_frame_dim = int(math.ceil(config.batch_size**.5))
    sample_image = sess.run(dcgan.sampler, feed_dict={dcgan.z: z})
    save_image(sample_image, [image_frame_dim, image_frame_dim], './samples/single_img_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))


def main():
  z = np.random.rand(DIM)
  print(z)
  network, tf_session = restore_model()
  OPTION = 1
  generate_image(z, network, session, OPTION)

main()
