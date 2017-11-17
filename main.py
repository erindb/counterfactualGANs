#! /usr/bin/env python

import math
import numpy as np
import tensorflow as tf

import model
import ops
import utils

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", 64, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", True, "True for visualizing, False for nothing [False]")

flags.DEFINE_integer("vector_opt", 1, "0 for all zeros, 1 for random uniform, 2 to load vector, 3 to cf sample")
flags.DEFINE_string("vector", None, "If vector_opt is 2 or 3, name of .npy file to read in")
flags.DEFINE_boolean("gif", False, "Make gif of base images instead of creating new images.")
FLAGS = flags.FLAGS

DIM = 100
VIZ_OPTION = 1 #Visualization option
NUM_FRAMES = 10 #For making .gifs
BASE_VECTORS = ['origin.npy', 'no_smile.npy', 'man.npy', 'glasses_mustache.npy'] #the array files for our base images

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True

def load_vector(z=None):
    vec = np.empty(shape=(FLAGS.batch_size, DIM))
    base = np.load(FLAGS.vector)
    for i in range(FLAGS.batch_size):
        vec[i] = base
    return vec

def restore_model():    
    sess = tf.Session(config=run_config)
    dcgan = model.DCGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_height=FLAGS.output_height,
        output_width=FLAGS.output_width,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        dataset_name=FLAGS.dataset,
        input_fname_pattern=FLAGS.input_fname_pattern,
        crop=FLAGS.crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir)
    
    dcgan.load(FLAGS.checkpoint_dir)  # by here, we should have our recovered model
    return dcgan, sess
  
def generate_image(z, dcgan, session, visualize_option):
    image_frame_dim = int(math.ceil(FLAGS.batch_size**.5))
    sample_image = session.run(dcgan.sampler, feed_dict={dcgan.z: z})
    utils.save_images(sample_image, [image_frame_dim, image_frame_dim], './samples/single_img.png')

def gaussian_cf_sampler(z):
    sigma = math.sqrt(1.0 / 12)
    cf_sample = np.empty(shape=(FLAGS.batch_size, DIM))
    for i in range(FLAGS.batch_size):
        cf_sample[i] = sigma * np.random.randn(DIM) + z
    return cf_sample

def esm_cf_sampler(z, stickiness):
    cf_sample = np.empty(shape=(FLAGS.batch_size, DIM))
    for i in range(FLAGS.batch_size):
        for j in range(DIM):
            if (np.random.uniform() >= stickiness):
                cf_sample[i][j] = np.random.uniform(-1, 1)
            else:
                cf_sample[i][j] = z[j]
    return cf_sample

def main():
    if (FLAGS.vector_opt == 0):
        z = np.zeros(shape=(FLAGS.batch_size, DIM))
    elif (FLAGS.vector_opt == 1):
        z = np.random.uniform(-0.5, 0.5, size=(FLAGS.batch_size, DIM))
    elif (FLAGS.vector_opt == 2):
        z = load_vector()
    elif (FLAGS.vector_opt == 3):
        z = gaussian_cf_sampler(np.load(FLAGS.vector))
    elif (FLAGS.vector_opt == 4):
        z = esm_cf_sampler(np.load(FLAGS.vector), 0.5)
    else:
        print("Invalid value for vector_opt argument. 0-4 are the only acceptable values.")
        print("Use -h or --help flags for more information.")
        return
 
    np.save("./prev_img_vector.npy", z)  # save our vectors to a file, so if we like one we can replicate
    network, tf_session = restore_model()
    
    if (FLAGS.gif):
        base_images = [np.load('./counterfactualGANs/base_vectors/%s' % i) for i in BASE_VECTORS]
        utils.make_gif(base_images, 'bases.gif')
    else:
        generate_image(z, network, tf_session, VIZ_OPTION)

main()
