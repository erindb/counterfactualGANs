#! /usr/bin/env python

import numpy as np
import tensorflow as tf

from model import *
from utils import *

DIM = ???


def generate_image(z):
	# generate image! :)
	network = # DCGAN model.load(checkpoint files)
	'''
    THIS IS THE KEY
	  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0

	'''
    img_vector = network.generator(z)
    # use imsave() and save_images() to save into a path (from utils.py)
    
    
	pass
	## for reference: (from `DCGAN-tensorflow/utils.py`)
	# def visualize(sess, dcgan, config, option):
	#   image_frame_dim = int(math.ceil(config.batch_size**.5))
	#   if option == 0:
	#     z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
	#     samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
	#     save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))


def main():
	z = np.rand(DIM)
	generate_image(z)

main()
