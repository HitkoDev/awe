import os

import tensorflow as tf

from model2 import model

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

image_size = 299

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', '', 'Model to restore')

model.load_weights(FLAGS.model)

model.save('converted.h5')
