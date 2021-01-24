import os

import tensorflow as tf

from model2 import model

image_size = 299

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', '', 'Model to restore')

model.load_weights(FLAGS.model, by_name=True, skip_mismatch=True)

model.save('converted.h5')
