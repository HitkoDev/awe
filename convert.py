import os
import random

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from dataset import AWEDataset, load_img

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

image_size = 299

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', '', 'Model to restore')

model = tf.keras.applications.InceptionResNetV2(
    include_top=False,
    input_shape=(image_size, image_size, 3),
    pooling='max'
)

model.layers.append(tf.keras.layers.Dense(256, activation=None))
model.layers.append(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))


model.load_weights(FLAGS.model)

model.save('converted.h5')
