from __future__ import (absolute_import, division, generators, print_function,
                        unicode_literals, with_statement)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tf_slim as slim

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS


def AWE_model(input, reuse=False):
    with tf.compat.v1.name_scope("model"):
        with tf.compat.v1.variable_scope("conv1"):
            net = tf.compat.v1.layers.conv2d(input, 32, [4, 4], activation=tf.nn.leaky_relu, padding='SAME',
                                             kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), reuse=reuse)
            net = tf.compat.v1.layers.max_pooling2d(net, [3, 3], [1, 1], padding='SAME')

        with tf.compat.v1.variable_scope("conv2"):
            net = tf.compat.v1.layers.conv2d(net, 32, [4, 4], activation=tf.nn.leaky_relu, padding='SAME',
                                             kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), reuse=reuse)
            net = tf.compat.v1.layers.max_pooling2d(net, [3, 3], [1, 1], padding='SAME')

        with tf.compat.v1.variable_scope("conv3"):
            net = tf.compat.v1.layers.conv2d(net, 64, [3, 3], activation=tf.nn.leaky_relu, padding='SAME',
                                             kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), reuse=reuse)
            net = tf.compat.v1.layers.max_pooling2d(net, [2, 2], [1, 1], padding='SAME')

        with tf.compat.v1.variable_scope("conv4"):
            net = tf.compat.v1.layers.conv2d(net, 64, [3, 3], activation=tf.nn.leaky_relu, padding='SAME',
                                             kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), reuse=reuse)
            net = tf.compat.v1.layers.max_pooling2d(net, [2, 2], [1, 1], padding='SAME')

        """with tf.compat.v1.variable_scope("conv5"):
            net = tf.compat.v1.layers.conv2d(net, 2, [1, 1], activation=None, padding='SAME',
                                             kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), reuse=reuse)
            net = tf.compat.v1.layers.max_pooling2d(net, [2, 2], [1, 1], padding='SAME')"""

        net = tf.compat.v1.layers.flatten(net)
        net = tf.compat.v1.layers.dense(net, 128)

    return net


def contrastive_loss(model1, model2, y, margin_sim, margin_dis):
    with tf.compat.v1.name_scope("contrastive-loss"):
        distance = tf.sqrt(tf.reduce_sum(input_tensor=tf.pow(model1 - model2, 2), axis=1, keepdims=True))
        similarity = y * tf.square(tf.maximum((distance - margin_sim), 0))                                           # keep the similar label (1) close to each other
        dissimilarity = (1 - y) * tf.square(tf.maximum((margin_dis - distance), 0))        # give penalty to dissimilar label if the distance is bigger than margin
        return tf.reduce_mean(input_tensor=dissimilarity + similarity) / 2
