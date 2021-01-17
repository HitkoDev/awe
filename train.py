from __future__ import (absolute_import, division, generators, print_function,
                        unicode_literals, with_statement)

import os

import tensorflow as tf

from dataset import AWEDataset
from inception import inception_v4
from model import *

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 10, 'Batch size')
flags.DEFINE_integer('train_iter', 2000, 'Total training iter')
flags.DEFINE_integer('step', 5, 'Save after ... iteration')
flags.DEFINE_integer('image_size', 224, 'Image size')
flags.DEFINE_string('data_dir', './images/converted', 'Dataset dir')

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.compat.v1.disable_eager_execution()

if __name__ == "__main__":
    # setup dataset
    train_dataset = AWEDataset(os.path.join(FLAGS.data_dir, 'train'))
    test_dataset = AWEDataset(os.path.join(FLAGS.data_dir, 'test'))
    # model = AWE_model
    model = inception_v4
    placeholder_shape = (None, FLAGS.image_size, FLAGS.image_size, 3)
    print("placeholder_shape", placeholder_shape)
    colors = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', '#ff00ff', '#990000', '#999900', '#009900', '#009999']

    # Setup network
    left = tf.compat.v1.placeholder(tf.float32, placeholder_shape, name='left')
    right = tf.compat.v1.placeholder(tf.float32, placeholder_shape, name='right')
    with tf.compat.v1.name_scope("similarity"):
        label = tf.compat.v1.placeholder(tf.int32, [None, 1], name='label')  # 1 if same, 0 if different
        label_float = tf.cast(label, dtype=tf.float32)
    margin = 0.5
    left_output = model(left, reuse=False)
    right_output = model(right, reuse=True)
    loss = contrastive_loss(left_output, right_output, label_float, margin)

    # Setup Optimizer
    global_step = tf.Variable(0, trainable=False)

    # starter_learning_rate = 0.0001
    # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
    # tf.scalar_summary('lr', learning_rate)
    # train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

    train_step = tf.compat.v1.train.AdamOptimizer(0.001).minimize(loss, global_step=global_step)

    # Start Training
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        # setup tensorboard
        tf.compat.v1.summary.scalar('step', global_step)
        tf.compat.v1.summary.scalar('loss', loss)
        for var in tf.compat.v1.trainable_variables():
            tf.compat.v1.summary.histogram(var.op.name, var)
        merged = tf.compat.v1.summary.merge_all()
        writer = tf.compat.v1.summary.FileWriter('train.log', sess.graph)

        # train iter
        for i in range(FLAGS.train_iter):
            batch_left, batch_right, batch_similarity = train_dataset.get_batch(FLAGS.batch_size, FLAGS.image_size)

            _, l, summary_str = sess.run([train_step, loss, merged],
                                         feed_dict={left: batch_left, right: batch_right, label: batch_similarity})

            writer.add_summary(summary_str, i)
            print("\r#%d - Loss" % i, l)

            if (i + 1) % FLAGS.step == 0:
                test_left, test_right, test_similarity = test_dataset.get_batch(FLAGS.batch_size, FLAGS.image_size, False)
                loss, _ = sess.run([loss, merged], feed_dict={left: test_left, right: test_right, label: test_similarity})
                print("\rValidation loss", l)

                saver.save(sess, "model/model.ckpt")
