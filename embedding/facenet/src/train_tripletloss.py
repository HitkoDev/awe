"""Training a face recognizer with TensorFlow based on the FaceNet paper
FaceNet: A Unified Embedding for Face Recognition and Clustering: http://arxiv.org/abs/1503.03832
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import, division, print_function

import argparse
import importlib
import itertools
import math
import os.path
import random
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import skimage.io
import tensorflow as tf
from six.moves import xrange  # @UnresolvedImport
from tensorflow.python.ops import data_flow_ops

import facenet
import lfw
from awe_dataset import AWEDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def main(args):

    network = importlib.import_module(args.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)

    # Write arguments to a text file
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))

    # Store some git revision info in a text file in the log directory
    src_path, _ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    train_set = facenet.get_dataset(args.data_dir)

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    if args.pretrained_model:
        print('Pre-trained model: %s' % os.path.expanduser(args.pretrained_model))

    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs)

    with tf.Graph().as_default():
        tf.compat.v1.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # Placeholder for the learning rate
        learning_rate_placeholder = tf.compat.v1.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.compat.v1.placeholder(tf.int32, name='batch_size')

        phase_train_placeholder = tf.compat.v1.placeholder(tf.bool, name='phase_train')

        image_paths_placeholder = tf.compat.v1.placeholder(tf.string, shape=(None, 3), name='image_paths')
        image_placeholder = tf.compat.v1.placeholder(tf.int8, shape=(None, 3, args.image_size, args.image_size, 3), name='image_in')
        labels_placeholder = tf.compat.v1.placeholder(tf.int64, shape=(None, 3), name='labels')

        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                              dtypes=[tf.int8, tf.int64],
                                              shapes=[(3, args.image_size, args.image_size, 3), (3,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([
            image_placeholder,
            # image_paths_placeholder,
            labels_placeholder
        ])

        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                """file_contents = tf.io.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)

                if args.random_crop:
                    image = tf.image.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    image = tf.image.resize_with_crop_or_pad(image, args.image_size, args.image_size)
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)

                #pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))"""
                images.append(tf.image.per_image_standardization(filename))
            images_and_labels.append([
                images,
                label
            ])

        image_batch, labels_batch = tf.compat.v1.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder,
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        labels_batch = tf.identity(labels_batch, 'label_batch')

        # Build the inference graph
        prelogits, _ = network.inference(image_batch, args.keep_probability,
                                         phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,
                                         weight_decay=args.weight_decay)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        # Split embeddings into anchor, positive and negative and calculate triplet loss
        anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, args.embedding_size]), 3, 1)
        triplet_loss = facenet.triplet_loss(anchor, positive, negative, args.alpha)

        learning_rate = tf.compat.v1.train.exponential_decay(learning_rate_placeholder, global_step,
                                                             args.learning_rate_decay_epochs * args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.compat.v1.summary.scalar('learning_rate', learning_rate)

        # Calculate the total losses
        regularization_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([triplet_loss] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op, opt = facenet.train(total_loss, global_step, args.optimizer,
                                      learning_rate, args.moving_average_decay, tf.compat.v1.global_variables())

        # Create a saver
        saver = tf.compat.v1.train.Saver(tf.compat.v1.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.compat.v1.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer(), feed_dict={phase_train_placeholder: True})
        sess.run(tf.compat.v1.local_variables_initializer(), feed_dict={phase_train_placeholder: True})

        summary_writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.compat.v1.train.start_queue_runners(coord=coord, sess=sess)

        train_dataset_path = os.path.join(args.data_dir, 'train')
        test_dataset_path = os.path.join(args.data_dir, 'test')

        train_dataset = AWEDataset(train_dataset_path)
        test_dataset = AWEDataset(test_dataset_path)

        with sess.as_default():

            if args.pretrained_model:
                print('Restoring pretrained model: %s' % args.pretrained_model)
                saver.restore(sess, os.path.expanduser(args.pretrained_model))

            # Training and validation loop
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
                # Train for one epoch
                train(args, sess, train_dataset, epoch,
                      image_placeholder,
                      # image_paths_placeholder,
                      labels_placeholder, labels_batch,
                      batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue, global_step,
                      embeddings, total_loss, train_op, summary_op, summary_writer, args.learning_rate_schedule_file,
                      args.embedding_size, anchor, positive, negative, triplet_loss)

                # Save variables and the metagraph if it doesn't exist already
                if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
                    os.makedirs(model_dir)
                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                imgs = test_dataset.images
                img = []
                c_same = 0
                c_diff = 0
                for i in range(len(imgs)):
                    im2 = imgs[i]
                    if len(im2) > 1:
                        random.shuffle(im2)
                        img.append([
                            im2[0]['src'],
                            im2[1]['src'],
                            True
                        ])
                        c_same += 1
                while c_diff < (c_same * 5) or (c_diff + c_same) % 3 != 0:
                    a = random.randint(0, len(imgs) - 1)
                    b = random.randint(0, len(imgs) - 1)
                    if a != b:
                        img.append([
                            imgs[a][-1]['src'],
                            imgs[b][-1]['src'],
                            False
                        ])
                        c_diff += 1
                random.shuffle(img)
                is_same = [x[2] for x in img]
                img = [y for x in img for y in x[0:-1]]

                evaluate(sess, img, embeddings, labels_batch,
                         image_placeholder,
                         # image_paths_placeholder,
                         labels_placeholder,
                         batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, is_same, args.batch_size,
                         args.lfw_nrof_folds, log_dir, step, summary_writer, args.embedding_size, args)

    return model_dir


def train(args, sess, dataset, epoch, image_paths_placeholder, labels_placeholder, labels_batch,
          batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, input_queue, global_step,
          embeddings, loss, train_op, summary_op, summary_writer, learning_rate_schedule_file,
          embedding_size, anchor, positive, negative, triplet_loss):
    import imgaug
    import imgaug.augmenters as iaa
    batch_number = 0
    augmentation = iaa.Sequential([iaa.Rotate((-170, 170))])

    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    while batch_number < args.epoch_size:
        # Sample people randomly from the dataset
        image_paths, num_per_class = sample_people(dataset, args.people_per_batch, args.images_per_person)

        print('Running forward pass on sampled images: ', end='')
        start_time = time.time()
        nrof_examples = args.people_per_batch * args.images_per_person
        labels_array = np.reshape(np.arange(nrof_examples), (-1, 3))
        image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))
        images_array = []
        for p in image_paths_array:
            imgs = []
            for path in p:
                image = np.array(skimage.io.imread(path))
                if len(image.shape) == 2:
                    image = np.stack([image, image, image], axis=2)
                image = image[:, :, 0:3]
                w, h, c = image.shape
                r = math.ceil((w ** 2 + h ** 2) ** 0.5)
                pw = math.ceil((r - w) / 2)
                ph = math.ceil((r - h) / 2)
                image = np.pad(image, ((pw, pw), (ph, ph), (0, 0)))
                m = path[:-4] + '.npy'
                with open(m, 'rb') as file:
                    mask = np.load(file)
                mask = np.pad(mask, ((pw, pw), (ph, ph)))

                # Augmenters that are safe to apply to masks
                # Some, such as Affine, have settings that make them unsafe, so always
                # test your augmentation on masks
                MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                                   "Fliplr", "Flipud", "CropAndPad",
                                   "Affine", "PiecewiseAffine"]

                def hook(images, augmenter, parents, default):
                    """Determines which augmenters to apply to masks."""
                    return augmenter.__class__.__name__ in MASK_AUGMENTERS

                # Store shapes before augmentation to compare
                image_shape = image.shape
                mask_shape = mask.shape
                # Make augmenters deterministic to apply similarly to images and masks
                det = augmentation.to_deterministic()
                image = det.augment_image(image)
                mask = det.augment_image(mask)
                # Verify that shapes didn't change
                assert image.shape == image_shape, "Augmentation shouldn't change image size"
                assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
                horizontal_indicies = np.where(np.any(mask, axis=0))[0]
                vertical_indicies = np.where(np.any(mask, axis=1))[0]
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                x2 += 1
                y2 += 1
                mask_out = image * np.stack([mask, mask, mask], axis=2)
                out = mask_out[y1:y2, x1:x2]
                out = cv2.resize(out, dsize=(args.image_size, args.image_size))
                #cv2.imwrite('verify.png', out)
                imgs.append(out)
            images_array.append(imgs)
        images_array = np.array(images_array)

        sess.run(enqueue_op, {
            # image_paths_placeholder: image_paths_array,
            image_paths_placeholder: images_array,
            labels_placeholder: labels_array
        })
        emb_array = np.zeros((nrof_examples, embedding_size))
        nrof_batches = int(np.ceil(nrof_examples / args.batch_size))
        for i in range(nrof_batches):
            batch_size = min(nrof_examples - i * args.batch_size, args.batch_size)
            emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                                                                       learning_rate_placeholder: lr, phase_train_placeholder: True})
            emb_array[lab, :] = emb
        print('%.3f' % (time.time() - start_time))

        # Select triplets based on the embeddings
        print('Selecting suitable triplets for training')
        triplets, nrof_random_negs, nrof_triplets = select_triplets(emb_array, num_per_class,
                                                                    np.reshape(images_array, (-1, args.image_size, args.image_size, 3)), args.people_per_batch, args.alpha)
        selection_time = time.time() - start_time
        print('(nrof_random_negs, nrof_triplets) = (%d, %d): time=%.3f seconds' %
              (nrof_random_negs, nrof_triplets, selection_time))

        # Perform training on the selected triplets
        nrof_batches = int(np.ceil(nrof_triplets * 3 / args.batch_size))
        triplet_paths = list(itertools.chain(*triplets))
        labels_array = np.reshape(np.arange(len(triplet_paths)), (-1, 3))
        triplet_paths_array = np.reshape(np.expand_dims(np.array(triplet_paths), 1), (-1, 3, args.image_size, args.image_size, 3))
        sess.run(enqueue_op, {image_paths_placeholder: triplet_paths_array, labels_placeholder: labels_array})
        nrof_examples = len(triplet_paths)
        train_time = 0
        i = 0
        emb_array = np.zeros((nrof_examples, embedding_size))
        loss_array = np.zeros((nrof_triplets,))
        summary = tf.compat.v1.Summary()
        step = 0
        while i < nrof_batches:
            start_time = time.time()
            batch_size = min(nrof_examples - i * args.batch_size, args.batch_size)
            feed_dict = {batch_size_placeholder: batch_size, learning_rate_placeholder: lr, phase_train_placeholder: True}
            err, _, step, emb, lab = sess.run([loss, train_op, global_step, embeddings, labels_batch], feed_dict=feed_dict)
            emb_array[lab, :] = emb
            loss_array[i] = err
            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f' %
                  (epoch, batch_number + 1, args.epoch_size, duration, err))
            batch_number += 1
            i += 1
            train_time += duration
            summary.value.add(tag='loss', simple_value=err)

        # Add validation loss and accuracy to summary
        #pylint: disable=maybe-no-member
        summary.value.add(tag='time/selection', simple_value=selection_time)
        summary_writer.add_summary(summary, step)
    return step


def select_triplets(embeddings, nrof_images_per_class, image_paths, people_per_batch, alpha):
    """ Select the triplets for training
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []

    print(embeddings.shape)

    # VGG Face: Choosing good triplets is crucial and should strike a balance between
    #  selecting informative (i.e. challenging) examples and swamping training with examples that
    #  are too hard. This is achieve by extending each pair (a, p) to a triplet (a, p, n) by sampling
    #  the image n at random, but only between the ones that violate the triplet loss margin. The
    #  latter is a form of hard-negative mining, but it is not as aggressive (and much cheaper) than
    #  choosing the maximally violating example, as often done in structured output learning.

    for i in xrange(people_per_batch):
        nrof_images = int(nrof_images_per_class[i])
        for j in xrange(1, nrof_images):
            a_idx = emb_start_idx + j - 1
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            for pair in xrange(j, nrof_images):  # For every possible positive pair.
                p_idx = emb_start_idx + pair
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN
                # all_neg = np.where(np.logical_and(neg_dists_sqr-pos_dist_sqr<alpha, pos_dist_sqr<neg_dists_sqr))[0]  # FaceNet selection
                all_neg = np.where((neg_dists_sqr - pos_dist_sqr) < alpha)[0]  # VGG Face selecction
                nrof_random_negs = all_neg.shape[0]
                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    # print('Triplet %d: (%d, %d, %d), pos_dist=%2.6f, neg_dist=%2.6f (%d, %d, %d, %d, %d)' %
                    #    (trip_idx, a_idx, p_idx, n_idx, pos_dist_sqr, neg_dists_sqr[n_idx], nrof_random_negs, rnd_idx, i, j, emb_start_idx))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)


def sample_people(dataset, people_per_batch, images_per_person):
    nrof_images = people_per_batch * images_per_person

    # Sample classes from the dataset
    nrof_classes = len(dataset.images)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    image_paths = []
    num_per_class = []
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths) < nrof_images:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset.images[class_index])
        image_indices = np.arange(nrof_images_in_class)
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class, images_per_person, nrof_images - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset.images[class_index][j]['src'] for j in idx]
        sampled_class_indices += [class_index] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1

    return image_paths, num_per_class


def evaluate(sess, image_paths, embeddings, labels_batch, image_paths_placeholder, labels_placeholder,
             batch_size_placeholder, learning_rate_placeholder, phase_train_placeholder, enqueue_op, actual_issame, batch_size,
             nrof_folds, log_dir, step, summary_writer, embedding_size, args):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Running forward pass on LFW images: ', end='')

    nrof_images = len(actual_issame) * 2
    assert(len(image_paths) == nrof_images)
    labels_array = np.reshape(np.arange(nrof_images), (-1, 3))
    image_paths_array = np.reshape(np.expand_dims(np.array(image_paths), 1), (-1, 3))

    images_array = []
    for p in image_paths_array:
        imgs = []
        for path in p:
            image = np.array(skimage.io.imread(path))
            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=2)
            image = image[:, :, 0:3]
            out = cv2.resize(image, dsize=(args.image_size, args.image_size))
            imgs.append(out)
        images_array.append(imgs)
    images_array = np.array(images_array)

    sess.run(enqueue_op, {image_paths_placeholder: images_array, labels_placeholder: labels_array})
    emb_array = np.zeros((nrof_images, embedding_size))
    nrof_batches = int(np.ceil(nrof_images / batch_size))
    label_check_array = np.zeros((nrof_images,))
    for i in xrange(nrof_batches):
        batch_size = min(nrof_images - i * batch_size, batch_size)
        emb, lab = sess.run([embeddings, labels_batch], feed_dict={batch_size_placeholder: batch_size,
                                                                   learning_rate_placeholder: 0.0, phase_train_placeholder: False})
        emb_array[lab, :] = emb
        label_check_array[lab] = 1
    print('%.3f' % (time.time() - start_time))

    assert(np.all(label_check_array == 1))

    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)

    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.compat.v1.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir, 'lfw_result.txt'), 'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.compat.v1.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)


def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='./logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='./models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='../../images/converted')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=189)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=224)
    parser.add_argument('--people_per_batch', type=int,
                        help='Number of people per batch.', default=189)
    parser.add_argument('--images_per_person', type=int,
                        help='Number of images per person.', default=3)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=99)
    parser.add_argument('--alpha', type=float,
                        help='Positive to negative triplet distance margin.', default=0.2)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=512)
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                        'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAM')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=5)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_dir', type=str,
                        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))