#!/usr/bin/env python3
from argparse import ArgumentParser
from importlib import import_module
import os

import json
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.compat.v1.disable_eager_execution()

parser = ArgumentParser(description='Convert model to a standalone format.')

# Required

parser.add_argument(
    '--experiment_root', required=True,
    help='Location used to store checkpoints and dumped data.')

# Optional

parser.add_argument(
    '--checkpoint', default=None,
    help='Name of checkpoint file of the trained network within the experiment '
         'root. Uses the last checkpoint if not provided.')

parser.add_argument(
    '--quiet', action='store_true', default=False,
    help='Don\'t be so verbose.')


def main():
    # Verify that parameters are set correctly.
    args = parser.parse_args()

    # Possibly auto-generate the output filename.
    if args.filename is None:
        basename = os.path.basename(args.dataset)
        args.filename = os.path.splitext(basename)[0] + '_embeddings.h5'
    args.filename = os.path.join(args.experiment_root, args.filename)

    # Load the args from the original experiment.
    args_file = os.path.join(args.experiment_root, 'args.json')

    if os.path.isfile(args_file):
        if not args.quiet:
            print('Loading args from {}.'.format(args_file))
        with open(args_file, 'r') as f:
            args_resumed = json.load(f)

        # Add arguments from training.
        for key, value in args_resumed.items():
            args.__dict__.setdefault(key, value)

        # A couple special-cases and sanity checks
        if (args_resumed['crop_augment']) == (args.crop_augment is None):
            print('WARNING: crop augmentation differs between training and '
                  'evaluation.')
        args.image_root = args.image_root or args_resumed['image_root']
    else:
        raise IOError('`args.json` could not be found in: {}'.format(args_file))

    if not args.quiet:
        print('Evaluating using the following parameters:')
        for key, value in sorted(vars(args).items()):
            print('{}: {}'.format(key, value))

    images_ph = tf.compat.v1.placeholder(tf.float32, shape=(None, args.net_input_height, args.net_input_width, 3), name='input_placeholder')

    # Create the model and an embedding head.
    model = import_module('nets.' + args.model_name)
    head = import_module('heads.' + args.head_name)

    endpoints, _ = model.endpoints(images_ph, is_training=False)
    with tf.compat.v1.name_scope('head'):
        endpoints = head.head(endpoints, args.embedding_dim, is_training=False)

    with tf.compat.v1.Session() as sess:
        # Initialize the network/load the checkpoint.
        if args.checkpoint is None:
            checkpoint = tf.train.latest_checkpoint(args.experiment_root)
        else:
            checkpoint = os.path.join(args.experiment_root, args.checkpoint)
        if not args.quiet:
            print('Restoring from checkpoint: {}'.format(checkpoint))
        tf.compat.v1.train.Saver().restore(sess, checkpoint)

        tf.compat.v1.saved_model.simple_save(sess, 'converted', {'input_placeholder': images_ph}, {'embeddings': endpoints['emb']})


if __name__ == '__main__':
    main()
