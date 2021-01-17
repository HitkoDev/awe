import io
import os
import random

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from dataset import AWEDataset, load_img

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

image_size = 299

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('epoch', 0, 'Epoch to start from')
flags.DEFINE_string('model', '', 'Model to restore')

train_dataset = AWEDataset(os.path.join('./images/converted', 'train'))
test_dataset = AWEDataset(os.path.join('./images/converted', 'test'))

labels_map = {}
for i in range(len(train_dataset.images)):
    labels_map[train_dataset.images[i][0]['class']] = i

epoch = 0


class TrainEpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, ep, logs=None):
        global epoch
        epoch = ep


def train():
    global epoch
    n = 2
    i = 0
    a = []
    b = []
    c = {}
    while True:
        random.shuffle(train_dataset.images)
        for x in train_dataset.images:
            random.shuffle(x)
            for im in x:
                print(epoch)
                if epoch < 100:
                    p = im['src']
                    if p not in c:
                        c[p] = load_img(im['src'], image_size, aug=False)
                    a.append(c[p])
                    b.append(labels_map[im['class']])

                a.append(load_img(im['src'], image_size))
                b.append(labels_map[im['class']])
            if i == n:
                yield (np.array(a) / 255., np.array(b))
                i = 0
                a = []
                b = []
            i += 1


def test():
    test_images = [x for y in test_dataset.images for x in y]
    a = []
    b = []
    for x in test_images:
        a.append(load_img(x['src'], image_size, False))
        b.append(labels_map[x['class']])
    while True:
        yield (np.array(a) / 255., np.array(b))


model = tf.keras.applications.InceptionResNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(image_size, image_size, 3),
    pooling='max'
)

model.layers.append(tf.keras.layers.Dense(256, activation=None))
model.layers.append(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))

# Compile the model
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.0001,
    decay_steps=10000,
    decay_rate=0.1,
    staircase=False
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr_schedule, amsgrad=True),
    loss=tfa.losses.TripletSemiHardLoss()
)

if FLAGS.model:
    model.load_weights(FLAGS.model)

# Train the network
history = model.fit(
    train(),
    epochs=300,
    steps_per_epoch=100,
    validation_data=test(),
    validation_steps=1,
    initial_epoch=FLAGS.epoch,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(filepath='./model/model.{epoch:02d}-{val_loss:.2f}.h5', save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        TrainEpochCallback()
    ]
)
