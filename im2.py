import os
import random

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

from dataset import AWEDataset, load_img
from model2 import image_size, model

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
    while True:
        a = []
        b = []
        c = []
        ds = train_dataset.get_epoch(2, image_size, 150)
        for k in ds:
            a.append(k[0])
            b.append(k[1])
            c.append(k[2])
        yield ([a, b], c)


def test():
    train_images = [x for y in train_dataset.images for x in y]
    test_images = [x for y in test_dataset.images for x in y]
    a = []
    b = []
    while True:
        imgs = [x for x in train_images]
        random.shuffle(imgs)
        random.shuffle(test_images)
        imgs2 = test_images[0:150]
        a = []
        b = []
        c = []
        i = 0
        lm = len(imgs2) / 2
        for t in imgs2:
            if len(a) < lm:
                loop = 0
                while len(b) < len(a):
                    if i >= len(imgs):
                        i = 0
                        if len(a) == loop:
                            break
                        loop = len(a)
                    if imgs[i] and imgs[i]['class'] == t['class']:
                        a.append(load_img(t['src'], image_size, False, False))
                        c.append(1.)
                        b.append(load_img(imgs[i]['src'], image_size, True, False))
                        imgs[i] = False
                        break
                    i += 1
            else:
                loop = 0
                while len(b) < len(a):
                    if i >= len(imgs):
                        i = 0
                        if len(a) == loop:
                            break
                        loop = len(a)
                    if imgs[i] and imgs[i]['class'] != t['class']:
                        a.append(load_img(t['src'], image_size, False, False))
                        c.append(0.)
                        b.append(load_img(imgs[i]['src'], image_size, True, False))
                        imgs[i] = False
                        break
                    i += 1

        print(lm, len(a))

        yield ([np.array(a), np.array(b)], np.reshape(np.array(c), (-1, 1)))


img1 = tf.keras.Input(shape=(image_size, image_size, 3))
img2 = tf.keras.Input(shape=(image_size, image_size, 3))
f1 = model(tf.keras.applications.inception_resnet_v2.preprocess_input(img1))
f2 = model(tf.keras.applications.inception_resnet_v2.preprocess_input(img2))
diff = tf.keras.layers.Lambda(lambda x: tf.sqrt(tf.reduce_sum(input_tensor=tf.pow(x[0] - x[1], 2), axis=1, keepdims=True)))([f1, f2])

model = tf.keras.Model(inputs=[img1, img2], outputs=diff)

model.save('m2.h5')

# Compile the model
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.0001,
    decay_steps=15,
    decay_rate=0.1,
    staircase=True
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr_schedule, amsgrad=True),
    loss=tfa.losses.ContrastiveLoss(0.5)
)

if FLAGS.model:
    model.load_weights(FLAGS.model)

# Train the network
history = model.fit(
    train(),
    epochs=300,
    steps_per_epoch=1,
    validation_data=test(),
    validation_steps=1,
    initial_epoch=FLAGS.epoch,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(filepath='./model/model.{epoch:02d}-{val_loss:.2f}.h5', save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        TrainEpochCallback()
    ]
)
