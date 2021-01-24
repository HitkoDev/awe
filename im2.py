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

train1_dataset = AWEDataset(os.path.join('./images/nomask', 'train'))
test1_dataset = AWEDataset(os.path.join('./images/nomask', 'test'))

train_dataset = AWEDataset(os.path.join('./images/converted', 'train'))
test_dataset = AWEDataset(os.path.join('./images/converted', 'test'))

labels_map = {}
for i in range(len(train_dataset.images)):
    labels_map[train_dataset.images[i][0]['class']] = i

epoch = 0


img = tf.keras.Input(shape=(image_size, image_size, 3))
f = model(tf.keras.applications.inception_resnet_v2.preprocess_input(img))

model = tf.keras.Model(inputs=img, outputs=f)

model.save('m2.h5')


class TrainEpochCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, ep, logs=None):
        global epoch
        epoch = ep


def train():
    global epoch, model
    n = 10
    i = 0
    a = []
    b = []
    c = {}
    while True:
        dataset = train_dataset
        mask = True
        if epoch < 75:
            dataset = train1_dataset
            mask = False
        random.shuffle(dataset.images)
        for x in dataset.images:
            random.shuffle(x)
            img = []
            lbl = []
            for im in x:
                # Include original images in the first 1/2 of traing samples
                if epoch < 150:
                    p = im['src']
                    if p not in c:
                        c[p] = load_img(im['src'], image_size, mask=mask, aug=False)
                    img.append(c[p])
                    lbl.append(labels_map[im['class']])

                img.append(load_img(im['src'], image_size, mask=mask))
                lbl.append(labels_map[im['class']])
            # select pairs with greatest distance
            pred = model.predict(np.array(img))
            m2 = []
            for i1 in range(len(pred) - 1):
                for i2 in range(i1 + 1, len(pred)):
                    dist = (np.sum((pred[i1] - pred[i2])**2))**0.5
                    m2.append([i1, i2, dist])
            m2 = sorted(m2, key=lambda a_entry: -a_entry[2])
            a.append(img[m2[0][0]])
            b.append(lbl[m2[0][0]])
            a.append(img[m2[0][1]])
            b.append(lbl[m2[0][1]])
            for en in m2[1:]:
                if en[0] != m2[0][0] and en[1] != m2[0][0] and en[0] != m2[0][1] and en[1] != m2[0][1]:
                    a.append(img[en[0]])
                    b.append(lbl[en[0]])
                    a.append(img[en[1]])
                    b.append(lbl[en[1]])
                    break

            if i == n:
                yield (np.array(a), np.array(b))
                i = 0
                a = []
                b = []
            i += 1


def test():
    global epoch
    test_images = [x for y in test_dataset.images for x in y]
    a = []
    b = []
    for x in test_images:
        a.append(load_img(x['src'], image_size, False, False))
        b.append(labels_map[x['class']])
    test1_images = [x for y in test1_dataset.images for x in y]
    c = []
    d = []
    for x in test1_images:
        c.append(load_img(x['src'], image_size, False, False))
        d.append(labels_map[x['class']])
    while True:
        if epoch < 75:
            yield (np.array(c), np.array(d))
        else:
            yield (np.array(a), np.array(b))


# Compile the model
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.00001,
    decay_steps=1000,
    decay_rate=0.1,
    staircase=True
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr_schedule),
    loss=tfa.losses.TripletHardLoss(soft=True)
)

if FLAGS.model:
    model.load_weights(FLAGS.model, by_name=True, skip_mismatch=True)

# Train the network
history = model.fit(
    train(),
    epochs=300,
    steps_per_epoch=20,
    validation_data=test(),
    validation_steps=1,
    initial_epoch=FLAGS.epoch,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(filepath='./model/model.{epoch:02d}-{val_loss:.2f}.h5', save_best_only=True),
        tf.keras.callbacks.TensorBoard(log_dir='./logs'),
        TrainEpochCallback()
    ]
)
