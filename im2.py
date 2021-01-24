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
        imgs = []
        random.shuffle(train_dataset.images)
        for x in train_dataset.images:
            random.shuffle(x)
            img = []
            lbl = []
            for im in x:
                # Include original images in the first 1/2 of traing samples
                if epoch < 150:
                    p = im['src']
                    if p not in c:
                        c[p] = load_img(im['src'], image_size, aug=False)
                    img.append(c[p])
                    lbl.append(labels_map[im['class']])

                img.append(load_img(im['src'], image_size))
                lbl.append(labels_map[im['class']])
            # select pairs with greatest distance
            pred = model.predict(np.array(img))
            for i in range(len(pred)):
                imgs.append([
                    img[i],
                    lbl[i],
                    pred[i]
                ])

        same = []
        diff = []
        for i1 in range(len(imgs) - 1):
            for i2 in range(i1 + 1, len(imgs)):
                im1 = imgs[i1]
                im2 = imgs[i2]
                dist = (np.sum((im1[2] - im2[2])**2))**0.5
                if im1[1] == im2[1]:
                    same.append([
                        i1,
                        i2,
                        dist
                    ])
                else:
                    diff.append([
                        i1,
                        i2,
                        dist
                    ])
        same = sorted(same, key=lambda a_entry: -a_entry[2])
        diff = sorted(diff, key=lambda a_entry: -a_entry[2])
        used = set()
        a1 = []
        b1 = []
        added = True
        while added:
            added = False
            for el in same[:len(same) // 2]:
                if el[0] not in used and el[1] not in used:
                    used.add(el[0])
                    a1.append(imgs[el[0]][0])
                    b1.append(imgs[el[0]][1])
                    used.add(el[1])
                    a1.append(imgs[el[1]][0])
                    b1.append(imgs[el[1]][1])
                    added = True

            for el in diff:
                if el[0] not in used and el[1] not in used:
                    used.add(el[0])
                    a1.append(imgs[el[0]][0])
                    b1.append(imgs[el[0]][1])
                    used.add(el[1])
                    a1.append(imgs[el[1]][0])
                    b1.append(imgs[el[1]][1])

        a = []
        b = []
        c = 0
        for k in range(len(a1)):
            a.append(a1[k])
            b.append(b1[k])
            if len(a) == 2 * n:
                yield (np.array(a), np.array(b))
                a = []
                b = []
                c += 1
        print(c)


def test():
    test_images = [x for y in test_dataset.images for x in y]
    a = []
    b = []
    for x in test_images:
        a.append(load_img(x['src'], image_size, False, False))
        b.append(labels_map[x['class']])
    while True:
        yield (np.array(a), np.array(b))


# Compile the model
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.00001,
    decay_steps=3000,
    decay_rate=0.1,
    staircase=True
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr_schedule),
    loss=tfa.losses.TripletHardLoss(margin=0.2)
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
