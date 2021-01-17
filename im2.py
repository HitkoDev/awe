import io
import os
import random

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from dataset import AWEDataset, load_img

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

train_dataset = AWEDataset(os.path.join('./images/converted', 'train'))
test_dataset = AWEDataset(os.path.join('./images/converted', 'test'))

labels_map = {}
for i in range(len(train_dataset.images)):
    labels_map[train_dataset.images[i][0]['class']] = i


def train():
    n = 25
    i = 0
    a = []
    b = []
    c = {}
    while True:
        random.shuffle(train_dataset.images)
        for x in train_dataset.images:
            random.shuffle(x)
            for im in x:
                p = im['src']
                if p not in c:
                    c[p] = load_img(im['src'], 160, aug=False)
                a.append(c[p])
                b.append(labels_map[im['class']])

                # a.append(load_img(im['src'], 160))
                # b.append(labels_map[im['class']])
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
        a.append(load_img(x['src'], 160, False))
        b.append(labels_map[x['class']])
    while True:
        yield (np.array(a) / 255., np.array(b))


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=64, kernel_size=4, padding='same', activation='relu', input_shape=(160, 160, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(filters=32, kernel_size=4, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation=None),  # No activation on final dense layer
    tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))  # L2 normalize embeddings
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.00001),
    loss=tfa.losses.TripletSemiHardLoss()
)

# Train the network
history = model.fit(
    train(),
    epochs=300,
    steps_per_epoch=30,
    validation_data=test(),
    validation_steps=1
)
