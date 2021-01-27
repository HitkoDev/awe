import os

import numpy as np
import tensorflow as tf

from dataset import AWEDataset, load_img

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

image_size = 299

train_dataset = AWEDataset(os.path.join('./images/converted', 'train'))
test_dataset = AWEDataset(os.path.join('./images/converted', 'test'))

model = tf.keras.models.load_model('converted.h5')

ds = {}

for tr in train_dataset.images[0:20]:
    imgs = []
    for im in tr:
        img = load_img(im['src'], image_size, True, False)
        imgs.append(img)

    r = model.predict(np.array(imgs))
    r = np.mean(r, axis=0)
    ds[tr[0]['class']] = r

for ts in test_dataset.images:
    imgs = []
    for im in ts:
        img = load_img(im['src'], image_size, False, False)
        imgs.append(img)
    r = model.predict(np.array(imgs))
    for pred in r:
        for k, v in ds.items():
            dist = np.sum((pred - v)**2)**0.5
            if ts[0]['class'] == k:
                print("True", dist)
            else:
                print("False", dist)
