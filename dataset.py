import glob
import math
import os
import random

import cv2
import imgaug.augmenters as iaa
import numpy as np

augmentation = iaa.Sequential([iaa.Rotate((-170, 170))])


class AWEDataset(object):

    def __init__(self, path):
        super().__init__()
        images = {}
        classes = []
        for d in os.listdir(path):
            for lr in ['L', 'R']:
                c = '{}_{}'.format(d, lr)
                classes.append(c)
                dir = os.path.join(path, d, lr)
                for f in glob.glob(dir + '/*.png'):
                    if c not in images:
                        images[c] = []
                    images[c].append({
                        "subject": d,
                        "lr": lr,
                        "src": f,
                        "mask": f[:-4] + '.npy',
                        "class": c
                    })
        self.images = [images[k] for k in images]
        self.classes = classes

    def get_epoch(self, size, image_size, mask=True):
        img = []

        all = [x for y in self.images for x in y]
        random.shuffle(all)

        i = 0
        loop = 0
        while True:
            if i >= len(all):
                i = 0
                if len(img) == loop:
                    break
                loop = len(img)

            im = all[i]
            if im:
                for j in range(len(all) - i - 1):
                    im2 = all[j + i + 1]
                    if im2 and im2['class'] == im['class']:
                        all[i] = False
                        all[j + i] = False
                        img.append([
                            im['src'],
                            im2['src'],
                            1
                        ])
                        break
            i += 1

        while True:
            if i >= len(all):
                i = 0
                if len(img) == loop:
                    break
                loop = len(img)

            im = all[i]
            if im:
                for j in range(len(all) - i - 1):
                    im2 = all[j + i + 1]
                    if im2 and im2['class'] != im['class']:
                        all[i] = False
                        all[j + i + 1] = False
                        img.append([
                            im['src'],
                            im2['src'],
                            0
                        ])
                        break
            i += 1

        random.shuffle(img)
        n = math.ceil(len(img) / size)
        ds = []
        for i in range(n):
            im = img[i * size:min(len(img), (i + 1) * size)]
            is_same = [x[2] for x in im]
            left = [load_img(x[0], image_size, mask) for x in im]
            right = [load_img(x[1], image_size, mask) for x in im]
            ds.append([
                np.array(left),
                np.array(right),
                np.reshape(np.array(is_same), (-1, 1))
            ])

        return ds


def load_img(path, image_size, mask=True):
    image = cv2.imread(path)
    if not mask:
        return cv2.resize(image, dsize=(image_size, image_size))
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
    out = cv2.resize(out, dsize=(image_size, image_size))
    #cv2.imwrite('verify.png', out)
    return out
