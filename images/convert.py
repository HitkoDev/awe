import csv
import os
import json
from os import path
import cv2
import shutil
import numpy as np
import skimage.io

images = {}

with open('awe-translation.csv', 'r') as f:
    reader = csv.reader(f)
    c = [x for x in reader][1:]
    images = {x[1]: {"src": x[0], "subject": int(x[2])} for x in c}

map = {}
for x in os.listdir('AWEDataset'):
    if os.path.isdir(os.path.join('AWEDataset', x)):
        with open(os.path.join('AWEDataset', x, 'annotations.json'), 'r') as f:
            d = json.load(f)
            for k in d['data']:
                i = d['data'][k]
                m = "{}/{}".format(x, i['file'])
                s = images[m]
                lr = i['d'].upper()
                if 'test' in s['src']:
                    dir = 'test'
                else:
                    dir = 'train'
                fn = s['src'][-8:-4]
                target = os.path.join('converted', dir, x, lr, fn)
                path = os.path.join('AWEForSegmentation', s['src'])
                if not os.path.exists(os.path.dirname(target)):
                    os.makedirs(os.path.dirname(target))

                img = cv2.imread(path)
                mask = cv2.imread(path.replace(dir, '{}annot'.format(dir)))
                bb = cv2.imread(path.replace(dir, '{}annot_rect'.format(dir)))
                w, h, c = img.shape

                gray = cv2.cvtColor(bb, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
                contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = contours[0] if len(contours) == 2 else contours[1]
                mask_path = target + '.npy'
                c = contours[0]
                if len(contours) > 1:
                    if lr == 'L':
                        for cc in contours[1:]:
                            x1 = min([p[0][0] for p in c if p[0][0] >= 0])
                            x2 = min([p[0][0] for p in cc if p[0][0] >= 0])
                            if x1 < x2:
                                c = cc
                    else:
                        for cc in contours[1:]:
                            x1 = min([p[0][0] for p in c if p[0][0] >= 0])
                            x2 = min([p[0][0] for p in cc if p[0][0] >= 0])
                            if x1 > x2:
                                c = cc

                m = np.full((w, h), False)
                y1 = min([p[0][1] for p in c if p[0][1] >= 0])
                y2 = max([p[0][1] for p in c if p[0][1] >= 0]) + 1
                x1 = min([p[0][0] for p in c if p[0][0] >= 0])
                x2 = max([p[0][0] for p in c if p[0][0] >= 0]) + 1
                s = mask[y1:y2, x1:x2, 1] > 0
                m[y1:y2, x1:x2] = s
                if dir == 'train':
                    shutil.copy(path, target + '.png')
                    with open(mask_path, 'wb+') as file:
                        np.save(file, m)
                else:
                    mask_out = img * np.stack([m, m, m], axis=2)
                    out = mask_out[y1:y2, x1:x2]
                    cv2.imwrite(target + '.png', out)
