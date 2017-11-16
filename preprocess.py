# -*- coding: utf-8 -*-

import os
import random
import pickle
import urllib.request
import numpy as np

from scipy.misc import imread, imsave


idols = [
    'Hanayo',
    'Honoka',
    'Nico',
    'Kotori',
    'Umi',
    'Eli',
    'Rin',
    'Maki',
    'Nozomi',
    'Kanan',
    'Ruby',
    'Hanamaru',
    'Yoshiko',
    'Riko',
    'You',
    'Chika',
    'Mari',
    'Dia',
]


def download():
    print("downloading images...")
    if not os.path.isdir('images'):
        os.mkdir('images')
    step = 0
    with open('data/urls.txt', mode='r') as f:
        for line in f:
            step += 1
            print(f"working on {step}th line", end='\r')
            url = line.strip()
            if not url:
                continue
            filename = url.split('/')[-5]
            filepath = os.path.join('images', filename)
            if not os.path.isfile(filepath):
                urllib.request.urlretrieve(url, filepath)
    print("\ndone")


def preprocess(cut=0.95):
    print("creating dataset...")
    if not os.path.isdir('faces'):
        os.mkdir('faces')
    root = 'images'
    filenames = os.listdir(root)
    data = []
    step = 0
    for filename in filenames:
        step += 1
        print(f"working on {step}th line", end='\r')
        if filename.split('.')[-1] != 'png':
            continue
        tokens = filename.split('_')
        freeze = False
        for token in tokens:
            for label, idol in enumerate(idols):
                if token == idol:
                    freeze = True
                    break
            if freeze:
                break
        if not freeze:
            continue
        image = imread(os.path.join(root, filename))[13:77, 35:99, :3]
        imsave('faces/{:04d}_{}.jpg'.format(len(data), idol), image)
        image = np.rollaxis(image / 255, 2)
        hot = np.zeros(len(idols))
        hot[label] = 1
        data.append((image, hot))
    pickle.dump(data, open('data/all.dat', 'wb'))
    random.shuffle(data)
    n = int(len(data) * cut)
    pickle.dump(data[:n], open('data/train.dat', 'wb'))
    pickle.dump(data[n:], open('data/test.dat', 'wb'))
    print("\ndone")
    print("total {} idols".format(len(data)))


if __name__ == '__main__':
    download()
    preprocess()
