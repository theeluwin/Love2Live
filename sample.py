# -*- coding: utf-8 -*-

try:
    import cv2
    import torch
except:
    pass

import os
import pickle
import torch
import numpy as np

from torch import FloatTensor as FT
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
from scipy.misc import imsave
from model import Love2Live
from preprocess import idols
from utils import SchoolIdolFestival


def denormalize(x):
    x = np.rollaxis(x, 2)
    x = np.rollaxis(x, 2)
    return x * 255


def ae(model, target='test', gpu=False):
    if not os.path.isdir('results'):
        os.mkdir('results')
    if not os.path.isdir(f'results/ae_{target}'):
        os.mkdir(f'results/ae_{target}')
    dataset = SchoolIdolFestival(target)
    batch_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    image, hot = next(iter(dataloader))
    labels = [h.numpy().argmax() for h in hot]
    x = V(image.float(), requires_grad=False)
    c = V(hot.float(), requires_grad=False)
    x = x.cuda() if gpu else x
    c = c.cuda() if gpu else c
    mu, ls = model.encoder(x, c)
    z = model.sample(mu, ls)
    x_ = model.decoder(z, c)
    x_ = x_.cpu() if gpu else x_
    for i in range(batch_size):
        imsave('results/ae_{}/{}_{}.jpg'.format(target, i + 1, idols[labels[i]]), denormalize(x_.data[i].numpy()))


def random(model, gpu=False, num_samples=20):
    if not os.path.isdir('results'):
        os.mkdir('results')
    if not os.path.isdir('results/random'):
        os.mkdir('results/random')
    for i, idol in enumerate(idols):
        hot = np.zeros((num_samples, len(idols)))
        hot[:, i] = 1
        c = V(FT(hot), requires_grad=False)
        c = c.cuda() if gpu else c
        x_ = model.predict(c)
        x_ = x_.cpu() if gpu else x_
        for j in range(num_samples):
            imsave('results/random/{}_{}.jpg'.format(idol, j + 1), denormalize(x_.data[j].numpy()))


def interpolate(model, gpu=False, num_samples=20, num_lim=10):
    if not os.path.isdir('results'):
        os.mkdir('results')
    if not os.path.isdir('results/interpolate'):
        os.mkdir('results/interpolate')
    dataset = SchoolIdolFestival('all')
    idx = lambda idol: idols.index(idol)
    pairs = [
        (idx('Yoshiko'), idx('Riko')),
        (idx('Maki'), idx('Nico')),
        (idx('Ruby'), idx('Dia')),
        (idx('Hanayo'), idx('Rin')),
        (idx('Maki'), idx('Yoshiko')),
    ]
    for source, target in pairs:
        for i in range(num_samples):
            hot = np.zeros((num_lim + 1, len(idols)))
            for lim in range(num_lim + 1):
                hot[lim, source] = 1 - 0.1 * lim
                hot[lim, target] = 0.1 * lim
            c = V(FT(hot), requires_grad=False)
            c = c.cuda() if gpu else c
            mu = V(torch.zeros(1, 1024), requires_grad=False)
            ls = V(torch.zeros(1, 1024), requires_grad=False)
            mu = mu.cuda() if gpu else mu
            ls = ls.cuda() if gpu else ls
            z = model.sample(mu, ls).repeat(num_lim + 1, 1)
            x_ = model.decoder(z, c)
            x_ = x_.cpu() if gpu else x_
            canvas = np.zeros((64, 64 * (num_lim + 1), 3))
            for lim in range(num_lim + 1):
                canvas[:, lim * 64: (lim + 1) * 64, :] = denormalize(x_.data[lim].numpy())
            imsave('results/interpolate/{}_{}_{}.jpg'.format(idols[source], idols[target], i), canvas)


if __name__ == '__main__':
    gpu = True
    name = 'love2live.{}'.format('gpu' if gpu else 'cpu')
    model = Love2Live(gpu=gpu)
    model.load_state_dict(torch.load(f'pts/{name}.pt'))
    model.eval()
    ae(model, 'train', gpu=gpu)
    ae(model, 'test', gpu=gpu)
    random(model, gpu=gpu)
    interpolate(model, gpu=gpu)
