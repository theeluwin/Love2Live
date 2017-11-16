# -*- coding: utf-8 -*-

try:
    import cv2
    import torch
except:
    pass

import os
import time
import torch
import pickle

from torch.nn import BCELoss
from torch.optim import Adam
from torch.autograd import Variable as V
from torch.utils.data import DataLoader
from model import Love2Live
from preprocess import idols
from utils import SchoolIdolFestival


def train(gpu=False, num_epochs=500):
    name = 'love2live.{}'.format('gpu' if gpu else 'cpu')
    model = Love2Live(gpu=gpu)
    if not os.path.isdir('pts'):
        os.mkdir('pts')
    print("loading a pre-trained model...")
    modelpath = f'pts/{name}.pt'
    optimpath = f'pts/{name}.optim.pt'
    if os.path.isfile(modelpath):
        model.load_state_dict(torch.load(modelpath))
        print("successfully loaded the pre-trained model")
    else:
        print("no pre-trained model found")
    model.train()
    optim = Adam(model.parameters())
    if os.path.isfile(optimpath):
        optim.load_state_dict(torch.load(optimpath))
    bce = BCELoss(size_average=False)
    dataset = SchoolIdolFestival('train')
    batch_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    start = time.time()
    total_batches = len(dataset) // batch_size
    for epoch in range(1, num_epochs + 1):
        lost = []
        for batch, (image, hot) in enumerate(dataloader):
            x = V(image.float(), requires_grad=False)
            c = V(hot.float(), requires_grad=False)
            x = x.cuda() if gpu else x
            c = c.cuda() if gpu else c
            mu, ls = model.encoder(x, c)
            z = model.sample(mu, ls)
            x_ = model.decoder(z, c)
            rc_loss = bce(x_, x) / 64 / 64 / 3
            kl_loss = 0.5 * (ls.exp() + mu.pow(2) - 1 - ls).mean(1).sum()
            loss = (rc_loss + kl_loss) / batch_size
            optim.zero_grad()
            loss.backward()
            optim.step()
            lost.append(loss.data[0])
        print("[epoch {:4d}] loss: {:9.6f}\r".format(epoch, sum(lost) / len(lost)))
    end = time.time()
    print("training done in {:.2f} seconds".format(end - start))
    torch.save(model.state_dict(), modelpath)
    torch.save(optim.state_dict(), optimpath)


if __name__ == '__main__':
    train(gpu=True)
