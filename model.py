# -*- coding: utf-8 -*-

try:
    import cv2
    import torch
except:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable as V
from utils import initialize_weights
from preprocess import idols


class Encoder(nn.Module):

    def __init__(self, gpu=False):
        super(Encoder, self).__init__()
        self.gpu = gpu
        self.channel = 3
        self.c_dim = len(idols)
        self.width = 64
        self.height = 64
        self.flat = 512 * (self.width // 16) * (self.height // 16)
        conv_seq = [
            nn.Conv2d(self.channel + self.c_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(),
            nn.Conv2d(256, 512, 4, 2, 1),
        ]
        fc_seq = [
            nn.Linear(self.flat, 1024),
            nn.BatchNorm1d(1024),
            nn.Tanh(),
        ]
        conv_seq = [m.cuda() for m in conv_seq] if self.gpu else conv_seq
        fc_seq = [m.cuda() for m in fc_seq] if self.gpu else fc_seq
        self.conv = nn.Sequential(*conv_seq)
        self.fc = nn.Sequential(*fc_seq)
        self.mu = nn.Linear(1024, 1024)
        self.ls = nn.Linear(1024, 1024)
        self.mu = self.mu.cuda() if self.gpu else self.mu
        self.ls = self.ls.cuda() if self.gpu else self.ls
        initialize_weights(self)

    def forward(self, x, c):
        c = c.repeat(1, self.width * self.height).view(-1, self.c_dim, self.width, self.height)
        out = self.conv(torch.cat([x, c], 1))
        out = out.view(-1, self.flat)
        out = self.fc(out)
        return self.mu(out), self.ls(out)


class Decoder(nn.Module):

    def __init__(self, gpu=False):
        super(Decoder, self).__init__()
        self.gpu = gpu
        self.channel = 3
        self.c_dim = len(idols)
        self.width = 64
        self.height = 64
        self.flat = 512 * (self.width // 16) * (self.height // 16)
        fc_seq = [
            nn.Linear(1024 + self.c_dim, self.flat),
            nn.BatchNorm1d(self.flat),
            nn.Tanh(),
        ]
        deconv_seq = [
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(),
            nn.ConvTranspose2d(64, self.channel, 4, 2, 1),
            nn.Sigmoid(),
        ]
        fc_seq = [m.cuda() for m in fc_seq] if self.gpu else fc_seq
        deconv_seq = [m.cuda() for m in deconv_seq] if self.gpu else deconv_seq
        self.fc = nn.Sequential(*fc_seq)
        self.deconv = nn.Sequential(*deconv_seq)
        initialize_weights(self)

    def forward(self, z, c):
        out = self.fc(torch.cat([z, c], 1)).view(-1, 512, self.width // 16, self.height // 16)
        out = self.deconv(out)
        return out


class Love2Live(nn.Module):

    def __init__(self, gpu=False):
        super(Love2Live, self).__init__()
        self.gpu = gpu
        self.encoder = Encoder(gpu=self.gpu)
        self.decoder = Decoder(gpu=self.gpu)

    def sample(self, mu, ls):
        eps = V(torch.randn(mu.size()), requires_grad=False)
        if self.gpu:
            eps = eps.cuda()
        return mu + (ls / 2).exp() * eps

    def forward(self, x, c):
        mu, ls = self.encoder(x, c)
        z = self.sample(mu, ls)
        return self.decoder(z, c)

    def predict(self, c):
        batch_size = c.size()[0]
        mu = V(torch.zeros(batch_size, 1024), requires_grad=False)
        ls = V(torch.zeros(batch_size, 1024), requires_grad=False)
        mu = mu.cuda() if self.gpu else mu
        ls = ls.cuda() if self.gpu else ls
        z = self.sample(mu, ls)
        return self.decoder(z, c)
