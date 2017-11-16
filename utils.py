# -*- coding: utf-8 -*-

import pickle
import torch.nn as nn

from torch.utils.data import Dataset


class SchoolIdolFestival(Dataset):

    def __init__(self, target='train'):
        self.data = pickle.load(open(f'data/{target}.dat', 'rb'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 1e-2)
            m.bias.data.zero_()
