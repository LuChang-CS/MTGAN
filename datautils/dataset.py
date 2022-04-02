import os

import torch
import numpy as np


class Dataset:
    def __init__(self, inputs, device):
        self.data = inputs
        self.device = device

        self.size = len(inputs[0])

    def __len__(self):
        return self.size

    def __getitem__(self, indices):
        data = [torch.tensor(x[indices], device=self.device) for x in self.data]
        return data


class DatasetReal:
    def __init__(self, path, device=None):
        self.path = path
        self.device = device
        print('loading real data ...')
        print('\tloading real training data ...')
        self.train_set = self._load('train.npz')
        print('\tloading real test data ...')
        self.test_set = self._load('test.npz')

    def _load(self, filename):
        data = np.load(os.path.join(self.path, filename))
        x, lens = data['x'].astype(np.float32), data['lens'].astype(np.int64)
        dataset = Dataset((x, lens), self.device)
        return dataset


class DatasetRealNext:
    def __init__(self, path, device=None):
        self.path = path
        self.device = device
        print('loading real next data ...')
        print('\tloading real next training data ...')
        self.train_set = self._load('train.npz')

    def _load(self, filename):
        data = np.load(os.path.join(self.path, filename))
        x, lens, y = data['x'].astype(np.float32), data['lens'].astype(np.int64), data['y'].astype(np.float32)
        dataset = Dataset((x, lens, y), self.device)
        return dataset
