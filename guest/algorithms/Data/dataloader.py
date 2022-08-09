# dataloader for NNs' training and testing.


import numpy as np
import pandas as pd
import sklearn.model_selection
import os
import torch
import random
from torch.utils.data import Dataset


class trafficDataset(Dataset):

    # get all npz data from the path
    # shuffle the data randomly to make the datapoints disorder
    # pick the first 80% for training and left 20% for testing
    def __init__(self, path, batches, device, state=None, split='train'):
        self.path = path
        self.batches = batches
        self.device = device
        self.split = split
        self.state = state
        self.index = os.listdir(self.path)
        np.random.set_state(self.state)
        np.random.shuffle(self.index)
        self.train = self.index[0: int(0.8 * len(self.index))]
        self.test = self.index[int(0.8 * len(self.index)):]

    # data['a'] is the input data, data['b'] is the labels.
    # if the labels is stored in 1, 2, 3. remember to modify them to [1, 0, 0], [0, 1, 0]...
    def __getitem__(self, item):
        if self.split == 'train':
            data = np.load(self.path + '/' + self.train[item], allow_pickle=True)
            x = data['a']
            label = data['b']
            # if label == 1:
            #     label = [1, 0, 0]
            # elif label == 2:
            #     label = [0, 1, 0]
            # elif label == 3:
            #     label = [0, 0, 1]
            # elif label == 4:
            #     label = [1, 1, 0]
            # elif label == 5:
            #     label = [1, 0, 1]
            # elif label == 6:
            #     label = [0, 1, 1]

        else:
            data = np.load(self.path + '/' + self.test[item], allow_pickle=True)
            x = data['a']
            label = data['b']
            # if label == 1:
            #     label = [1, 0, 0]
            # elif label == 2:
            #     label = [0, 1, 0]
            # elif label == 3:
            #     label = [0, 0, 1]
            # elif label == 4:
            #     label = [1, 1, 0]
            # elif label == 5:
            #     label = [1, 0, 1]
            # elif label == 6:
            #     label = [0, 1, 1]

        # convert data to torch format
        x = torch.tensor(x, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return x, label

    def __len__(self):
        return len(self.train) if self.split == 'train' else len(self.test)
