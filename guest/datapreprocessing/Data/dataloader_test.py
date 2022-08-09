import numpy as np
import pandas as pd
import sklearn.model_selection
import os
import torch
import random
from torch.utils.data import Dataset


class trafficDataset(Dataset):
    def __init__(self, path, batches, device, state=None):
        self.path = path
        self.batches = batches
        self.device = device
        self.state = state
        self.index = os.listdir(self.path)

    def __getitem__(self, item):
        data = np.load(self.path + '/' + self.index[item])
        x = data['a']
        label = data['b']
        x = torch.tensor(x, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return x, label

    def __len__(self):
        return len(self.index)
