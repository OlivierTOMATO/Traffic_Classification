# TWO layer CNN model.
# number filters, input_size, hidden_size remains to be set
# conv1d + BN + RELU + MaxPooling + conv1d + BN + RELU + MaxPooling + Linear + RELU + Linear
# refer my report to see the model structure

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchsummary import summary


class cnn_model(nn.Module):
    def __init__(self, num_filters, input_size, hidden_size):
        super(cnn_model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layer1 = nn.Sequential(
            nn.Conv1d(self.input_size, num_filters, kernel_size=5),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(num_filters, hidden_size, kernel_size=5),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.fc = nn.Sequential(
            # nn.Linear(512, 128),
            nn.Linear(hidden_size * 9, 32),
            nn.ReLU(),
            nn.Linear(32, 3))

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.layer1(x)
        out = self.layer2(out)
        # out = self.layer3(out)
        # out = torch.max(out, dim=-1)[0]
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

#
# summary(cnn_model(num_filters=64, hidden_size=128, input_size=4).to("cuda:0"), input_size=(50, 4), batch_size=20)
