# RNN + CNN model.
# input size should be changed according to real input
# 1-layers LSTM + onv1d + BN + RELU + MaxPooling + Linear + RELU + Linear
# refer my report to see the model structure

import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F


class rnn_cnn(nn.Module):
    def __init__(self, num_filters, num_layers, batch_size, hidden_size, device):
        super(rnn_cnn, self).__init__()
        self.batch_size = batch_size
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.device = device
        self.hidden_size = hidden_size
        # self.layer1 = nn.Sequential(
        #     nn.LSTM(3, num_filters, self.num_layers, bias=True, bidirectional=True, batch_first=True, dropout=0.4),
        #     nn.BatchNorm1d(num_filters),
        #     nn.ReLU()
        # )
        self.layer1 = nn.LSTM(4, num_filters, self.num_layers, bias=True, dropout=0.4)
        self.layer2 = nn.Sequential(
            nn.Conv1d(num_filters, hidden_size, kernel_size=5),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 23, 32),
            nn.ReLU(),
            nn.Linear(32, 3))

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(1), self.num_filters).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.num_filters).to(self.device)
        out, _ = self.layer1(x, (h0, c0))
        out = out.transpose(1, 2)
        out = self.layer2(out)
        # out = torch.max(out, dim=-1)[0]
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
