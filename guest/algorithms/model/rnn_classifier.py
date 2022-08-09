# TWO layer RNN model.
# input size should be changed according to real input
# 2-layers LSTM + Linear + RELU + Linear
# refer my report to see the model structure

import torch
import torch.nn as nn
import numpy as np

from torchsummary import summary


class rnn_model(nn.Module):
    def __init__(self, num_filters, num_layers, device='cpu'):
        super(rnn_model, self).__init__()
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.device = device
        # self.layer1 = nn.Sequential(
        #     nn.LSTM(3, num_filters, self.num_layers, bias=True, bidirectional=True, batch_first=True, dropout=0.4),
        #     nn.BatchNorm1d(num_filters),
        #     nn.ReLU()
        # self.layer1 = nn.Sequential(
        #     nn.Conv1d(4, num_filters, kernel_size=10),
        #     nn.BatchNorm1d(num_filters),
        #     nn.ReLU(),
        #     # nn.MaxPool1d(kernel_size=2, stride=2)
        # )
        self.layer2 = nn.LSTM(4, num_filters, self.num_layers, bias=True)
        self.fc = nn.Sequential(
            nn.Linear(num_filters, 32),
            nn.ReLU(),
            nn.Linear(32, 3))

    def forward(self, x):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        # x = x.transpose(1, 2)
        # out = self.layer1(x)
        # out = out.transpose(1, 2)
        out = x
        h0 = torch.zeros(self.num_layers, out.size(1), self.num_filters).to(self.device)
        c0 = torch.zeros(self.num_layers, out.size(1), self.num_filters).to(self.device)
        out, (h0, c0) = self.layer2(out, (h0, c0))
        # out = out.transpose(1, 2)
        # noting: use max/ mean here, instead of the last output to get better result
        out = torch.max(out, dim=1)[0]
        # out = torch.mean(out, dim=1)
        # out = out[:, -1, :]
        out = self.fc(out)
        return out

# summary(cnn_rnn_model(num_filters=64, num_layers=2).to('cuda:2'), input_size=(50, 4), batch_size=20)

