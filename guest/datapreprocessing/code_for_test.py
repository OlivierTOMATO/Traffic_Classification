import numpy as np
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Data.dataloader_test import trafficDataset

model_used = torch.load('model/cnn_model.pkl')

testDataset = trafficDataset(path='./Data/multiple_traffic', batches=64, device='cpu')
testDataloader = DataLoader(testDataset, 8, shuffle=False)
device = 'cuda:2'

model_used.eval()
with torch.no_grad():
    model_result = []
    targets = []
    for i, batch in tqdm(enumerate(testDataloader), total=len(testDataloader), smoothing=0.9, leave=False):
        x, y = batch
        y_label = y.numpy()
        x, y = x.to(device), y.to(device)
        pred = model_used(x)
        # y_label = torch.max(y, 1)[1].detach().cpu().numpy()
        output_sig = torch.sigmoid(pred)
        model_result.extend(output_sig.detach().cpu().numpy())
        targets.extend(y_label)

    pred = np.array(np.array(model_result) > 0.5, dtype=float)
    targets = np.array(targets)
    c = np.c_[pred, targets]
    c = c[c[:, 3].argsort()]
    print(c)