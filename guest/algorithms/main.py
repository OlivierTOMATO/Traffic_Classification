# experiments of CNN RNN and RNN-CNN MODEL over the generated NPZ data.

from pathlib import Path
import torch
from sklearn import metrics
from sklearn.metrics import precision_score, multilabel_confusion_matrix
from tqdm import tqdm

# from Data.data_pro import data_pro
from Data.dataloader import trafficDataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
import numpy as np

from model.cnn_classifier import cnn_model
from model.rnn_classifier import rnn_model
from model.rnn_cnn import rnn_cnn

# choose device to run the model, where GPU is preferred
torch.backends.cudnn.enabled = False
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print('Loading data...')
state = np.random.get_state()

# use dataloader defined in Data/dataloader to read the data in.
# use Dataloader to load the read data, define batch_size here, which is important to the outcome.
# trainDataset = trafficDataset(path='./Data/test_sample(time)', batches=32, device='cpu', state=state, split='train')
# testDataset = trafficDataset(path='./Data/test_sample(time)', batches=32, device='cpu', state=state, split='test')
trainDataset = trafficDataset(path='./Data/multiple_traffic', batches=32, device='cpu', state=state, split='train')
testDataset = trafficDataset(path='./Data/multiple_traffic', batches=32, device='cpu', state=state, split='test')
trainDataloader = DataLoader(trainDataset, 64, shuffle=True)
testDataloader = DataLoader(testDataset, 64, shuffle=False)

# define the model to be trained here, push it into the device
model = cnn_model(num_filters=32, input_size=4, hidden_size=256).to(device)
# model = rnn_cnn(num_filters=64, num_layers=2, batch_size=32, device=device, hidden_size=128).to(device)
# model = rnn_model(num_filters=64, num_layers=2, device=device).to(device)

# Adam optimizer, learning rate set initially 0.01 and exponentially descend with gamma 0.95-0.96.
# define the Loss function, BCELoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
criterion = torch.nn.BCELoss()

num_classes = 3


def calculate_metrics(pred, target, threshold=0.5):
    pred = np.array(pred > threshold, dtype=float)
    # t = classification_report(target, pred, target_names=['HTTP', 'VOIP', 'RTP'])
    Accuracy_list.append(accuracy_score(target, pred))
    print('loss: {}, accuracy: {}, hamming_loss: {}, precision_score: {}'.format(train_loss_sum / len(trainDataloader),
                                                                                 accuracy_score(target, pred),
                                                                                 hamming_loss(target, pred),
                                                                                 precision_score(target, pred,
                                                                                                 average='samples')))
    print(metrics.classification_report(pred, target, digits=3))
    mcm = multilabel_confusion_matrix(target, pred)
    print(mcm)

    # show confusion metrix
    # plt.imshow(mcm, cmap=plt.cm.Blues)
    # indices = range(len(mcm))
    # classes = [1, 2, 3]
    # plt.xticks(indices, classes)
    # plt.yticks(indices, classes)
    # plt.colorbar()
    # plt.xlabel('guess')
    # plt.ylabel('fact')
    # plt.title("cnn")
    # print(mcm[1, 0])
    # for first_index in range(len(mcm)):
    #     for second_index in range(len(mcm[first_index])):
    #         plt.text(second_index, first_index, mcm[first_index, second_index])
    # plt.show()


num = 30
Loss_list = []
Accuracy_list = []
Loss_list_1 = []
Accuracy_list_1 = []


# training process
for epoch in range(num):
    model.train()
    train_loss_sum = 0
    accuracy = 0
    model_result = []
    targets = []
    for i, batch in tqdm(enumerate(trainDataloader), total=len(trainDataloader), smoothing=0.9, leave=False):
        # push the data to device and do prediction
        x, y = batch
        y_label = y.numpy()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        output_sig = torch.sigmoid(pred)
        # use loss function to backward the hyperparameter and optimizer the model.
        loss = criterion(output_sig, y)
        loss.backward()
        train_loss_sum += loss.detach().cpu()
        optimizer.step()
        model_result.extend(output_sig.detach().cpu().numpy())
        targets.extend(y_label)
    Loss_list_1.append(np.array(train_loss_sum / len(trainDataloader)))
    k = np.array(model_result)
    k = np.array(k > 0.5, dtype=float)
    Accuracy_list_1.append(accuracy_score(np.array(targets), k))

    # After one round of training, do testing with the left 20% data.
    test_loss_sum = 0
    model.eval()
    with torch.no_grad():
        model_result = []
        targets = []
        for i, batch in tqdm(enumerate(testDataloader), total=len(testDataloader), smoothing=0.9, leave=False):
            x, y = batch
            y_label = y.numpy()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            # y_label = torch.max(y, 1)[1].detach().cpu().numpy()
            output_sig = torch.sigmoid(pred)
            loss = criterion(output_sig, y)
            test_loss_sum += loss.detach().cpu()
            model_result.extend(output_sig.detach().cpu().numpy())
            targets.extend(y_label)
        print("test:")
        Loss_list.append(np.array(test_loss_sum / len(testDataloader)))
        # output the testing result in the console.
        calculate_metrics(np.array(model_result), np.array(targets))

    ExpLR.step()

# code for drawing the convergence curve
# x1 = range(0, num)
# x2 = range(0, num)
# y1 = Accuracy_list
# y2 = Loss_list
# y3 = Accuracy_list_1
# y4 = Loss_list_1
# # plt.subplot(2, 1, 1)
# plt.plot(x1, y1, '--', label='test')
# plt.plot(x1, y3, '-', label='train')
# # plt.ylim((0, 1))
# plt.title('LSTM+CNN')
# plt.ylabel('accuracy')
# plt.xlabel('epoches')
# plt.legend()
# plt.show()
# plt.plot(x2, y2, '-', label='test')
# plt.plot(x2, y4, '-', label='train')
# plt.title('LSTM+CNN')
# plt.xlabel('epoches')
# plt.ylabel('loss')
# plt.legend()
# plt.show()
# plt.savefig("accuracy_loss.jpg")
