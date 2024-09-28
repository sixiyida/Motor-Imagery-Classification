import mne
import matplotlib.pyplot as pl
import numpy as np
import os

filename = "dataset/A03T.gdf"
raw = mne.io.read_raw_gdf(filename)

print(raw.info)
print(raw.ch_names)


# Find the events time positions


# Pre-load the data
raw.load_data()

# Filter the raw signal with a band pass filter in 7-35 Hz
raw.filter(7., 35., fir_design='firwin')

# Remove the EOG channels and pick only desired EEG channels
raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                       exclude='bads')

events, events_dict = mne.events_from_annotations(raw)

print('Number of events:',len(events))
print(events_dict)
print(events)

tmin, tmax = 1., 4.

event_id = dict({'769': 7,'770': 8,'771': 9,'772': 10})
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)



# Getting labels and changing labels from 7,8,9,10 -> 1,2,3,4
labels = epochs.events[:,-1] - 7

data = epochs.get_data()

print(labels)
print(data.shape)

import pywt


# signal is decomposed to level 5 with 'db4' wavelet

def wpd(X):
    coeffs = pywt.WaveletPacket(X, 'db4', mode='symmetric', maxlevel=5)
    return coeffs


def feature_bands(x):
    Bands = np.empty((8, x.shape[0], x.shape[1], 30))  # 8 freq band coefficients are chosen from the range 4-32Hz

    for i in range(x.shape[0]):
        for ii in range(x.shape[1]):
            pos = []
            C = wpd(x[i, ii, :])
            pos = np.append(pos, [node.path for node in C.get_level(5, 'natural')])

            for b in range(1, 9):
                Bands[b - 1, i, ii, :] = C[pos[b]].data

    return Bands


wpd_data = feature_bands(data)

import matplotlib.pyplot as plt
from mne.decoding import CSP # Common Spatial Pattern Filtering
from sklearn import preprocessing
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ShuffleSplit

# Cross Validation Split 交叉验证拆分
cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)

from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

acc = []
ka = []
prec = []
recall = []



class Classifier(nn.Module):
    def __init__(self, num_layers=1):
        super(Classifier, self).__init__()
        self.dropout_rate = 0.5
        self.l2_regularization = 0.01

        # First Layer
        self.fc1 = nn.Linear(32, 124)
        self.fc1.weight.data.uniform_(0, 1)  # kernel_initializer
        self.fc1.bias.data.zero_()
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout_rate)

        # Intermediate Layers
        self.layers = nn.ModuleList()
        for itr in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(124, 124),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            )
            self.layers.append(layer)

        # Last Layer
        self.fc2 = nn.Linear(124, 4)
        self.fc2.weight.data.uniform_(0, 1)
        self.fc2.bias.data.zero_()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # First Layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Intermediate Layers
        for layer in self.layers:
            x = layer(x)

        # Last Layer
        x = self.fc2(x)
        return x

print(labels.shape)

for train_idx, test_idx in cv.split(labels):
    print(train_idx,test_idx)
    Csp = []
    ss = []
    epoch_list = []
    loss_list = []

    label_train, label_test = labels[train_idx], labels[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    # CSP filter applied separately for all Frequency band coefficients

    Csp = [CSP(n_components=4, reg=None, log=True, norm_trace=False) for _ in range(8)]

    ss = preprocessing.StandardScaler()

    X_train = ss.fit_transform(
        np.concatenate(tuple(Csp[x].fit_transform(wpd_data[x, train_idx, :, :], label_train) for x in range(8)),
                       axis=-1))

    X_test = ss.transform(
        np.concatenate(tuple(Csp[x].transform(wpd_data[x, test_idx, :, :]) for x in range(8)), axis=-1))

    X_train = torch.Tensor(X_train)
    X_test = torch.Tensor(X_test)
    y_train = torch.Tensor(y_train).long()
    y_test = torch.Tensor(y_test).long()

    trainset = Data.TensorDataset(X_train,y_train)

    model = Classifier()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.01,weight_decay=0.01)

    num_epochs = 300

    #nn.fit(trainset, criterion=criterion, optimizer=optimizer, epochs=300)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, modellabels = data
           # print(modellabels)
            optimizer.zero_grad()
            outputs = model(inputs)
            #outputs_softmax=nn.functional.softmax(outputs)
            loss = criterion(outputs, modellabels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #print('loss:',loss.item())

        print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / len(trainloader)))
        epoch_list.append(epoch)
        loss_list.append(running_loss / len(trainloader))

    # plt.plot(epoch_list,loss_list)
    # plt.title("Loss-Epoch")
    # plt.ylabel("Loss")
    # plt.xlabel("Epoch")
    # plt.show()

    y_pred = nn.functional.softmax(model.forward(X_test),dim=1)

    y_pred = y_pred.detach().numpy()

    print(y_pred)

    pred = (y_pred == y_pred.max(axis=-1)[:, None]).astype(int)

    y_test = nn.functional.one_hot(y_test,num_classes=4)

    print(pred)

    y_test = y_test.detach().numpy()

    print(y_pred.shape)

    print(y_test.shape)

    acc.append(accuracy_score(y_test.argmax(axis=-1), pred.argmax(axis=-1)))
    ka.append(cohen_kappa_score(y_test.argmax(axis=-1), pred.argmax(axis=-1)))
    prec.append(precision_score(y_test.argmax(axis=-1), pred.argmax(axis=-1), average='weighted'))
    recall.append(recall_score(y_test.argmax(axis=-1), pred.argmax(axis=-1), average='weighted'))

import pandas as pd

scores = {'Accuracy':acc,'Kappa':ka,'Precision':prec,'Recall':recall}

Es = pd.DataFrame(scores)

avg = {'Accuracy':[np.mean(acc)],'Kappa':[np.mean(ka)],'Precision':[np.mean(prec)],'Recall':[np.mean(recall)]}

Avg = pd.DataFrame(avg)


T = pd.concat([Es,Avg])

T.index = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','Avg']
T.index.rename('Fold',inplace=True)

print(T)
