import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.data import Data

import torch.optim as optim
import torch_geometric.transforms as T
import argparse
import scipy.sparse as sp
import numpy as np
import pandas as pd
import math
import pickle as pkl
import networkx as nx
import time
import matplotlib.pyplot as plt
import random
from osgeo import gdal
from models import GAT, GCN
start = time.time()
NUM_CLASSES = 2

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)



def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()

    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

# define a function to compute the accuracy of models
def accuracy_calculate(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct/len(labels)

# define a function to compute the adjacency matrix
def adj_calculate(graph):
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    adj = sparse_to_tuple(adj)
    adj = adj[0]
    adj = adj.transpose()
    adj = torch.from_numpy(adj).long()
    return adj

# read data
data = pd.read_csv('F:/GCN/point_coo.csv')
features = torch.from_numpy(data.drop(['FID', 'POINT_X', 'POINT_Y', 'au'], axis=1).values).float()  # Evidence layer data
labels = torch.tensor([data['au']], dtype=torch.long).squeeze()  # labels

# Randomly select training and validation data
def random_select(data):
    au = torch.tensor(data[data['au'] == 1].FID.values)
    noau = torch.tensor(data[data['au'] == 0].FID.values)
    rate = 0.2
    S = round(rate * len(au))
    idx_test_au = torch.LongTensor(random.sample(range(au.size(0)), S))
    idx_test_noau = torch.LongTensor(random.sample(range(noau.size(0)), S))
    test_au = torch.sort(idx_test_au).values
    test_noau = torch.sort(idx_test_noau).values
    test_au = torch.index_select(au, 0, test_au)
    test_noau = torch.index_select(noau, 0, test_noau)
    idx_test = torch.LongTensor(torch.cat([test_au, test_noau], dim=0))

    def del_tensor(input, delete):
        input_list = input.numpy().tolist()
        for i in range(len(delete)):
            input_list.remove(delete[i].item())

        return torch.Tensor(input_list)

    train_au = del_tensor(au, test_au)
    train_noau = del_tensor(noau, test_noau)
    idx_train = torch.cat([train_au, train_noau], dim=0).type(torch.long)

    return idx_train, idx_test

idx = random_select(data)

# input graph
rf1 = open('graph_150.pkl', 'rb')
# compute the adjacency matrix
graph = pkl.load(rf1)
edge_index = adj_calculate(graph)

# GAT
model = GAT(nfeat=features.shape[1],
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
# GCN
# model = GCN(nfeat=features.shape[1],
#             nclass=labels.max().item() + 1,
#             dropout=args.dropout)
print(model)


optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
device = torch.device('cpu')
model.to(device)
labels = labels.to(device)
idx_train = idx[0].to(device)
idx_test = idx[1].to(device)
features = features.to(device)
adj = edge_index.to(device)


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy_calculate(output[idx_train], labels[idx_train])
    loss_train.backward()
    loss_train = float(np.array(loss_train.detach().numpy()))
    acc_train = float(np.array(acc_train.detach().numpy()))

    optimizer.step()

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy_calculate(output[idx_test], labels[idx_test])
    loss_test = float(np.array(loss_test.detach().numpy()))
    acc_test = float(np.array(acc_test.detach().numpy()))


    print('Epoch: {:04d}'.format(epoch),
          'loss_train: {:.4f}'.format(loss_train),
          'acc_train: {:.4f}'.format(acc_train),
          'loss_test: {:.4f}'.format(loss_test),
          'acc_test: {:.4f}'.format(acc_test),
          'time: {:.4f}s'.format(time.time() - t))
    return loss_train, acc_train, loss_test, acc_test


t_total = time.time()

loss_list_train = []
loss_list_test = []
accuracy_train=[]
accuracy_test=[]

max_test = 0
epoch_best = 0
for epoch in range(1, args.epochs+1):
    train_result = train(epoch)
    loss_list_train.append(train_result[0])
    accuracy_train.append(train_result[1])

    loss_list_test.append(train_result[2])
    accuracy_test.append(train_result[3])

    if (train_result[3] >= max_test):
        max_test = train_result[3]
        model_path = "model_best.pth"
        torch.save(model, model_path)
        epoch_best = epoch

    if (epoch % 100 == 0):
        a = str(epoch)
        model_out_path = "model_" + a + ".pth"
        torch.save(model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# # loss curve
epochs = range(len(loss_list_train))
plt.plot(epochs, loss_list_train, 'b', label='Loss Value')
plt.plot(epochs, accuracy_train, 'r', label='Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title("Train")
plt.legend(loc='best')
plt.show()

###################################
# predict
probability_value = []
model = torch.load("F:/GAT/model_best.pth")
print("Predict:")
output = model(features, adj)
probability = nn.functional.softmax(output, dim=1)
probab = probability.detach().numpy()
probability_value.append(probab[:, 1])
result = probability_value[0].reshape(243, 171)

# # Get coordinate and projection information
Baguio = gdal.Open('F:/Baguio/GAUGE100/composite.tif')  # Read a TIF image from the study area
projection = Baguio.GetProjection()
transform = Baguio.GetGeoTransform()
driver = gdal.GetDriverByName('GTiff')

# # Write the result to the tif
dst_ds = driver.Create('F:/GAT/Result_6heads_1.2/GAT6h.tif', 171, 243, 1, gdal.GDT_Float64)
dst_ds.SetGeoTransform(transform)
dst_ds.SetProjection(projection)
dst_ds.GetRasterBand(1).WriteArray(result)
dst_ds.FlushCache()
dst_ds = None