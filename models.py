import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GAT(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(GAT, self).__init__()

        self.gat1 = GATConv(nfeat, 8, heads=6, concat=False)
        self.gat2 = GATConv(8, 16, heads=6, concat=False)
        self.gat3 = GATConv(16, 8, heads=1, concat=False)
        self.gat4 = GATConv(8, nclass, heads=1, concat=False)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gat1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gat2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gat3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat4(x, adj)
        return F.log_softmax(x, dim=1)




class GCN(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(GCN, self).__init__()

        self.gcn1 = GCNConv(nfeat, 8)
        self.gcn2 = GCNConv(8, 16)
        self.gcn3 = GCNConv(16, 8)
        self.gcn4 = GCNConv(8, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gcn2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gcn3(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn4(x, adj)
        return F.log_softmax(x, dim=1)
