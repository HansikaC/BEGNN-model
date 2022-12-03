import torch
import torch.nn as nn
# from dgl.nn import GatedGraphConv as GraphN
from .graph_edge_weight import GraphEdgeWeight as GraphN

'''Here in this class we build the Gated GNN model.'''

class GNN(nn.Module):

    def forward(self, features, g, edge_weight):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edge_weights=edge_weight)
        return h

    def __init__(self,in_feats,n_hidden,n_classes,n_layers,activation,dropout,normalization='none'):
        super(GNN, self).__init__()
        self.layers = nn.ModuleList()

        # input layer to GNN
        self.layers.append(GraphN(in_feats, n_hidden, activation=activation, norm=normalization))

        # hidden layers of GNN
        for i in range(n_layers - 1):
            self.layers.append(GraphN(n_hidden, n_hidden, activation=activation, norm=normalization))

        # output layer of GNN
        self.layers.append(GraphN(n_hidden, n_classes, norm=normalization))
        self.dropout = nn.Dropout(p=dropout)

    