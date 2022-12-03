import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn import GATConv as GATnn

'''
The GATConv of DGL is a Graph Attention Network. In the following class we use Torch modules for GAT
For Graph attention layer we took reference from Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>
'''

class GAT(nn.Module):

    def __init__(self,num_layers,in_dim,num_hidden,num_classes,heads,activation,feat_drop=0,attn_drop=0,negative_slope=0.2,residual=False):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        # input projection
        self.gat_layers.append(GATnn(in_dim, num_hidden, heads[0],feat_drop, attn_drop, negative_slope, False, self.activation))
        
        # hidden layers of GAT
        for l in range(1, num_layers):
            # because of multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATnn(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        
        # output of attention network
        self.gat_layers.append(GATnn(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))
    
    '''The forward method is used to compute graph attention network layer.'''
    def forward(self, inputs, g):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
            
        # output 
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits