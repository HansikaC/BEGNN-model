import torch as th
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .gnn import GNN
from .gat import GAT

'''
Here we create Bert classifier, BertGNN and Bert GAT.

In Bert Classifier we tokenize from a pretrained model, here we are using 'roberta_base';
In Bert GNN we are training both BERT and GNN models
IN Bert GAT we are combining both BERT and GNNs features that are extracted and we use softmax at the end
'''

class BertClassifier(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit


class BertGNN(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gnn_layers=2, n_hidden=200, dropout=0.5):
        super(BertGNN, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gnn = GNN(in_feats=self.feat_dim,n_hidden=n_hidden,n_classes=nb_class,n_layers=gnn_layers-1,activation=F.elu,dropout=dropout)

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gnn_logit = self.gnn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        gnn_pred = th.nn.Softmax(dim=1)(gnn_logit)
        pred = (gnn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred
    
class BertGAT(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gnn_layers=2, heads=8, n_hidden=32, dropout=0.5):
        super(BertGAT, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gnn = GAT(num_layers=gnn_layers-1,in_dim=self.feat_dim,num_hidden=n_hidden,num_classes=nb_class,heads=[heads] * (gnn_layers-1) + [1],activation=F.elu,feat_drop=dropout,attn_drop=dropout)

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gnn_logit = self.gnn(g.ndata['cls_feats'], g)[idx]
        gnn_pred = th.nn.Softmax(dim=1)(gnn_logit)
        pred = (gnn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred
