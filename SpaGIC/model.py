# _*_coding : utf-8_*_
# @Time : 2023/10/29 16:48
# @Author :刘威
# @File :model
# @Project :SpaGIC

import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from layers import GraphConvolution

class SpaGIC(Module):
    def __init__(self, in_dim, out_dim, dropout=0.0, act=F.relu_):
        super(SpaGIC, self).__init__()
        self.in_dim = in_dim
        self.dim1 = 256
        self.dim2 = 64

        self.dropout = dropout
        self.act = act

        self.gcn1_en = GraphConvolution(self.in_dim, self.dim1)
        self.gcn2_en = GraphConvolution(self.dim1, self.dim2)

        self.gcn2_de = GraphConvolution(self.dim2, self.dim1)
        self.gcn1_de = GraphConvolution(self.dim1, self.in_dim)

        self.hid1_bn = torch.nn.BatchNorm1d(self.dim1)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, self.training)

        # encoder
        x = self.gcn1_en(x, adj)
        x = self.act(x)
        x = self.gcn2_en(x, adj)

        emb = x

        # decoder
        x = self.gcn2_de(x, adj)
        x = self.act(x)
        x = self.gcn1_de(x, adj)

        return emb, x