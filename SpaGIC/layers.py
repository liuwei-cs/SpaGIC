# _*_coding : utf-8_*_
# @Time : 2023/10/29 16:39
# @Author :刘威
# @File :layers
# @Project :SpaGIC

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

# GCN
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_dim, out_dim, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Parameter(torch.FloatTensor(in_dim, out_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            
    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_dim) + ' -> ' \
               + str(self.out_dim) + ')'
