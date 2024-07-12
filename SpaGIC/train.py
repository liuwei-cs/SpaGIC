# _*_coding : utf-8_*_
# @Time : 2023/10/29 16:57
# @Author :刘威
# @File : train
# @Project :SpaGIC

import torch
from preprocess import *
import time
import random
import numpy as np
from model import SpaGIC
from tqdm import tqdm
from torch import nn
from utils import *


class Train():
    def __init__(self,
                 adata,
                 device=torch.device('cpu'),
                 learning_rate=0.001,
                 weight_decay=1e-4,
                 epochs=500,
                 in_dim=3000,
                 out_dim=64,
                 random_seed=2023,
                 mse_weight=60,
                 graph_weight=0.01,
                 nce_weight=0.01,
                 coord_type='spatial',
                 n_neighbor=5,
                 n_cluster=7,
                 protocol='10X'
                 ):
        self.adata = adata.copy()
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.random_seed = random_seed
        self.mse_weight = mse_weight
        self.graph_weight = graph_weight
        self.nce_weight = nce_weight
        self.n_cluster = n_cluster

        fix_seed(self.random_seed)

        if 'highly_variable' not in self.adata.var.keys():
            preprocess(self.adata)

        if 'feature' not in self.adata.obsm.keys():
            get_feature(self.adata)

        if 'similarity' not in self.adata.obsp.keys() and 'adj' not in self.adata.obsp.keys():
            get_similarity(self.adata, coord_type=coord_type)

        if 'adj' not in self.adata.obsp.keys():
            get_graph(self.adata, n_neighbor=n_neighbor)

        self.features = torch.FloatTensor(self.adata.obsm['feature'].copy()).to(self.device)
        self.adj = self.adata.obsp['adj']

        self.in_dim = self.features.shape[1]
        self.out_dim = out_dim

        self.normalized_adj = normalize_adj(self.adj.copy(), protocol=protocol).to(self.device)
        self.adj = torch.FloatTensor(self.adj.copy()).to(self.device)

    # model training
    def train(self):
        self.model = SpaGIC(self.in_dim, self.out_dim).to(self.device)

        self.BCE_loss = nn.BCEWithLogitsLoss()
        self.KL_loss = nn.KLDivLoss(reduction='batchmean')

        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate,
                                          weight_decay=self.weight_decay)
        print('Begin to train...')

        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            self.emb, self.x_ = self.model(self.features, self.normalized_adj)

            self.adj_ = torch.mm(self.emb, self.emb.T)

            self.loss_adj_1 = self.BCE_loss(self.adj_, self.adj)
            self.loss_adj_2 = self.KL_loss(F.log_softmax(self.adj, dim=1) + 1e-8, self.adj_.softmax(dim=1) + 1e-8)
            self.loss_NCE = Noise_Cross_Entropy(self.emb, self.adj)
            self.loss_feature = F.mse_loss(self.x_, self.features)

            loss = self.mse_weight * self.loss_feature + self.graph_weight * (self.loss_adj_1 + self.loss_adj_2) \
                   + self.nce_weight * self.loss_NCE

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Optimization finished!")

        with torch.no_grad():
            self.model.eval()

            self.emb, self.x_ = self.model(self.features, self.normalized_adj)
            self.emb = self.emb.detach().cpu().numpy()
            self.x_ = self.x_.detach().cpu().numpy()

            self.adata.obsm['emb'] = self.emb
            self.adata.obsm['rec_feat'] = self.x_

            return self.adata



