# _*_coding : utf-8_*_
# @Time : 2023/10/19 20:50
# @Author :刘威
# @File :preprocess
# @Project :SpaGIC

import os
import torch
import random
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from scipy.linalg import block_diag
from utils import *

def preprocess(adata):
    # adata.layers['raw'] = adata.X.copy()      # Data denoising
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)

def get_feature(adata):
    if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
        feat_mtx = adata.X.toarray()[:, adata.var['highly_variable']]
    else:
        feat_mtx = adata.X[:, adata.var['highly_variable']]
    adata.obsm['feature'] = feat_mtx
    # print(adata)

def get_similarity(adata, coord_type='spatial', nearest=4):
    if coord_type == 'spatial':
        coordinate = adata.obsm['spatial']
    elif coord_type == 'img_loc':
        coordinate = np.array([adata.obs['array_row'].values, adata.obs['array_col'].values])
        coordinate = coordinate.T
        adata.obsm['img_loc'] = coordinate
    distance = pairwise_distances(coordinate, metric="euclidean")
    dist_sorted = distance.copy()
    dist_sorted.sort()
    m = 2
    l = int(np.sum(np.sum(dist_sorted[:, 1:nearest+1], axis=1) / nearest) / adata.n_obs * m)
    print(f'Recommended l is {l}')
    pos_sim = np.exp(-0.5 * (distance ** 2) / (l ** 2))
    adata.obsp['pos_sim'] = pos_sim
    adata.obsp['distance'] = distance
    adata.obsp['similarity'] = normalize_sim(pos_sim, 'minmax')

# Z-score/min-max Normalization
def normalize_sim(sim, type='minmax'):
    sim = sim.copy()
    for i in range(sim.shape[0]):
        if type == 'Z-score':
            mean = sim[i].mean()
            std = sim[i].std()
            sim[i] = (sim[i] - mean) / std
        elif type == 'minmax':
            minV = sim[i].min()
            maxV = sim[i].max()
            sim[i] = (sim[i] - minV) / (maxV - minV)

    return sim

def get_graph(adata, n_neighbor=5):
    similarity = adata.obsp['similarity']
    n_spot = similarity.shape[0]
    interaction = np.zeros([n_spot, n_spot])

    for i in range(n_spot):
        sim = similarity[i, :]
        sim_idx = sim.argsort()
        interaction[i][sim_idx[-(n_neighbor + 1):]] = 1

    adata.obsm['neighbors'] = interaction
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj > 1, 1, adj)

    adata.obsp['adj'] = adj

def normalize_adj(adj, protocol='10X'):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    degree = np.array(adj.sum(1))
    d_inv_sqrt = np.power(degree, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

    if protocol in ['Stereo-seq', 'Slide-seqV2']:
        return sparse_mx_to_torch_sparse_tensor(adj)
    else:
        return torch.FloatTensor(adj.toarray())

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

# Joint analysis
def integrate_adata(adata_list, coord_type='spatial', n_neighbor=5):
    adj_list = []
    for single_adata in adata_list:
        if 'similarity' not in single_adata.obsp.keys():
            get_similarity(single_adata, coord_type=coord_type)
        if 'adj' not in single_adata.obsp.keys():
            get_graph(single_adata, n_neighbor=n_neighbor)

        adj_list.append(single_adata.obsp['adj'])
        # print(single_adata)
    batch_adata = sc.concat(adata_list, keys=None)
    adjacency = block_diag(*adj_list)
    batch_adata.obsp['adj'] = adjacency

    preprocess(batch_adata)
    get_feature(batch_adata)
    print(batch_adata)

    return batch_adata

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
