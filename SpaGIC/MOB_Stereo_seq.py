# _*_coding : utf-8_*_
# @Time : 2024/1/2 11:18
# @Author :刘威
# @File :MOB_Stereo_seq
# @Project :SpaGIC

import os
import torch
import pandas as pd
import scanpy as sc
from preprocess import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from utils import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from train import Train

# 小鼠嗅球数据集
if __name__ == "__main__":
    # Run device, by default, the package is implemented on 'cpu'. We recommend using GPU.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available(), device)

    # the location of R, which is necessary for mclust algorithm. Please replace the path below with local R installation path
    os.environ['R_HOME'] = '/home/wei/miniconda3/envs/R/lib/R'

    savepath = '../Result/Domains/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    dataset = 'Mouse_Olfactory_Bulb'
    # the number of clusters
    n_cluster = 7

    n_neighbor = 15
    epochs = 500
    coord_type = 'spatial'
    weights = [[5, 0.01, 0.01]]

    # read data
    file_fold = '../../Data/' + dataset + '/'
    adata = sc.read_h5ad(file_fold + 'filtered_feature_bc_matrix.h5ad')
    adata.var_names_make_unique()
    print(adata)

    for mse_weight, graph_weight, nce_weight in weights:
        set = str(n_neighbor) + "_" + str(mse_weight) + "_" + str(graph_weight) \
              + "_" + str(nce_weight) + "_" + str(epochs)
        print(set)
        print("*" * 70)

        model = Train(adata, device=device, epochs=epochs, mse_weight=mse_weight,
                      graph_weight=graph_weight, nce_weight=nce_weight, coord_type=coord_type,
                      n_neighbor=n_neighbor, n_cluster=n_cluster, protocol='Stereo-seq')

        # train model
        adata = model.train()

        method = 'mclust'  # mclust, leiden, louvain, kmeans
        cluster(adata, n_cluster, method)
        getScore(adata, key='domain')

        SC = adata.uns['SC']
        print(dataset, ' SC: ', SC)

    pl = ['#1f77b4ff', '#ff7f0eff', '#2ca02cff', '#d62728ff', '#9467bdff', '#8c564bff', '#e377c2ff']

    # sc.set_figure_params(vector_friendly=True, figsize=(4.5 / 2.54, 3.5 / 2.54))
    ax = sc.pl.spatial(adata, color='domain', spot_size=40, frameon=False, show=False, palette=pl,
                       title='SpaGIC_MOB')
    ax[0].invert_yaxis()
    plt.savefig(savepath + f'SpaGIC_MOB_sp.png', bbox_inches='tight', dpi=300)
    # plt.savefig(savepath + f'SpaGIC_MOB_sp.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    #### SingleDomian visual
    print('MOB_SingleDomian Start...')
    raw_domain_types = adata.obs['domain'].unique()
    # print(raw_domain_types)     # [7, 2, 3, 1, 4, 5, 6]
    new_domain_types = []
    for i in raw_domain_types:
        adata.obs['domain'].replace(i, str(i), inplace=True)
        new_domain_types.append(str(i))
    new_domain_types.sort()
    print(new_domain_types)

    zeros = np.zeros([adata.n_obs, n_cluster])
    matrix_cell_type = pd.DataFrame(zeros, index=adata.obs_names, columns=new_domain_types)
    for cell in list(adata.obs_names):
        ctype = adata.obs.loc[cell, 'domain']
        matrix_cell_type.loc[cell, ctype] = 1
    adata.obs[matrix_cell_type.columns] = matrix_cell_type.astype(str)

    # sc.set_figure_params(vector_friendly=True, figsize=(2.4/2.54, 2/2.54))
    fig, axes = plt.subplots(nrows=1, ncols=len(raw_domain_types), figsize=(4/2.54 * 7, 3.1/2.54))
    for j in new_domain_types:
        idx = int(j) - min(raw_domain_types)
        ax = sc.pl.spatial(adata, color=j, spot_size=40, frameon=False, show=False, palette=['gray', pl[idx]],
                           ax=axes[idx], legend_loc=None)
        ax[0].invert_yaxis()
    plt.savefig(savepath + f'SpaGIC_MOB_SingleDomian.png', bbox_inches='tight', dpi=300)
    # plt.savefig(savepath + f'SpaGIC_MOB_SingleDomian.pdf', bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()
    print('MOB_SingleDomian Finish!')

    #### MarkerGene visual
    print('MOB_MarkerGene start...')
    markers = ['Apod', 'Mbp', 'Nrgn', 'Pcp4', 'Gabra1', 'Slc6a11', 'Cck']

    # sc.set_figure_params(vector_friendly=True, figsize=(2.4/2.54, 2/2.54))
    fig, axes = plt.subplots(nrows=1, ncols=len(markers) + 1, figsize=(4/2.54 * 7, 3.1/2.54))
    for i in range(len(markers)):
        sc.pl.spatial(adata, color=markers[i], spot_size=40, frameon=False, show=False, ax=axes[i],
                      colorbar_loc=None)    # , layer='raw'
        axes[i].axis('off')
        axes[i].invert_yaxis()
        scalar_mappable = axes[i].get_children()[0]
        scalar_mappable.set_clim(0, 3.5)
    axes[-1].axis('off')
    colorbar = fig.colorbar(scalar_mappable, orientation='vertical', shrink=0.6, ax=axes[-1],
                            fraction=1, pad=-0.3, label='Expression')
    colorbar.set_ticks([0, 3.5])
    plt.savefig(savepath + f'SpaGIC_MOB_MarkGene.png', bbox_inches='tight', dpi=300)
    # plt.savefig(savepath + f'SpaGIC_MOB_MarkGene.pdf', bbox_inches='tight', dpi=300)
    plt.close()

    print('MOB_MarkerGene Finish!')
