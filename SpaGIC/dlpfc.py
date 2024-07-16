# _*_coding : utf-8_*_
# @Time : 2023/11/29 16:58
# @Author :刘威
# @File :dlpfc
# @Project :SpaGIC

import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
from preprocess import *
from utils import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from train import Train

if __name__ == "__main__":
    # Run device, by default, the package is implemented on 'cpu'. We recommend using GPU.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available(), device)

    # the location of R, which is necessary for mclust algorithm. Please replace the path below with local R installation path
    os.environ['R_HOME'] = '/home/wei/miniconda3/envs/R/lib/R'

    datasets = ['151507', '151508', '151509', '151510', '151669', '151670',
                '151671', '151672', '151673', '151674', '151675', '151676']
    datasets = ['151673']
    # the number of clusters
    n_clusters = [7, 7, 7, 7, 5, 5, 5, 5, 7, 7, 7, 7]
    n_clusters = [7]

    n_neighbor = 5
    epochs = 500
    coord_type = 'spatial'
    weights = [[60, 0.01, 0.01]]

    adatas = []
    for i in range(len(datasets)):
        dataset = datasets[i]
        # read data
        file_fold = '../Data/DLPFC/' + dataset
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()

        # add ground_truth
        df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
        adata.obs['ground_truth'] = df_meta['layer_guess']
        # filter out NA nodes
        adata = adata[~pd.isnull(adata.obs['ground_truth'])]

        # # Ground Truth
        # sc.pl.spatial(adata, img_key="hires", color='ground_truth', title=f'{dataset}', frameon=False, show=False)
        # plt.savefig('../Result/Ground_Truth/' + f'{dataset}.png', bbox_inches='tight', dpi=300)
        # plt.savefig('../Result/Ground_Truth/' + f'{dataset}.pdf', bbox_inches='tight', dpi=300)
        # # plt.show()
        # plt.close()

        adatas.append(adata)
        # print(adata)

    for mse_weight, graph_weight, nce_weight in weights:
        ari_ = []
        ari_ref_ = []
        for i in range(len(datasets)):
            dataset = datasets[i]
            n_cluster = n_clusters[i]
            adata = adatas[i]
            set = str(n_neighbor) + "_" + str(mse_weight) + "_" + str(graph_weight) \
                   + "_" + str(nce_weight) + "_" + str(epochs)
            print(set)
            print("*" * 70)
            model = Train(adata, device=device, epochs=epochs, mse_weight=mse_weight,
                          graph_weight=graph_weight, nce_weight=nce_weight, coord_type=coord_type,
                          n_neighbor=n_neighbor, n_cluster=n_cluster, protocol='10X')

            # train model
            adata = model.train()

            method = 'mclust'  # mclust, leiden, louvain, kmeans
            cluster(adata, n_cluster, method)
            getScore(adata, key='domain')
            ari = adata.uns['ARI']
            ari_.append(round(ari, 3))
            # refining
            refine(adata, radius=50, key='domain')
            getScore(adata, key='domain_refined')
            ari_ref = adata.uns['ARI']
            ari_ref_.append(round(ari_ref, 3))
            print(dataset, ' ARI:', ari, ' ARI_ref:', ari_ref)

            DLPFC_visual(adata, dataset, key='domain_refined')

            # clear memory
            import gc
            gc.collect()
            del adata

        print(ari_)
        print(ari_ref_)
