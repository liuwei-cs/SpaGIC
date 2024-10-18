# _*_coding : utf-8_*_
# @Time : 2024/3/28 16:23
# @Author :刘威
# @File :DLPFC_Integrate_WithHarmony
# @Project :SpaGIC

import os
import torch
import pandas as pd
import scanpy as sc
from preprocess import *
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import harmonypy as hm

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from train import Train

if __name__ == "__main__":
    # Run device, by default, the package is implemented on 'cpu'. We recommend using GPU.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available(), device)

    # the location of R, which is necessary for mclust algorithm. Please replace the path below with local R installation path
    os.environ['R_HOME'] = '/home/wei/miniconda3/envs/R/lib/R'

    savepath = '../Result/Integration/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    datasets_df = pd.DataFrame([['151507', '151508', '151509', '151510'],
                                ['151669', '151670', '151671', '151672'],
                                ['151673', '151674', '151675', '151676']])
    datasets_df.index = ['Sample1', 'Sample2', 'Sample3']
    datasets_df.columns = ['Location1', 'Location2', 'Location3', 'Location4']
    batch_tag = 'Sample3'
    datasets = datasets_df.loc[batch_tag].values    # Sample
    # datasets = datasets_df[batch_tag].values        # Location
    dataset_tags = datasets

    n_neighbor = 5
    n_cluster = 7 if batch_tag != 'Sample2' else 5

    epochs = 500
    coord_type = 'spatial'
    weights = [[60, 0.01, 0.01]]

    adata_list = []
    for i in range(len(datasets)):
        dataset = datasets[i]
        # print(dataset)
        file_fold = '../../Data/DLPFC/' + str(dataset)
        adata = sc.read_visium(file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        adata.var_names_make_unique()

        # add ground_truth
        df_meta = pd.read_csv(file_fold + '/metadata.tsv', sep='\t')
        adata.obs['ground_truth'] = df_meta['layer_guess'].values
        # filter out NA nodes
        adata = adata[~pd.isnull(adata.obs['ground_truth'])]

        adata.obs_names = [x + '_' + dataset for x in adata.obs_names]
        adata.obs['batch_name'] = [dataset] * adata.X.shape[0]
        adata_list.append(adata)

    batch_adata = integrate_adata(adata_list, coord_type=coord_type, n_neighbor=n_neighbor)
    for mse_weight, graph_weight, nce_weight in weights:
        set = batch_tag + "_" + str(n_neighbor) + "_" + str(mse_weight) + "_" + str(graph_weight) \
            + "_" + str(nce_weight) + "_" + str(epochs)
        print(set)
        print("*" * 70)

        # define model
        model = Train(batch_adata, device=device, epochs=epochs, mse_weight=mse_weight,
                      graph_weight=graph_weight, nce_weight=nce_weight, coord_type=coord_type,
                      n_neighbor=n_neighbor, n_cluster=n_cluster, protocol='10X')
        # run model
        batch_adata = model.train()

        # Harmony
        data_mat = batch_adata.obsm['emb'].copy()
        vars_use = ['batch_name']
        meta_data = batch_adata.obs[vars_use]
        # Run Harmony
        ho = hm.run_harmony(data_mat, meta_data, vars_use)
        batch_adata.obsm['Harmony_Revised'] = ho.Z_corr.T

        method = 'mclust'  # mclust, leiden, louvain, kmeans
        cluster(batch_adata, n_clusters=n_cluster, method=method, key='Harmony_Revised')

        domain_tag = 'domain'
        embed_tag = 'Harmony_Revised'

        # filter out NA nodes
        sub_adata = batch_adata[~pd.isnull(batch_adata.obs['ground_truth'])]

        # sc.set_figure_params(vector_friendly=True, figsize=(4.5 / 2.54, 4.5 / 2.54))
        fig, ax_list = plt.subplots(1, 4, figsize=(5 / 2.54 * 4, 5 / 2.54))
        ### Plotting UMAP before batch effect correction
        sc.pp.pca(sub_adata)
        sc.pp.neighbors(sub_adata, use_rep='X_pca')
        sc.tl.umap(sub_adata)
        sc.pl.umap(sub_adata, color='batch_name', title='Uncorrected', ax=ax_list[0], frameon=False,
                   show=False, size=3, legend_fontsize=5)
        ### Plotting UMAP after batch effect correction
        sc.pp.neighbors(sub_adata, use_rep=embed_tag)
        sc.tl.umap(sub_adata)
        sc.pl.umap(sub_adata, color='batch_name', ax=ax_list[1], title='Batch corrected', frameon=False,
                   show=False, size=3, legend_fontsize=5)
        
        sc.pl.umap(sub_adata, color='ground_truth', ax=ax_list[2], title='Ground Truth', frameon=False,
                   show=False, size=3, legend_fontsize=5)
        sc.pl.umap(sub_adata, color=domain_tag, ax=ax_list[3], title='Predict Domain', frameon=False,
                   show=False, size=3, legend_fontsize=5)
        plt.savefig(savepath + f'SpaGIC_{batch_tag}_umap_hm.png', bbox_inches='tight', dpi=300)
        # plt.savefig(savepath + f'SpaGIC_{batch_tag}_umap_hm.pdf', bbox_inches='tight', dpi=300)
        # plt.show()
        plt.close()

        # scores
        ARI = adjusted_rand_score(sub_adata.obs['ground_truth'], sub_adata.obs[domain_tag])
        NMI = normalized_mutual_info_score(sub_adata.obs['ground_truth'], sub_adata.obs[domain_tag])
        print(f'{batch_tag}, ARI: {ARI}; NMI: {NMI}')

        #### single slice visual
        # sc.set_figure_params(vector_friendly=True, figsize=(3.2 / 2.54, 3.5 / 2.54))
        fig, axs = plt.subplots(1, len(datasets), figsize=(4.5 * len(datasets), 3))
        for idx, single_adata in enumerate(adata_list):
            single_adata.obs[domain_tag] = batch_adata[batch_adata.obs['batch_name'] == datasets[idx]].obs[domain_tag].values
            # filter out NA nodes
            sub_adata = single_adata[~pd.isnull(single_adata.obs['ground_truth'])]
            refine(sub_adata, radius=50, key=domain_tag)
            key = 'domain_refined'
            ARI = adjusted_rand_score(sub_adata.obs['ground_truth'], sub_adata.obs[key])
            NMI = normalized_mutual_info_score(sub_adata.obs['ground_truth'], sub_adata.obs[key])
            print(f'{datasets[idx]}, ARI: {ARI}; NMI: {NMI}')
            sc.pl.spatial(sub_adata, img_key="hires", ax=axs[idx], color=key, frameon=False, legend_fontsize=8,
                          title=f'{datasets[idx]}_ARI=%.4f-NMI=%.4f' % (ARI, NMI), show=False)
        plt.savefig(savepath + f'SpaGIC_{batch_tag}_sp_hm.png', bbox_inches='tight', dpi=300)
        # plt.savefig(savepath + f'SpaGIC_{batch_tag}_sp_hm.pdf', bbox_inches='tight', dpi=300)
        # plt.show()
        plt.close()