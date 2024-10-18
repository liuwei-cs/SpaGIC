# _*_coding : utf-8_*_
# @Time : 2023/10/29 18:55
# @Author :刘威
# @File :utils
# @Project :SpaGIC

import numpy as np
import pandas as pd
from scipy.sparse import issparse
import scanpy as sc
from sklearn.metrics import pairwise_distances, adjusted_rand_score, normalized_mutual_info_score, silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import seaborn as sns
import os

def getScore(adata, key='domain'):
    if 'ground_truth' in adata.obs.keys():
        # filter out NA nodes
        adata_ = adata[~pd.isnull(adata.obs['ground_truth'])]
        ARI = adjusted_rand_score(adata_.obs['ground_truth'], adata_.obs[key])
        adata.uns['ARI'] = ARI
        NMI = normalized_mutual_info_score(adata_.obs['ground_truth'], adata_.obs[key])
        adata.uns['NMI'] = NMI
    else:
        SC = silhouette_score(adata.obsm['emb_pca'], adata.obs[key])
        adata.uns['SC'] = SC
        DB = davies_bouldin_score(adata.obsm['emb_pca'], adata.obs[key])
        adata.uns['DB'] = DB

def cluster(adata, n_clusters=7, method='mclust', key = 'emb_pca', start=2.0, step=0.01, max_run=50):
    print('clustering...')
    # single slice
    if key == 'emb_pca':
        pca = PCA(n_components=20, random_state=2023)
        embedding = pca.fit_transform(adata.obsm['emb'].copy())
        adata.obsm['emb_pca'] = embedding

    if method == 'mclust':
        adata = mclust_R(adata, used_obsm=key, n_clusters=n_clusters)
        adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
        res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, step=step, max_run=max_run)
        sc.tl.leiden(adata, random_state=2023, resolution=res)
        adata.obs['domain'] = adata.obs['leiden'].astype('category')
    elif method == 'louvain':
        res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, step=step, max_run=max_run)
        sc.tl.louvain(adata, random_state=2023, resolution=res)
        adata.obs['domain'] = adata.obs['louvain'].astype('category')
    elif method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters).fit(adata.obsm['emb_pca'])
        adata.obs['domain'] = kmeans.labels_.astype(str)
        adata.obs['domain'] = adata.obs['domain'].astype('category')

def mclust_R(adata, n_clusters, modelNames='EEE', used_obsm='emb', random_seed=2023):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects

    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri

    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), n_clusters, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')

    return adata

# louvain / leiden
def search_res(adata, n_clusters, use_rep='emb', method='louvain', start=2.0, step=0.01, max_run=50, prior=True):
    sc.pp.neighbors(adata, use_rep=use_rep)
    res=start
    print("Start at res = ", res, "step = ", step)
    if method == 'leiden':
        sc.tl.leiden(adata, random_state=2023, resolution=res)
        count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
        print('resolution={}, cluster number={}'.format(res, count_unique))
        run = 1
        while count_unique != n_clusters:
            run += 1
            old_sign = 1 if (count_unique < n_clusters) else -1
            res = res + step * old_sign
            res = round(res, 4)
            sc.tl.leiden(adata, random_state=2023, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
            new_sign = 1 if (count_unique < n_clusters) else -1
            if new_sign != old_sign:
                step = round(step / 2, 4)
                print("Step changed to", step)
            if run >= max_run:
                print("The maximum number of searches has been reached! Exact resolution not found!")
                break
    elif method == 'louvain':
        sc.tl.louvain(adata, random_state=2023, resolution=res)
        count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
        print('resolution={}, cluster number={}'.format(res, count_unique))
        run = 1
        while count_unique != n_clusters:
            run += 1
            old_sign = 1 if (count_unique < n_clusters) else -1
            res = res + step * old_sign
            res = round(res, 4)
            sc.tl.louvain(adata, random_state=2023, resolution=res)
            count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique())
            print('resolution={}, cluster number={}'.format(res, count_unique))
            new_sign = 1 if (count_unique < n_clusters) else -1
            if new_sign != old_sign:
                step = step / 2
                print("Step changed to", step)
            if run >= max_run:
                print("The maximum number of searches has been reached! Exact resolution not found!")
                break

    print("recommended res = ", str(res))
    return res

# optional
def refine(adata, radius=50, key='domain'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    if 'distance' in adata.obsp.keys():
        distance = adata.obsp['distance']
    else:
        position = adata.obsm['spatial']
        distance = pairwise_distances(position, metric="euclidean")

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    adata.obs['domain_refined'] = new_type

# Calculate InfoNCE-like loss. Considering spatial neighbors as positive pairs for each spot
def Noise_Cross_Entropy(emb, adj):
    sim = cosine_sim_tensor(emb)
    sim_exp = torch.exp(sim)

    # negative pairs
    n = torch.mul(sim_exp, 1 - adj).sum(axis=1)
    # positive pairs
    p = torch.mul(sim_exp, adj).sum(axis=1)

    ave = torch.div(p, n)
    loss = -torch.log(ave).mean()

    return loss

# Calculate cosine similarity.
def cosine_sim_tensor(emb):
    M = torch.matmul(emb, emb.T)
    length = torch.norm(emb, p=2, dim=1)
    Norm = torch.matmul(length.reshape((emb.shape[0], 1)), length.reshape((emb.shape[0], 1)).T) + -5e-12      # reshape((emb.shape[0], 1))
    M = torch.div(M, Norm)
    if torch.any(torch.isnan(M)):
        M = torch.where(torch.isnan(M), torch.full_like(M, 0.4868), M)

    return M

# DLPFC
def DLPFC_visual(adata, dataset, key='domain'):
    savepath = '../Result/Domains/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    ARI = adata.uns['ARI']
    NMI = adata.uns['NMI']

    sc.pl.spatial(adata, img_key="hires", color=key, frameon=False,
                  title=f'{dataset}_ARI=%.4f-NMI=%.4f' % (ARI, NMI),  show=False)
    plt.savefig(savepath + f'SpaGIC_{dataset}_sp.png', bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()

    # PAGA
    sc.pp.neighbors(adata, use_rep='emb_pca')
    sc.tl.umap(adata)

    plt.rcParams["figure.figsize"] = (3, 3)
    sc.tl.paga(adata, groups=key)
    sc.pl.paga_compare(adata, legend_fontsize=10, frameon=False, size=20, title=f'{dataset}_PAGA',
                       legend_fontoutline=2, show=False)
    plt.savefig(savepath + f'SpaGIC_{dataset}_paga.png', bbox_inches='tight', dpi=300)
    # plt.show()
    plt.close()