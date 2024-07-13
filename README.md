# SpaGIC: Graph-informed clustering in spatial transcriptomics via self-supervised contrastive learning

![](https://github.com/liuwei-cs/SpaGIC/SpaGIC.jpg)

## Overview
SpaGIC is a novel graph-based self-supervised contrastive learning framework for effective spatial transcriptomics analysis. In the workflow of SpaGIC, a GCN-based auto-encoder is utilized to learn spots representation by iteratively aggregating information from neighboring nodes. Initially, the framework employs the KNN algorithm to construct an adjacency graph according to spots spatial location, and takes preprocessed gene expression data as nodes feature (Fig. A). Based on the embeddings, SpaGIC reconstructs the adjacency to enhance spots representation through maximizing graph structural mutual information both edge-wise and local neighborhood-wise in a self-supervised manner (Fig. B). Concurrently, an InfoNCE-like contrastive learning loss is integrated into the model training to preserve the spatial neighbor information by grouping spatially adjacent spots and separating spatially non-adjacent spots in latent space. (Fig. C). Furthermore, to fully exploit the gene expression profiles, SpaGIC reconstructs the raw gene expression matrix from latent embeddings using a decoder, while adhering to the constraint of MSE (Mean Squared Error) loss (Fig. D). The resulting output from SpaGIC is versatile and can be applied across a range of downstream spatial transcriptomics analysis tasks, such as spatial domain identification, data denoising, visualization, trajectory inference and multi-slice joint analysis (Fig. E).

## Requirements
You'll need to install the following packages in order to run the codes.
* python==3.8
* torch>=2.1.0
* cudnn>=11.8
* numpy==1.24.3
* pandas==2.0.3
* matplotlib==3.7.3
* scanpy==1.9.6
* anndata==0.8.0
* scipy==1.10.1
* scikit-learn==1.3.2
* tqdm==4.66.1
* rpy2==3.5.14
* R==4.3.1
* h5py==3.10.0
* leidenalg==0.10.2
* louvain==0.8.1

## Data availability
All datasets used in this work can be downloaded from the link below. 
(1) the LIBD human dorsolateral prefrontal cortex (DLPFC) dataset: http://spatial.libd.org/spatialLIBD
(2) the 10x Visium human breast cancer dataset: https://www.10xgenomics.com/datasets/human-breast-cancer-block-a-section-1-1-standard-1-1-0 
(3) the anterior section of the 10x Visium mouse brain: https://www.10xgenomics.com/resources/datasets/mouse-brain-serial-section-1-sagittal-anterior-1-standard-1-1-0
(4) the Stereo-seq mouse olfactory bulb dataset: https://github.com/STOmics/SAW/tree/main/Test_Data
(5) the mouse Slide-seqV2 olfactory bulb dataset: https://singlecell.broadinstitute.org/single_cell/study/SCP815/highly-sensitive-spatial-transcriptomics-at-near-cellular-resolution-with-slide-seqv2#study-summary
(6) the STARmap mouse visual cortex dataset: https://drive.google.com/drive/folders/1I1nxheWlc2RXSdiv24dex3YRaEh780my?usp=sharing.
(7) the osmFISH mouse somatosensory cortex dataset: https://linnarssonlab.org/osmFISH/

## Citation
Liu et al. 
