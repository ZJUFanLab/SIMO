import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm
import scanpy as sc
from .helper import *
from .clustering import ConsensusCluster
from sklearn.preprocessing import scale
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix
from sklearn.cluster import  AgglomerativeClustering
from scipy.stats import rankdata


def rank_genes_groups(input_adata, log=True,groupby=None,layer=None,use_raw=False,method='wilcoxon'):
    """
    Ranks and evaluates differential gene expression across groups within an AnnData object.

    Args:
    input_adata (AnnData): Input AnnData object containing gene expression data.
    log (bool): If True, retains log fold changes in the output; if False, calculates linear fold changes. Defaults to True.
    groupby (str, optional): Column name in .obs that defines which groups to compare. If None, the function does not proceed.
    layer (str, optional): Specifies which layer of the data to use in the analysis. If None, uses .X by default.
    use_raw (bool): If True, uses the raw attribute of AnnData for calculations. Defaults to False.
    method (str): Statistical method used for ranking genes; supported methods include 'wilcoxon', 't-test', etc. Defaults to 'wilcoxon'.

    Returns:
    df (DataFrame): A pandas DataFrame containing the ranked genes along with their group associations and statistical measures such as log fold changes or recalculated fold changes based on the 'log' parameter.
    """
    adata = input_adata.copy()
    if log:
        sc.tl.rank_genes_groups(adata, groupby=groupby, method=method, layer=layer, use_raw=use_raw)
        df = sc.get.rank_genes_groups_df(adata, group=None, key='rank_genes_groups')
        
    else:
        sc.tl.rank_genes_groups(adata, groupby=groupby, method=method, layer=layer, use_raw=use_raw)
        df = sc.get.rank_genes_groups_df(adata, group=None, key='rank_genes_groups')
        df = df.drop(columns=['logfoldchanges'])
        
        obs_tidy = extract_exp(adata, layer=layer)
        obs_tidy.index = adata.obs[groupby].tolist()
        obs_tidy.columns = adata.var.index.tolist()

        cell_types = adata.obs[groupby].unique()
        fold_changes = []

        for cell_type in cell_types:
            mean_exp_current = obs_tidy.loc[obs_tidy.index == cell_type].mean()
            mean_exp_others = obs_tidy.loc[obs_tidy.index != cell_type].mean()
            fold_change = mean_exp_current / (mean_exp_others + 1e-9)
            fold_changes.append(pd.DataFrame({
                'group': cell_type,
                'names': mean_exp_current.index,
                'logfoldchanges': fold_change.values
            }))
        fold_change_df = pd.concat(fold_changes)
        df = df.merge(fold_change_df, on=['group', 'names'], how='left')
        
    
    return df

def regulation_analysis(adata1, 
                        adata2, 
                        marker1_by='annotation',
                        marker2_by='annotation',
                        layer1='data',
                        layer2='data',
                        method='wilcoxon',
                        use_raw=False,
                        feature_transfer_df=None, 
                        deg_pval_threshold=0.05,
                        log=True,
                       subset=None):
    """
    Performs gene regulation analysis between two AnnData datasets, focusing on differential expression and feature interaction.

    Parameters:
        adata1 (AnnData): First dataset for analysis.
        adata2 (AnnData): Second dataset for analysis.
        marker1_by (str): Annotation column in `adata1` to group by for differential expression analysis. Default is 'annotation'.
        marker2_by (str): Annotation column in `adata2` to group by for differential expression analysis. Default is 'annotation'.
        layer1 (str): Specifies the data layer to use from `adata1`. Default is 'data'.
        layer2 (str): Specifies the data layer to use from `adata2`. Default is 'data'.
        method (str): Statistical method for gene ranking (e.g., 'wilcoxon'). Default is 'wilcoxon'.
        use_raw (bool): If True, uses raw data for calculations. Default is False.
        feature_transfer_df (DataFrame): DataFrame containing feature interaction information, essential for cross-dataset comparisons.
        deg_pval_threshold (float): p-value threshold for filtering significant differential expression. Default is 0.05.
        log (bool): If True, retains log-transformed fold changes; if False, uses absolute fold changes. Default is True.
        subset (list, optional): List of groups to subset the data before analysis. If provided, restricts analysis to these groups.

    Returns:
        DataFrame: A comprehensive DataFrame containing detailed results of the gene regulation analysis. Fields include combinations of features, fold changes, scores, adjusted p-values, Pearson correlation coefficients, and counts of data points analyzed.

    Detailed Workflow:
    """
    if subset is not None:
        ann1 = adata1[adata1.obs[marker1_by].isin(subset)]
        ann2 = adata2[adata2.obs[marker2_by].isin(subset)]
    else:
        ann1 = adata1.copy()
        ann2 = adata2.copy()

    final_result = pd.DataFrame(columns=['combos', 'feature1', 'feature2', 'group', 'logfc1', 'logfc2', 
                                         'max_x', 'max_y', 'score1', 'score2', 'pvals1', 'pvals2', 
                                         'pvals_adj1', 'pvals_adj2', 'pearson_correlation', 'pearson_pvalue', 'number'])

    # Perform gene ranking and filtering for adata1
    df1 = rank_genes_groups(ann1, groupby=marker1_by, layer=layer1, use_raw=use_raw, method=method,log=log)
    df_filtered1 = df1[df1['pvals_adj'] < deg_pval_threshold]

    # Perform gene ranking and filtering for adata2
    df2 = rank_genes_groups(ann2, groupby=marker2_by, layer=layer2, use_raw=use_raw, method=method,log=log)
    df_filtered2 = df2[df2['pvals_adj'] < deg_pval_threshold]

    existing_features1 = set(df_filtered1['names'])
    existing_features2 = set(df_filtered2['names'])

    # Filter feature_transfer_df based on existing features
    filtered_feature_transfer_df = feature_transfer_df[
        (feature_transfer_df['feature1'].isin(existing_features1)) & 
        (feature_transfer_df['feature2'].isin(existing_features2))
    ]

    # Iterate over filtered_feature_transfer_df and compute analysis results
    for idx, row in tqdm(filtered_feature_transfer_df.iterrows(), total=len(filtered_feature_transfer_df), desc="Processing"):
        df1_row = df_filtered1[df_filtered1['names'] == row['feature1']]
        df2_row = df_filtered2[df_filtered2['names'] == row['feature2']]
        merged_df = pd.merge(df1_row, df2_row, on='group', how='inner')
        
        merged_df = merged_df.dropna()

        if len(merged_df) < 2:
            continue

        max_x = max(abs(merged_df['logfoldchanges_x']))
        max_y = max(abs(merged_df['logfoldchanges_y']))

        pearson_corr, pearson_pvalue = pearsonr(merged_df['logfoldchanges_x'], merged_df['logfoldchanges_y'])
        
        for _, sub_row in merged_df.iterrows():
            final_result = final_result.append({
                'combos': row['feature1'] + '::' + row['feature2'],
                'feature1': row['feature1'],
                'feature2': row['feature2'],
                'group': sub_row['group'],
                'logfc1': sub_row['logfoldchanges_x'],
                'logfc2': sub_row['logfoldchanges_y'],
                'max_x': max_x,
                'max_y': max_y,
                'score1': sub_row['scores_x'],
                'score2': sub_row['scores_y'],
                'pvals1': sub_row['pvals_x'],
                'pvals2': sub_row['pvals_y'],
                'pvals_adj1': sub_row['pvals_adj_x'],
                'pvals_adj2': sub_row['pvals_adj_y'],
                'pearson_correlation': pearson_corr,
                'pearson_pvalue': pearson_pvalue,
                'number': len(merged_df)
            }, ignore_index=True)
            
        

    return final_result


def smooth_expression(df, pos, k=10):
    """
    Smooth the expression matrix.

    Parameters:
        df (DataFrame): DataFrame containing the expression matrix, with rows as cells and columns as genes.
        pos (ndarray): Spatial position information of cells, where each row represents the coordinates of a cell.
        k (int): Number of nearest neighbors to consider. Default is 10.

    Returns:
        smoothed_df (DataFrame): DataFrame of the smoothed expression matrix.
    """
    # Use KNN to compute the indices of the k nearest neighbors for each cell
    nbrs = NearestNeighbors(n_neighbors=k).fit(pos)
    distances, indices = nbrs.kneighbors(pos)

    # Smooth the expression for each cell
    smoothed_expression = []
    for i in range(len(df)):
        neighbors_indices = indices[i]
        neighbor_expression = df.iloc[neighbors_indices]
        smoothed_expression.append(np.mean(neighbor_expression, axis=0))

    # Convert smoothed_expression to a DataFrame and retain the original DataFrame's row index
    smoothed_df = pd.DataFrame(smoothed_expression, columns=df.columns, index=df.index)
    return smoothed_df

def cross_modal_smooth(df1, pos1, df2, pos2, k=10):
    """
    Performs cross-modal smoothing between two datasets to integrate and smooth modal information based on spatial proximity.

    Args:
        df1 (DataFrame): First dataset containing modal information for each cell.
        pos1 (array-like): Positional coordinates of cells in the first dataset.
        df2 (DataFrame): Second dataset containing modal information for each cell.
        pos2 (array-like): Positional coordinates of cells in the second dataset.
        k (int): Number of nearest neighbors to consider in smoothing process. Default is 10.

    Returns:
        tuple: A tuple of two DataFrames:
            - modal_df1_smoothed: Smoothed modal information for the first dataset.
            - modal_df2_smoothed: Smoothed modal information for the second dataset.
    """
    # Build KNN model on dataset 2
    nbrs2 = NearestNeighbors(n_neighbors=k).fit(pos2)
    distances2, indices2 = nbrs2.kneighbors(pos1)

    # Smooth the modal information for each cell in dataset 1
    smoothed_modal_df1 = []
    for i in range(len(df1)):
        neighbors_indices = indices2[i]
        neighbor_modal = df2.iloc[neighbors_indices]
        smoothed_modal_df1.append(np.mean(neighbor_modal, axis=0))

    # Convert the smoothed modal information to a DataFrame and retain the original DataFrame's row index
    modal_df1_smoothed = pd.DataFrame(smoothed_modal_df1, columns=df2.columns, index=df1.index)

    nbrs1 = NearestNeighbors(n_neighbors=k).fit(pos1)
    distances1, indices1 = nbrs1.kneighbors(pos2)
    # Similarly, smooth the modal information for each cell in dataset 2
    smoothed_modal_df2 = []
    for i in range(len(df2)):
        neighbors_indices = indices1[i]
        neighbor_modal = df1.iloc[neighbors_indices]
        smoothed_modal_df2.append(np.mean(neighbor_modal, axis=0))

    # Convert the smoothed modal information to a DataFrame and retain the original DataFrame's row index
    modal_df2_smoothed = pd.DataFrame(smoothed_modal_df2, columns=df1.columns, index=df2.index)

    return modal_df1_smoothed, modal_df2_smoothed

def min_max_scale_columns(df, min_val=0, max_val=1, epsilon=0.01):
    scaled_df = df.copy()
    for column in df.columns:
        min_val_col = df[column].min()
        max_val_col = df[column].max()
        scaled_df[column] = min_val + (max_val - min_val) * ((df[column] - min_val_col) / (max_val_col - min_val_col))
        scaled_df[column] += abs(scaled_df[column].min()) + epsilon
    return scaled_df

def weighted_cor(mat, weighted_mat, method='pearson', na_zero=True):
    if method == 'spearman':
        mat = np.apply_along_axis(rankdata, 0, mat)
    mat = scale(mat)
    mat[np.isnan(mat)] = np.nan
    cov = np.dot(np.dot(mat.T, weighted_mat), mat)
    diag = np.sqrt(np.diag(cov)[:, None] * np.diag(cov)[None, :])
    cor = cov / diag
    if na_zero:
        cor[np.where(np.isnan(cor))] = 0
    return cor

def cor_remove(cor_df, ave_cor_cut=0.5, min_n=5, max_n=100, na_diag=False):
    cor_mat = cor_df.values  # Convert DataFrame to numpy array
    gene_names = cor_df.index  # Extract gene names
    if na_diag:
        np.fill_diagonal(cor_mat, np.nan)
    cor_mat_temp = np.copy(cor_mat)
    cor_ave = np.nanmean(cor_mat_temp, axis=1)
    cor_max = np.nanmax(cor_mat_temp)
    if (cor_max < ave_cor_cut) or (cor_mat_temp.shape[0] < min_n):
        cor_mat_temp = np.zeros((1, 1))
        gene_names = [gene_names[0]]  # Keep only the first gene name
    else:
        cor_min = np.nanmin(cor_ave)
        cor_min_idx = np.nanargmin(cor_ave)
        idx = 1
        while ((cor_min < ave_cor_cut and idx <= (cor_mat_temp.shape[0] - min_n)) or
               ((cor_mat_temp.shape[0] - idx)) >= max_n):
            cor_mat_temp = np.delete(cor_mat_temp, cor_min_idx, axis=0)
            cor_mat_temp = np.delete(cor_mat_temp, cor_min_idx, axis=1)
            gene_names = np.delete(gene_names, cor_min_idx)  # Remove corresponding gene name
            cor_ave = np.nanmean(cor_mat_temp, axis=1)
            cor_min = np.nanmin(cor_ave)
            cor_min_idx = np.nanargmin(cor_ave)
            idx += 1
    # Convert numpy array back to DataFrame
    cor_mat_temp_df = pd.DataFrame(cor_mat_temp, index=gene_names, columns=gene_names)
    return cor_mat_temp_df

def cc_gene_k(df, cor_df, con_df, k=8, avg_con_min=0.5, avg_cor_min=0.5, min_feature=10, max_feature=100):
    res = {}
    for k_i in np.unique(df.cluster):
        print(k_i, ' ')
        gene_k = df.feature[df.cluster==k_i]
        cor_df_temp = cor_df.loc[gene_k, gene_k]
        cor_df_temp = cor_remove(cor_df_temp, max_n=1000, min_n=1, ave_cor_cut=avg_cor_min)
        gene_k = cor_df_temp.index.tolist()
        if not gene_k:
            gene_k = None
        con_df_temp = con_df.loc[gene_k, gene_k]
        con_df_temp = cor_remove(con_df_temp, max_n=max_feature, min_n=1, ave_cor_cut=avg_con_min)
        gene_k = con_df_temp.index.tolist()
        if not gene_k:
            gene_k = None
        res[f'k{k_i}'] = gene_k

    res = {k: v for k, v in res.items() if v is not None and len(v) >= min_feature}
    
    data = [(gene, k[1:]) for k, v in res.items() for gene in v]
    df_res = pd.DataFrame(data, columns=['combos', 'module'])

    return df_res

def spatial_regulation(adata1, 
                       adata2,
                       regulation_df,
                       feature_transfer_df=None, 
                       layer1='data',
                       layer2='data',
                       smooth_k=10,
                       sigma=140, 
                       correlation='pearson',
                       epsilon=1e-3,
                       minK=2,
                       maxK=8,
                       rep=20,
                       resample_proportion=0.5,
                       cc_k=8,
                       avg_con_min=0.5,
                       avg_cor_min=0.5,
                       min_feature=20,
                       max_feature=200
                      ):
    """
    Performs a comprehensive spatial regulation analysis between two AnnData datasets, evaluating gene regulation through spatial proximity and expression smoothing.

    Parameters:
        adata1 (AnnData): First spatial dataset containing expression data and spatial coordinates.
        adata2 (AnnData): Second spatial dataset containing expression data and spatial coordinates.
        regulation_df (DataFrame): DataFrame containing predefined gene regulation pairs; if None, will be derived from `feature_transfer_df`.
        feature_transfer_df (DataFrame, optional): DataFrame containing feature transfer information between two datasets.
        layer1 (str): Layer from which to extract expression data in `adata1`. Default is 'data'.
        layer2 (str): Layer from which to extract expression data in `adata2`. Default is 'data'.
        smooth_k (int): Number of nearest neighbors for smoothing expression data. Default is 10.
        sigma (float): Standard deviation for Gaussian kernel in spatial smoothing. Default is 140.
        correlation (str): Method of correlation ('pearson' or 'spearman') to use in weighted correlation calculations. Default is 'pearson'.
        epsilon (float): Small constant to avoid division by zero in scaling. Default is 1e-3.
        minK (int): Minimum number of clusters for consensus clustering. Default is 2.
        maxK (int): Maximum number of clusters for consensus clustering. Default is 8.
        rep (int): Number of repetitions for consensus clustering. Default is 20.
        resample_proportion (float): Proportion of sample to resample in each repetition for consensus clustering. Default is 0.5.
        cc_k (int): Number of nearest neighbors in consensus clustering to consider for correlation. Default is 8.
        avg_con_min (float): Minimum average connectivity threshold to filter gene clusters. Default is 0.5.
        avg_cor_min (float): Minimum average correlation threshold to filter gene clusters. Default is 0.5.
        min_feature (int): Minimum number of features per cluster. Default is 20.
        max_feature (int): Maximum number of features per cluster. Default is 200.

    Returns:
        dict: A dictionary containing multiple outputs:
            - 'wcor': Weighted correlation matrix.
            - 'cc': Consensus clustering results including feature clusters.
            - 'reg_df': Regulation dataframe with scaled regulation score.
            - 'pos': Combined positional coordinates of both datasets.
    """
    print("Starting spatial regulation analysis...")

    # Extracting features from correlation dataframe
    if regulation_df is  None:
        regulation_df = feature_transfer_df[feature_transfer_df.feature1.isin(adata1.var.index) & feature_transfer_df.feature2.isin(adata2.var.index)]
        
    feature_lst1 = regulation_df.feature1.unique().tolist()
    feature_lst2 = regulation_df.feature2.unique().tolist()
        
    # Extracting expression data and spatial positions
    df1 = extract_exp(adata1, layer=layer1, gene=feature_lst1)
    df2 = extract_exp(adata2, layer=layer2, gene=feature_lst2)
    pos1 = extract_reduction(adata1, use_rep='spatial', column_names=['x', 'y'])
    pos2 = extract_reduction(adata2, use_rep='spatial', column_names=['x', 'y'])

    # Smoothing expression data
    smoothed_df1 = smooth_expression(df1, pos1, k=smooth_k)
    smoothed_df2 = smooth_expression(df2, pos2, k=smooth_k)
    smoothed_df1_cross, smoothed_df2_cross = cross_modal_smooth(df1, pos1, df2, pos2, k=smooth_k)

    # Combining smoothed expression data
    combined_df_m1 = pd.concat([smoothed_df1, smoothed_df2_cross], axis=0)
    combined_df_m2 = pd.concat([smoothed_df2, smoothed_df1_cross], axis=0)
    combined_df_m2 = combined_df_m2.reindex(combined_df_m1.index)

    # Filtering feature transfer dataframe
    filtered_feature_transfer_df = feature_transfer_df[
        (feature_transfer_df['feature1'].isin(set(combined_df_m1.columns))) &
        (feature_transfer_df['feature2'].isin(set(combined_df_m2.columns)))
    ]

    # Scaling expression data
    scaled_combined_df_m1 = min_max_scale_columns(combined_df_m1, epsilon=epsilon)
    scaled_combined_df_m2 = min_max_scale_columns(combined_df_m2, epsilon=epsilon)

    # Making regulation data
    print("Processing spatial regulation...")
    reg_df = pd.DataFrame(index=combined_df_m1.index)
    for idx, row in tqdm(filtered_feature_transfer_df.iterrows(), total=len(filtered_feature_transfer_df),
                         desc="Processing"):
        exp1 = scaled_combined_df_m1[row['feature1']]
        exp2 = scaled_combined_df_m2[row['feature2']]
        zscore = exp2 / exp1
        reg_df[row['feature1'] + "::" + row['feature2']] = zscore

    # Make spatial kernel
    pos = np.concatenate((pos1, pos2), axis=0)
    dist_mat = distance_matrix(pos, pos)
    kern_mat = np.exp(-1 * (dist_mat ** 2) / (2 * (sigma ** 2)))
    # Weighted correlation
    wcor_mat = weighted_cor(reg_df.values, kern_mat, method=correlation)
    # Weighted distance
    pairwise_distances = pdist(wcor_mat)
    wcor_dis = squareform(pairwise_distances)
    
    # Consensus cluster
    print("Clustering feautres...")
    cc = ConsensusCluster(
        cluster=AgglomerativeClustering,
        L=minK,
        K=maxK,
        H=rep,
        resample_proportion=resample_proportion)
    cc.fit(wcor_dis)
    clusters = cc.predict()

    # Filter clusters
    print("Filtering feautres...")
    df = pd.DataFrame({'feature': reg_df.columns, 'cluster': clusters})
    cor_df = pd.DataFrame(wcor_mat, index=reg_df.columns, columns=reg_df.columns)
    con_df = pd.DataFrame(cc.Mk[cc.bestK - minK], index=reg_df.columns, columns=reg_df.columns)
    cc_res = cc_gene_k(df, 
                       cor_df, 
                       con_df, 
                       k=cc_k, 
                       avg_con_min=avg_con_min, 
                       avg_cor_min=avg_cor_min, 
                       min_feature=min_feature, 
                       max_feature=max_feature)
    
    cc_res['feature1'] = cc_res.combos.str.split('::').str[0]
    cc_res['feature2'] = cc_res.combos.str.split('::').str[1]
    
    output = {'wcor': cor_df, 'cc': cc_res, 'reg_df': reg_df, 'pos': pos}
    return output

def module_analysis(anndata, 
                   layer='data',
                    feature=None,
                   smooth_mode='spatial',
                   smooth_k=10,
                   sigma=140, 
                    correlation='pearson',
                       minK=2,
                       maxK=8,
                       rep=20,
                       resample_proportion=0.5,
                       cc_k=8,
                       avg_con_min=0.5,
                       avg_cor_min=0.5,
                       min_feature=20,
                       max_feature=200
                      ):
    """
    Performs spatial module analysis on a single-cell dataset by integrating spatial and expression data to discover functional gene modules.

    Parameters:
        anndata (AnnData): Single-cell dataset to analyze.
        layer (str): Specific layer of the AnnData object from which to extract expression data. Default is 'data'.
        feature (list, optional): List of specific features to include in the analysis; if None, all features are used.
        smooth_mode (str): Method of smoothing, can be 'spatial' using spatial coordinates or 'reduction' using dimensionality reduction coordinates. Default is 'spatial'.
        smooth_k (int): Number of nearest neighbors to consider in smoothing expression data. Default is 10.
        sigma (float): Standard deviation for Gaussian kernel in spatial smoothing. Default is 140.
        correlation (str): Method of correlation to use in weighted correlation calculations; 'pearson' or 'spearman'. Default is 'pearson'.
        minK (int): Minimum number of clusters for consensus clustering. Default is 2.
        maxK (int): Maximum number of clusters for consensus clustering. Default is 8.
        rep (int): Number of repetitions for consensus clustering to ensure stability. Default is 20.
        resample_proportion (float): Proportion of sample to resample in each repetition for consensus clustering. Default is 0.5.
        cc_k (int): Number of nearest neighbors in consensus clustering to consider for correlation. Default is 8.
        avg_con_min (float): Minimum average connectivity threshold to filter feature clusters. Default is 0.5.
        avg_cor_min (float): Minimum average correlation threshold to filter feature clusters. Default is 0.5.
        min_feature (int): Minimum number of features per cluster. Default is 20.
        max_feature (int): Maximum number of features per cluster. Default is 200.

    Returns:
        dict: A dictionary containing:
            - 'wcor': Weighted correlation matrix.
            - 'cc': Results from the consensus clustering.
    """
    print("Starting spatial module analysis...")


    # Extracting expression data and spatial positions
    df = extract_exp(anndata, layer=layer,gene=feature )
        
    pos = extract_reduction(anndata, use_rep='spatial', column_names=['x', 'y'])
    if smooth_mode == 'spatial':
        smooth_reduc= pos.copy()
        
    if smooth_mode == 'reduction':
        smooth_reduc= extract_reduction(anndata, use_rep='reduction')

    # Smoothing expression data
    smoothed_df = smooth_expression(df, smooth_reduc, k=smooth_k)

    # Make spatial kernel
    dist_mat = distance_matrix(pos, pos)
    kern_mat = np.exp(-1 * (dist_mat ** 2) / (2 * (sigma ** 2)))
    # Weighted correlation
    wcor_mat = weighted_cor(smoothed_df.values, kern_mat, method=correlation)
    # Weighted distance
    pairwise_distances = pdist(wcor_mat)
    wcor_dis = squareform(pairwise_distances)
    
    # Consensus cluster
    print("Clustering feautres...")
    cc = ConsensusCluster(
        cluster=AgglomerativeClustering,
        L=minK,
        K=maxK,
        H=rep,
        resample_proportion=resample_proportion)
    cc.fit(wcor_dis)
    clusters = cc.predict()

    # Filter clusters
    print("Filtering feautres...")
    df = pd.DataFrame({'feature': smoothed_df.columns, 'cluster': clusters})
    cor_df = pd.DataFrame(wcor_mat, index=smoothed_df.columns, columns=smoothed_df.columns)
    con_df = pd.DataFrame(cc.Mk[cc.bestK - minK], index=smoothed_df.columns, columns=smoothed_df.columns)
    cc_res = cc_gene_k(df, 
                       cor_df, 
                       con_df, 
                       k=cc_k, 
                       avg_con_min=avg_con_min, 
                       avg_cor_min=avg_cor_min, 
                       min_feature=min_feature, 
                       max_feature=max_feature)
    
    output = {'wcor': cor_df, 'cc': cc_res}
    
    return output
