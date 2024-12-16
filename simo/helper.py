import numpy as np
from anndata import AnnData
import anndata as ad
import scanpy as sc
import pandas as pd
from typing import Optional
import scipy.sparse
from typing import Optional, Union
from sklearn.preprocessing import normalize
import sklearn.utils.extmath
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from scipy.sparse import issparse
from scipy.stats import pearsonr
import math
import ot

###
### Alignment
###

def load_data(path):
    """
    Load data.
    
    Args:
        path: Path of anndata.
    """
    ann = ad.read_h5ad(path)
    return ann

def process_anndata(adata_raw, highly_variable_genes=True, normalize_total=True,
                    log1p=True, scale=True, pca=True, neighbors=True,
                    umap=True, n_top_genes=3000, n_comps=100,
                    mode='rna', ndim=30):
    """
    Processes raw AnnData objects based on specified modes and methods, preparing data for subsequent analyses.

    Args:
    adata_raw: AnnData object containing raw count data.
    highly_variable_genes (bool): If True, identifies highly variable genes. Default is True.
    normalize_total (bool): If True, normalizes total counts across all cells. Default is True.
    log1p (bool): If True, applies log1p transformation to the data. Default is True.
    scale (bool): If True, scales the data to unit variance. Default is True.
    pca (bool): If True, performs Principal Component Analysis (PCA). Default is True.
    neighbors (bool): If True, computes the neighborhood graph of cells using the cosine metric. Default is True.
    umap (bool): If True, performs Uniform Manifold Approximation and Projection (UMAP). Default is True.
    n_top_genes (int): Number of top highly variable genes to identify. Default is 3000.
    n_comps (int): Number of principal components to compute in PCA. Default is 100.
    mode (str): Specifies the mode of the data ('rna' or 'atac'). Default is 'rna'.
    ndim (int): Number of dimensions to use in PCA when computing neighbors. Default is 30.

    Returns:
    adata: Processed AnnData object. If 'rna' mode is selected, processes include identification of highly variable genes, normalization, log transformation, scaling, PCA, and neighbor graph computation. For 'atac' mode, it includes LSI processing and neighbor graph computation based on LSI components.

    Detailed processing steps vary based on the mode:

    For 'rna': Processing includes gene variability filtering, total count normalization, log transformation, data scaling, PCA, and neighbor computation.
    For 'atac': Includes Latent Semantic Indexing (LSI) and neighbor computation based on the LSI components.
    The function outputs comprehensive status updates during the processing, and returns the processed AnnData object suitable for downstream analysis like clustering or trajectory inference.
    """
    adata = adata_raw.copy()
    # Processing for RNA data
    if mode == 'rna':
        print("Processing RNA data...")
        
        if highly_variable_genes:
            print("Identifying highly variable genes...")
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3", span=0.6, min_disp=0.1)

        if normalize_total:
            print("Normalizing total counts...")
            adata.layers["counts"] = adata.X.copy()
            sc.pp.normalize_total(adata)

        if log1p:
            print("Applying log1p transformation...")
            sc.pp.log1p(adata)

        print("Saving pre-log1p counts to a layer...")
        adata.layers["data"] = adata.X.copy()

        if scale:
            print("Scaling the data...")
            sc.pp.scale(adata)

        if pca:
            print("Performing PCA...")
            sc.tl.pca(adata, n_comps=n_comps, svd_solver="auto")
            
        if neighbors:
            print("Calculating neighbors based on cosine metric...")
            sc.pp.neighbors(adata, metric="cosine", n_pcs=ndim)
            
    # Processing for ATAC data
    elif mode == 'atac':
        print("Processing ATAC data...")
        
        print("Running LSI...")
        run_lsi(adata, n_components=n_comps, n_iter=15)
        
        if neighbors:
            print("Calculating neighbors based on cosine metric using X_lsi...")
            sc.pp.neighbors(adata, metric="cosine", use_rep="X_lsi", n_pcs=ndim)
            
    if umap:
        print("Performing UMAP...")
        sc.tl.umap(adata)
    
    print("Processing completed.")
    return adata

def calculate_gene(adata, top_deg_num=100, maker_by="leiden"):
    """
    Calculate marker genes based on groupings/clusters in the AnnData object.

    Args:
        adata (AnnData): AnnData object.
        top_deg_num (int): Number of top marker genes to retrieve. Defaults to 100.
        maker_by (str): Key for group information based on which marker genes will be calculated. Defaults to "leiden".

    Returns:
        adata (AnnData): Updated AnnData object with a new 'group_marker' variable in `.var` attribute.
    """
    sc.tl.rank_genes_groups(adata, maker_by, method='wilcoxon', use_raw=False)
    marker_df = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(top_deg_num)
    marker_array = np.array(marker_df)
    marker_array = np.ravel(marker_array)
    marker_array = np.unique(marker_array)
    marker = list(marker_array)
    group_marker_genes = marker
    is_group_marker = pd.Series(False, index=adata.var_names)
    is_group_marker[group_marker_genes] = True
    adata.var['group_marker'] = is_group_marker.values
    return adata

def find_marker(
    adata1: AnnData, 
    adata2: AnnData, 
    gene_selection_method: str = None,
    deg_num: int = 10, 
    marker1_by: str = "leiden", 
    marker2_by: str = "leiden", 
    min_cells: int = 0,
    return_anndata: bool = False
) -> tuple:
    """
    Find marker genes between two AnnData objects based on specified groupings/clusters.

    Args:
        adata1 (AnnData): First AnnData object.
        adata2 (AnnData): Second AnnData object.
        gene_selection_method (str): Method for selecting marker genes. Options: 'deg' for differential expression genes,'hvg' for highly variable genes, or None for all common genes. Defaults to None.
        deg_num (int): Number of top marker genes to retrieve. Defaults to 10.
        marker1_by (str): Key for group information in adata1 based on which marker genes will be calculated. Defaults to "leiden".
        marker2_by (str): Key for group information in adata2 based on which marker genes will be calculated. Defaults to "leiden".
        min_cells (int): Minimum number of cells expressing a gene to be considered. Defaults to 0.

    Returns:
        Tuple containing updated adata1 and adata2 with calculated marker genes based on common genes.
    """
    #print("Finding marker genes...")

    # Filter genes based on min_cells threshold
    if min_cells > 0:
        sc.pp.filter_genes(adata1, min_cells=min_cells)
        sc.pp.filter_genes(adata2, min_cells=min_cells)   
    
    # Select common genes between adata1 and adata2
    common_genes = np.intersect1d(adata1.var.index, adata2.var.index)
    print(f"Number of common genes: {len(common_genes)}")
    adata1 = adata1[:, common_genes]
    adata2 = adata2[:, common_genes]
    
    if gene_selection_method=='deg':
        # Calculate marker genes for adata1 and adata2 based on specified groupings
        #print(f"Calculating marker genes based on '{marker1_by}' for adata1...")
        adata1 = calculate_gene(adata1, top_deg_num=deg_num, maker_by=marker1_by)
        
        #print(f"Calculating marker genes based on '{marker2_by}' for adata2...")
        adata2 = calculate_gene(adata2, top_deg_num=deg_num, maker_by=marker2_by)
        
        #print("Marker gene calculation completed.")
        lst1 = list(adata1.var.index[adata1.var['group_marker']])
        lst2 = list(adata2.var.index[adata2.var['group_marker']])
        lst1.extend(lst2)
        gene_list = np.unique(lst1).tolist()
        
    elif gene_selection_method=='hvg':
        lst1 = list(adata1.var.index[adata1.var['highly_variable']])
        lst2 = list(adata2.var.index[adata2.var['highly_variable']])
        lst1.extend(lst2)
        gene_list = np.unique(lst1).tolist()
        
    else:
        gene_list = np.unique(list(adata1.var.index)).tolist()
        
    if return_anndata:
        return adata1, adata2
    else:
        return gene_list
    

def average_expression(adata: AnnData, avg_by: str = 'leiden', layer: str = None) -> pd.DataFrame:
    """
    Calculate the average gene expression for each category/group defined by 'avg_by'.

    Args:
        adata (AnnData): AnnData object.
        avg_by (str): Key for grouping categories based on which average expression will be calculated. Defaults to 'leiden'.
        layer (str): Optional - Layer of data to calculate average expression from. Defaults to None (use adata.X).

    Returns:
        mean_expression_df (pd.DataFrame): DataFrame containing mean expression values for each category.
    """
    unique_categories = np.unique(adata.obs[avg_by])
    mean_expression_by_category = []
    
    if layer is None:
        data_to_avg = adata.X
    elif layer in adata.layers:
        data_to_avg = adata.layers[layer]
    else:
        raise ValueError(f"Layer '{layer}' not found in adata.layers.")
    
    for category in unique_categories:
        category_indices = adata.obs[avg_by] == category
        gene_expression = data_to_avg[category_indices]
        mean_expression = np.mean(gene_expression, axis=0)
        mean_expression = np.asarray(mean_expression).ravel()
        mean_expression_by_category.append(mean_expression)
    
    mean_expression_df = pd.DataFrame(mean_expression_by_category, columns=adata.var_names, index=unique_categories)
    return mean_expression_df


def map_test_to_reference(data, transfer_df, type_in_data='leiden'):
    """
    Map labels from test data to reference data based on a transfer probability DataFrame.

    Args:
        data (AnnData): Test data (AnnData object).
        transfer_df (DataFrame): DataFrame containing transfer probabilities between test and reference data.
        type_in_data (str): Key for the annotation type in the test data. Defaults to 'leiden'.

    Returns:
        filtered_data (AnnData): Test data with updated labels transferred from reference data.
    """
    melted_df = transfer_df.reset_index().melt(id_vars='index', var_name='Cell_Type', value_name='score')
    sorted_df = melted_df.sort_values(by='score', ascending=False)
    sorted_df = sorted_df.rename(columns={sorted_df.columns[0]: 'reference', sorted_df.columns[1]: 'test'})
    sorted_df = sorted_df[sorted_df['score'] > 0]
    
    filtered_data = data[data.obs[type_in_data].isin(sorted_df['test'].unique())]

    mapping_dict = {}
    for test_cell_type in sorted_df['test'].unique():
        test_rows = sorted_df[sorted_df['test'] == test_cell_type]
        best_match = test_rows[test_rows['score'] == test_rows['score'].max()]
        if not best_match.empty:
            mapping_dict[test_cell_type] = best_match['reference'].values[0]
    filtered_data.obs['annotation'] = filtered_data.obs[type_in_data].map(mapping_dict)
    
    return filtered_data        
        
def label_transfer(test_data,
                    reference_data,
                    layer=None,
                    test_avg_by = 'leiden',
                    reference_avg_by = 'leiden',
                    reg = 1,
                    cutoff = 0.5,
                    modality2_type = "pos"):
    """
    Perform transfer of labels from reference_data to test_data based on gene expression correlations.

    Args:
        test_data (AnnData): Data containing test samples.
        reference_data (AnnData): Data containing reference samples.
        layer (str, optional): Layer to use from AnnData object. Default is None.
        test_avg_by (str, optional): Method to average test data by. Default is 'leiden'.
        reference_avg_by (str, optional): Method to average reference data by. Default is 'leiden'.
        reg (int, optional): Regularization parameter for unbalanced optimal transport. Default is 1.
        cutoff (float, optional): Cutoff value for mapping probabilities. Default is 0.5.

    Returns:
        test_data (AnnData): Test data with transferred labels.
        reference_data (AnnData): Reference data.
        transfer_df (DataFrame): DataFrame containing transfer probabilities.
    """
    print("Performing label transfer...")

    print("Calculating average expression...")
    test_exp = average_expression(test_data, layer=layer, avg_by=test_avg_by).T
    reference_exp = average_expression(reference_data, layer=layer, avg_by=reference_avg_by).T
    
    # Compute correlation matrix
    print("Computing correlation matrix...")
    exp1 = reference_exp.T
    exp2 = test_exp.T
    correlation_df = pd.DataFrame(index=exp1.index, columns=exp2.index)
    for cell_type1 in exp1.index:
        for cell_type2 in exp2.index:
            gene_expr1 = exp1.loc[cell_type1]
            gene_expr2 = exp2.loc[cell_type2]
            correlation, _ = pearsonr(gene_expr1, gene_expr2)
            correlation_df.loc[cell_type1, cell_type2] = correlation
    correlation_df = scale_dataframe_to_01(correlation_df).astype('float')
    
    # Compute transfer probabilities using unbalanced optimal transport
    print("Performing unbalanced optimal transport...")
    if modality2_type == "neg":
        M = correlation_df.values
    else:
        M = 1 - correlation_df.values
        
    a = np.ones((correlation_df.shape[0],)) / correlation_df.shape[0]
    b = np.ones((correlation_df.shape[1],)) / correlation_df.shape[1]
    
    l2_uot = ot.unbalanced.mm_unbalanced(a, b, M, reg, div='kl')
    transfer_df = pd.DataFrame(l2_uot, index=correlation_df.index, columns=correlation_df.columns).astype(float)
    transfer_df = scale_dataframe_to_01(transfer_df).astype('float')
    
    # Apply cutoff to transfer probabilities and map labels
    print("Applying cutoff to transfer probabilities and mapping labels...")
    transfer_df = transfer_df.applymap(lambda x: 0 if x < cutoff else x)
    test_data = map_test_to_reference(test_data, transfer_df, type_in_data=test_avg_by)
    reference_data.obs['annotation'] = reference_data.obs[reference_avg_by]
    
    print("Label transfer completed.")
    return test_data, reference_data, transfer_df

###
### Extract
###

def extract_exp(data, layer=None, gene = None):
    """
    Extract gene expression data from the given data object.

    Args:
        data (AnnData): AnnData object.
        layer (str): Optional - Layer of data from which to extract expression data. Defaults to None (use data.X).
        gene (str or list): Optional - Gene name or list of gene names to extract expression data for.

    Returns:
        exp_data (pd.DataFrame): DataFrame containing gene expression data.
    """
    if layer is None:
        if issparse(data.X):
            expression_data = data.X.toarray()
        else:
            expression_data = data.X
    elif layer in data.layers:
        if issparse(data.layers[layer]):
            expression_data = data.layers[layer].toarray()
        else:
            expression_data = data.layers[layer]
    else:
        raise ValueError(f"Layer '{layer}' not found in data.layers.")

    exp_data = pd.DataFrame(expression_data)
    exp_data.columns = data.var.index.tolist()
    exp_data.index = data.obs.index.tolist()
    
    if gene is not None:
        exp_data = exp_data.loc[:,gene]
    
    return exp_data

def extract_reduction(data: AnnData, use_rep: str = 'reduction', column_names: list = None) -> pd.DataFrame:
    """
    Extract the reduced dimensions (e.g., PCA, tSNE) from an AnnData object.

    Args:
        data (AnnData): AnnData object.
        use_rep (str): Key for the representation in the 'obsm' attribute of AnnData objects. Defaults to 'reduction'.

    Returns:
        reduction_df (pd.DataFrame): DataFrame containing the reduced dimensions.
    """
    if column_names is not None:
        reduction_df = pd.DataFrame(data.obsm[use_rep], columns=column_names)
        reduction_df.index = data.obs.index.tolist()
    else:
        reduction_df = pd.DataFrame(data.obsm[use_rep])
        reduction_df.index = data.obs.index.tolist()
    return reduction_df

def scale_dataframe_to_01(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale values in a DataFrame to the range [0, 1].

    Args:
        df (pd.DataFrame): Input DataFrame containing numerical values.

    Returns:
        scaled_df (pd.DataFrame): DataFrame with values scaled to the range [0, 1].
    """
    min_value = df.min().min()
    max_value = df.max().max()
    scaled_df = (df - min_value) / (max_value - min_value)
    return scaled_df

###
### LSI
###

Array = Union[np.ndarray, scipy.sparse.spmatrix]
def tfidf(X: np.ndarray) -> np.ndarray:
    """
    Calculate TF-IDF (Term Frequency-Inverse Document Frequency) matrix.

    Args:
        X (np.ndarray): Input matrix or sparse matrix.

    Returns:
        np.ndarray: TF-IDF weighted matrix.
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf

def run_lsi(
    adata: AnnData, n_components: int = 20,
    use_highly_variable: Optional[bool] = None, **kwargs
) -> None:
    """
    Run Latent Semantic Indexing (LSI) on input AnnData object.

    Args:
        adata (AnnData): Annotated data object.
        n_components (int): Number of components for LSI.
        use_highly_variable (bool, optional): Whether to use highly variable genes. Defaults to None.
        **kwargs: Additional keyword arguments for randomized_svd.
    """
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = normalize(X, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi
    
# Covert a sparse matrix into a dense np array
to_dense_array = lambda X: X.toarray() if isinstance(X,scipy.sparse.csr.spmatrix) else np.array(X)

# Returns the data matrix or representation
extract_data_matrix = lambda adata,rep: adata.X if rep is None else adata.obsm[rep] 

###
### Graph
###

def construct_graph(X, k, mode= "connectivity", metric="minkowski",p=2):
    """
    Construct graph with KNN.
    
    Args:
        X: Input data containing features.
        k (int): Number of neighbors for each data point.
        mode (str): Optional - Mode for constructing the graph, either 'connectivity' or 'distance'. Defaults to 'connectivity'.
        metric (str): Optional - Name of the distance metric to use. Defaults to 'minkowski'.
        p (int): Optional - Parameter for the Minkowski metric. Defaults to 2.

    Returns:
        -The knn graph of input data. 
    """
    assert (mode in ["connectivity", "distance"]), "Norm argument has to be either one of 'connectivity', or 'distance'. "
    if mode=="connectivity":
        include_self=True
    else:
        include_self=False
    c_graph=kneighbors_graph(X, k, mode=mode, metric=metric, include_self=include_self,p=p)
    return c_graph

def distances_cal(graph,type_aware=None,aware_power=2):
    """
    Calculate distance between cells/spots based on graph.
    
    Args:
        graph: KNN graph.
        type_aware: A dataframe contains cells/spots id and type information.
        aware_power: Type aware parameter. The greater the parameter, the greater the distance between different areas/types of spots/cells in the graph.
        
    Returns:
        -The distance matrix of cells/spots. 
    """
    
    shortestPath=dijkstra(csgraph= csr_matrix(graph), directed=False, return_predecessors=False)
    if type_aware is not None:
        shortestPath = to_dense_array(shortestPath)
        shortestPath = pd.DataFrame(shortestPath)
        shortestPath.index = type_aware.index
        shortestPath.columns = type_aware.index
        shortestPath['id1']=shortestPath.index
        shortestPath = shortestPath.melt(id_vars=['id1'], var_name ='id2', value_name='value')

        meta1 = type_aware.copy()
        meta1.columns = ['id1','type1']
        meta2 = type_aware.copy()
        meta2.columns = ['id2','type2']

        shortestPath = pd.merge(shortestPath, meta1, on='id1',how="left")
        shortestPath = pd.merge(shortestPath, meta2, on='id2',how="left")

        shortestPath['same_type'] = shortestPath['type1']==shortestPath['type2']
        shortestPath.loc[(shortestPath.same_type == False), 'value'] = shortestPath.loc[(shortestPath.same_type == False), 'value']*aware_power
        shortestPath.drop(['type1','type2','same_type'],axis=1,inplace=True)
        shortestPath = shortestPath.pivot(index='id1', columns='id2',values = 'value')
        
        order = type_aware.index.tolist()
        shortestPath = shortestPath[order]
        shortestPath = shortestPath.loc[order]
        shortestPath = shortestPath.values

    the_max=np.nanmax(shortestPath[shortestPath != np.inf])
    shortestPath[shortestPath > the_max] = the_max
    C_dis=shortestPath/shortestPath.max()
    C_dis -= np.mean(C_dis)
    return C_dis

###
### Datasets
###

def intersect_datasets(data1, data2, by="annotation"):
    """
    Get the intersection of two AnnData objects based on annotation.

    Args:
        data1 (AnnData): The first AnnData object.
        data2 (AnnData): The second AnnData object.

    Returns:
        AnnData: The intersection of data1 and data2.
    """
    # Find the intersection of annotations
    common_annotations = list(set(data1.obs[by]).intersection(data2.obs[by]))

    # Filter data1 and data2 based on the common_annotations
    intersected_data1 = data1[data1.obs[by].isin(common_annotations)]
    intersected_data2 = data2[data2.obs[by].isin(common_annotations)]

    return intersected_data1, intersected_data2

def sort_datasets(adata1=None, adata2=None, suffix1=None, suffix2=None):
    """
    Sort and subset two AnnData objects based on their cell names.

    Args:
    - adata1 (AnnData): First AnnData object.
    - adata2 (AnnData): Second AnnData object.
    - suffix1 (str): Suffix used in the cell names of adata1 for sorting.
    - suffix2 (str): Suffix used in the cell names of adata2 for sorting.

    Returns:
    - Tuple containing sorted and subsetted adata1 and adata2 based on matching cell names.
    """
    cell_names1 = sorted([cell_name[:-len(suffix1)] for cell_name in adata1.obs_names if cell_name.endswith(suffix1)])
    cell_names2 = sorted([cell_name[:-len(suffix2)] for cell_name in adata2.obs_names if cell_name.endswith(suffix2)])
    merged_cell_names = intersect(cell_names1,cell_names2)
    adata1 = adata1[[cell_name + suffix1 for cell_name in merged_cell_names], :]
    adata2 = adata2[[cell_name + suffix2 for cell_name in merged_cell_names], :]
    return adata1, adata2

def intersect_cells(adata1=None, adata2=None, suffix1=None, suffix2=None):
    """
    Intersect and subset two AnnData objects based on intersecting cell names.

    Args:
        adata1 (AnnData): First AnnData object.
        adata2 (AnnData): Second AnnData object.
        suffix1 (str): Suffix used in the cell names of adata1 for intersecting cells.
        suffix2 (str): Suffix used in the cell names of adata2 for intersecting cells.

    Returns:
        Tuple containing intersected and subsetted adata1 and adata2 based on matching cell names.
    """
    cell_names1 = sorted([cell_name[:-len(suffix1)] for cell_name in adata1.obs_names if cell_name.endswith(suffix1)])
    cell_names2 = sorted([cell_name[:-len(suffix2)] for cell_name in adata2.obs_names if cell_name.endswith(suffix2)])
    merged_cell_names = list(zip(cell_names1, cell_names2))
    adata1 = adata1[[cell_name + suffix1 for cell_name, _ in merged_cell_names], :]
    adata2 = adata2[[cell_name + suffix2 for _, cell_name in merged_cell_names], :]
    return adata1, adata2

def intersect(lst1, lst2): 
    """
    Gets and returns intersection of two lists.

    Args:
        lst1: List
        lst2: List
    
    Returns:
        lst3: List of common elements.
    """
    
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3 

def subset_ndimension(adata1=None, adata2=None,dim_num =None):
    """
    Subset the n dimensions from reduction data in two AnnData objects.

    Args:
        adata1 (AnnData): First AnnData object.
        adata2 (AnnData): Second AnnData object.
        dim_num (int): Number of dimensions to subset. Defaults to None.

    Returns:
        adata1 (AnnData): Updated AnnData object 1 with subsetted dimensions.
        adata2 (AnnData): Updated AnnData object 2 with subsetted dimensions.
    """
    if dim_num is None:
        dim_num = min(adata1.obsm['reduction'].shape[1], adata2.obsm['reduction'].shape[1])
    if isinstance(adata1.obsm['reduction'], pd.DataFrame):
        adata1.obsm['reduction'] = adata1.obsm['reduction'].iloc[:, :dim_num]
        adata2.obsm['reduction'] = adata2.obsm['reduction'].iloc[:, :dim_num]
    elif isinstance(adata1.obsm['reduction'], np.ndarray):
        adata1.obsm['reduction'] = adata1.obsm['reduction'][:, :dim_num]
        adata2.obsm['reduction'] = adata2.obsm['reduction'][:, :dim_num]
    return adata1, adata2

def top_n(df, n=3, column='APM'):
    """
    Get a subset of the DataFrame according to the values of a column.
    
    """
    return df.sort_values(by=column, ascending=False)[:n]

def save_integrated_data(adata, name, path):
    integrated_emb = extract_reduction(adata, use_rep='integrated')
    integrated_umap = extract_reduction(adata, use_rep='X_umap')
    integrated_emb.to_csv(f"{path}/{name}_integrated_emb.csv", index=True)
    integrated_umap.to_csv(f"{path}/{name}_integrated_umap.csv", index=True)
    adata.obs.to_csv(f"{path}/{name}_integrated_obs.csv", index=True)
    adata.write_h5ad(f"{path}/{name}_integrated.h5ad")
    
def dist_cal(x1,y1,x2,y2):
    dist_x = x2 - x1
    dist_y = y2 - y1
    square_all = math.sqrt(dist_x*dist_x + dist_y*dist_y)
    return square_all

def scale_num(list):
    """
    Scale the input list.

    Args:
        list: List
    
    Returns:
        scale_list: List of scaled elements.
    """
    
    a = max(list)
    b = min(list)
    scale_list = []
    for i in list:
        scale_num = (i-b)/(a-b)
        scale_list.append(scale_num)
    return scale_list

###
### Simulation
###

def simulate_gene_exp(adata, pc = 0.25, factor = 1):
    adata_sim = adata.copy()
    df = extract_exp(adata_sim)
    # add pseudocounts 
    alpha = df.copy().to_numpy() + pc

    # get vector of total counts per spot
    n = df.sum(axis=1).to_numpy()

    # Simulate total counts using negative binomial
    mean = np.mean(n)
    var = np.var(n)*factor
    n = sample_nb(mean, var, len(n)).astype(int)

    # Reassign zero counts so we don't divide by 0 in future calcuation
    n[n == 0] = 1

    # convert to float
    alpha = np.array(alpha, dtype=np.float64)
    n = np.array(n, dtype=np.float64)

    # convert rows to unit vectors
    alpha = alpha/alpha.sum(axis=1)[:, None]

    dist = np.empty(df.shape)
    for i in range(alpha.shape[0]):
        dist[i] = np.random.multinomial(n[i], alpha[i])
    new_df = pd.DataFrame(dist, index= df.index, columns= df.columns)
    adata_sim.X = new_df
    return adata_sim

def sample_nb(m, v, n = 1):
    r = m**2/(v - m)
    p = m/v
    samples = np.random.negative_binomial(r, p, n)
    return samples


def shuffle_cells(anndata_obj, random_seed=1234):
    np.random.seed(random_seed)
    original_order = np.arange(anndata_obj.n_obs)
    shuffled_order = np.random.permutation(original_order)
    shuffled_anndata = anndata_obj[shuffled_order, :]
    return shuffled_anndata
