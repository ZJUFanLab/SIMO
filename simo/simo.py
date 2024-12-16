import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from anndata import AnnData
from scipy.optimize import nnls
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
import ot
from .helper import *

def alignment_1(
    adata1: AnnData,  
    adata2: AnnData,
    alpha: float = 0.1, 
    G_init = None,
    p_distribution = None, 
    q_distribution = None, 
    numItermax: int = 200, 
    norm: str = 'l2', 
    backend = ot.backend.NumpyBackend(),  
    return_obj: bool = False,
    verbose: bool = False, 
    k: int = 10,
    graph_mode: str = "connectivity",
    aware_st: bool = True,
    aware_sc: bool = True,
    aware_st_label: str = "type",
    aware_sc_label: str = "type",
    aware_power_st: int = 2,
    aware_power_sc: int = 2,
    **kwargs) :
    
    """
    Performs probabilistic alignment of spatial transcriptomic data to single-cell RNA-seq data using fused Gromov-Wasserstein optimal transport (FGW-OT).

    Args:
        adata1 (AnnData): AnnData object containing spatial transcriptomic data.
        adata2 (AnnData): AnnData object containing single-cell RNA-seq data.
        alpha (float): Alignment tuning parameter, controlling the balance between feature and spatial alignment. Values range from 0 to 1.
        G_init (np.ndarray, optional): Initial guess for the transport plan. Default is uniform distribution.
        p_distribution (np.ndarray, optional): Distribution of spots in `adata1`. Default is uniform.
        q_distribution (np.ndarray, optional): Distribution of cells in `adata2`. Default is uniform.
        numItermax (int): Maximum number of iterations for the optimization. Default is 200.
        norm (str): Normalization method for low-dimensional representations, options include 'l1', 'l2', and 'max'. Default is 'l2'.
        backend (ot.backend.Backend): Backend for computational operations, usually set to ot.backend.NumpyBackend().
        return_obj (bool): If True, returns the objective function value along with the optimal transport plan. Default is False.
        verbose (bool): If True, prints optimization loss at each iteration. Default is False.
        k (int): Number of neighbors for constructing k-nearest neighbor graphs. Default is 10.
        graph_mode (str): Type of graph to construct, either 'connectivity' or 'distance'. Default is 'connectivity'.
        aware_st (bool): If True, adjusts distances between spots based on metadata in `aware_st_label`. Default is True.
        aware_sc (bool): If True, adjusts distances between cells based on metadata in `aware_sc_label`. Default is True.
        aware_st_label (str): Metadata label in `adata1` used for distance adjustment. Default is 'type'.
        aware_sc_label (str): Metadata label in `adata2` used for distance adjustment. Default is 'type'.
        aware_power_st (int): Power for distance adjustment in spots. Default is 2.
        aware_power_sc (int): Power for distance adjustment in cells. Default is 2.
        **kwargs: Additional keyword arguments for lower-level functions.

    Returns:
        If return_obj is False:
            pd.DataFrame: DataFrame with alignment results, showing transport values between spots and cells.
        If return_obj is True:
            tuple: Transport plan and DataFrame with alignment results.
    """
    nx = backend
    n1 = adata1.shape[0]
    n2 = adata2.shape[0]
    
    print('Calculating dissimilarity using euclidean distance on scaled data...')
    
    
    ###
    adata_merge = ad.AnnData(X=adata1.layers["counts"]).concatenate(ad.AnnData(X=adata2.layers["counts"]),index_unique=None)
    # Process the merged anndata
    adata_merge = process_anndata(adata_merge,neighbors=False,umap=False,pca=False)
    reduc_spot = adata_merge.X[0:n1,]
    reduc_single = adata_merge.X[n1:n1+n2,]
    M = ot.dist(reduc_spot, reduc_single, metric='euclidean')
    M /= M.max()
    M = nx.from_numpy(M)
    ###

    # exp_spot = adata1.X
    # exp_single = adata2.X
    # M = ot.dist(exp_spot, exp_single, metric='euclidean')
    # M /= M.max()
    # M = nx.from_numpy(M)

    # Construct the graph
    if isinstance(adata1.obsm['spatial'], pd.DataFrame):
        location_array = adata1.obsm['spatial'].values
    else:
        location_array = adata1.obsm['spatial']

    if isinstance(adata2.obsm['reduction'], pd.DataFrame):
        reduction_array = adata2.obsm['reduction'].values
    else:
        reduction_array = adata2.obsm['reduction']
    
    reduction_array =normalize(reduction_array, norm=norm, axis=1)
    #Xgraph = construct_graph(location_array, k=k, mode=graph_mode, type_aware=spot_meta, aware_power=0)###break up
    print('Constructing '+str(graph_mode)+"...")
    print('k = '+str(k))
    Xgraph = construct_graph(location_array, k=k, mode=graph_mode)
    ygraph = construct_graph(reduction_array, k=k, mode=graph_mode)

    # Adjust the distance according to meta info
    type_aware1 = None
    type_aware2 = None
    if aware_st:
        print('aware_st = True')
        type_aware_dict = {
        'spot': pd.Series(adata1.obs.index.tolist(),index=adata1.obs.index.tolist()),
        'spot_type': pd.Series(adata1.obs[aware_st_label],index=adata1.obs.index.tolist())
        }
        type_aware1 = pd.DataFrame(type_aware_dict)
        print('aware power = '+str(aware_power_st))
    if aware_sc:
        print('aware_sc = True')
        type_aware_dict = {
        'single': pd.Series(adata2.obs.index.tolist(),index=adata2.obs.index.tolist()),
        'single_type': pd.Series(adata2.obs[aware_sc_label],index=adata2.obs.index.tolist())
        }
        type_aware2 = pd.DataFrame(type_aware_dict)
        print('aware power = '+str(aware_power_sc))
        
    Cx = distances_cal(Xgraph,type_aware=type_aware1, aware_power=aware_power_st)
    Cy = distances_cal(ygraph,type_aware=type_aware2, aware_power=aware_power_sc)
    Cx = nx.from_numpy(Cx)
    Cy = nx.from_numpy(Cy)

    # Init distributions
    if p_distribution is None:
        p = np.ones((n1,)) / n1
        p = nx.from_numpy(p)
    else:
        p = nx.from_numpy(p_distribution)
    if q_distribution is None:
        q = np.ones((n2,)) / n2
        q = nx.from_numpy(q)
    else:
        q = nx.from_numpy(q_distribution)

    # Run OT
    if G_init is not None:
        G_init = nx.from_numpy(G_init)
   
    print('Running OT...')
    print('alpha = '+str(alpha))
    pi, logw = fgw_ot(M, Cx, Cy, p, q, G_init = G_init, loss_fun='square_loss', alpha= alpha, log=True, numItermax=numItermax,verbose=verbose)
    pi = nx.to_numpy(pi)
    print("OT done!")
    #obj = nx.to_numpy(logw['fgw_dist'])
    out_data = pd.DataFrame(pi)
    out_data.columns = adata2.obs.index
    out_data.index =  adata1.obs.index

    # Filter the results
    out_data['spot']=out_data.index
    out_data = out_data.melt(id_vars=['spot'], var_name ='cell', value_name='value')
    out_data = out_data.sort_values(by="value",ascending=False)

    if return_obj:
        return pi, out_data
    return out_data


def fgw_ot(M, C1, C2, p, q, G_init = None, loss_fun='square_loss', alpha=0.1, armijo=False, log=False,numItermax=200,numItermaxEmd=10e6, use_gpu = False, **kwargs):
    """
    Adapted fused_gromov_wasserstein with G_init (inital mapping).
    Also added capability of utilizing different POT backends to speed up computation.
    
    For more info, see: https://pythonot.github.io/gen_modules/ot.gromov.html
    """
    p, q = ot.utils.list_to_array(p, q)
    p0, q0, C10, C20, M0 = p, q, C1, C2, M
    nx = ot.backend.get_backend(p0, q0, C10, C20, M0)
    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun)

    if G_init is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = (1/nx.sum(G_init)) * G_init
        if use_gpu:
            G0 = G0.cuda()
    def f(G):
        return ot.gromov.gwloss(constC, hC1, hC2, G)
    def df(G):
        return ot.gromov.gwggrad(constC, hC1, hC2, G)
    if log:
        res, log = ot.gromov.cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, log=True, numItermaxEmd=numItermaxEmd, **kwargs)

        fgw_dist = log['loss'][-1]

        log['fgw_dist'] = fgw_dist
        log['u'] = log['u']
        log['v'] = log['v']
        return res, log
    else:
        return ot.gromov.cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, **kwargs)
    
def alignment_2(
    adata1: AnnData,  
    adata2: AnnData,
    coor_df: pd.DataFrame,
    reg: float = 1, 
    cutoff: float = 0,
    layer: str = None,  
    transfer_obs1 = None,
    transfer_obs2 = None,
    adata1_avg_by = 'leiden',
    adata2_avg_by = 'leiden',
    modality2_type = "pos",
    p_distribution = None, 
    q_distribution = None, 
    norm: str = 'l2', 
    backend = ot.backend.NumpyBackend(),  
    k: int = 10,
    graph_mode: str = "connectivity",
    reverse=False,
    **kwargs):
    """
    Performs probabilistic alignment of spatial transcriptomic data to single-cell RNA-seq data using fused Gromov-Wasserstein optimal transport (FGW-OT).

    Args:
        adata1 (AnnData): AnnData object containing single-cell RNA-seq data.
        adata2 (AnnData): AnnData object containing single-cell omics data.
        coor_df (pd.DataFrame): DataFrame linking spots and cells, used for final result mapping.
        reg (float): Regularization parameter.
        cutoff (float): Cutoff parameter for deciding on label transfer.
        layer (str, optional): Specifies the data layer to use for analysis if not the default. Default is None.
        transfer_obs1 (str, optional): Observation key in `adata1` for pre-existing cell type annotations.
        transfer_obs2 (str, optional): Observation key in `adata2` for pre-existing cell type annotations.
        adata1_avg_by (str): Grouping variable for averaging data in `adata1`. Default is 'leiden'.
        adata2_avg_by (str): Grouping variable for averaging data in `adata2`. Default is 'leiden'.
        modality2_type (str): Specifies the type of modality for the second dataset, can be 'pos' for positional data. Default is 'pos'.
        p_distribution (np.ndarray, optional): Custom probability distribution for `adata1`. Default is None.
        q_distribution (np.ndarray, optional): Custom probability distribution for `adata2`. Default is None.
        norm (str): Normalization method for data matrices, options include 'l1', 'l2', 'max'. Default is 'l2'.
        backend (ot.backend.Backend): Computational backend used for operations, default is ot.backend.NumpyBackend().
        k (int): Number of neighbors for constructing k-nearest neighbor graphs. Default is 10.
        graph_mode (str): Type of graph to construct, either 'connectivity' or 'distance'. Default is 'connectivity'.
        reverse (bool): If True, reverse the role of `adata1` and `adata2` in label transfer. Default is False.
        **kwargs: Additional arguments for Gromov-Wasserstein calculation.

    Returns:
        out_data (pd.DataFrame): DataFrame containing alignment results with columns ['spot', 'cell', 'value'] indicating transport values.
        transfer_df (pd.DataFrame): DataFrame with label transfer results.
        obs_df1 (pd.DataFrame): DataFrame with observations from `adata1`.
        obs_df2 (pd.DataFrame): DataFrame with observations from `adata2`.
    """

    if transfer_obs1 is not None and transfer_obs2 is not None:
        print('Use existing types for analysis...')
        adata1.obs['annotation'] = adata1.obs[transfer_obs1]
        adata2.obs['annotation'] = adata2.obs[transfer_obs2]
    else:
        if reverse:
            adata2, adata1, transfer_df = label_transfer(test_data = adata2,
                                                                reference_data = adata1,
                                                                layer=layer,
                                                                test_avg_by=adata2_avg_by,
                                                                reference_avg_by=adata1_avg_by,
                                                                reg=reg,
                                                                cutoff = cutoff,
                                                                modality2_type = modality2_type)
        else:
            adata1, adata2, transfer_df = label_transfer(test_data = adata1,
                                                        reference_data = adata2,
                                                        layer=layer,
                                                        test_avg_by=adata1_avg_by,
                                                        reference_avg_by=adata2_avg_by,
                                                        reg=reg,
                                                        cutoff = cutoff,
                                                        modality2_type = modality2_type)
    
        obs_df1 = adata1.obs.copy()
        obs_df2 = adata2.obs.copy()
        
    # Get the intersecting cell types
    intersect_type = intersect(adata1.obs['annotation'].unique().tolist(), adata2.obs['annotation'].unique().tolist())

    # Initialize an empty DataFrame to store the results
    out_data = pd.DataFrame()

    nx = backend
    # Iterate over each intersect_type with a progress display
    for idx, cell_type in enumerate(intersect_type):
        print(f"Processing cell type {idx + 1}/{len(intersect_type)}: {cell_type}")

        # Select the corresponding cell types from ann1 and ann2
        ann1 = adata1[adata1.obs['annotation'] == cell_type]
        ann2 = adata2[adata2.obs['annotation'] == cell_type]

        # Perform uot_alignment on the subsets
        if k > min(ann1.shape[0], ann2.shape[0]):
            k = min(ann1.shape[0], ann2.shape[0])
            
        n1 = ann1.shape[0]
        n2 = ann2.shape[0]
        
        # Construct the graph
        #num_columns = min(ann1.obsm['reduction'].shape[1], ann2.obsm['reduction'].shape[1])
        reduction_array_1 = ann1.obsm['reduction']
        reduction_array_2 = ann2.obsm['reduction']
        reduction_array_1 = normalize(reduction_array_1, norm=norm, axis=1)
        reduction_array_2 = normalize(reduction_array_2, norm=norm, axis=1)

        # Construct graphs for alignment
        Xgraph = construct_graph(reduction_array_1, k=k, mode=graph_mode)
        ygraph = construct_graph(reduction_array_2, k=k, mode=graph_mode)

        # Calculate distances and convert to NetworkX graphs
        Cx = distances_cal(Xgraph)
        Cy = distances_cal(ygraph)
        
        Cx = nx.from_numpy(Cx)
        Cy = nx.from_numpy(Cy)
        
        # Init distributions
        if p_distribution is None:
            p = np.ones((n1,)) / n1
            p = nx.from_numpy(p)
        else:
            p = nx.from_numpy(p_distribution)
        if q_distribution is None:
            q = np.ones((n2,)) / n2
            q = nx.from_numpy(q)
        else:
            q = nx.from_numpy(q_distribution)
        
        pi, log = ot.gromov.gromov_wasserstein(Cx, Cy, p, q, log=True,**kwargs)

        alignment_result = pd.DataFrame(pi, 
                                        index=ann1.obs_names, 
                                        columns=ann2.obs_names).astype(float)
        alignment_result['cell1'] = alignment_result.index
        alignment_result = alignment_result.melt(id_vars=['cell1'], var_name='cell2', value_name='value')
        alignment_result = alignment_result.sort_values(by="value", ascending=False)
        alignment_result = alignment_result[alignment_result['value'] > 0]

        # Concatenate the alignment result with the existing results
        out_data = pd.concat([out_data, alignment_result])

    # Reset the index of the result DataFrame
    out_data.reset_index(drop=True, inplace=True)
    out_data = out_data[['cell1', 'cell2', 'value']]
    
    out_data = coor_df[['spot', 'cell']].merge(out_data[['cell1', 'cell2','value']], left_on='cell', right_on='cell1', how='left')
    out_data = out_data[['spot', 'cell2', 'value']]
    out_data.rename(columns={'cell2': 'cell'}, inplace=True)
    
    return out_data,transfer_df,obs_df1,obs_df2


def assign_coord_1(
    adata1: AnnData,
    adata2: AnnData,
    out_data: pd.DataFrame, 
    non_zero_probabilities: bool = True,
    no_repeated_cells: bool = True,
    top_num: int = None,
    expected_num: pd.DataFrame = None,
    pos_random: bool = False,
    layer: str = 'data',
    weighted_average: bool = False
    ) :
    """
    Assigns spatial coordinates to single cells based on the output of optimal transport alignment, using additional logic for cell distribution and optional randomness.

    Args:
        adata1 (AnnData): Spatial transcriptomic data, which includes spatial coordinates for each observation (spot).
        adata2 (AnnData): Single cell data for which spatial coordinates need to be inferred.
        out_data (pd.DataFrame): DataFrame containing alignment results between spots and cells with probability values indicating the strength of assignment.
        non_zero_probabilities (bool): If True, excludes any alignment results where the probability is zero. Default=True.
        no_repeated_cells (bool): If True, prevents the assignment of a single cell to multiple spots, ensuring each cell is assigned only once. Default=True.
        top_num (int, optional): Specifies the maximum number of cells that can be assigned to a single spot. If None, no limit is applied. Default=None.
        expected_num (pd.DataFrame, optional): Specifies a DataFrame with the expected number of cells per spot, which can help refine cell allocation. Default=None.
        pos_random (bool): If True, assigns coordinates randomly within a determined radius. If False, uses an algorithm based on similarity metrics to determine coordinates. Default=False.
        layer (str): Specifies the layer of the AnnData objects to use for extracting gene expression data, which is used in computing cell assignments. Default='data'.

    Returns:
        pd.DataFrame: A DataFrame that includes the original out_data columns plus columns for the assigned x and y coordinates ('Cell_xcoord' and 'Cell_ycoord').
    """
    
    print('Assigning spatial coordinates to cells...')
    print('random = '+str(pos_random))
    if top_num is not None:
        print("Maximum number of cells assigned to spots: "+str(top_num))
        cell_num = {spot: top_num for spot in pd.unique(adata1.obs_names)}
    if expected_num is not None:
        print("Determine the number of cells allocated to each spot based on the input information")
        cell_num = expected_num['cell_num'].to_dict()

    if non_zero_probabilities :
        out_data = out_data[out_data['value']>0]
    if no_repeated_cells :   
        out_data = out_data.sort_values(by="value",ascending=False)
        out_data = out_data[out_data.duplicated('cell') == False]

    meta1_dict = {
        'spot': pd.Series(adata1.obs.index.tolist(),index=adata1.obs.index.tolist()),
        'spot_type': pd.Series(adata1.obs['type'],index=adata1.obs.index.tolist())
    }
    meta1 = pd.DataFrame(meta1_dict)
    out_data = pd.merge(out_data, meta1, on='spot',how="left")

    meta2_dict = {
        'spot': pd.Series(adata2.obs.index.tolist(),index=adata2.obs.index.tolist()),
        'spot_type': pd.Series(adata2.obs['type'],index=adata2.obs.index.tolist())
    }
    meta2 = pd.DataFrame(meta2_dict)
    meta2.columns = ['cell','cell_type']
    out_data = pd.merge(out_data, meta2, on='cell',how="left")
        
    if layer is not None:
        adata1_copy = AnnData(extract_exp(adata1, layer=layer))
        adata2_copy = AnnData(extract_exp(adata2, layer=layer))
    else:
        adata1_copy = AnnData(adata1.X)
        adata2_copy = AnnData(adata2.X)
        
    adata1_copy.obs = adata1.obs
    adata2_copy.obs = adata2.obs

    res2 = pd.DataFrame(columns=adata2_copy.var_names, index=adata2_copy.obs['type'].astype('category').cat.categories)
    
    for clust in adata2_copy.obs['type'].astype('category').cat.categories: 
        res2.loc[clust] = adata2_copy[adata2_copy.obs['type'].isin([clust]),:].X.mean(0)
    res2 = np.array(res2)
    res2 = np.transpose(res2)  
    data = extract_exp(adata1_copy)
    ratio_df = pd.DataFrame(columns=adata2_copy.obs['type'].astype('category').cat.categories, index=pd.unique(out_data.spot))   
    for spot in pd.unique(out_data.spot):
        res1 = data.loc[spot]
        res1 = res1.T.values  # (gene_num, 1)
        res1 = res1.reshape(res1.shape[0],)
        ratio_sub = nnls(res2, res1)[0]
        
        ratio_sum = np.sum([ratio_sub], axis=1)[0]
        if ratio_sum == 0:  
            ratio_sub = [0] * len(ratio_sub)
        else:
            ratio_sub = (ratio_sub / ratio_sum).tolist()
        ratio_sub = np.round(np.array(ratio_sub)*cell_num[spot])
        ratio_df.loc[spot] = ratio_sub
    
    # Assign cell to spot
    decon_df = pd.DataFrame(columns=out_data.columns)
    for spot in pd.unique(out_data.spot):
        spot_ratio = ratio_df.loc[spot]
        spot_ot = out_data.loc[out_data['spot'] == spot]
        decon_spot1 = pd.DataFrame(columns=out_data.columns)
        for cluster_id in range(0,len(spot_ratio)):
            cluster = spot_ratio.index[cluster_id]
            decon_num = spot_ratio[cluster_id]
            decon_spot_ot = spot_ot.loc[spot_ot['cell_type'] == cluster][0:int(decon_num)]
            decon_spot1 = pd.concat([decon_spot1,decon_spot_ot])
        decon_num = decon_spot1.shape[0]
        if decon_num < cell_num[spot]:
            rest_spot_ot = spot_ot.drop(decon_spot1.index)
            rest_spot_ot = rest_spot_ot.sort_values(by="value",ascending=False)
            decon_spot2 = rest_spot_ot.iloc[0:(cell_num[spot]-decon_num)]
            decon_spot = pd.concat([decon_spot1,decon_spot2])
        elif decon_num > 0 :
            decon_spot = decon_spot1
        decon_df = pd.concat([decon_df,decon_spot])
        
    out_data = decon_df.groupby('spot').apply(lambda x: x.nlargest(cell_num[x.name], 'value'))

    # Adjust cell coord
    if pos_random:
        ## Calculate radius
        if isinstance(adata1.obsm['spatial'], pd.DataFrame):
            coord = adata1.obsm['spatial'].copy()
        else:
            coord = pd.DataFrame(adata1.obsm['spatial'], 
                                 columns=['x', 'y'], 
                                 index=adata1.obs.index)
        a = coord.to_numpy()
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(a)
        distances, indices = nbrs.kneighbors(coord)
        radius = distances[:, -1] / 2
        radius = radius.tolist()
        coord['spot'] = coord.index
        
        ## Calculate coord randomly  
        if len(list(out_data.index.names)) > 1:
            out_data.index = out_data.index.droplevel()
        df_meta = pd.merge(out_data,coord,on='spot',how="left")
        all_coord = df_meta[['x', 'y']].to_numpy()

        mean_radius = np.mean(radius)
        all_radius = [mean_radius] * all_coord.shape[0]

        length = np.random.uniform(0, all_radius)
        angle = np.pi * np.random.uniform(0, 2, all_coord.shape[0])
        x = all_coord[:, 0] + length * np.cos(angle)
        y = all_coord[:, 1] + length * np.sin(angle)
        cell_coord = {'Cell_xcoord': np.around(x, 2).tolist(), 'Cell_ycoord': np.around(y, 2).tolist()}
        df_cc = pd.DataFrame(cell_coord)
        df_meta = pd.concat([df_meta, df_cc], axis=1)
    else:
        ## Calculate radius
        if isinstance(adata1.obsm['spatial'], pd.DataFrame):
            coord = adata1.obsm['spatial'].copy()
        else:
            coord = pd.DataFrame(adata1.obsm['spatial'], 
                                 columns=['x', 'y'], 
                                 index=adata1.obs.index)
        a = coord.to_numpy()
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(a)
        distances, indices = nbrs.kneighbors(coord.values)
        radius = distances[:, -1] / 2
        radius = radius.tolist()
        mean_radius = np.mean(radius)

        if len(list(out_data.index.names)) > 1:
            out_data.index = out_data.index.droplevel()

        ## Calculate dist of two spots
        dist = pd.DataFrame(columns=coord.index,index = coord.index)
        dist['spot1']=dist.index
        dist = dist.melt(id_vars=['spot1'], var_name ='spot2', value_name='value')
        coord1 = coord.copy()
        coord1.columns = ['x1','y1']
        coord1['spot1']=coord1.index
        coord2 = coord.copy()
        coord2.columns = ['x2','y2']
        coord2['spot2']=coord2.index
        dist = pd.merge(dist, coord1, on='spot1',how="left")
        dist = pd.merge(dist, coord2, on='spot2',how="left")
        
        ## Filter dist dataframe with n_neighber to speed up
        a = coord[['x', 'y']].to_numpy()
        nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(a)
        distances, indices = nbrs.kneighbors(coord.values)
        indices = pd.DataFrame(indices)
        nn_dic = {indices.index.tolist()[i]:coord.index.tolist()[i] for i in range(coord.shape[0])}
        indices = indices.replace(nn_dic)
        indices = indices.rename(columns={0:'spot1'})
        nn_dist = indices.melt(id_vars=['spot1'], var_name ='num', value_name='spot2')
        nn_dist["spot_pair"] = nn_dist["spot1"]+nn_dist["spot2"]
        dist["spot_pair"] = dist["spot1"]+dist["spot2"]
        dist = dist[dist['spot_pair']. isin(nn_dist["spot_pair"])]
        dist = dist.drop(['spot_pair'],axis=1)
        
        ## Filter spot that asigned no cells
        spot_num = coord.shape[0]
        spot_has_cell_num = len(np.unique(out_data.spot))
        dist = dist[dist.spot1.isin(pd.unique(out_data.spot))&dist.spot2.isin(pd.unique(out_data.spot))]
        print('There are '+str(spot_num)+' spots and '+str(spot_has_cell_num)+' of them were assigned cells.')
        
        dist_dis = []
        for i in dist.index:
            dist_dis.append(dist_cal(dist.loc[i].x1,dist.loc[i].y1,dist.loc[i].x2,dist.loc[i].y2))
        dist.value = dist_dis

        ## Select nearest neighbors of each spot
        dist_closest = dist[dist['value'] > 0]
        dist_closest = dist_closest[dist_closest['value'] < 1.5*min(dist_closest.value)]
        num_closest = pd.value_counts(dist_closest.spot2).max()

        ## Make gene expression data of mappeds spot and cells
        exp_adata2 = extract_exp(adata2_copy)
        exp_adata2_copy = exp_adata2.copy()
        exp_adata2_copy['cell'] = exp_adata2_copy.index
        exp_mapped = pd.merge(out_data[['spot','cell']], exp_adata2_copy, on='cell',how="left")
        exp_mapped = exp_mapped.drop(['cell'],axis=1)
        exp_mapped = exp_mapped.groupby('spot')[np.setdiff1d(exp_mapped.columns, 'spot')].mean()

        df_meta = pd.DataFrame(columns = list(out_data.columns)+['x','y','Cell_xcoord','Cell_ycoord'] )
        for each_spot in np.unique(out_data.spot):
            each_spot_x = coord.loc[each_spot].x
            each_spot_y = coord.loc[each_spot].y

            ### Claculate dist to neighbors of each spot
            dist_of_each_spot = dist[dist.spot1==each_spot]
            dist_of_each_spot = dist_of_each_spot[dist_of_each_spot['value'] < 3*mean_radius]
            dist_of_each_spot = dist_of_each_spot[dist_of_each_spot['value'] > 0]
            if dist_of_each_spot.shape[0] == 0 :
                dist_of_each_spot = dist[dist.spot1==each_spot]
                
            ### Add pseudo cell when neighbors are insufficient
            if dist_of_each_spot.shape[0] < num_closest:
                x_sum = (dist_of_each_spot.shape[0]+1)*each_spot_x
                y_sum = (dist_of_each_spot.shape[0]+1)*each_spot_y
                x_pseudo = x_sum-sum(dist_of_each_spot.x2)
                y_pseudo = y_sum-sum(dist_of_each_spot.y2)
                value_pseudo = dist_cal(each_spot_x,each_spot_y,x_pseudo,y_pseudo)
                pseudo_data = [each_spot,each_spot,value_pseudo,each_spot_x,each_spot_y,x_pseudo,y_pseudo]
                pseudo_data = pd.DataFrame(pseudo_data,columns=[each_spot],index=dist_of_each_spot.columns).T
                dist_of_each_spot =  pd.concat([dist_of_each_spot,pseudo_data])
                
            if dist_of_each_spot.shape[0] > num_closest:
                dist_of_each_spot.nsmallest(num_closest,"value",keep='all')
            

            ### Extract ot output of each spot
            spot_cell_ot = out_data[out_data.spot==each_spot].copy()
            exp_mapped = exp_mapped[exp_adata2.columns]
            spot_cell_ot.loc[:,'Cell_xcoord'] = each_spot_x
            spot_cell_ot.loc[:,'Cell_ycoord'] = each_spot_y
            spot_cell_ot.loc[:,'x'] = each_spot_x
            spot_cell_ot.loc[:,'y'] = each_spot_y

            ### Align cells according to pearson correlation coefficient calculated with neighbor spots
            for cell_self in spot_cell_ot.cell:
                exp_cell = exp_adata2.loc[cell_self].values
                neighbor_pearson = []
                for neighbor_spot in dist_of_each_spot.spot2:
                    exp_spot = exp_mapped.loc[neighbor_spot].values
                    pc = pearsonr(exp_cell,exp_spot)
                    neighbor_pearson.append(pc[0])

                if len(neighbor_pearson)>2:
                    neighbor_pearson_scaled = scale_num(neighbor_pearson)###scale to 0-1
                elif len(neighbor_pearson)>1:
                    neighbor_pearson_scaled = neighbor_pearson
                elif len(neighbor_pearson)==1:
                    neighbor_pearson_scaled = [0]
                    
                dist_of_each_spot=dist_of_each_spot.copy()
                
                dist_of_each_spot.loc[:,'x_difference'] = dist_of_each_spot.x2 - dist_of_each_spot.x1
                dist_of_each_spot.loc[:,'y_difference'] = dist_of_each_spot.y2 - dist_of_each_spot.y1
                
                x_map = np.mean(dist_of_each_spot.x_difference * neighbor_pearson_scaled + dist_of_each_spot.x1)
                y_map = np.mean(dist_of_each_spot.y_difference * neighbor_pearson_scaled + dist_of_each_spot.y1)
                
                if weighted_average and np.sum(neighbor_pearson_scaled) != 0:
                    x_map = np.sum(dist_of_each_spot.x_difference * neighbor_pearson_scaled + dist_of_each_spot.x1) / np.sum(neighbor_pearson_scaled)
                    y_map = np.sum(dist_of_each_spot.y_difference * neighbor_pearson_scaled + dist_of_each_spot.y1) / np.sum(neighbor_pearson_scaled)
                else:
                    x_map = np.mean(dist_of_each_spot.x_difference * neighbor_pearson_scaled + dist_of_each_spot.x1)
                    y_map = np.mean(dist_of_each_spot.y_difference * neighbor_pearson_scaled + dist_of_each_spot.y1)
                
                spot_cell_ot.loc[spot_cell_ot.cell==cell_self,'Cell_xcoord'] = x_map
                spot_cell_ot.loc[spot_cell_ot.cell==cell_self,'Cell_ycoord'] = y_map

            ### Adjust coord to make cells more distributed
            if spot_cell_ot.shape[0] > 1:
                x_midpoint = np.mean(spot_cell_ot.Cell_xcoord)
                y_midpoint = np.mean(spot_cell_ot.Cell_ycoord)
                spot_cell_ot.Cell_xcoord = spot_cell_ot.Cell_xcoord + each_spot_x - x_midpoint
                spot_cell_ot.Cell_ycoord = spot_cell_ot.Cell_ycoord + each_spot_y - y_midpoint
                x_dif = spot_cell_ot.Cell_xcoord - each_spot_x
                y_dif = spot_cell_ot.Cell_ycoord - each_spot_y
                #### Restrict coord to the scope of the spot
                squ = x_dif * x_dif + y_dif * y_dif
                ratio = mean_radius/max(squ ** 0.5)
                spot_cell_ot.Cell_xcoord = x_dif * ratio + each_spot_x
                spot_cell_ot.Cell_ycoord = y_dif * ratio + each_spot_y
                
            df_meta = pd.concat([df_meta, spot_cell_ot])
            
    print('Assignment done!')
            
    return df_meta

def assign_coord_2(
    adata1: AnnData,
    adata2: AnnData,
    out_data: pd.DataFrame, 
    non_zero_probabilities: bool = True,
    no_repeated_cells: bool = True,
    top_num: int = None,
    expected_num: pd.DataFrame = None,
    pos_random: bool = False,
    weighted_average: bool = False
    ) :
    """
    Assigns spatial coordinates to cells in single-cell data based on their optimal transport alignment 
    to spatial transcriptomic data, considering alignment probabilities and distribution expectations.

    Parameters:
        adata1 (AnnData): Spatial transcriptomic data with known spot locations.
        adata2 (AnnData): Single cell data where spatial information is inferred.
        out_data (pd.DataFrame): Contains alignment results between spots and cells, including alignment 
                                 probabilities (values).
        non_zero_probabilities (bool): If True, discards alignment results with zero probabilities to 
                                       focus on significant alignments. Default is True.
        no_repeated_cells (bool): If True, ensures that each cell is assigned to only one spot, preventing 
                                  the same cell from being assigned multiple times. Default is True.
        top_num (int, optional): Limits the number of cells that can be assigned to a single spot. If None, 
                                 no limit is applied. Default is None.
        expected_num (pd.DataFrame, optional): Specifies expected cell counts per spot, which can be used to 
                                               guide the distribution of cells. Default is None.
        pos_random (bool): If True, assigns cell coordinates randomly within a predefined radius around each spot. 
                           If False, assigns coordinates based on a similarity measure (cosine similarity of gene 
                           expression profiles). Default is False.
        weighted_average (bool): If True, assigns coordinates to cells based on a weighted average 
                                 of the distances to neighboring spots, using the scaled similarity 
                                 between the cell and neighboring spots as weights. If False, assigns 
                                 coordinates based on the mean of the distances without weighting.
        ...

    Returns:
        pd.DataFrame: Returns a DataFrame with the original columns from `out_data` along with 'Cell_xcoord' 
                      and 'Cell_ycoord' columns representing the assigned spatial coordinates of each cell.
    """
    
    print('Assigning spatial coordinates to cells...')
    print('random = '+str(pos_random))
    if top_num is not None:
        print("Maximum number of cells assigned to spots: "+str(top_num))
        cell_num = {spot: top_num for spot in pd.unique(adata1.obs_names)}
    if expected_num is not None:
        print("Determine the number of cells allocated to each spot based on the input information")
        cell_num = expected_num['cell_num'].to_dict()

    if non_zero_probabilities :
        out_data = out_data[out_data['value']>0]
    if no_repeated_cells :   
        out_data = out_data.sort_values(by="value",ascending=False)
        out_data = out_data[out_data.duplicated('cell') == False]

    meta1_dict = {
        'spot': pd.Series(adata1.obs.index.tolist(),index=adata1.obs.index.tolist()),
        'spot_type': pd.Series(adata1.obs['type'],index=adata1.obs.index.tolist())
    }
    meta1 = pd.DataFrame(meta1_dict)
    out_data = pd.merge(out_data, meta1, on='spot',how="left")

    meta2_dict = {
        'spot': pd.Series(adata2.obs.index.tolist(),index=adata2.obs.index.tolist()),
        'spot_type': pd.Series(adata2.obs['type'],index=adata2.obs.index.tolist())
    }
    meta2 = pd.DataFrame(meta2_dict)
    meta2.columns = ['cell','cell_type']
    out_data = pd.merge(out_data, meta2, on='cell',how="left")

    # Assign cell to spot
    out_data = out_data.groupby('spot').apply(lambda x: x.nlargest(cell_num[x.name], 'value'))

    # Adjust cell coord
    if pos_random:
        ## Calculate radius
        if isinstance(adata1.obsm['spatial'], pd.DataFrame):
            coord = adata1.obsm['spatial'].copy()
        else:
            coord = pd.DataFrame(adata1.obsm['spatial'], 
                                 columns=['x', 'y'], 
                                 index=adata1.obs.index)
        a = coord.to_numpy()
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(a)
        distances, indices = nbrs.kneighbors(coord.values)
        radius = distances[:, -1] / 2
        radius = radius.tolist()
        coord['spot'] = coord.index
        
        ## Calculate coord randomly  
        if len(list(out_data.index.names)) > 1:
            out_data.index = out_data.index.droplevel()
        df_meta = pd.merge(out_data,coord,on='spot',how="left")
        all_coord = df_meta[['x', 'y']].to_numpy()

        mean_radius = np.mean(radius)
        all_radius = [mean_radius] * all_coord.shape[0]

        length = np.random.uniform(0, all_radius)
        angle = np.pi * np.random.uniform(0, 2, all_coord.shape[0])
        x = all_coord[:, 0] + length * np.cos(angle)
        y = all_coord[:, 1] + length * np.sin(angle)
        cell_coord = {'Cell_xcoord': np.around(x, 2).tolist(), 'Cell_ycoord': np.around(y, 2).tolist()}
        df_cc = pd.DataFrame(cell_coord)
        df_meta = pd.concat([df_meta, df_cc], axis=1)
    else:
        ## Calculate radius
        if isinstance(adata1.obsm['spatial'], pd.DataFrame):
            coord = adata1.obsm['spatial'].copy()
        else:
            coord = pd.DataFrame(adata1.obsm['spatial'], 
                                 columns=['x', 'y'], 
                                 index=adata1.obs.index)
        a = coord.to_numpy()
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(a)
        distances, indices = nbrs.kneighbors(coord.values)
        radius = distances[:, -1] / 2
        radius = radius.tolist()
        mean_radius = np.mean(radius)

        if len(list(out_data.index.names)) > 1:
            out_data.index = out_data.index.droplevel()

        ## Calculate dist of two spots
        dist = pd.DataFrame(columns=coord.index,index = coord.index)
        dist['spot1']=dist.index
        dist = dist.melt(id_vars=['spot1'], var_name ='spot2', value_name='value')
        coord1 = coord.copy()
        coord1.columns = ['x1','y1']
        coord1['spot1']=coord1.index
        coord2 = coord.copy()
        coord2.columns = ['x2','y2']
        coord2['spot2']=coord2.index
        dist = pd.merge(dist, coord1, on='spot1',how="left")
        dist = pd.merge(dist, coord2, on='spot2',how="left")
        
        dist_orig = dist.copy()###
        
        ## Filter dist dataframe with n_neighber to speed up
        a = coord[['x', 'y']].to_numpy()
        nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(a)
        distances, indices = nbrs.kneighbors(coord.values)
        indices = pd.DataFrame(indices)
        nn_dic = {indices.index.tolist()[i]:coord.index.tolist()[i] for i in range(coord.shape[0])}
        indices = indices.replace(nn_dic)
        indices = indices.rename(columns={0:'spot1'})
        nn_dist = indices.melt(id_vars=['spot1'], var_name ='num', value_name='spot2')
        nn_dist["spot_pair"] = nn_dist["spot1"]+nn_dist["spot2"]
        dist["spot_pair"] = dist["spot1"]+dist["spot2"]
        dist = dist[dist['spot_pair']. isin(nn_dist["spot_pair"])]
        dist = dist.drop(['spot_pair'],axis=1)
        
        ## Filter spot that asigned no cells
        spot_num = coord.shape[0]
        spot_has_cell_num = len(np.unique(out_data.spot))
        dist = dist[dist.spot1.isin(pd.unique(out_data.spot))&dist.spot2.isin(pd.unique(out_data.spot))]
        print('There are '+str(spot_num)+' spots and '+str(spot_has_cell_num)+' of them were assigned cells.')
        
        
        dist_dis = []
        for i in dist.index:
            dist_dis.append(dist_cal(dist.loc[i].x1,dist.loc[i].y1,dist.loc[i].x2,dist.loc[i].y2))
        dist.value = dist_dis

        ## Select nearest neighbors of each spot
        dist_closest = dist[dist['value'] > 0]
        if dist_closest.shape[0] == 0:
            print("Too few spots have been assigned cells. Terminating the function.")
            return pd.DataFrame() 
        dist_closest = dist_closest[dist_closest['value'] < 1.5*min(dist_closest.value)]
        num_closest = pd.value_counts(dist_closest.spot2).max()

        ## Make embedding data of mappeds spot and cells
        emb_adata2 = extract_reduction(adata2)
        emb_adata2_copy = emb_adata2.copy()
        emb_adata2_copy['cell'] = emb_adata2_copy.index
        emb_mapped = pd.merge(out_data[['spot','cell']], emb_adata2_copy, on='cell',how="left")
        emb_mapped = emb_mapped.drop(['cell'],axis=1)
        # emb_mapped = emb_mapped.groupby('spot')[np.setdiff1d(emb_mapped.columns, 'spot')].mean()
        cols_to_average = emb_mapped.columns.difference(['spot'])
        emb_mapped = emb_mapped.groupby('spot')[cols_to_average].mean()

        df_meta = pd.DataFrame(columns = list(out_data.columns)+['x','y','Cell_xcoord','Cell_ycoord'] )
        for each_spot in np.unique(out_data.spot):
            each_spot_x = coord.loc[each_spot].x
            each_spot_y = coord.loc[each_spot].y

            ### Claculate dist to neighbors of each spot
            dist_of_each_spot = dist[dist.spot1==each_spot]
            dist_of_each_spot = dist_of_each_spot[dist_of_each_spot['value'] < 3*mean_radius]
            dist_of_each_spot = dist_of_each_spot[dist_of_each_spot['value'] > 0]
            if dist_of_each_spot.shape[0] == 0 :
                if dist[dist.spot1==each_spot].shape[0] > 0 :
                    dist_of_each_spot = dist[dist.spot1==each_spot]
                else:
                    dist_dis_orig = []
                    for i in dist_orig.index:
                        dist_dis_orig.append(dist_cal(dist_orig.loc[i].x1,
                                                 dist_orig.loc[i].y1,
                                                 dist_orig.loc[i].x2,
                                                 dist_orig.loc[i].y2))
                    dist_orig.value = dist_dis_orig
                    dist_no_neighbor = dist_orig[dist_orig.spot1==each_spot]
                    dist_no_neighbor = dist_no_neighbor[dist_no_neighbor.spot2.isin(pd.unique(out_data.spot))]
                    dist_of_each_spot = dist_no_neighbor
                
            ### Add pseudo cell when neighbors are insufficient
            if dist_of_each_spot.shape[0] < num_closest:
                x_sum = (dist_of_each_spot.shape[0]+1)*each_spot_x
                y_sum = (dist_of_each_spot.shape[0]+1)*each_spot_y
                x_pseudo = x_sum-sum(dist_of_each_spot.x2)
                y_pseudo = y_sum-sum(dist_of_each_spot.y2)
                value_pseudo = dist_cal(each_spot_x,each_spot_y,x_pseudo,y_pseudo)
                pseudo_data = [each_spot,each_spot,value_pseudo,each_spot_x,each_spot_y,x_pseudo,y_pseudo]
                pseudo_data = pd.DataFrame(pseudo_data,columns=[each_spot],index=dist_of_each_spot.columns).T
                dist_of_each_spot =  pd.concat([dist_of_each_spot,pseudo_data])
                
            if dist_of_each_spot.shape[0] > num_closest:
                dist_of_each_spot.nsmallest(num_closest,"value",keep='all')
            

            ### Extract ot output of each spot
            spot_cell_ot = out_data[out_data.spot==each_spot].copy()
            emb_mapped = emb_mapped[emb_adata2.columns]
            spot_cell_ot.loc[:,'Cell_xcoord'] = each_spot_x
            spot_cell_ot.loc[:,'Cell_ycoord'] = each_spot_y
            spot_cell_ot.loc[:,'x'] = each_spot_x
            spot_cell_ot.loc[:,'y'] = each_spot_y

            ### Align cells according to pearson correlation coefficient calculated with neighbor spots
            for cell_self in spot_cell_ot.cell:
                emb_cell = emb_adata2.loc[cell_self].values
                neighbor_similarity = []
                for neighbor_spot in dist_of_each_spot.spot2:
                    emb_spot = emb_mapped.loc[neighbor_spot].values
                    sim = 1 - cosine(emb_cell, emb_spot)
                    neighbor_similarity.append(sim)
                    
                neighbor_similarity_scaled = neighbor_similarity

                if len(neighbor_similarity)>2:
                    neighbor_similarity_scaled = scale_num(neighbor_similarity)###scale to 0-1
                elif len(neighbor_similarity)>0:
                    neighbor_similarity_scaled = neighbor_similarity
                    
                dist_of_each_spot=dist_of_each_spot.copy()
                
                dist_of_each_spot.loc[:,'x_difference'] = dist_of_each_spot.x2 - dist_of_each_spot.x1
                dist_of_each_spot.loc[:,'y_difference'] = dist_of_each_spot.y2 - dist_of_each_spot.y1
                
                if weighted_average and np.sum(neighbor_similarity_scaled) != 0:
                    x_map = np.sum(dist_of_each_spot.x_difference * neighbor_similarity_scaled + dist_of_each_spot.x1) / np.sum(neighbor_similarity_scaled)
                    y_map = np.sum(dist_of_each_spot.y_difference * neighbor_similarity_scaled + dist_of_each_spot.y1) / np.sum(neighbor_similarity_scaled)
                else:
                    x_map = np.mean(dist_of_each_spot.x_difference * neighbor_similarity_scaled + dist_of_each_spot.x1)
                    y_map = np.mean(dist_of_each_spot.y_difference * neighbor_similarity_scaled + dist_of_each_spot.y1)
                
                spot_cell_ot.loc[spot_cell_ot.cell==cell_self,'Cell_xcoord'] = x_map
                spot_cell_ot.loc[spot_cell_ot.cell==cell_self,'Cell_ycoord'] = y_map

            ### Adjust coord to make cells more distributed
            if spot_cell_ot.shape[0] > 1:
                x_midpoint = np.mean(spot_cell_ot.Cell_xcoord)
                y_midpoint = np.mean(spot_cell_ot.Cell_ycoord)
                spot_cell_ot.Cell_xcoord = spot_cell_ot.Cell_xcoord + each_spot_x - x_midpoint
                spot_cell_ot.Cell_ycoord = spot_cell_ot.Cell_ycoord + each_spot_y - y_midpoint
                x_dif = spot_cell_ot.Cell_xcoord - each_spot_x
                y_dif = spot_cell_ot.Cell_ycoord - each_spot_y
                #### Restrict coord to the scope of the spot
                squ = x_dif * x_dif + y_dif * y_dif
                ratio = mean_radius/max(squ ** 0.5)
                spot_cell_ot.Cell_xcoord = x_dif * ratio + each_spot_x
                spot_cell_ot.Cell_ycoord = y_dif * ratio + each_spot_y
                
            df_meta = pd.concat([df_meta, spot_cell_ot])
            
    print('Assignment done!')
            
    return df_meta

def omics_alignment(adata_st, 
                    adata_rna, 
                    adata_gam, 
                    adata_atac, 
                    gene_list1_params, 
                    gene_list2_params,
                    alignment_1_params,
                    assign_coord_1_params,
                    alignment_2_params,
                    assign_coord_2_params,
                    batch_size=None
                   ):
    # Step 1: Find marker genes in the first dataset
    print("\033[1;36;40mFinding marker genes...\033[0m")
    gene_list1 = find_marker(adata_st, 
                             adata_rna, 
                             **gene_list1_params)

    # Step 2: Perform alignment 1
    print("\033[1;36;40mPerforming alignment 1...\033[0m")
    if batch_size is not None:
        out1 = alignment_1_batch(adata1=adata_st[:, gene_list1].copy(),
                        adata2=adata_rna[:, gene_list1].copy(),
                        batch_size=batch_size,
                        **alignment_1_params)
    else:
        out1 = alignment_1(adata1=adata_st[:, gene_list1].copy(),
                        adata2=adata_rna[:, gene_list1].copy(),
                        **alignment_1_params)

    # Step 3: Assign coordinates based on alignment 1
    print("\033[1;36;40mAssigning coordinates based on alignment 1...\033[0m")
    map1 = assign_coord_1(adata1=adata_st[:, gene_list1].copy(), 
                          adata2=adata_rna[:, gene_list1].copy(), 
                          out_data=out1,
                          **assign_coord_1_params)

    # Step 4: Extract cells from RNA data based on spatial mapping
    tmp = adata_rna[map1.cell]
    tmp.obsm['spatial'] = map1[['Cell_xcoord','Cell_ycoord']].values

    # Step 5: Find marker genes in the second dataset
    print("\033[1;36;40mFinding marker genes...\033[0m")
    gene_list2 = find_marker(tmp, 
                             adata_gam, 
                             **gene_list2_params)

    # Step 6: Perform alignment 2
    print("\033[1;36;40mPerforming alignment 2...\033[0m")
    out2,_,_,_ = alignment_2(adata1=tmp[:, gene_list2].copy(), 
                       adata2=adata_gam[:, gene_list2].copy(), 
                       coor_df=map1,
                       **alignment_2_params)

    # Step 7: Assign coordinates based on alignment 2
    print("\033[1;36;40mAssigning coordinates based on alignment 2...\033[0m")
    map2 = assign_coord_2(adata1=adata_st, 
                          adata2=adata_atac, 
                          out_data=out2,
                          **assign_coord_2_params)
    
    results = {
        "alignment_modality1": out1,
        "coord_modality1": map1,
        "alignment_modality2": out2,
        "coord_modality2": map2
    }
    return results


def alignment_1_batch(
    adata1: AnnData,  
    adata2: AnnData,
    alpha: float = 0.1, 
    G_init = None,
    p_distribution = None, 
    q_distribution = None, 
    numItermax: int = 200, 
    norm: str = 'l2', 
    backend = ot.backend.NumpyBackend(),  
    return_obj: bool = False,
    verbose: bool = False, 
    k: int = 10,
    graph_mode: str = "connectivity",
    aware_st: bool = True,
    aware_sc: bool = True,
    aware_st_label: str = "type",
    aware_sc_label: str = "type",
    aware_power_st: int = 2,
    aware_power_sc: int = 2,
    batch_size: int = 5,           # New parameter specifying the number of splits
    **kwargs
) :
    # Split the dataset into multiple subsets
    total_cells = adata2.shape[0]
    cells_per_split = total_cells // batch_size
    print(f'Splitting dataset into {batch_size} subsets...')
    out_data_list = []
    for i in range(batch_size):
        # Determine the indices for the current subset
        start_idx = i * cells_per_split
        end_idx = (i + 1) * cells_per_split if i < batch_size - 1 else total_cells
        
        # Extract the current subset of adata2
        adata2_split = adata2[start_idx:end_idx, :]
        
        # Perform alignment on the current subset
        print(f'Running alignment on subset {i+1}/{batch_size}...')
        out_data_split = alignment_1(adata1, adata2_split, alpha, G_init, p_distribution, q_distribution, numItermax, norm, backend, return_obj, verbose, k, graph_mode, aware_st, aware_sc, aware_st_label, aware_sc_label, aware_power_st, aware_power_sc,  **kwargs)
        
        # Concatenate the results from this subset to the list
        out_data_list.append(out_data_split)

    # Concatenate the results from all subsets
    out_data = pd.concat(out_data_list)

    return out_data