import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, norm
from sklearn.decomposition import PCA
from PyComplexHeatmap import ClusterMapPlotter, HeatmapAnnotation, anno_simple

from .helper import *

def rotate_points(x, y, angle):
    rad = np.radians(angle)  
    cos, sin = np.cos(rad), np.sin(rad)
    rotation_matrix = np.array([[cos, -sin], [sin, cos]]) 

    points = np.array([x, y])
    x_rot, y_rot = rotation_matrix.dot(points) 
    
    return x_rot, y_rot



def plot_scatter(df, x_col, y_col, color_col, ax=None, 
                 palette=None, x_label=None, y_label=None, show_ticks=True, title=None, size=None, marker='o', 
                 rotation_angle=0, reverse_y=False ,reverse_x=False,
                 hue_order=None,show=True,linewidth=0,set_equal=True,figsize=(5,4),save_path=None):
    df['x_rot'], df['y_rot'] = rotate_points(df[x_col], df[y_col], rotation_angle)
    """
    Plots a scatter plot with options for color, size, rotation, and other visual adjustments.

    Parameters:
        df (DataFrame): The DataFrame containing data to plot.
        x_col (str): Column name used for x-axis values.
        y_col (str): Column name used for y-axis values.
        color_col (str): Column name used for coloring points.
        ax (matplotlib Axes, optional): Axes object on which to draw the plot. Defaults to current axes if None.
        palette (str or sequence, optional): Color palette for the points.
        x_label (str, optional): Label text for the x-axis.
        y_label (str, optional): Label text for the y-axis.
        show_ticks (bool): If False, both x and y axis ticks will be hidden.
        title (str, optional): Title for the plot. Defaults to 'Scatter plot of {y_col} vs {x_col}' if None.
        size (int, optional): Size of the scatter plot points.
        marker (str): Style of the plot markers.
        rotation_angle (float): Degrees to rotate points around the origin.
        reverse_y (bool): If True, inverts the y-axis.
        reverse_x (bool): If True, inverts the x-axis.
        hue_order (list, optional): Specifies the order of the hue levels.
        show (bool): If True, displays the plot. Useful in non-interactive environments.
        linewidth (float): Width of the lines that make up the plot marker edges.
        set_equal (bool): If True, sets the aspect of the plot to be equal, maintaining the scale on axes.
        figsize (tuple): Dimensions of the figure in inches (width, height).
        save_path (str, optional): Path where the plot will be saved. If specified, the plot is saved to this location.
    """
    if reverse_y:
        df['y_rot'] = -df['y_rot']
    if reverse_x:
        df['x_rot'] = -df['x_rot']
        
    if ax is None:
        ax = plt.gca()
    ax.figure.set_size_inches(figsize)
    sns.scatterplot(data=df, 
                    x='x_rot', 
                    y='y_rot', 
                    hue=color_col, 
                    palette=palette, 
                    s=size, 
                    marker=marker, 
                    legend='full', 
                    hue_order=hue_order, 
                    ax=ax, 
                    linewidth=linewidth)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Scatter plot of {y_col} vs {x_col}')
    
    ax.set_xlabel(x_label if x_label else x_col)
    ax.set_ylabel(y_label if y_label else y_col)
    
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    
    ax.legend(title=color_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    if set_equal:
        ax.axis('equal')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight') 
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
        
def sdplot(anndata,
           use_rep='spatial',
           color_col='cell_type',
           size=20,
           marker='o',
           rotation_angle=-90,
           show_ticks=False,
           x_label='Spatial_x',
           y_label='Spatial_y',
           palette='Set1',
           show=False,
           set_equal=False,
           highlight=None,
           title=None,
           highlight_color=['#D8281A','#CDCECA'],
           ax=None,
           linewidth=0,
           ncols=5,
           figsize=(4.5,4),
           save_path=None):
    """
    Visualizes spatial data from an AnnData object, plotting cells based on spatial coordinates 
    with color coding according to cell type or other specified categories. The function allows for
    customization of the plot appearance and offers the option to highlight specific categories for 
    detailed analysis.

    Parameters:
        anndata (AnnData): Annotated data matrix where observations (cells) are stored in rows and 
            variables (genes) in columns. This dataset should have spatial coordinates and other 
            annotations.
        use_rep (str): Key under which the spatial coordinates are stored in anndata.obsm. Default is 'spatial'.
        color_col (str): Name of the column in anndata.obs used to color the data points. Default is 'cell_type'.
        size (int): Size of the points in the scatter plot. Default is 20.
        marker (str): Marker style of the scatter plot points. Default is 'o'.
        rotation_angle (float): Angle in degrees to rotate the plot. Default is -90.
        show_ticks (bool): If False, will hide the axis ticks. Default is False.
        x_label (str): Label for the x-axis. Default is 'Spatial_x'.
        y_label (str): Label for the y-axis. Default is 'Spatial_y'.
        palette (str or sequence): Colors to use for plotting categories. Default is 'Set1'.
        show (bool): Whether to display the plot. If False, the plot is closed after rendering. Useful for saving to files without displaying.
        set_equal (bool): If True, sets the aspect of the plot to be equal, maintaining the scale on axes. Default is False.
        highlight (list or str, optional): List of categories within the color_col to highlight in separate subplots. 
            If None, no highlights are made. If specified, other categories are colored using a secondary color.
        title (str, optional): Title for the plot. If not provided, no title is set.
        highlight_color (list): Colors to use for highlighting the selected categories and the others. Default is ['#D8281A','#CDCECA'].
        ax (matplotlib Axes, optional): Pre-existing axes for the plot. If None, a new figure and axes are created.
        linewidth (int): Width of the marker edges. Default is 0.
        ncols (int): Number of columns in the subplot grid. Default is 5.
        figsize (tuple): Dimension of the figure in inches, given as (width, height). Default is (4.5, 4).
        save_path (str, optional): Path to save the plot. If provided, the plot is saved to this path.
    """
    
    pos_df = pd.DataFrame(anndata.obsm[use_rep], columns=['x', 'y'])
    pos_df.index = anndata.obs.index.tolist()
    plot_df = pd.concat([anndata.obs[[color_col]], pos_df], axis=1)
    
    if highlight is not None:
        palette_colors = {'Selected cells': highlight_color[0],'Other': '#CDCECA'}
        
        if isinstance(highlight, str):
            highlight = [highlight]

        if len(highlight)<ncols:
            ncols = len(highlight)
        n_features = len(highlight) 
        nrows = -(-n_features // ncols) 
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))

        if n_features > 1:
            axes = axes.flatten()

        for idx, feature in enumerate(highlight):
            ax = axes[idx] if n_features > 1 else axes

            plot_df['highlight'] = plot_df[color_col].apply(lambda x: 'Other' if x != feature else 'Selected cells')

            plot_scatter(plot_df,
                         x_col='x',
                         y_col='y',
                         color_col='highlight',
                         size=size,
                         marker=marker,
                         rotation_angle=rotation_angle,
                         show_ticks=show_ticks,
                         x_label=x_label,
                         y_label=y_label,
                         palette=palette_colors,
                         show=False,
                         set_equal=set_equal,
                         title=feature,
                        ax=ax,
                        linewidth=linewidth,
                        figsize=figsize)
        plt.tight_layout()
        for idx in range(n_features, nrows*ncols):
            fig.delaxes(axes.flatten()[idx])

        if save_path:
            plt.savefig(save_path, bbox_inches='tight') 
            print(f"Plot saved to {save_path}")
        if show:
            plt.show()
        else:
            plt.close()


    else:
        plot_scatter(plot_df,
                     x_col='x',
                     y_col='y',
                     color_col=color_col,
                     size=size,
                     marker=marker,
                     rotation_angle=rotation_angle,
                     show_ticks=show_ticks,
                     x_label=x_label,
                     y_label=y_label,
                     palette=palette,
                     show=show,
                     set_equal=set_equal,
                     title=title,
                     ax=ax,
                     figsize=figsize,
                     linewidth=linewidth,save_path=save_path)

def func(x, a, b):
    return a * x + b

def cor_plot(x_data, y_data, group, palette='Set1', confidence_interval=0.95, fit_color='r', fit_linestyle='--',figsize=(6, 4),show=True,save_path=None):
    """
    Creates a scatter plot with a regression line, confidence intervals, and Pearson correlation statistics,
    illustrating the relationship and statistical confidence between two datasets.

    Parameters:
        x_data (array-like): Independent variable data points used for the scatter plot and regression line.
        y_data (array-like): Dependent variable data points used for the scatter plot and regression line.
        group (array-like): Categorical data used for coloring points in the scatter plot according to different groups.
        palette (str): Color palette name or a list of colors to use for different groups in the scatter plot. Default is 'Set1'.
        confidence_interval (float): Confidence level for the prediction interval around the fit line. Default is 0.95.
        fit_color (str): Color for the fit line. Default is 'red'.
        fit_linestyle (str): Line style for the fit line. Default is '--'.
        figsize (tuple): Size of the figure in inches, specified as (width, height). Default is (6, 4).
        show (bool): If True, displays the plot; if False, closes the figure to prevent rendering. Useful in scripts and non-interactive contexts. Default is True.
        save_path (str, optional): Path to save the plot. If provided, the plot is saved to this path. Default is None.

    Returns:
        None: This function does not return a value but displays or saves a plot based on input parameters.
    """
    plt.figure(figsize=figsize)

    sns.scatterplot(x=x_data, y=y_data, hue=group, palette=palette)
    
    popt, pcov = curve_fit(func, x_data, y_data)
    x_fit = np.linspace(min(x_data), max(x_data), 100)
    y_fit = func(x_fit, *popt)
    plt.plot(x_fit, y_fit, color=fit_color, linestyle=fit_linestyle, label=f'Fit: a={popt[0]:.2f}, b={popt[1]:.2f}')
    
    perr = np.sqrt(np.diag(pcov))
    z_score = np.abs(norm.ppf((1 - confidence_interval) / 2))
    lower = func(x_fit, *(popt - z_score * perr))
    upper = func(x_fit, *(popt + z_score * perr))
    plt.fill_between(x_fit, lower, upper, color='gray', alpha=0.2, label=f'{confidence_interval*100}% Confidence Interval')
    
    pearson_corr, pearson_pvalue = pearsonr(x_data, y_data)
    plt.title(f'Pearson Correlation: {pearson_corr:.2f}, p-value: {pearson_pvalue:.2f}', loc='right')
    
    plt.xlabel('logfc1')
    plt.ylabel('logfc2')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight') 
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

def module_pca_plot(sr_out,
                   color_by='module',
                   figsize=(5, 4),
                   palette='magma',
                   save_path=None,
                   show=True,
                   prefix='Module '):
    """
    Plots a PCA visualization of weighted correlation data derived from consensus clustering,
    highlighting the relationship among different modules.

    Parameters:
        sr_out (dict): A dictionary containing output from a consensus clustering analysis, specifically:
            - 'cc': A DataFrame that includes cluster assignments and possibly other metrics per feature.
            - 'wcor': A DataFrame representing weighted correlations between features.
        color_by (str): The column in the 'cc' DataFrame used to determine the color of points in the plot. 
                        Default is 'module', which assumes there is a 'module' column in 'cc' DataFrame.
        figsize (tuple): Size of the figure in inches, specified as (width, height). Default is (5, 4).
        palette (str): Color palette to use for distinguishing between different clusters. Default is 'magma'.
        save_path (str, optional): Path to save the plot. If provided, the plot is saved to this path.
        show (bool): If True, displays the plot; if False, closes the figure to prevent rendering. This is
                     useful in scripts and non-interactive environments. Default is True.
        prefix (str): Prefix to prepend to cluster labels in the plot. Default is 'Module '.
    """
    features = sr_out['cc']['combos']
    clusters = sr_out['cc'][color_by]
    clusters = prefix + clusters.astype(str)
    
    scaler = StandardScaler()
    scaled_wcor = scaler.fit_transform(sr_out['wcor'].loc[features, features])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_wcor)

    pca_result_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
    pca_result_df['Cluster'] = clusters

    plt.figure(figsize=figsize)
    sns.scatterplot(data=pca_result_df, x='PC1', y='PC2', hue='Cluster', palette=palette, marker='o', alpha=1)

    plt.title('PCA Visualization with Consensus Clustering')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(False)
    plt.legend(title='Module', loc='center left', bbox_to_anchor=(1, 0.5))
    if  save_path:
        plt.savefig(save_path, bbox_inches='tight') 
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close() 

def module_heatmap_plot(sr_out, 
                        module_name=['1','2'], 
                        cmap='mako',
                        annotation_cmap='Set1',
                        label='Exp level',
                        prefix='Module ',
                        figsize=(4, 4),
                        save_path=None,
                       show=True,
                       row_split_gap=0.5,
                       col_split_gap=0.5):
    """
    Plots a heatmap with annotations for the specified modules using weighted correlation data. 
    This visualization is enhanced with module-specific color coding and optional annotations.

    Parameters:
        sr_out (dict): A dictionary containing the output from a consensus clustering analysis, specifically:
                       'cc' - a DataFrame that includes cluster assignments per feature.
                       'wcor' - a DataFrame representing weighted correlations between features.
        module_name (list of str): Names of the modules to include in the plot. Default is ['1', '2'].
        cmap (str): Color map for the heatmap values. Default is 'mako'.
        annotation_cmap (str): Color map for annotations used to distinguish modules. Default is 'Set1'.
        label (str): Label for the heatmap's color bar. Default is 'Exp level'.
        prefix (str): Prefix to prepend to module labels in the plot. Default is 'Module '.
        figsize (tuple): Dimension of the figure in inches (width, height). Default is (4, 4).
        save_path (str, optional): Path to save the plot. If provided, the plot is saved to this path.
        show (bool): If True, displays the plot; if False, closes the figure without displaying it. Useful for saving the figure without rendering it in a GUI. Default is True.
        row_split_gap (float): Gap size between different rows in the heatmap when split by module. Default is 0.5.
        col_split_gap (float): Gap size between different columns in the heatmap when split by module. Default is 0.5.
    """
    plt.figure(figsize=figsize)
    features = sr_out['cc']['combos']
    clusters = sr_out['cc']['module']
    wcor_df = pd.DataFrame(sr_out['wcor'].loc[features, features].values, index=clusters, columns=clusters)
    
    tmp = sr_out['cc']
    filtered_tmp = tmp[tmp['module'].isin(module_name)]
    features = sr_out['cc']['combos']
    wcor_df = pd.DataFrame(sr_out['wcor'].loc[features, features].values, index=features, columns=features)
    meta_df = filtered_tmp
    meta_df.index = meta_df.combos.tolist()
    meta_df['module'] = prefix + meta_df['module'].astype(str)
    
    module_name_ordered = ['Module ' + name for name in module_name]
    meta_df['module'] = pd.Categorical(meta_df['module'], categories=module_name_ordered)

    tmp = plt.get_cmap(annotation_cmap)
    colors = {module_name_ordered[i]: tmp.colors[i] for i in range(len(module_name_ordered))}
    
    col_ha = HeatmapAnnotation(Module=anno_simple(meta_df.module, legend=True, colors=colors, height=2), axis=1, label_kws={'visible': False})
    row_ha = HeatmapAnnotation(Module=anno_simple(meta_df.module, legend=False, colors=colors, height=2), axis=0, label_kws={'visible': False})
    
    cm = ClusterMapPlotter(data=wcor_df, top_annotation=col_ha, left_annotation=row_ha,
                           show_rownames=False, show_colnames=False,
                           col_split=meta_df.module, row_split=meta_df.module, cmap=cmap, label=label,
                           rasterized=True,row_split_gap=row_split_gap,col_split_gap=col_split_gap)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')  
        print(f"Plot saved to {save_path}")
    if not show:
        plt.close()

def parse_cutoff(value, scores):
    if isinstance(value, str) and value.startswith('q'):
        percentile = int(value[1:])
        return np.percentile(scores, percentile)
    else:
        return value

def apply_cutoffs(df, min_cutoff, max_cutoff, i=None,col='score'):
    scores = df[col].tolist()
    if isinstance(min_cutoff, list):
        min_value = parse_cutoff(min_cutoff[i], scores)
    else:
        min_value = parse_cutoff(min_cutoff, scores)
    if isinstance(max_cutoff, list):
        max_value = parse_cutoff(max_cutoff[i], scores)
    else:
        max_value = parse_cutoff(max_cutoff, scores)
    
    df[col] = np.clip(df[col], min_value, max_value)

def module_spatial_plot(adata,
                        sr_out,
                        module_name,
                        feature_name,
                        use_raw=False,
                        cmap='viridis',
                        rotation_angle=-90,
                        reverse_x=False,
                        reverse_y=False,
                        set_equal=False,
                        set_grid=False,
                        set_xticks=False,
                        set_yticks=False,
                        size=10, 
                        figsize=(10,4),
                        wspace=0.1,
                        hspace=0,
                        min_cutoff=None,
                        max_cutoff=None,
                        show=True,
                        return_data=False,
                       save_path=None,
                       xlabel='Spatial_x',
                       ylabel='Spatial_y'):  
    """
    Plots spatial gene expression heatmaps for specified modules, allowing visualization of gene scores across different spatial areas.

    Parameters:
        adata (AnnData): Annotated data matrix from single-cell experiments that includes spatial coordinates.
        sr_out (dict): Dictionary containing the outputs from spatial analysis, including the 'cc' DataFrame with modules and features.
        module_name (str or list): Single module name or list of module names for which to plot gene expression.
        feature_name (str): The column in 'cc' DataFrame that identifies the gene sets for scoring.
        use_raw (bool): If True, uses raw gene expression data for scoring. Default is False.
        cmap (str): Color map for visualizing gene expression intensity. Default is 'viridis'.
        rotation_angle (float): Angle in degrees to rotate the plot coordinates. Default is -90.
        reverse_x (bool): If True, reverses the x-axis after applying rotation. Default is False.
        reverse_y (bool): If True, reverses the y-axis after applying rotation. Default is False.
        set_equal (bool): If True, sets an equal scaling (aspect ratio) on the axes. Default is False.
        set_grid (bool): If True, displays a grid on the plot. Default is False.
        set_xticks (bool): If False, x-axis ticks are not displayed. Default is False.
        set_yticks (bool): If False, y-axis ticks are not displayed. Default is False.
        size (int): Size of the scatter plot points. Default is 10.
        figsize (tuple): Size of the figure in inches (width, height). Default is (10, 4).
        wspace (float): The width of the spaces between subplots when there are multiple modules. Default is 0.1.
        hspace (float): The height of the spaces between subplots when there are multiple modules. Default is 0.
        min_cutoff (float, optional): Minimum cutoff for data included in the plot.
        max_cutoff (float, optional): Maximum cutoff for data included in the plot.
        show (bool): If True, displays the plot. If False, closes the plot without displaying, which is useful for saving without rendering.
        return_data (bool): If True, returns the data used in the plot. Default is False.
        save_path (str, optional): If provided, saves the plot to the given path.
        xlabel (str): Label for the x-axis. Default is 'Spatial_x'.
        ylabel (str): Label for the y-axis. Default is 'Spatial_y'.
    """
    tmp = adata.copy()
    cc_df = sr_out['cc']
    scores_data = {}
    
    if isinstance(module_name, list):
        fig, axes = plt.subplots(1, len(module_name), figsize=(figsize[0] * len(module_name), figsize[1]))
        fig.subplots_adjust(wspace=wspace,hspace=hspace)  
        for i, name in enumerate(module_name):
            gene_list = cc_df[feature_name][cc_df.module==name].unique()
            score_name = f'score_{name}'
            sc.tl.score_genes(tmp, gene_list=gene_list, use_raw=use_raw, score_name=score_name)
            scores_data[score_name] = tmp.obs[score_name]
            data = {
                'x': tmp.obsm['spatial'][:,0],
                'y': tmp.obsm['spatial'][:,1],
                'score': tmp.obs[score_name]
            }
            df = pd.DataFrame(data)
            
            apply_cutoffs(df, min_cutoff, max_cutoff, i)
            
            df['x_rot'], df['y_rot'] = rotate_points(df['x'], df['y'], rotation_angle)
            
            if reverse_y:
                df['y_rot'] = -df['y_rot']
            if reverse_x:
                df['x_rot'] = -df['x_rot']
            
            ax = axes[i] if len(module_name) > 1 else axes
            im = ax.scatter(df['x_rot'], df['y_rot'], c=df['score'], cmap=cmap, alpha=1, s=size, label=name)
            ax.set_title("Module: " + name)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(set_grid)
            if not set_xticks:
                ax.set_xticks([])  
            if not set_yticks:
                ax.set_yticks([])  
            if set_equal:
                ax.axis('equal')
            fig.colorbar(im, ax=ax) 
        if save_path:
            plt.savefig(save_path, bbox_inches='tight') 
            print(f"Plot saved to {save_path}")
        if show:
            plt.show()
        else:
            plt.close()
            
        if return_data:
            scores_df = pd.DataFrame(scores_data)
            scores_df.index = tmp.obs_names
            return scores_df

    else:
        gene_list = cc_df[feature_name][cc_df.module==module_name].unique()
        sc.tl.score_genes(tmp, gene_list=gene_list, use_raw=use_raw)
        data = {
            'x': tmp.obsm['spatial'][:,0],
            'y': tmp.obsm['spatial'][:,1],
            'score': tmp.obs['score']
        }
        df = pd.DataFrame(data)
        
        if return_data:
            scores_df = pd.DataFrame({'score': tmp.obs['score']})
            scores_df.index = tmp.obs_names  
            return scores_df
        
        apply_cutoffs(df, min_cutoff, max_cutoff)
        
        df['x_rot'], df['y_rot'] = rotate_points(df['x'], df['y'], rotation_angle)
        
        if reverse_y:
            df['y_rot'] = -df['y_rot']
        if reverse_x:
            df['x_rot'] = -df['x_rot']
        
        plt.figure(figsize=figsize)
        im = plt.scatter(df['x_rot'], df['y_rot'], c=df['score'], cmap=cmap, alpha=1, s=size, label=module_name)
        plt.colorbar(im, label='Intensity') 
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Intensity Scatter Plot')
        plt.grid(set_grid)
        if set_equal:
            plt.axis('equal')
        plt.legend()
        if not set_xticks:
            plt.xticks([])  
        if not set_yticks:
            plt.yticks([])
        if save_path:
            plt.savefig(save_path, bbox_inches='tight') 
            print(f"Plot saved to {save_path}")
        if show:
            plt.show()
        else:
            plt.close()
            
def module_dot_plot(adata,
                        sr_out,
                        module_name,
                        feature_name,
                        groupby='cell_type',
                        prefix='Module:',
                        use_raw=False,
                        show=True,
                    level=None,
                    save_path=None): 
    """
    Creates dot plots to visualize the expression of genes grouped into specified modules across different cell types or other specified groups.

    Parameters:
        adata (AnnData): Annotated data matrix from single-cell experiments, typically containing gene expression data along with various annotations.
        sr_out (dict): A dictionary containing results from a spatial or modular analysis, typically including a DataFrame with modules and associated genes.
        module_name (list of str): List of module names for which gene expression scores will be calculated and plotted.
        feature_name (str): The column name in the 'cc' DataFrame of 'sr_out' that contains the gene names associated with each module.
        groupby (str): The column in 'adata.obs' to group the data by in the plot (e.g., cell type, treatment group). Default is 'cell_type'.
        prefix (str): Prefix to add to the score names that will be generated in 'adata.obs'. This helps to identify the newly created columns. Default is 'Module:'.
        use_raw (bool): Whether to use raw gene expression data for calculating scores. If False, uses normalized data. Default is False.
        show (bool): If True, the plot will be displayed using plt.show(). If False, the plot will not be displayed, which is useful for saving the plot without rendering it in interactive environments. Default is True.
        level (list, optional): Optional specific order for the categories in the 'groupby' column, which can be used to define the order of plotting categories. Default is None.
        save_path (str, optional): If provided, the plot will be saved to this path. Useful for creating files for reports or presentations. Default is None.
    """ 
    tmp = adata.copy()
    if level is not None:
        tmp.obs[groupby].cat.set_categories(level, inplace=True)
        
    cc_df = sr_out['cc']
    for name in module_name:
        gene_list = cc_df[feature_name][cc_df.module==name].unique()
        sc.tl.score_genes(tmp, gene_list=gene_list, use_raw=use_raw, score_name=prefix+name)
    scores = [col for col in tmp.obs.columns if col.startswith(prefix)]
    sc.pl.dotplot(tmp, scores, groupby=groupby,show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight') 
        print(f"Plot saved to {save_path}")
    if not show:
        plt.close()

def module_vilolin_plot(adata,
                        sr_out,
                        module_name,
                        feature_name,
                        groupby='cell_type',
                        prefix='Module:',
                        use_raw=False,
                        palette='Set1',
                        show=True,
                        return_data=False,
                        level=None,
                        save_path=None): 
    """
    Creates violin plots to visualize the distribution of gene expression scores for specified modules across different groups of cells.

    Parameters:
        adata (AnnData): Annotated data matrix from single-cell experiments, containing gene expression data and metadata.
        sr_out (dict): Dictionary containing outputs from a clustering or module analysis, including gene sets in 'cc'.
        module_name (list of str): List of module names to plot, corresponding to clusters or groups of genes.
        feature_name (str): The column in 'cc' DataFrame that identifies the gene sets for scoring.
        groupby (str): The metadata column in 'adata' to group by on the x-axis of the violin plot. Default is 'cell_type'.
        prefix (str): Prefix added to the score names that are calculated and added to 'adata.obs'. Default is 'Module:'.
        use_raw (bool): If True, uses raw gene expression data for calculating scores. Default is False.
        palette (str): Color palette for the plots, defining colors for each group or cell type. Default is 'Set1'.
        show (bool): If True, displays the plot; if False, does not display the plot, which is useful for saving plots without rendering them in interactive environments. Default is True.
        return_data (bool): If True, returns the DataFrame containing the scores instead of plotting. Default is False.
        level (list, optional): Custom order for categories in the 'groupby' column to dictate their plotting order. Default is None.
        save_path (str, optional): Path to save the plot image. If provided, the plot is saved to this path.
    """
    tmp = adata.copy()
    if level is not None:
        tmp.obs[groupby].cat.set_categories(level, inplace=True)
    cc_df = sr_out['cc']
    for name in module_name:
        gene_list = cc_df[feature_name][cc_df.module==name].unique()
        sc.tl.score_genes(tmp, gene_list=gene_list, use_raw=use_raw, score_name=prefix+name)
    scores = [col for col in tmp.obs.columns if col.startswith(prefix)]
    if return_data:
        return tmp.obs[scores,groupby]
    else: 
        sc.pl.violin(tmp, scores, groupby=groupby,palette=palette,show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight') 
            print(f"Plot saved to {save_path}")
        if not show:
            plt.close()
                
def plot_feature(df, x_col, y_col, ax=None, 
                 palette=None, x_label=None, y_label=None, show_ticks=True, title=None, size=None, marker='o', 
                 rotation_angle=0, reverse_y=False, reverse_x=False,
                 hue_order=None, show=True, linewidth=0, set_equal=True,figsize=(5,4)):

    df['x_rot'], df['y_rot'] = rotate_points(df[x_col], df[y_col], rotation_angle)
    
    if reverse_y:
        df['y_rot'] = -df['y_rot']
    if reverse_x:
        df['x_rot'] = -df['x_rot']
        
    if ax is None:
        ax = plt.gca()
    ax.figure.set_size_inches(figsize)
    sns.scatterplot(data=df, 
                    x='x_rot', 
                    y='y_rot', 
                    hue='gene', 
                    palette=palette, 
                    s=size, 
                    marker=marker, 
                    legend='full', 
                    hue_order=hue_order, 
                    ax=ax, 
                    linewidth=linewidth)
    
    norm = plt.Normalize(df['gene'].min(), df['gene'].max())
    sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
    sm.set_array([])

    # Remove the legend and add a colorbar
    ax.get_legend().remove() 
    cbar = ax.figure.colorbar(sm, ax=ax) 

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Scatter plot of {y_col} vs {x_col}')
    
    ax.set_xlabel(x_label if x_label else x_col)
    ax.set_ylabel(y_label if y_label else y_col)
    
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    
    if set_equal:
        ax.axis('equal')
    if show:
        plt.show()
        
def sfplot(anndata, layer='data', use_rep='spatial', features=None,
           min_cutoff=None, max_cutoff=None, size=20, marker='o',
           rotation_angle=-90, show_ticks=False, x_label='Spatial_x',
           y_label='Spatial_y', palette='viridis', show=True, set_equal=False,
           linewidth=0, figsize=(5,4),save_path=None,ncols=5):
    """
    Plots the spatial distribution of features in spatially resolved data.

    Parameters:
        anndata (AnnData): Annotated data matrix containing spatial data and annotations.
        layer (str): The layer of data to be used (default is 'data').
        use_rep (str): The type of representation data to be used (default is 'spatial').
        features (list of str): List of features to be plotted (default is None).
        min_cutoff (float): Minimum value used to filter the data (default is None).
        max_cutoff (float): Maximum value used to filter the data (default is None).
        size (int): Size of the plotted points (default is 20).
        marker (str): Marker style for the plotted points (default is 'o').
        rotation_angle (int): Rotation angle for x and y axis labels (default is -90).
        show_ticks (bool): Whether to display tick labels (default is False).
        x_label (str): Label for the x-axis (default is 'Spatial_x').
        y_label (str): Label for the y-axis (default is 'Spatial_y').
        palette (str): Color palette for the plot (default is 'viridis').
        show (bool): Whether to display the plot (default is True).
        set_equal (bool): Whether to set the aspect ratio of the plot to be equal (default is False).
        linewidth (int): Width of lines to plot (default is 0).
        figsize (tuple): Figure size (default is (5, 4)).
        save_path (str): Path to save the plot image (default is None).
        ncols (int): Number of columns for subplots organization (default is 5).
    """
    if isinstance(features, str):
        features = [features]
    
    if len(features)<ncols:
        ncols = len(features)

    n_features = len(features)
    nrows = -(-n_features // ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0] * ncols, figsize[1] * nrows))
    
    if n_features == 1:
        axes = [axes]
        
    for idx, feature in enumerate(features):
        if n_features > 1:
            ax = axes[idx//ncols][idx%ncols] if nrows > 1 else axes[idx]
        else:
            ax = axes[idx]

        plot_df = extract_reduction(anndata, use_rep=use_rep, column_names=['x', 'y'])
        if feature in anndata.obs.columns:
            plot_df['gene'] = anndata.obs[feature].values
        else:
            exp_values = extract_exp(anndata, layer=layer, gene=feature).values
            plot_df['gene'] = exp_values

        apply_cutoffs(plot_df, min_cutoff, max_cutoff, i=None, col='gene')

        plot_feature(plot_df,
                     x_col='x',
                     y_col='y',
                     ax=ax,
                     palette=palette,
                     x_label=x_label,
                     y_label=y_label,
                     title=feature,
                     size=size,
                     marker=marker,
                     rotation_angle=rotation_angle,
                     show_ticks=show_ticks,
                     linewidth=linewidth,
                     set_equal=set_equal,
                     figsize=figsize,
                     show=False) 
    
    plt.tight_layout()
    
    for idx in range(n_features, nrows*ncols):
        fig.delaxes(axes.flatten()[idx])
        
    if save_path:
        plt.savefig(save_path, bbox_inches='tight') 
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

            
def spatial_lineplot(adata, df, score_name, coord='x', label='spot', line_color='#C65885', point_color='#C5C5C5',
                     xlabel='x coordinate', ylabel='module score', title=None, edge_color='#C5C5C5', 
                     point_alpha=0.5, l_width=2,p_width=0.2,size=30,scale=True, root_df=None,show=True,save_path=None,figsize=(4.5,4)):
    """
    Generates a line plot to visualize the relationship between spatial coordinates and module scores.

    Parameters:
        adata (AnnData): Annotated data matrix containing spatial data and annotations.
        df (DataFrame): DataFrame containing module scores and spatial coordinates.
        score_name (str): Name of the module score column in the DataFrame.
        coord (str): Name of the spatial coordinate to be used on the x-axis (default is 'x').
        label (str): Label for the data points (default is 'spot').
        line_color (str): Color for the fitted curve (default is '#C65885').
        point_color (str): Color for the data points (default is '#C5C5C5').
        xlabel (str): Label for the x-axis (default is 'x coordinate').
        ylabel (str): Label for the y-axis (default is 'module score').
        title (str): Title for the plot (default is None).
        edge_color (str): Color for the edges of data points (default is '#C5C5C5').
        point_alpha (float): Transparency of data points (default is 0.5).
        l_width (int): Line width for the fitted curve (default is 2).
        p_width (float): Width of the point's edge line (default is 0.2).
        size (int): Size of the data points (default is 30).
        scale (bool): Whether to scale the module scores (default is True).
        root_df (DataFrame): DataFrame containing root cell coordinates for calculating distances (default is None).
        show (bool): If True, displays the plot (default is True).
        save_path (str): Path to save the plot image (default is None).
        figsize (tuple): Figure size (default is (4.5, 4)).
    """
    df['cell'] = df.index
    pos = extract_reduction(adata, use_rep='spatial', column_names=['x', 'y'])
    pos['cell'] = pos.index
    df_plot = df[['cell', score_name]].merge(pos, on='cell', how='inner')
    if scale:
        scaler = StandardScaler()
        df_plot[score_name] = scaler.fit_transform(df_plot[score_name].values.reshape(-1, 1))
        
    if root_df is not None:
        root_df['cell'] = root_df.index
        dists = []
        for index, row in df_plot.iterrows():
            dist = np.sqrt((root_df['x'] - row['x'])**2 + (root_df['y'] - row['y'])**2).mean()
            dists.append(dist)
        df_plot[coord] = dists

    model = smf.ols(f'{score_name} ~ {coord}', data=df_plot).fit()
    p_value = model.f_pvalue

    coefficients = np.polyfit(df_plot[coord], df_plot[score_name], 1)
    linear_polynomial = np.poly1d(coefficients)

    y_vals_for_plot = np.linspace(df_plot[coord].min(), df_plot[coord].max(), 500)
    score_vals_for_plot = linear_polynomial(y_vals_for_plot)

    plt.figure(figsize=figsize)
    plt.scatter(df_plot[coord], df_plot[score_name], label=label,s=size, color=point_color, edgecolors=edge_color, alpha=point_alpha,linewidths=p_width)
    plt.plot(y_vals_for_plot, score_vals_for_plot, color=line_color, label='Fitted Curve', linewidth=l_width)

    equation_text = f'Equation: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}\n' + f'p-value: {p_value:.4e}'

    plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round", alpha=0.5, color='wheat'))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight') 
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()
        
import networkx as nx
def plot_network_graph(df, module_names, prefix='Module', cmap='coolwarm', alpha=1,save_path=None,show=True):
    """
    Generates network graphs for each specified module to visualize the interaction between modules and their components.

    Parameters:
        df (DataFrame): DataFrame containing module names and their interaction components. It must have columns for 'module', 'combos', 'pearson_pvalue', and 'pearson_correlation'.
        module_names (list of str): List of module names to be plotted. Each module will be plotted in a separate graph.
        prefix (str, optional): Prefix to append to module names in the plot, defaults to 'Module'.
        cmap (str, optional): Colormap for mapping the Pearson correlation values to colors, defaults to 'coolwarm'.
        alpha (float, optional): Opacity for the node colors, defaults to 1.
        save_path (str, optional): If provided, the path where the plot image will be saved. If None, the plot will not be saved.
        show (bool, optional): If True, the plot will be displayed. If False, the plot will be closed after creation; defaults to True.
    """
    num_modules = len(module_names)
    fig, axs = plt.subplots(1, num_modules, figsize=(num_modules * 6, 4)) 

    for i, module_name in enumerate(module_names):
        ax = axs[i] if num_modules > 1 else axs
        module_df = df[df.module == module_name].copy()
        module_df.module = prefix + module_df.module

        G = nx.DiGraph()

        for _, row in module_df.iterrows():
            G.add_edge(row['module'], row['combos'])

        module_nodes = [node for node in G.nodes() if node.startswith('Module')]

        module_color = 'lightgray'

        combos_node_sizes = dict(zip(module_df['combos'], -np.log10(module_df['pearson_pvalue'])))
        combos_node_colors = dict(zip(module_df['combos'], module_df['pearson_correlation']))

        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if node in module_nodes:
                node_colors.append(module_color)
                node_sizes.append(1500)
            else:
                if node in combos_node_sizes:
                    size = combos_node_sizes[node]
                    node_sizes.append(0 + 500 * size)
                    pearson_value = combos_node_colors[node]
                    norm = plt.Normalize(vmin=module_df['pearson_correlation'].min(), vmax=module_df['pearson_correlation'].max())
                    cmap = plt.cm.get_cmap(cmap)
                    color = cmap(norm(pearson_value))
                    rgba_color = list(color)
                    rgba_color[3] = alpha
                    node_colors.append(tuple(rgba_color))
                else:
                    node_colors.append('gray')
                    node_sizes.append(1500)

        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes, edge_color='#8B8B8A', arrows=False, width=2, ax=ax)

        norm = plt.Normalize(vmin=module_df['pearson_correlation'].min(), vmax=module_df['pearson_correlation'].max())  
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        plt.colorbar(sm, label='Pearson Correlation', ax=ax)

        tmp = combos_values_rounded = [round(value, 1) for value in combos_node_sizes.values()]
        sizes = [min(tmp), max(tmp)]
        legend_sizes = [0 + 500 * size for size in sizes]
        legend = [ax.scatter([], [], s=size, label=f'{label} size: {size/500}', color='gray') for size, label in zip(legend_sizes, ['Min', 'Max'])]
        ax.legend(handles=legend, title='Node Size', loc='lower right', fontsize=8, markerscale=0.3, bbox_to_anchor=(1.8, 0.5))

        ax.set_title(f'Network Plot of {prefix} {module_name}')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight') 
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()

import plotly.io as pio
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def plot_3d(anndatas, color_cols, titles, use_rep='spatial', sizes=[1, 1, 1], zs=[0, 0.5, 1], color_palettes=None, output_format='png', save_path=None,show=True):
    """
    Generates a 3D scatter plot.

    Parameters:
        anndatas (list): List of AnnData objects containing spatial data and annotations.
        color_cols (list): List of column names in the AnnData objects to be used as colors for the scatter plot.
        titles (list): List of titles for each subplot.
        use_rep (str): Representation of the data to be used (default is 'spatial').
        sizes (list): List of marker sizes for each subplot (default is [1, 1, 1]).
        zs (list): List of z-coordinates for each subplot (default is [0, 0.5, 1]).
        color_palettes (list): List of color palettes for each subplot (default is None).
        output_format (str): Output format for the plot ('png' or 'notebook', default is 'png').
        save_path (str): Path to save the plot image (default is None).
        show (bool): If True, displays the plot (default is True).
    """
    if not isinstance(anndatas, list):
        raise TypeError("anndatas must be a list")
    if not isinstance(color_cols, list):
        raise TypeError("color_cols must be a list")
    if not isinstance(titles, list):
        raise TypeError("titles must be a list")
    if len(anndatas) != len(color_cols) or len(anndatas) != len(titles):
        raise ValueError("anndatas, color_cols, and titles must have the same length")

    if output_format == 'png':
        pio.renderers.default = "png"
    elif output_format == 'notebook':
        pio.renderers.default = "notebook"
    else:
        raise ValueError("Invalid output_format. Choose 'png' or 'notebook'.")

    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    for idx, (anndata, color_col, z, size, title, color_palette) in enumerate(zip(anndatas, color_cols, zs, sizes, titles, color_palettes)):
        pos_df = pd.DataFrame(anndata.obsm[use_rep], columns=['x', 'y'])
        pos_df['z'] = z 
        pos_df[color_col] = anndata.obs[color_col].values

        trace = go.Scatter3d(
            x=pos_df['x'], y=pos_df['y'], z=pos_df['z'],
            mode='markers',
            marker=dict(
                size=size,
                color=[color_palette[val] for val in pos_df[color_col]],
                opacity=1
            ),
            name=title
        )
        fig.add_trace(trace)

    fig.update_layout(title='3D Scatter Plot',
                      scene=dict(
                          xaxis_title='X',
                          yaxis_title='Y',
                          zaxis_title='Z'
                      ))
    if save_path:
        fig.write_html(save_path)
    if show:
        fig.show()
    else:
        fig.close()
        
        