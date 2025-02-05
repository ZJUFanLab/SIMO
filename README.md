

# SIMO v1.0.0 <img src='images/icon.png' align="right" height="200" /></a>

# Spatial integration of multi-omics single-cell data

### Penghui Yang, ..., Xiaohui Fan*



SIMO is a computational method, which perform Spatial Integration of Multi-Omics single-cell datasets through probabilistic alignment.

![Image text](images/overview.png)

## Requirements and Installation
### Installation of SIMO

[![anndata 0.9.2](https://img.shields.io/badge/anndata-0.9.2-green)](https://pypi.org/project/anndata/0.9.2/)
[![h5py 3.10.0](https://img.shields.io/badge/h5py-3.10.0-red)](https://pypi.org/project/h5py/3.10.0/)
[![igraph 0.10.8](https://img.shields.io/badge/igraph-0.10.8-yellow)](https://pypi.org/project/python-igraph/0.10.8/)
[![jupyterlab 3.6.7](https://img.shields.io/badge/jupyterlab-3.6.7-brightgreen)](https://pypi.org/project/jupyterlab/3.6.7/)
[![kaleido 0.2.1](https://img.shields.io/badge/kaleido-0.2.1-lightgrey)](https://pypi.org/project/kaleido/0.2.1/)
[![louvain 0.7.1](https://img.shields.io/badge/louvain-0.7.1-orange)](https://pypi.org/project/louvain/0.7.1/)
[![matplotlib 3.5.2](https://img.shields.io/badge/matplotlib-3.5.2-blueviolet)](https://pypi.org/project/matplotlib/3.5.2/)
[![networkx 3.1](https://img.shields.io/badge/networkx-3.1-blue)](https://pypi.org/project/networkx/3.1/)
[![notebook 6.3.0](https://img.shields.io/badge/notebook-6.3.0-critical)](https://pypi.org/project/notebook/6.3.0/)
[![pot 0.8.2](https://img.shields.io/badge/pot-0.8.2-9cf)](https://pypi.org/project/POT/0.8.2/)
[![numpy 1.22.4](https://img.shields.io/badge/numpy-1.22.4-ff69b4)](https://pypi.org/project/numpy/1.22.4/)
[![pandas 1.4.3](https://img.shields.io/badge/pandas-1.4.3-success)](https://pypi.org/project/pandas/1.4.3/)
[![PyComplexHeatmap 1.6.7](https://img.shields.io/badge/PyComplexHeatmap-1.6.7-important)](https://pypi.org/project/PyComplexHeatmap/1.6.7/)
[![scikit-learn 1.2.0](https://img.shields.io/badge/scikit--learn-1.2.0-informational)](https://pypi.org/project/scikit-learn/1.2.0/)
[![scipy 1.8.1](https://img.shields.io/badge/scipy-1.8.1-lightblue)](https://pypi.org/project/scipy/1.8.1/)
[![scanpy 1.9.1](https://img.shields.io/badge/scanpy-1.9.1-brightred)](https://pypi.org/project/scanpy/1.9.1/)
[![scikit-misc 0.1.4](https://img.shields.io/badge/scikit--misc-0.1.4-orange)](https://pypi.org/project/scikit-misc/0.1.4/)
[![leidenalg 0.10.0](https://img.shields.io/badge/leidenalg-0.10.0-yellowgreen)](https://pypi.org/project/leidenalg/0.10.0/)

```
# We recommend using Anaconda, and then you can create a new environment.
# Create and activate Python environment
conda create -n simo python=3.8
conda activate simo
# Installation of SIMO
pip install simo-omics
```
## Tutorials

We have applied SIMO on datasets, here we give step-by-step tutorials for all application scenarios. And preprocessed datasets used can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1smPbTfkkd_Pvk0hXzIZrXk55aEbnXPVl?usp=sharing).


* [Spatial integration of the mouse cortex](tutorial/mouse_brain.ipynb)
* [Spatial integration of the human myocardial infarction](tutorial/human_heart.ipynb)

## About
Should you have any questions, please feel free to contact the author of the manuscript, Mr. Penghui Yang (yangph@zju.edu.cn).

## References
Penghui Yang, Kaiyu Jin, Yue Yao, Lijun Jin, Xin Shao, Chengyu Li, Xiaoyan Lu, and Xiaohui Fan. Spatial integration of multi-omics single-cell data with SIMO. Nat Commun 16, 1265 (2025)