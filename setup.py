from setuptools import setup, find_packages

setup(
    name='simo-omics',
    version='v1.0.0',
    author='Penghui Yang',
    author_email='yangph@zju.edu.cn',
    description='SIMO: Spatial Integration of Multi-Omics single-cell datasets through probabilistic alignment',
    long_description='SIMO is an advanced Python package designed for the spatial integration of multi-omics data from single-cell datasets, utilizing probabilistic alignment techniques to accurately depict single-cell maps and reconstruct the spatial distribution of various biological molecules, thereby providing crucial insights into the complex multimodal heterogeneity and topological patterns of cells essential for advancing spatial biology.',
    long_description_content_type='text/markdown',
    url='https://github.com/ZJUFanLab/SIMO',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.8',
    install_requires=[
        'anndata==0.9.2',
        'h5py==3.10.0',
        'igraph==0.10.8',
        'jupyterlab==3.6.7',
        'kaleido==0.2.1',
        'louvain==0.7.1',
        'matplotlib==3.5.2',
        'networkx==3.1',
        'notebook==6.3.0',
        'pot==0.8.2',
        'numpy==1.22.4',
        'pandas==1.4.3',
        'PyComplexHeatmap==1.6.7',
        'scikit-learn==1.2.0',
        'scipy==1.8.1',
        'scanpy==1.9.1',
        'scikit-misc==0.1.4',
        'leidenalg==0.10.0'
    ],
)
