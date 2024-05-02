# Towards Quantifying the Effect of Datasets for Benchmarking

This repository contains the source code to reproduce the results and analysis of the paper

> Towards Quantifying the Effect of Datasets for Benchmarking: A Look at Tabular Machine Learning
> Ravin Kohli, Matthias Feurer, Bernd Bischl, Katharina Eggensperger, Frank Hutter
> Data-centric Machine Learning Research (DMLR) Workshop at ICLR 2024

The code is provided as-is and we will neither maintain it nor provide bug fixes.

## Installation
```
git clone https://github.com/automl/dmlr-iclr24-datasets-for-benchmarking
cd tabular_data_experiments
conda create -n tabular_data_experiments python=3.10
conda activate tabular_data_experiments
conda install swig

# Install for usage
pip install .

# Install for development
make install-dev
```

## 3rd-party source code

Our code is heavily inspired by the great [source code](https://github.com/LeoGrin/tabular-benchmark) published alongside the paper [*Why do tree-based models still outperform deep learning on tabular data?*](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html) by Leo Grinsztajn, Edouard Oyallon and Gael Varoquaux.

## Data

The raw data can be found [here](https://bwsyncandshare.kit.edu/s/scAezD4GstL3szw).

## Visualizations

We provide the following notebooks for visualization:

### dataset_and_suites.ipynb

Contains code that creates the table used throughout the paper.

### result_analysis.ipynb

Contains code that creates the figures used throughout the paper.