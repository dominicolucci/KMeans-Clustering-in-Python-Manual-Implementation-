# KMeans Clustering from Scratch

This repository contains a custom implementation of the K-Means clustering algorithm using only NumPy. It includes:
- `KMeans.py`: A Python module implementing the K-Means algorithm with support for both Euclidean (L2) and Manhattan (L1) distance metrics.
- `KMeans.ipynb`: A Jupyter notebook demonstrating the usage of the `KMeans` class with example datasets and visualizations.

## File Descriptions

### `KMeans.py`
This file defines a `KMeans` class with the following features:
- Supports Euclidean (`order=2`) and Manhattan (`order=1`) distance metrics.
- Initializes cluster centers between the 10th and 90th percentiles of each feature.
- Uses early stopping if cluster assignments do not change between iterations.

Key methods:
- `fit(X)`: Fits the KMeans model to data `X`.
- `predict(X)`: Returns the cluster index for each sample in `X`.

### `KMeans.ipynb`
This notebook demonstrates:
- How to load and visualize data.
- How to use the custom `KMeans` class.
- Visualizations of cluster assignments before and after convergence.
- The effect of varying `k` and distance metrics.

## Requirements

- Python 3.6+
- NumPy
- Matplotlib (used in the notebook)
- Jupyter Notebook

Install dependencies using:
```bash
pip install numpy matplotlib notebook
