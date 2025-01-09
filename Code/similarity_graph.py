import numpy as np
import pandas as pd
import networkx as nx
import joblib
import os

from settings.path_settings import path_data
from settings.general_settings import *


# description: function to calculate the Gaussian kernel distance matrix
# parameters:
    # X: np.array, input array containing the data points
    # (sigma: hyperparameter to be set in the general settings)
def gaussian_kernel_matrix(X: np.array):
    distances = np.sum((X[:, np.newaxis] - X) ** 2, axis=-1)
    kernel_matrix = np.exp(-distances / (2 * sigma ** 2))
    np.fill_diagonal(kernel_matrix, 0, wrap=False)
    return kernel_matrix


# description: function f to specify the construction of the edges in the similarity graph
# parameters:
    # k: float, similarity value, entry of the Gaussian kernel matrix
    # (weighted: bool, hyperparameter to set in the general settings,
    #           determines one of the two approaches to calculate the edge weights, unweighted edges if set to False)
    # (epsilon: hyperparameter to set in the general settings)
def func_f(k: float):
    if k >= epsilon:
        if weighted:
            return k
        else:
            return 1
    else:
        return 0


# description: construct a similarity graph from a data set
# parameters:
    # df: pd.Dataframe, data set to construct the graph from
    # (weighted: bool, hyperparameter to set in the general settings,
    #           determines one of the two approaches to calculate the edge weights, unweighted edges if set to False)
    # (load_K: bool, hyperparameter to set in the general settings,
    #           whether to load the Gaussian kernel matrix from the files or to calculate it)
    # (sigma, epsilon: hyperparameters to set in the general settings)
def get_similarity_graph(df: pd.DataFrame):
    if load_K and os.path.exists(path_data + ''.join(['K_matrix_sigma=', str(sigma), '_eps=', str(epsilon), '.joblib'])):
        K = joblib.load(os.path.join(path_data + ''.join(['K_matrix_sigma=', str(sigma), '_eps=', str(epsilon), '.joblib'])))
    else:
        K = gaussian_kernel_matrix(X=df.to_numpy())
        joblib.dump(K, os.path.join(path_data + ''.join(['K_matrix_sigma=', str(sigma), '_eps=', str(epsilon), '.joblib'])))
    E = pd.DataFrame(K).applymap(lambda x: func_f(x)).to_numpy()
    G = nx.from_numpy_array(E)
    joblib.dump(G, os.path.join(
        path_data + ''.join(['G_sigma=', str(sigma), '_eps=', str(epsilon), '_weighted=', str(weighted), '.joblib'])))
    return G