import pandas as pd
import numpy as np
import networkx as nx
import joblib
import os

from similarity_graph import gaussian_kernel_matrix
from settings.general_settings import *
from settings.path_settings import *


# description: calculates the amount of corresponding zero-valued genes
# parameters:
    # center: pd.Series, vector of the cluster center
    # singleton_sample: pd.Series, vector of the sample in the singleton cluster
def get_second_cluster_dist(center: pd.Series, singleton_sample: pd.Series):
    X = pd.concat([center, singleton_sample], axis=1).to_numpy().T
    return len(np.where(X.mean(axis=0) == 0)[0])


# description: calculates the similarity score based on the Gaussion kernel
# parameters:
    # center: pd.Series, vector of the cluster center
    # singleton_sample: pd.Series, vector of the sample in the singleton cluster
def get_cluster_dist(center: pd.Series, singleton_sample: pd.Series):
    X = pd.concat([center, singleton_sample], axis=1).to_numpy().T

    # calculate the Gaussion kernel matrix using the same parameters used during clustering
    # (for this reason they only have to be set once in the general settings)
    K = gaussian_kernel_matrix(X=X)
    return K[0,1]


# description: determine the cluster that the singleton observation will be reassigned to
# parameters:
    # df_center: pd.Dataframe, contains all the vectors of the cluster centers
    # singleton_sample: pd.Series, vector of the sample in the singleton cluster
def get_closest_cluster(df_center: pd.DataFrame, singleton_sample: pd.Series):
    # Step 1: calculate the distance to all the clusters using the Gaussian kernel similarity matrix
    list_distances = [get_cluster_dist(center=df_center.iloc[i], singleton_sample=singleton_sample) for i in np.arange(df_center.shape[0])]

    # check if the similarity scores are unique
    if len(set(list_distances)) < df_center.shape[0]:
        list_max = np.where(list_distances==np.array(list_distances).min())[0].tolist()

        # Step 2: calculate the amount of corresponding zero-valued genes
        list_zero_dist = [get_second_cluster_dist(center=df_center.iloc[i], singleton_sample=singleton_sample) for i in np.arange(df_center.shape[0])]

        # determine cluster with max similarity score that has the most corresponding zero-valued genes
        max_idx = list_max[0]
        zero_count = list_zero_dist[max_idx]
        for idx in list_max[1:]:
            if list_zero_dist[idx] > zero_count:
                max_idx = idx
                zero_count = list_zero_dist[idx]
        return df_center.index[max_idx]
    else:
        # return the name of the cluster with the highest similarity
        return df_center.index[np.array(list_distances).argmax()]


# description: determine the center of a given cluster
# parameters:
    # df: pd.Dataframe, contains all the observations and their respective clustering labels
    # label: int, name of the cluster whose center is to calculate
def get_cluster_center(df: pd.DataFrame, label: int):
    return df.loc[df['clustering_label']==label].drop(columns=['clustering_label']).mean(axis=0).to_frame(name=label).T


# description: identify the clusters which contain extactly one or more than one observation
# parameters:
    # df_cluster: pd.Dataframe, contains all the observations and their respective clustering labels
def identify_singletons(df_cluster: pd.DataFrame):
    singleton_clusters = [i+1 for i in np.where(df_cluster[df_cluster.columns[-2:]].groupby(by='clustering_label').count()==1)[0]]
    non_singleton_clusters = [i+1 for i in np.where(df_cluster[df_cluster.columns[-2:]].groupby(by='clustering_label').count()>1)[0]]
    return singleton_clusters, non_singleton_clusters


# description: return the list of new cluster labels where the observations in singleton clusters were reassigned
# parameters:
    # df: pd.Dataframe, contains all the oberservations
    # labels: list, contains the list of labels resulting from the HCS clustering
def adapt_singletons(df: pd.DataFrame, labels: list):
    df_cluster = df.join(pd.Series(data=labels, index=df.index, name='clustering_label'))
    singleton_clusters, non_singleton_clusters = identify_singletons(df_cluster=df_cluster)
    if len(non_singleton_clusters)==0:
        print('warning: all clusters are singletons')
        return labels
    else:
        # calculate the centers for each cluster
        df_center = pd.concat([get_cluster_center(df_cluster, label=label) for label in non_singleton_clusters])

        # determine the new label for the samples in singleton clusters
        df_cluster['new_label'] = df_cluster.apply(lambda x: get_closest_cluster(df_center=df_center, singleton_sample=x[:-1]) if x['clustering_label'] in singleton_clusters else x['clustering_label'], axis=1)
        new_labels = df_cluster['new_label'].apply(lambda x: int(x)).T.values
        joblib.dump(new_labels, os.path.join(
            path_data + ''.join(['new_labels_sigma=', str(sigma), '_eps=', str(epsilon), '_weighted=', str(weighted), '.joblib'])))
        return new_labels