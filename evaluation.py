import pandas as pd
import numpy as np
import os
import joblib
import networkx as nx
import matplotlib.pyplot as plt

from settings.general_settings import *
from settings.path_settings import *
from initial_analysis import subset_data


# description: get the table displaying the cluster sizes
# parameters:
    # new_labels: list, output of the adapt_singletons() function
def get_cluster_sizes(new_labels: list):
    return pd.DataFrame(np.unique(new_labels, return_counts=True), index=['class', 'size']).T.sort_values(
        by=['size'],ascending=False).reset_index(drop=True)


# description: get the necessary information to construct the bar chart
# parameters:
    # df_joined: pd.Dataframe, result of the load_data(add_labels=True) function
    # new_labels: list, output of the adapt_singletons() function
def get_bar_chart_data(df_joined: pd.DataFrame, new_labels: list):
    df_subset_l = subset_data(df_joined, drop_label=False)

    # join actual classes and clustering labels together
    df_comp = df_subset_l[['Class']].join(pd.DataFrame(new_labels, index=df_subset_l.index, columns=['clustering_label']))

    # count the size of the classes to divide by them later to get the distribution in percent
    class_sizes = pd.DataFrame(np.unique(df_subset_l['Class'].values, return_counts=True),
                               index=['class', 'size']).T.sort_values(by=['size'], ascending=False).reset_index(drop=True)

    # count how often the different clustering labels appear for one class and then divide by the total number of observations for one class
    class_sizes['distr_labels'] = class_sizes.apply(lambda x: {
        val: (df_comp.loc[(df_comp['Class'] == x['class']) & (df_comp['clustering_label'] == val)].shape[0]) / (
        x['size']) for val in list(set(df_comp.loc[df_comp['Class'] == x['class']]['clustering_label'].values))},
                                                    axis=1)
    return class_sizes
