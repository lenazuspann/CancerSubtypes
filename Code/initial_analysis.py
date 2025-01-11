import pandas as pd
import numpy as np
import joblib
import os

from settings.path_settings import *
from settings.general_settings import *


# creating tables from pandas data frames: df_summary.to_latex(index=True, float_format="{:.4f}".format)


# description: load the data set from the csv-file
# parameters:
    # add_labels: bool=False, needs to be set to True if the labels should be added to the data frame
    # (note: True is useful for evaluating results or getting an overview of the data)
def load_data(add_labels: bool=False):
    df_rna = pd.read_csv(os.path.join(path_data, 'RNA_data.csv'), index_col=0)
    if add_labels:
        df_labels = pd.read_csv(os.path.join(path_data, 'RNA_labels.csv'), index_col=0)
        df_joined = df_rna.join(df_labels)
        return df_joined
    else:
        return df_rna


# description: gives an overview of the values distributed amoung the classes and the relative amount of zeros
# parameters:
    # df_joined: pd.Dataframe, result of the above function with add_labels=True
def get_data_overview(df_joined: pd.DataFrame):
    df_summary = df_joined[['Class']].reset_index(names='Percentage').groupby(by=['Class']).count().map(lambda x: x/df_joined.shape[0])
    df_summary['description'] = df_summary.apply(lambda x: dict_class_description[x.name], axis=1)
    df_summary['only zero-valued attributes'] = df_summary.apply(
        lambda x: (df_joined.loc[df_joined['Class'] == x.name].loc[:, (df_joined.loc[df_joined['Class'] == x.name] == 0).all(
        axis=0)].shape[1]) / (df_joined.shape[1]-1), axis=1)
    df_summary['attributes with at least one zero-valued sample'] = df_summary.apply(
        lambda x: (df_joined.loc[df_joined['Class'] == x.name].loc[:, (df_joined.loc[df_joined['Class'] == x.name] == 0).any(
        axis=0)].shape[1]) / (df_joined.shape[1]-1), axis=1)
    df_summary['only non-zero-valued attributes'] = df_summary.apply(
        lambda x: (df_joined.loc[df_joined['Class'] == x.name].loc[:,(df_joined.loc[df_joined['Class'] == x.name] != 0).all(
        axis=0)].shape[1]) / ( df_joined.shape[1]-1), axis=1)
    return df_summary


# description: drops the classes specified in the list classes in general_settings.py
# parameters:
    # df_joined: pd.Dataframe containing the values and the classes
    # drop_label: bool=False, whether to keep the column containing the class label
def subset_data(df_joined: pd.DataFrame, drop_label: bool=False):
    df_subset = df_joined.loc[df_joined['Class'].isin(classes)]
    if drop_label:
        return df_subset.drop(columns=['Class'])
    else:
        return df_subset

