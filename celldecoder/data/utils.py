import torch
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def construct_batch(batch_size, num_nodes):
    batch = torch.cat(
        [torch.ones(num_nodes).long() * i for i in range(batch_size)], dim=0
    )
    return batch


def construct_inner_edge(edge_index, batch_size, num_nodes):
    shift = torch.LongTensor([[num_nodes, num_nodes]]).T.to(edge_index.device)
    e = torch.cat([edge_index + shift * i for i in range(batch_size)], dim=-1)
    return e


def construct_cross_edge(edge_index, batch_size, num_nodes1, num_nodes2):
    shift = torch.LongTensor([[num_nodes1, num_nodes2]]).T.to(edge_index.device)
    e = torch.cat([edge_index + shift * i for i in range(batch_size)], dim=-1)
    return e


def construct_cross_edge_both(edge_index, batch_size, num_nodes1, num_nodes2):
    shift0 = num_nodes1 + num_nodes2
    shift1 = torch.LongTensor([[0, num_nodes1]]).T.to(edge_index.device)
    edge_index = torch.cat([edge_index + shift0 * i for i in range(batch_size)], dim=-1)
    e = torch.cat([edge_index + shift1 for i in range(batch_size)], dim=-1)
    return e


def data_bootstrapping(data: pd.DataFrame, max_cell=10000):
    labels_class = np.unique(data[:, -1])
    max_val = min(pd.value_counts(data[:, -1]).max(), max_cell)
    bootsted_data = np.empty(shape=(1, data.shape[1]), dtype=np.float32)
    for cell_class in labels_class:
        tmp = data[data[:, -1] == cell_class]
        idx = np.random.choice(range(len(tmp)), max_val)
        feature = tmp[idx]
        bootsted_data = np.r_[bootsted_data, feature]
    return np.delete(bootsted_data, 0, axis=0)


def data_split(data, ratio: float = 0.3, key_names="cell_type", stratified=True):
    """
    @description  :split for reference and query
    ---------
    @param  : data: Anndata
            data.obs[key_names] cell_type
            stratification  split
    -------
    @Returns  : X_train, X_test, y_train, y_test
    -------
    """
    x = np.array(data.obs_names)
    y = np.array(data.obs[key_names])
    if stratified == True:
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=ratio, random_state=42, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=ratio, random_state=42
        )

    return X_train, X_test, y_train, y_test
