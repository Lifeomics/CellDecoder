import torch
from collections import Counter
import numpy as np

import pandas as pd
import os.path as osp
import json
import scanpy as sc
import pandas as pd
from tqdm import tqdm


def get_edges(ppi, node2id):
    """return edge_index (tensor [E,2]),edge_feature (tensor [E,F])
    ppi : pandas.DataFrame, columns=['protein1','protein2','score']
    node2id : dict
    """
    vs = ppi.values
    edges = []
    ef = []
    for p1, p2, c in vs:
        if p1 not in node2id or p2 not in node2id:
            continue
        e = [node2id[p1], node2id[p2]]
        edges.append(e)
        ef.append(c)
    return torch.LongTensor(edges), torch.FloatTensor(ef)


def get_x(hcc, id2node, node2id):
    """return graph features
    hcc : pandas.DataFrame, columns=['sampleid','protein1','protein2',...]
    id2node : dict
    node2id : dict
    """
    node_names = hcc.columns[1:]
    xs = {}
    for row in hcc.values:
        person = row[0]
        nodes = row[1:]
        x = torch.zeros((len(id2node), 1))
        for i, v in enumerate(nodes):
            node_name = node_names[i]
            if node_name not in node2id:
                continue
            x[node2id[node_name]] = v
        xs[person] = x
    return xs


def merge_x_dict(x1, x2):
    x_dict = {}
    for k in x1:
        x = torch.concat([x1[k], x2[k]], dim=-1)
        x_dict[k] = x
    return x_dict


def get_xy(xd, yd):
    """return grpah features (tensor),label (tensor)
    xd : dict
    yd : dict
    """
    xs = []
    ys = []
    for k, v in xd.items():
        if k not in yd:
            continue
        ys.append(yd[k])
        xs.append(xd[k])
    xs = torch.stack(xs)
    ys = torch.Tensor(ys)
    return xs, ys


def build_links(graph, node2id, e, device):
    """build inner and cross links for the next layer
    @ graph : one hierachical graph layer
    @ node2id : node2id for lower layer
    @ e : edges for lower layer
    @ return : hid2id,cross_links,inner_links
    """
    d = {}
    for hid, ns in graph.items():
        for n in ns:
            if n in node2id:
                if hid not in d:
                    d[hid] = []
                d[hid].append(node2id[n])
    assert (
        sum(np.array(list(Counter([len(ns) for hid, ns in d.items()]).values())) == 0)
        == 0
    )
    id2hid = sorted(list(d.keys()))
    hid2id = dict(zip(id2hid, range(len(id2hid))))

    cross_links = []
    for hid, nids in d.items():
        hid = hid2id[hid]
        for nid in nids:
            cross_links.append([nid, hid])
    cross_links = torch.LongTensor(cross_links).T

    # affinity
    c1 = cross_links
    i1 = e
    N1 = len(node2id)
    N2 = len(hid2id)
    tc1 = torch.sparse_coo_tensor(c1, torch.ones(c1.shape[1]), size=(N1, N2)).to_dense()
    ti1 = torch.sparse_coo_tensor(i1, torch.ones(i1.shape[1]), size=(N1, N1)).to_dense()
    ti2 = tc1.T @ ti1 @ tc1
    inner_links = (ti2 > 0).nonzero().T
    return hid2id, cross_links, inner_links


def load_data(
    dataroot,
    fn_feature="features.csv",
    fn_edges="filter_ppi.txt",
    fn_label="sampleid.csv",
    fn_hierarchy="hierarchy_graph.json",
    fn_h5ad=None,
    device="cpu",
    cls2id=None,
):
    """
    features: columns=[sample_id, protein1,protein2,...]
    edges: columns=[protein1,protein2,score]
    label: columns=[sample_id,class,classname,...]
    """
    print(f"Loading Raw data from {dataroot}")
    if not fn_h5ad:
        features = pd.read_csv(osp.join(dataroot, fn_feature))
        ppi = pd.read_csv(osp.join(dataroot, fn_edges))
        sampleid = pd.read_csv(osp.join(dataroot, fn_label))
        hierarchy_graph = json.load(open(osp.join(dataroot, fn_hierarchy)))
    else:
        data = sc.read_h5ad(osp.join(dataroot, fn_h5ad))
        features = pd.DataFrame(data.X, columns=data.var_names).reset_index()
        ppi = data.uns["ppi"]
        sampleid = data.obs["cell_type"].reset_index().reset_index()
        if not cls2id:
            id2cls = sorted(np.unique(sampleid["cell_type"]))
            cls2id = dict(zip(id2cls, range(len(id2cls))))
        sampleid["index"] = sampleid["cell_type"].apply(lambda x: cls2id[x])
        hierarchy_graph = eval(data.uns["hierarchy"])

    y_dict = dict(sampleid.values[:, :2])
    id2node = set(ppi.values[:, :-1].flatten())
    node2id = dict(zip(sorted(list(id2node)), range(len(id2node))))
    # number of features not exist
    assert (
        len(id2node - set(features.columns.values)) == 0
    ), f"#{len(id2node-set(features.columns.values))} nodes in PPI do not have features"

    x_dict = get_x(features, id2node, node2id)
    x, y = get_xy(x_dict, y_dict)
    assert x.shape[0] == y.shape[0]
    ei, ef = get_edges(ppi, node2id)
    assert ei.shape[0] == ef.shape[0] and ei.shape[1] == 2
    ei = ei.T

    print("Building inner and cross links")
    hid2id = node2id
    inner_links = ei
    all_cross_links = []
    all_inner_links = []
    hid2ids = [node2id]
    for graph in tqdm(hierarchy_graph[::-1]):
        hid2id, cross_links, inner_links = build_links(
            graph, hid2id, inner_links, device
        )
        all_cross_links.append(cross_links)
        all_inner_links.append(inner_links)
        hid2ids.append(hid2id)

    description = (
        f"#node {len(id2node)} \t #graph {len(y)} \n"
        + f"Shape: feature {x.shape};\t label {y.shape};\t edges {ei.shape};\t edge_feat {ef.shape}\n"
        + f"cross links ,{[l.shape for l in all_cross_links]}\n"
        + f"inner links ,{[l.shape for l in all_inner_links]}\n"
        + f"dataroot {dataroot}\n fn_feature {fn_feature} \n fn_edges {fn_edges} \n fn_label {fn_label} \n fn_hierarchy {fn_hierarchy}"
    )

    data = dict(
        zip(
            "x y ei ef inner_links cross_links hid2ids cls2id description".split(),
            [
                x,
                y,
                ei,
                ef,
                all_inner_links,
                all_cross_links,
                hid2ids,
                cls2id,
                description,
            ],
        )
    )
    return data
