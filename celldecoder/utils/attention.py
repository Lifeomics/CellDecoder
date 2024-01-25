import pandas as pd
import json
from tqdm import tqdm
import torch
import numpy as np


def get_atten_score(att_path, layer="0"):
    """ """
    with open(att_path, "r") as f:
        attention_dict_all = json.load(f)

    return attention_dict_all[layer]


def process_attention(dataset, attentions, num_nodes=None, cross=True, hierarchy=3):
    """
    cross: 是否在cross 层用gat，if false，则用gat encoder

    """
    print("Processing Raw Attentions...")
    hid2ids_dict = dataset.raw_data["hid2ids"]
    attention_dict = {f"{i}": {} for i in range(hierarchy + 1)}
    pos = tqdm(attentions)
    id2hid = []
    for i in range(len(hid2ids_dict)):
        id2hid.append(dict(zip(hid2ids_dict[i].values(), hid2ids_dict[i].keys())))

    # 获得原始edge信息
    edges = []
    edge_index, inner_edge_indexs, cross_edge_indexs, _ = dataset.metadata
    if cross:
        for e in cross_edge_indexs:
            e = e.cpu().numpy().tolist()
            edges.append([(e[0][i], e[1][i]) for i in range(len(e[0]))])
    else:
        edge_index = edge_index.cpu().numpy().tolist()
        edges.append(
            [(edge_index[0][i], edge_index[1][i]) for i in range(len(edge_index[0]))]
        )
        for e in inner_edge_indexs:
            e = e.cpu().numpy().tolist()
            edges.append([(e[0][i], e[1][i]) for i in range(len(e[0]))])

    # 处理attention
    for att_batch in pos:
        for l, a in enumerate(att_batch):
            # 一个dataloader内部
            (edge_index, att) = a

            if cross:
                # scr: l; dst: l+1
                edge_index = edge_index[
                    :, edge_index[0] < (num_nodes[l] + num_nodes[l + 1])
                ]
                edge_index = torch.stack(
                    (edge_index[0], edge_index[1] - num_nodes[l]), 0
                )
            edge_index = edge_index.cpu().numpy().tolist()
            edge_set = [
                (edge_index[0][i], edge_index[1][i]) for i in range(len(edge_index[0]))
            ]

            att_dict = dict(zip(edge_set, att.detach().cpu().numpy().tolist()))

            for edge_ind in edges[l]:
                src, dst = edge_ind
                src = id2hid[l][src]
                if cross:
                    dst = id2hid[l + 1][dst]
                else:
                    dst = id2hid[l][dst]
                edge = str(src) + "," + str(dst)

                if edge not in attention_dict[str(l)]:
                    attention_dict[str(l)][edge] = []
                attention_dict[str(l)][edge].append(
                    att_dict[edge_ind] + [np.mean(att_dict[edge_ind])]
                )

    for l in attention_dict.keys():
        for edge in attention_dict[l].keys():
            # 对每条边的所有data做平均
            att_matrix = attention_dict[l][edge]
            mean_att = np.mean(att_matrix, axis=0).reshape(-1).tolist()
            attention_dict[l][edge] = mean_att

    return attention_dict
