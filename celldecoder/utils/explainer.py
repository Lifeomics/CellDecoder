import imp
from math import sqrt
from random import shuffle
from typing import Optional
import time
from inspect import signature
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing

# from torch_geometric.explain import GNNExplainer
from torch_geometric.loader import DataLoader

# from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph, to_networkx, get_num_hops
from ppi.data.utils import construct_inner_edge

EPS = 1e-15


def process_attention_v1(
    dataset, attentions, num_nodes=None, cross=True, hierarchy=3, return_sample=False
):
    """
    cross: 是否在cross 层用gat，if false，则用gat encoder

    """
    print("Processing Raw Attentions...")
    hid2ids_dict = dataset.raw_data["hid2ids"]
    attention_dict = {f"{i}": {} for i in range(hierarchy + 1)}
    pos = tqdm(
        attentions
    )  # [graph, layer, (edge_index, att)] # edge_index: [2, E], att: [E, n_heads]
    id2hid = []  # [layer, {id:hid}]
    for i in range(len(hid2ids_dict)):
        id2hid.append(dict(zip(hid2ids_dict[i].values(), hid2ids_dict[i].keys())))

    # 获得原始edge信息
    edges = []  # [layer, E, 2 ]
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
    for att_batch in pos:  # [layer, (edge_index, att)]
        for l, a in enumerate(att_batch):  # l: index, a: (edge_index, att)
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
            edge_index = edge_index.cpu().numpy().tolist()  # [2, E]
            edge_set = [
                (edge_index[0][i], edge_index[1][i]) for i in range(len(edge_index[0]))
            ]  # [E, 2]

            att_dict = dict(
                zip(edge_set, att.detach().cpu().numpy().tolist())
            )  # {(n1,n2):[n_heads]}
            for edge_ind in edges[l]:  # edge_ind [ E, 2 ]
                src, dst = edge_ind  # [2]
                src = id2hid[l][src]  # turn hid
                if cross:
                    dst = id2hid[l + 1][dst]
                else:
                    dst = id2hid[l][dst]
                edge = str(src) + "," + str(dst)

                if edge not in attention_dict[str(l)]:
                    attention_dict[str(l)][edge] = []
                attention_dict[str(l)][edge].append(
                    att_dict[edge_ind] + [np.mean(att_dict[edge_ind])]
                )  # [layer, edge, sample, [n_heads, mean]]

    if not return_sample:
        for l in attention_dict.keys():
            for edge in attention_dict[l].keys():
                # 对每条边的所有data做平均
                att_matrix = attention_dict[l][edge]
                mean_att = np.mean(att_matrix, axis=0).reshape(-1).tolist()
                attention_dict[l][edge] = mean_att

    return attention_dict


def get_edges_data(dataset, cross):
    # 获得原始edge信息
    edges = []  # [layer, E, 2 ]
    edge_index, inner_edge_indexs, cross_edge_indexs, _ = dataset.metadata
    if cross:
        for e in cross_edge_indexs:
            edges.append(e.T.cpu())
    else:
        edges.append(edge_index.T.cpu())
        for e in inner_edge_indexs:
            edges.append(e.T.cpu())
    return edges


def filter_edges(attentions, edges_data, cross, num_nodes):
    num_layer = len(attentions[0])
    num_sample = len(attentions)
    org_edges = [attentions[0][l][0] for l in range(num_layer)]  # [layer, [2, E]]
    filt_idxs = [[] for _ in range(num_layer)]
    filt_edges = [[] for _ in range(num_layer)]
    for l in range(num_layer):
        edge_index = edges_data[l]  # edges in data [E, 2]
        org_edge = org_edges[l].T.numpy()  # edges in calculation [E, 2]
        if cross:
            # scr: l; dst: l+1
            edge_index = edge_index[
                :, edge_index[0] < (num_nodes[l] + num_nodes[l + 1])
            ]
            edge_index = torch.stack((edge_index[0], edge_index[1] - num_nodes[l]), 0)
        edge_index = edge_index.cpu().numpy().tolist()
        edge_set = set(tuple(e) for e in edge_index)  # [E, 2]

        for ei, edge in enumerate(tqdm(org_edge)):
            if tuple(edge) in edge_set:
                filt_idxs[l].append(ei)
                filt_edges[l].append(edge)
        filt_idxs[l] = torch.LongTensor(np.array(filt_idxs[l]))
        filt_edges[l] = torch.LongTensor(np.array(filt_edges[l])).T
    return filt_idxs, filt_edges


def transpose_attns(attentions, filt_idxs):
    num_layer = len(attentions[0])
    num_sample = len(attentions)
    attns = []  # [layer, E, sample, n_heads+1]
    for layer in tqdm(range(num_layer)):
        sample_attn = []
        for sample in range(num_sample):
            attn = attentions[sample][layer][1]
            attn = torch.cat(
                [attn, torch.mean(attn, dim=-1, keepdim=True)], dim=-1
            )  # [E, n_heads+1]
            sample_attn.append(attn)
        sample_attn = torch.stack(sample_attn, dim=0)  # [sample, E, n_heads+1]
        sample_attn = torch.transpose(sample_attn, 0, 1)  # [E, sample, n_heads+1]
        sample_attn = sample_attn[filt_idxs[layer]]  # [E, sample, n_heads+1]
        attns.append(sample_attn)
    return attns


def process_attention(
    dataset, attentions, num_nodes=None, cross=True, hierarchy=3, return_sample=False
):
    """
    cross: 是否在cross 层用gat，if false，则用gat encoder
    attentions : [graph, layer, (edge_index, att)] # edge_index: [2, E], att: [E, n_heads]
    output: attention_dict : [layer, edge, sample, [n_heads, mean]]

    """
    print("Processing Raw Attentions...")
    assert cross == False, "faster version does not support cross"
    hid2ids_dict = dataset.raw_data["hid2ids"]
    attention_dict = {f"{i}": {} for i in range(hierarchy + 1)}
    id2hid = []  # [layer, {id:hid}]
    for i in range(len(hid2ids_dict)):
        id2hid.append(dict(zip(hid2ids_dict[i].values(), hid2ids_dict[i].keys())))
    edges = get_edges_data(dataset, cross)

    # Assume: for each layer, edge_index for each sample is the same
    # input: attentions : [sample, layer, (edge_index, att)], edge_index: [2, E], att: [E, n_heads]

    # filter edge_index
    print("filtering edges...")
    filt_idxs, filt_edges = filter_edges(attentions, edges, cross, num_nodes)

    # transpose attention and filt attentions
    print("filtering attentions...")
    attns = transpose_attns(attentions, filt_idxs)

    # turn to readable dict
    print("transforming readable...")
    num_layer = len(attentions[0])
    for l in range(num_layer):
        for ei, edge_ind in enumerate(tqdm(filt_edges[l].T)):  # [2]
            src, dst = edge_ind.numpy()  # [2]
            src = id2hid[l][src]  # turn hid
            dst = id2hid[l + 1][dst] if cross else id2hid[l][dst]
            edge = str(src) + "," + str(dst)
            attention_dict[str(l)][edge] = attns[l][ei].numpy()

    # average samples
    if not return_sample:
        for l in attention_dict.keys():
            for edge in attention_dict[l].keys():
                # 对每条边的所有data做平均
                att_matrix = attention_dict[l][edge]
                mean_att = np.mean(att_matrix, axis=0).reshape(-1).tolist()
                attention_dict[l][edge] = mean_att
    return attention_dict


def emask2dict(inner_edge_masks, inner_edge_indexs, hid2ids_dict):
    inner_edge_masks_dict = {}

    for l in range(len(inner_edge_masks)):
        inner_edge_mask = inner_edge_masks[l]
        inner_edge_index = inner_edge_indexs[l]
        assert len(inner_edge_mask) == inner_edge_index.size(1)
        assert max(inner_edge_index[0]) <= max(hid2ids_dict[l].values())

        inner_edge_masks_dict[l] = {}
        id2hid = dict(zip(hid2ids_dict[l].values(), hid2ids_dict[l].keys()))

        for i in range(inner_edge_mask.size(0)):
            u, v = int(inner_edge_index[0][i]), int(inner_edge_index[1][i])
            src, dst = id2hid[u], id2hid[v]
            k = src + "," + dst
            inner_edge_masks_dict[l][k] = float(inner_edge_mask[i])

    return inner_edge_masks_dict


def process_full_graph(metadata, inner_edge_masks, cross_edge_masks=None):
    edge_index, inner_edge_indexs, cross_edge_indexs, num_nodes = metadata

    full_edge_index = []
    full_edge_index.append(edge_index)

    for l in range(len(inner_edge_indexs) - 1):
        inner_edge_index = inner_edge_indexs[l] + sum(num_nodes[: l + 1])
        full_edge_index.append(inner_edge_index)

    full_edge_mask = torch.cat(inner_edge_masks)
    full_edge_index = torch.cat(full_edge_index, dim=1)
    assert len(full_edge_index[0]) == len(full_edge_mask)

    for l in range(len(cross_edge_indexs)):
        cross_edge_index = torch.stack(
            [
                cross_edge_indexs[l][0] + sum(num_nodes[:l]),
                cross_edge_indexs[l][1] + sum(num_nodes[: l + 1]),
            ],
            dim=0,
        )
        full_edge_index = torch.cat((full_edge_index, cross_edge_index), dim=1)
    if cross_edge_masks is None:
        cross_edge_masks = torch.ones(len(full_edge_index[0]) - len(full_edge_mask))

    full_edge_mask = torch.cat((full_edge_mask, cross_edge_masks), dim=0)
    assert full_edge_mask.size(0) == full_edge_index.size(1)

    return full_edge_index, full_edge_mask


def get_model_att(test_dataset, model, device):
    """output attentions for test_dataset of current model"""
    model.eval()
    with torch.no_grad():
        all_atts = []
        for data in test_dataset:
            data = data.to(device)
            data.batch = torch.LongTensor([0] * data.num_nodes).to(device)
            _, attentions = model(data.x, data.batch, return_attention_weights=True)
            all_atts.append(attentions)
    return all_atts


def set_multimasks(
    model,
    inner_edge_masks,
    inner_edge_indexs,
    cross_edge_masks=None,
    cross_edge_indexs=None,
    batch_size=None,
    apply_sigmoid=True,
    device="cpu",
):
    """pyG version request:"""
    layer = 0
    for module in model.modules():
        if isinstance(module, MessagePassing):
            edge_index = inner_edge_indexs[layer]
            mask = inner_edge_masks[layer]
            loop_mask = edge_index[0] != edge_index[1]
            module.explain = True
            if batch_size is None:
                module._edge_mask = mask.to(device)
                module._loop_mask = loop_mask.to(device)
                module._apply_sigmoid = apply_sigmoid
            else:
                module._edge_mask = mask.repeat(batch_size).to(device)
                module._loop_mask = loop_mask.repeat(batch_size).to(device)
                module._apply_sigmoid = apply_sigmoid

            layer += 1
            if layer == len(inner_edge_indexs):
                break


def clear_masks(model: torch.nn.Module):
    """Clear all masks from the model."""
    layer = 0
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.explain = False
            module._edge_mask = None
            module._loop_mask = None
            module._apply_sigmoid = True

            layer += 1
    return module


class PPIExplainer(torch.nn.Module):
    r"""The GNN-Explainer model from the `"GNNExplainer: Generating
    Explanations for Graph Neural Networks"
    <https://arxiv.org/abs/1903.03894>`_ paper for identifying compact subgraph
    structures and small subsets node features that play a crucial role in a
    GNN’s node-predictions.


    Args:
        model (torch.nn.Module): The GNN module to explain.
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`100`)
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.01`)
        return_type (str, optional): Denotes the type of output from
            :obj:`model`. Valid inputs are :obj:`"log_prob"` (the model
            returns the logarithm of probabilities), :obj:`"prob"` (the
            model returns probabilities), :obj:`"raw"` (the model returns raw
            scores) and :obj:`"regression"` (the model returns scalars).
            (default: :obj:`"log_prob"`)
        feat_mask_type (str, optional): Denotes the type of feature mask
            that will be learned. Valid inputs are :obj:`"feature"` (a single
            feature-level mask for all nodes), :obj:`"individual_feature"`
            (individual feature-level masks for each node), and :obj:`"scalar"`
            (scalar mask for each each node). (default: :obj:`"feature"`)
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in :attr:`~torch_geometric.nn.models.GNNExplainer.coeffs`.
    """

    coeffs = {
        "edge_size": 0.005,  # 原参数
        "edge_reduction": "sum",
        "node_feat_size": 1.0,
        "node_feat_reduction": "mean",
        "edge_ent": 1.0,  # 原参数
        "node_feat_ent": 0.1,
    }

    def __init__(
        self,
        model,
        dataset,
        epochs: int = 100,
        lr: float = 0.01,
        return_type: str = "log_prob",
        # feat_mask_type: str = 'feature',
        log: bool = True,
        device: str = "cuda",
        explain_cross: bool = False,
        train_sample_gt=0,
        ce_loss_gt=0,
        **kwargs,
    ):
        super().__init__()
        self.epochs = epochs
        self.lr = lr
        self.coeffs.update(kwargs)
        self.model = model
        self.return_type = return_type
        self.log = log

        # assert feat_mask_type in ['feature', 'individual_feature', 'scalar']
        # self.feat_mask_type = feat_mask_type

        self.coeffs.update(kwargs)

        metadata = dataset.metadata
        self.dataset = dataset
        self.edge_index, inner_edge_indexs, cross_edge_indexs, self.num_nodes = metadata
        self.inner_edge_indexs = []
        self.cross_edge_indexs = []
        self.edge_index = self.edge_index.to(device)  # 第一层
        self.inner_edge_indexs.append(self.edge_index)

        for i in range(len(self.num_nodes) - 2):
            inner_edge_index = inner_edge_indexs[i].to(device)
            self.inner_edge_indexs.append(inner_edge_index)
            cross_edge_index = cross_edge_indexs[i].to(device)
            self.cross_edge_indexs.append(cross_edge_index)

        self.num_features = dataset.num_features

        self.explain_cross = explain_cross
        self.device = device
        self.ce_loss_gt = ce_loss_gt
        self.train_sample_gt = train_sample_gt

    def _initialize_masks(self):
        N = self.num_nodes[0]
        F = self.num_features
        std = 0.1

        # if self.feat_mask_type == 'individual_feature':
        #     self.node_feat_mask = torch.nn.Parameter(torch.randn(N, F) * std)
        # elif self.feat_mask_type == 'scalar':
        #     self.node_feat_mask = torch.nn.Parameter(torch.randn(N, 1) * std)
        # else:
        #     self.node_feat_mask = torch.nn.Parameter(torch.randn(1, F) * std)

        self.inner_edge_masks = []
        if self.explain_cross:
            self.cross_edge_masks = []
        else:
            self.cross_edge_masks = None

        std = torch.nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * N))
        self.inner_edge_masks.append(
            torch.nn.Parameter(torch.randn(self.edge_index.size(1)) * std)
        )

        for i in range(len(self.num_nodes) - 2):
            num_nodes1 = self.num_nodes[i]
            std = torch.nn.init.calculate_gain("relu") * sqrt(2.0 / (2 * num_nodes1))
            inner_edge_index = self.inner_edge_indexs[i + 1]
            self.inner_edge_masks.append(
                torch.nn.Parameter(torch.randn(inner_edge_index.size(1)) * std)
            )

            if self.explain_cross:
                num_nodes2 = self.num_nodes[i + 1]
                std = torch.nn.init.calculate_gain("relu") * sqrt(
                    2.0 / (2 * (num_nodes1 + num_nodes2))
                )
                cross_edge_index = self.cross_edge_indexs[i]
                self.cross_edge_masks.append(
                    torch.nn.Parameter(torch.randn(cross_edge_index.size(1)) * std)
                )

    def _clear_masks(self):
        clear_masks(self.model)
        self.node_feat_masks = None
        self.inner_edge_masks = None
        self.cross_edge_masks = None

    def get_loss(self, out: Tensor, prediction: Tensor, **kwargs):
        log_logits = self._to_log_prob(out)
        loss = self._loss(log_logits, prediction, **kwargs)
        return loss

    def _loss(self, log_logits, prediction):
        # log_logits: [B, c] , prediction: [B,]
        loss = 0
        for i in range(len(prediction)):
            loss += -log_logits[i, prediction[i]]

        masks = self.inner_edge_masks
        if self.explain_cross:
            masks += self.cross_edge_masks
        for edge_mask in masks:
            m = edge_mask.sigmoid()

            edge_reduce = getattr(torch, self.coeffs["edge_reduction"])
            loss = loss + self.coeffs["edge_size"] * edge_reduce(m)
            ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
            loss = loss + self.coeffs["edge_ent"] * ent.mean()

        # m = self.node_feat_mask.sigmoid()
        # node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        # loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m)
        # ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        # loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss

    @torch.no_grad()
    def get_initial_prediction(self, dataset: list, **kwargs):
        batch = None
        C = self.dataset.num_classes

        data_idx_dict = {i: [] for i in range(C)}
        predictions = []
        for i in range(len(dataset)):
            data = dataset[i]
            data = data.to(self.device)
            batch = torch.zeros(data.x.shape[0], dtype=int, device=self.device)
            out = self.model(data.x, batch, **kwargs)
            if self.return_type == "regression":
                prediction = out
            else:
                prediction = out.argmax(dim=-1)

            data_idx_dict[prediction.item()].append(i)
            predictions.append(prediction)

        predictions = torch.LongTensor(predictions).to(self.device)

        return predictions, data_idx_dict

    def _to_log_prob(self, out):
        return torch.log_softmax(out, dim=1)

    def explain_graph(self, dataset, batch_size, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to explain the prediction made by the GNN for a graph.

        Args:
            dataset: list of datas
            batch_size: batch_size when training masks
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        self.model.eval()
        self._clear_masks()

        # Get the initial prediction (logits) and labels
        predictions, data_idx_dict = self.get_initial_prediction(dataset, **kwargs)

        for i, data in enumerate(dataset):
            data.orig_idx = i

        total_orig_acc = 0
        total_acc = 0

        # 对于每一个预测的label输出解释subgraph
        inner_edge_mask_dict = {}
        cross_edge_mask_dict = {}
        for c in data_idx_dict:
            # 预测为label c的data
            self._initialize_masks()
            self.to(self.device)
            parameters = self.inner_edge_masks
            if self.explain_cross:
                parameters += self.cross_edge_masks

            optimizer = torch.optim.Adam(parameters, lr=self.lr)
            data_index = data_idx_dict[c]
            if len(data_index) == 0:
                print("NO graph has predict label {}!".format(c))
                continue

            # 筛选训练样本
            if self.train_sample_gt:
                # 用ground true label来过滤训练样本
                c_datas = [data for i, data in enumerate(dataset) if data.y == c]
            else:
                # 用predict label来过滤训练样本
                c_datas = [
                    data for i, data in enumerate(dataset) if i in set(data_index)
                ]
            train_loader = DataLoader(c_datas, batch_size, shuffle=True)
            if self.log:  # pragma: no coverf
                pbar = tqdm(total=self.epochs)
                pbar.set_description(
                    "Explaining for prediction of label {}...".format(c)
                )

            t0 = time.time()
            for epoch in range(1, self.epochs + 1):
                for data in train_loader:
                    set_multimasks(
                        self.model,
                        self.inner_edge_masks,
                        self.inner_edge_indexs,
                        self.cross_edge_masks,
                        self.cross_edge_indexs,
                        data.num_graphs,
                        apply_sigmoid=True,
                        device=self.device,
                    )
                    data = data.to(self.device)

                    optimizer.zero_grad()
                    out = self.model(x=data.x, batch=data.batch, **kwargs)
                    if self.ce_loss_gt:
                        loss = self.get_loss(out, data.y)
                    else:
                        loss = self.get_loss(out, predictions[data.orig_idx])
                    loss.backward()
                    optimizer.step()

                if self.log:  # pragma: no cover
                    pbar.update(1)

            if self.log:  # pragma: no cover
                pbar.close()

            ## evaluation
            orig_acc = 0
            after_acc = 0
            predict_match = 0

            with torch.no_grad():
                set_multimasks(
                    self.model,
                    self.inner_edge_masks,
                    self.inner_edge_indexs,
                    self.cross_edge_masks,
                    self.cross_edge_indexs,
                    apply_sigmoid=True,
                    device=self.device,
                )

                for data in [
                    data for i, data in enumerate(dataset) if i in set(data_index)
                ]:
                    data = data.to(self.device)
                    out = self.model(
                        x=data.x,
                        batch=torch.zeros(
                            data.x.shape[0], dtype=int, device=self.device
                        ),
                    )
                    prediction = predictions[data.orig_idx]
                    after_pred = out.argmax()

                    orig_acc += prediction.eq(data.y).item()
                    after_acc += after_pred.eq(data.y).item()
                    predict_match += prediction.eq(after_pred).item()

            print(
                "[Finish Explaining Label {}] num_graphs={}, orig acc={:.4f}, after acc={:.4f}, predict match={:.2f}%".format(
                    c,
                    len(data_index),
                    orig_acc / len(data_index),
                    after_acc / len(data_index),
                    predict_match / len(data_index) * 100,
                )
            )

            ## output
            inner_edge_masks = []  # [hierarchy, E_inner] [层数，每层内的边数] 每一条边都有一个mask score
            cross_edge_masks = []  # [hierarchy, E_cross]
            for i, mask in enumerate(self.inner_edge_masks):
                inner_edge_masks.append(mask.detach().cpu().sigmoid())
            if self.explain_cross:
                for i in self.cross_edge_masks:
                    cross_edge_masks.append(i.detach().cpu().sigmoid())
                # node_feat_mask = self.node_feat_mask.detach().sigmoid().squeeze()

            inner_edge_mask_dict[c] = inner_edge_masks
            cross_edge_mask_dict[c] = cross_edge_masks

            self._clear_masks()

            total_orig_acc += orig_acc
            total_acc += after_acc
        print(
            "total orig acc:{}, total after acc:{}".format(
                total_orig_acc / len(dataset), total_acc / len(dataset)
            )
        )

        return inner_edge_mask_dict, cross_edge_mask_dict

    def explain(self, args, **kwargs):
        # explain one single graph
        inner_edge_mask_dict, cross_edge_mask_dict = self.explain_graph(
            self.dataset.datas, args.batch_size, **kwargs
        )

        # gat encoder: edge_mask * edge attention score, 输出的mask score是否乘以每条边的attention
        if args.multi_atten == 1 and args.encoder == "gat":
            best_attentions = get_model_att(self.dataset.datas, self.model, self.device)
            attention_dict = process_attention(
                self.dataset, best_attentions, hierarchy=args.hierarchy, cross=False
            )

        output_dict = {}  # key: classes, value: {层：{边： mask score}}
        for c in inner_edge_mask_dict:
            class_mask = emask2dict(
                inner_edge_mask_dict[c],
                self.inner_edge_indexs,
                self.dataset.raw_data["hid2ids"],
            )

            if args.multi_atten == 1 and args.encoder == "gat":
                for l in class_mask.keys():
                    mask = class_mask[l]
                    atten_dict = attention_dict[str(l)]
                    for e in atten_dict:
                        mask[e] = mask[e] * atten_dict[e][-1]
                    class_mask[l] = mask
            output_dict[c] = class_mask

        ## 把每个类别的mask拍平成vector，可视化每个类别mask的correlation
        if args.correlation:
            mask_vecs = []
            for c in inner_edge_mask_dict:
                inner_edge_masks = inner_edge_mask_dict[c]
                mask_vec = np.concatenate(inner_edge_masks, axis=0)
                mask_vecs.append(mask_vec)

            mask_vecs_np = np.stack(mask_vecs, axis=0)
            coef = np.corrcoef(mask_vecs_np)
            ax = sns.heatmap(coef, vmax=1)
            plt.show()
            plt.savefig(
                os.path.join(
                    args.save_dir, args.encoder + "_emask_correlation_ppiexplainer.png"
                ),
                format="PNG",
            )

        return output_dict

    def __repr__(self):
        return f"{self.__class__.__name__}()"
