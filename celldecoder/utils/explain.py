import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import os
from torch_geometric.data import DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm, trange
import random


def saliency_map(input_grads):  # [node, dim]
    # print('saliency_map')
    node_saliency_map = []
    for n in range(input_grads.shape[0]):  # nth node
        node_grads = input_grads[n, :]
        node_saliency = torch.norm(F.relu(node_grads)).item()
        node_saliency_map.append(node_saliency)
    node_saliency_map = torch.tensor(node_saliency_map)
    return node_saliency_map  # [node, 1]


def grad_cam(final_conv_acts, final_conv_grads):  # [node, dim]
    # print('grad_cam')
    node_heat_map = []
    alphas = torch.mean(
        final_conv_grads, axis=0
    )  # mean gradient for each feature (512x1)
    for n in range(final_conv_acts.shape[0]):  # nth node
        node_heat = F.relu(alphas @ final_conv_acts[n]).item()
        node_heat_map.append(node_heat)
    node_heat_map = torch.tensor(node_heat_map)
    return node_heat_map


def get_grads(model, dataloader, device):
    ys = []
    grads = []  # [layer, num_samples, node, dim]
    fmaps = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def append(cont, index, x):
        if len(cont) > index:
            cont[index].append(x)
        elif len(cont) == index:
            cont.append([x])
        else:
            raise ValueError(f"len(cont)={len(cont)} < index={index}")

    for data in tqdm(dataloader):
        data = data.to(device)
        batch = data.batch
        batch_size = batch[-1].item() + 1

        optimizer.zero_grad()
        preds = model.forward_explain(data.x, data.batch)
        loss = F.cross_entropy(preds, data.y)
        loss.backward()

        ys.append(data.y)
        num = len(model.inner_feature_maps)

        for i in range(num):
            x = model.inner_feature_maps[i]
            append(fmaps, i, x.view(batch_size, -1, x.shape[-1]).detach().cpu())
            x = model.inner_grads[i]
            append(grads, i, x.view(batch_size, -1, x.shape[-1]).detach().cpu())

    ys = torch.cat(ys).cpu()
    for i in trange(num):
        grads[i] = torch.cat(grads[i], dim=0)
        fmaps[i] = torch.cat(fmaps[i], dim=0)
        print(grads[i].shape, fmaps[i].shape)
    return grads, fmaps, ys


def get_id2nodes(hid2ids):
    ls = list(range(len(hid2ids)))
    for k, v in hid2ids.items():
        ls[v] = k
    return ls


def gen_explains(fmaps, grads, explain_method):
    exps = []  # [layer,sample,node_explain]
    for layer in range(len(fmaps)):
        fmap = fmaps[layer]
        grad = grads[layer]
        exp_layer = []
        for sample_idx in tqdm(range(len(fmap))):
            if explain_method == "grad":
                exp = saliency_map(grad[sample_idx])
            elif explain_method == "grad_cam":
                exp = grad_cam(fmap[sample_idx], grad[sample_idx])
            else:
                raise NotImplementedError(f"Unknown explain method {explain_method}")
            exp_layer.append(exp)
        exp_layer = torch.stack(exp_layer, dim=0)
        exps.append(exp_layer)
    return exps


class FeatExplainer:
    def __init__(self, device, fmaps, grads, explain_method, batch_size=-1):
        self.device = device
        self.fmaps = fmaps
        self.grads = grads
        # self.fmaps = [fmap.to(device) for fmap in fmaps]
        # self.grads = [grad.to(device) for grad in grads]
        self.explain_method = explain_method
        self.batch_size = batch_size

    def gen_explains_gpu(self):
        fmaps, grads, explain_method, batch_size = (
            self.fmaps,
            self.grads,
            self.explain_method,
            self.batch_size,
        )
        # fmaps : [layer,num_samples,node,dim]
        exps = []  # [layer,sample,node_explain]
        for layer in trange(len(fmaps)):
            if batch_size > 0:
                exp_layer = []
                for batch in range(0, len(fmaps[layer]), self.batch_size):
                    fmap = fmaps[layer][batch : batch + self.batch_size].to(self.device)
                    grad = grads[layer][batch : batch + self.batch_size].to(self.device)
                    exp = self.gen_explains_layer_gpu(fmap, grad, explain_method).cpu()
                    exp_layer.append(exp)
                exp_layer = torch.cat(exp_layer, dim=0)
            else:
                fmap = fmaps[layer].to(self.device)
                grad = grads[layer].to(self.device)
                exp_layer = self.gen_explains_layer_gpu(
                    fmap, grad, explain_method
                ).cpu()
            exps.append(exp_layer)
        return exps

    def gen_explains_layer_gpu(self, fmap, grad, explain_method):
        # grad : [num_samples,node,dim]
        if explain_method == "grad":
            exp = self.grad_layer_gpu(grad)
        elif explain_method == "grad_cam":
            exp = self.grad_cam_layer_gpu(fmap, grad)
        else:
            raise NotImplementedError(f"Unknown explain method {explain_method}")
        return exp

    @torch.no_grad()
    def grad_layer_gpu(self, grad):
        # grad : [num_samples,node,dim]
        exp = torch.norm(F.relu(grad), dim=-1)  # [num_samples,node]
        return exp

    @torch.no_grad()
    def grad_cam_layer_gpu(self, fmap, grad):
        # grad : [num_samples,node,dim]
        alphas = torch.mean(grad, dim=1, keepdim=True)  # [num_samples, 1, dim]
        exp = F.relu((alphas * fmap).sum(dim=-1))  # [num_samples, node]
        return exp
