import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool
from torch_scatter import scatter
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv
from torch import nn


def activate_func(x, func):
    if func == "tanh":
        return torch.tanh(x)
    elif hasattr(F, func):
        return getattr(F, func)(x)
    elif func == "":
        pass
    else:
        raise TypeError("PyTorch does not support activation function {}".format(func))
    return x


def mean_pool():
    def pool(x, edge_index, num_nodes):
        res = x[edge_index[0]]
        res = scatter(res, edge_index[1], dim=0, dim_size=num_nodes, reduce="mean")
        return res

    return pool


def attn_pool(dim, args):
    return GATConv(
        dim,
        dim,
        heads=args.heads,
        concat=False,
        dropout=args.dropout,
        add_self_loops=args.add_self_loop,
    )


def gatconv(in_dim, dim, out_dim, args):
    class MCONV(torch.nn.Module):
        def __init__(self, in_channels, dim, out_channels):
            super().__init__()
            self.conv = GATConv(
                in_channels, dim, concat=False, heads=args.heads, dropout=args.dropout
            )
            self.bn = BatchNorm1d(dim)

        def forward(self, x, edge_index):
            x = self.conv(x, edge_index)
            x = self.bn(x)
            return x.relu()

    return MCONV(in_dim, dim, out_dim)


def get_inner_convs(args, in_channels, dim, out_channels, layer):
    if args.encoder == "gin":

        def ginconv(dim):
            return GINConv(
                Sequential(
                    Linear(dim, dim), BatchNorm1d(dim), ReLU(), Linear(dim, dim), ReLU()
                )
            )

        conv1 = GINConv(
            Sequential(
                Linear(in_channels, dim),
                BatchNorm1d(dim),
                ReLU(),
                Linear(dim, dim),
                ReLU(),
            )
        )
        inner_convs = torch.nn.ModuleList([ginconv(dim) for i in range(layer)])
    elif args.encoder == "gat":
        conv1 = gatconv(in_channels, dim, dim, args)
        inner_convs = torch.nn.ModuleList(
            [gatconv(dim, dim, dim, args) for i in range(layer)]
        )
    else:
        raise NotImplementedError(f"encoder {args.encoder} not implemented")
    return conv1, inner_convs


def get_cross_convs(args, in_channels, dim, out_channels, layer):
    if args.pool == "mean":
        cross_convs = [mean_pool() for i in range(layer)]
    elif args.pool == "gat":
        cross_convs0 = [mean_pool() for i in range(layer)]
        cross_convs1 = nn.ModuleList([attn_pool(dim, args) for i in range(layer)])
        cross_convs = [cross_convs0, cross_convs1]
    else:
        raise NotImplementedError(f"encoder {args.encoder} not implemented")
    return cross_convs


class Net(torch.nn.Module):
    def __init__(self, in_channels, dim, out_channels, dataset, args=None, device=None):
        super().__init__()
        self.args = args
        self.encoder = args.encoder

        if args.add_one_hot:
            print(f"add one hot : in_dim from {in_channels} to {in_channels+dim}")
            in_channels += dim
            self.node_embs = torch.nn.Parameter(torch.Tensor(args.num_nodes, dim))
            self.node_embs.data.uniform_(-1, 1)

        self.in_channels = in_channels
        metadata = dataset.metadata
        self.dataset = dataset
        (
            self.edge_index,
            self.inner_edge_indexs,
            self.cross_edge_indexs,
            self.num_nodes,
        ) = metadata
        self.edge_index = self.edge_index.to(device)
        self.inner_edge_indexs = [i.to(device) for i in self.inner_edge_indexs]
        self.cross_edge_indexs = [i.to(device) for i in self.cross_edge_indexs]

        self.conv1, self.inner_convs = get_inner_convs(
            args, in_channels, dim, out_channels, len(self.inner_edge_indexs)
        )
        self.cross_convs = get_cross_convs(
            args, in_channels, dim, out_channels, len(self.cross_edge_indexs)
        )
        if args.pool == "gat":
            self.cross_convs0, self.cross_convs1 = self.cross_convs

        num_node_last = self.num_nodes[-2]
        if args.skip_raw:
            self.lin1 = Linear(num_node_last * dim + self.num_nodes[0], dim)
        else:
            self.lin1 = Linear(num_node_last * dim, dim)

        self.lin2 = Linear(dim, out_channels)

    def forward_cross(self, x, batch_size, i):
        args = self.args
        num_nodes1 = self.num_nodes[i]
        num_nodes2 = self.num_nodes[i + 1]
        if args.pool == "gat":
            # first do mean pool
            cross_edge_index = self.cross_edge_indexs[i]
            cross_edge_index0 = self.dataset.construct_cross_edge(
                cross_edge_index, batch_size, num_nodes1, num_nodes2, i
            )
            x0 = x
            x0 = x0.view(batch_size, num_nodes1, -1)
            x1 = self.cross_convs0[i](
                x, cross_edge_index0, num_nodes2 * batch_size
            )  # get new layer x初始值
            x1 = x1.view(batch_size, num_nodes2, -1)
            x = torch.cat([x0, x1], dim=1)
            x = x.view(batch_size * (num_nodes1 + num_nodes2), -1)

            # then attention
            cross_edge_index1 = self.dataset.construct_cross_edge_both(
                cross_edge_index, batch_size, num_nodes1, num_nodes2, i
            )
            x = self.cross_convs1[i](x, cross_edge_index1)

            # new mask
            mask = torch.arange(num_nodes2 + num_nodes1) >= num_nodes1
            mask = mask.repeat(batch_size)
            x = x[mask]

            # activation
            x = activate_func(x, args.pool_act)

        elif args.pool == "mean":
            cross_edge_index = self.cross_edge_indexs[i]
            cross_edge_index = self.dataset.construct_cross_edge(
                cross_edge_index, batch_size, num_nodes1, num_nodes2, i
            )
            x = self.cross_convs[i](x, cross_edge_index, num_nodes2 * batch_size)
        else:
            raise NotImplementedError(f"pool {self.pool} not implemented")
        return x

    def forward_inner(self, x, batch_size, i):
        args = self.args
        num_nodes1 = self.num_nodes[i]
        num_nodes2 = self.num_nodes[i + 1]
        inner_edge_index = self.inner_edge_indexs[i]
        inner_edge_index = self.dataset.construct_inner_edge(
            inner_edge_index, batch_size, num_nodes2, i
        )
        x = self.inner_convs[i](x, inner_edge_index)
        return x

    def forward(self, x, batch):
        args = self.args
        attentions = []

        batch_size = batch[-1].item() + 1

        org_x = x.view(batch_size, -1)
        if args.add_one_hot:
            node_embs = self.node_embs
            node_embs = node_embs.expand(batch_size, *node_embs.shape)
            node_embs = node_embs.reshape(x.shape[0], -1)
            x = torch.cat([x, node_embs], dim=-1)

        edge_index = self.dataset.construct_inner_edge(
            self.edge_index, batch_size, self.num_nodes[0], -1
        )
        x = self.conv1(x, edge_index)

        for i in range(len(self.num_nodes) - 2):
            x = self.forward_cross(x, batch_size, i)
            x = self.forward_inner(x, batch_size, i)

        x = x.view(batch_size, -1)
        if args.skip_raw:
            x = torch.cat([x, org_x], dim=-1)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.lin2(x)

        return x


class NetFeat(Net):
    def __init__(self, in_channels, dim, out_channels, dataset, args=None, device=None):
        super().__init__(in_channels, dim, out_channels, dataset, args=args, device=device)
        self.internal_embeddings = {}
        
    def forward(self, x, batch):
        args = self.args

        batch_size = batch[-1].item() + 1
        nhid = args.nhid
        org_x = x.view(batch_size, -1)
        if args.add_one_hot:
            node_embs = self.node_embs
            node_embs = node_embs.expand(batch_size, *node_embs.shape)
            node_embs = node_embs.reshape(x.shape[0], -1)
            x = torch.cat([x, node_embs], dim=-1)
            self.internal_embeddings['add_one_hot'] = x.detach().cpu().reshape(batch_size, -1, x.shape[-1])

        edge_index = self.dataset.construct_inner_edge(self.edge_index, batch_size, self.num_nodes[0], -1)
        
        x = self.conv1(x, edge_index)
        self.internal_embeddings['conv1'] = x.detach().cpu().reshape(batch_size, -1, nhid)
        for i in range(len(self.num_nodes) - 2):
            x = self.forward_cross(x, batch_size, i)
            
            self.internal_embeddings[f'cross_{i}'] = x.detach().cpu().reshape(batch_size, -1, nhid)
            
            x = self.forward_inner(x, batch_size, i)
            
            self.internal_embeddings[f'inner_{i}'] = x.detach().cpu().reshape(batch_size, -1, nhid)
            
        x = x.view(batch_size, -1)
        if args.skip_raw:
            x = torch.cat([x, org_x], dim=-1)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.lin2(x)
        
        self.internal_embeddings[f'output'] = x.detach().cpu()
        return x

class MLP(Net):
    def __init__(self, in_channels, dim, out_channels, dataset, args=None, device=None):
        super().__init__(in_channels, dim, out_channels, dataset, args, device)
        self.lin1 = Linear(self.num_nodes[0], dim)

    def forward(self, x, batch):
        args = self.args
        batch_size = batch[-1].item() + 1
        x = x.view(batch_size, -1)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.lin2(x)

        return x


class MLP_infer(Net):
    def __init__(self, in_channels, dim, out_channels, dataset, args=None, device=None):
        super().__init__(in_channels, dim, out_channels, dataset, args, device)

    def forward(self, x, batch):
        args = self.args
        batch_size = batch[-1].item() + 1
        x = x.view(batch_size, -1)
        x = F.linear(x, self.lin1.weight[:, -self.num_nodes[0] :], self.lin1.bias)
        x = x.relu()
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.lin2(x)

        return x
