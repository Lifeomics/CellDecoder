from .bnn import Net as NetO
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, GATConv


def gatconv(in_dim, dim, out_dim, args):
    class MCONV(torch.nn.Module):
        def __init__(self, in_channels, dim, out_channels):
            super().__init__()
            self.conv = GATConv(
                in_channels, dim, concat=False, heads=args.heads, dropout=args.dropout
            )
            self.bn = BatchNorm1d(dim)

        def forward(self, x, edge_index, return_attention_weights=False):
            if return_attention_weights:
                x, (edge_index_, alpha) = self.conv(
                    x, edge_index, return_attention_weights=True
                )
                x = self.bn(x)
                return x.relu(), (edge_index_, alpha)
            else:
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


class Net(NetO):
    def __init__(self, in_channels, dim, out_channels, dataset, args=None, device=None):
        super().__init__(
            in_channels, dim, out_channels, dataset, args=args, device=device
        )
        self.conv1, self.inner_convs = get_inner_convs(
            args, self.in_channels, dim, out_channels, len(self.inner_edge_indexs)
        )

    def forward_inner(self, x, batch_size, i, return_attention_weights=False):
        args = self.args
        num_nodes1 = self.num_nodes[i]
        num_nodes2 = self.num_nodes[i + 1]
        inner_edge_index = self.inner_edge_indexs[i]
        inner_edge_index = self.dataset.construct_inner_edge(
            inner_edge_index, batch_size, num_nodes2, i
        )
        if return_attention_weights:
            x, (edge_index_, alpha) = self.inner_convs[i](
                x, inner_edge_index, return_attention_weights=True
            )
            return x, (edge_index_, alpha)
        else:
            x = self.inner_convs[i](x, inner_edge_index)
            return x

    def forward(self, x, batch, return_attention_weights=False):
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
        if return_attention_weights and self.encoder == "gat":
            x, (edge_index_, alpha) = self.conv1(
                x, edge_index, return_attention_weights=True
            )
            attentions.append((edge_index_.detach().cpu(), alpha.detach().cpu()))
        else:
            x = self.conv1(x, edge_index)

        for i in range(len(self.num_nodes) - 2):
            x = self.forward_cross(x, batch_size, i)
            if return_attention_weights and self.encoder == "gat":
                x, (edge_index_, alpha) = self.forward_inner(
                    x, batch_size, i, return_attention_weights=True
                )
                attentions.append((edge_index_.detach().cpu(), alpha.detach().cpu()))
            else:
                x = self.forward_inner(x, batch_size, i)

        x = x.view(batch_size, -1)
        if args.skip_raw:
            x = torch.cat([x, org_x], dim=-1)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.lin2(x)

        if return_attention_weights and self.encoder == "gat":
            return x, attentions
        else:
            return x
