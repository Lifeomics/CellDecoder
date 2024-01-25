from .bnn import Net as NetO
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, GATConv


class Net(NetO):
    def clean_explain(self):
        self.inner_grads = {}
        self.inner_feature_maps = {}

    def add_explain(self, x):
        x.requires_grad_(True)
        idx = len(self.inner_feature_maps)
        self.inner_feature_maps[idx] = x

        def hook(grad):
            self.inner_grads[idx] = grad

        x.register_hook(hook)

    def forward_explain(self, x, batch):
        self.clean_explain()

        args = self.args
        attentions = []

        batch_size = batch[-1].item() + 1

        org_x = x.view(batch_size, -1)
        if args.add_one_hot:
            node_embs = self.node_embs
            node_embs = node_embs.expand(batch_size, *node_embs.shape)
            node_embs = node_embs.reshape(x.shape[0], -1)
            x = torch.cat([x, node_embs], dim=-1)

        self.add_explain(x)

        edge_index = self.dataset.construct_inner_edge(
            self.edge_index, batch_size, self.num_nodes[0], -1
        )
        x = self.conv1(x, edge_index)

        for i in range(len(self.num_nodes) - 2):
            x = self.forward_cross(x, batch_size, i)
            self.add_explain(x)
            x = self.forward_inner(x, batch_size, i)

        x = x.view(batch_size, -1)
        if args.skip_raw:
            x = torch.cat([x, org_x], dim=-1)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.lin2(x)

        return x
