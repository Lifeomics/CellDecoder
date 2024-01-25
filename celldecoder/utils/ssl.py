import torch.nn.functional as F
import torch
import numpy as np


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score


@torch.no_grad()
def graph_classification_evaluation(model, pooler, dataloader, device="cuda:0"):
    model.eval()
    ys = []
    preds = []
    for data in dataloader:
        ys.append(data.y)
        data = data.to(device)
        out = model.embed(data.x, data.edge_index, data.batch)
        out = pooler(out, data.batch)
        preds.append(out.cpu())
    x = torch.cat(preds).numpy()
    y = torch.cat(ys).numpy()
    test_f1, test_std = evaluate_graph_embeddings_using_svm(x, y)
    print(f"#Test : {test_f1}±{test_std}")
    return test_f1


from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from .data import *


def evaluate_graph_embeddings_using_svm(embeddings, labels):
    result = []
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    # for train_index, test_index in kf.split(embeddings, labels):
    folds = get_stratified_fold(
        embeddings, labels, n_splits=10, val_ratio=0, show_index=False
    )
    for train_index, val_index, test_index in folds:
        x_train = embeddings[train_index]
        x_test = embeddings[test_index]
        y_train = labels[train_index]
        y_test = labels[test_index]
        params = {"C": [1e-3, 1e-2, 1e-1, 1, 10]}
        svc = SVC(random_state=42)
        clf = GridSearchCV(svc, params)
        clf.fit(x_train, y_train)

        preds = clf.predict(x_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")
        result.append([f1, acc])
    test_f1 = np.mean(result, axis=0)
    test_std = np.std(result, axis=0)
    return test_f1, test_std


class GraphMAELoss(torch.nn.Module):
    def __init__(
        self,
        encoder,
        encoder_to_decoder,
        decoder,
        enc_mask_token,
        criterion,
        mask_rate=0.1,
    ):
        super(GraphMAELoss, self).__init__()
        self._replace_rate = 0
        self._drop_edge_rate = 0
        self._mask_rate = mask_rate
        self._mask_token_rate = 1 - self._replace_rate
        self._decoder_type = "gin"

        self.encoder = encoder
        self.encoder_to_decoder = encoder_to_decoder
        self.decoder = decoder
        self.criterion = criterion
        self.enc_mask_token = enc_mask_token

    def encoding_mask_noise(self, edge_index, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[
                perm_mask[: int(self._mask_token_rate * num_mask_nodes)]
            ]
            noise_nodes = mask_nodes[
                perm_mask[-int(self._replace_rate * num_mask_nodes) :]
            ]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[
                :num_noise_nodes
            ]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        return edge_index, out_x, (mask_nodes, keep_nodes)

    # def mask_attr_prediction(self, x,edge_index):
    def forward(self, x, edge_index, x_init=None):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(
            edge_index, x, self._mask_rate
        )

        if self._drop_edge_rate > 0:
            pass
            # use_g, masked_edges = drop_edge(pre_use_g, self._drop_edge_rate, return_edges=True)
        else:
            use_g = pre_use_g

        enc_rep = self.encoder(use_x, use_g)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type != "mlp":
            # * remask, re-mask
            rep[mask_nodes] = 0  #

        if self._decoder_type == "mlp":
            recon = self.decoder(rep)
        else:
            recon = self.decoder(rep, pre_use_g)

        if x_init is None:
            x_init = x
        x_init = x_init[mask_nodes]
        x_rec = recon[mask_nodes]

        loss = self.criterion(x_rec, x_init)
        return loss
