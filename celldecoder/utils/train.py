from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
from .misc import EarlyStopping
from tqdm import tqdm
import os


def cal_metric(preds, targets, metric):
    if metric == "accuracy":
        return accuracy_score(targets, preds)
    elif metric == "f1_macro":
        return f1_score(targets, preds, average="macro")
    elif metric == "f1_micro":
        return f1_score(targets, preds, average="micro")
    elif metric == "f1_weighted":
        return f1_score(targets, preds, average="weighted")
    else:
        raise NotImplementedError(f"{metric} not implemented")
    return None


def cal_metric_dict(preds, targets, metrics):
    metric_dict = {k: cal_metric(preds, targets, k) for k in metrics}
    return metric_dict


class Trainer:
    def __init__(self, model, optimizer, metrics, args=None, writer=None):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.metrics = metrics
        self.writer = writer

    def train(self, model, train_loader, optimizer, device):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data.x, data.batch)
            loss = F.cross_entropy(output, data.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * data.num_graphs
        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def test(self, model, loader, device, metrics=["accuracy", "f1_macro"]):
        model.eval()
        ys = []
        preds = []
        for data in loader:
            ys.append(data.y)
            data = data.to(device)
            out = model(data.x, data.batch).cpu()
            preds.append(out)
        ys = torch.cat(ys).numpy()
        preds = F.softmax(torch.cat(preds), dim=1).numpy()
        preds = np.argmax(preds, axis=1)
        metric_dict = cal_metric_dict(preds, ys, metrics)
        return metric_dict

    def train_till_end(self, dataset, device):
        args = self.args
        patience, batch_size, max_epochs, early_metric = (
            args.patience,
            args.batch_size,
            args.max_epochs,
            args.early_metric,
        )
        save_model_path = args.ckpt_path
        model, optimizer, writer = self.model, self.optimizer, self.writer

        train_loader = DataLoader(dataset.train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(dataset.val_dataset, batch_size)

        earlystop = EarlyStopping(mode="max", patience=patience)
        best_val, best_metric_dict = 0, {}

        with tqdm(range(max_epochs)) as bar:
            for epoch in bar:
                loss = self.train(model, train_loader, optimizer, device)

                train_metric_dict = self.test(model, train_loader, device)
                metric_dict = self.test(model, val_loader, device)
                val_metric = metric_dict[early_metric]

                if val_metric > best_val:
                    best_val = val_metric
                    best_metric_dict = metric_dict
                    if save_model_path:
                        os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
                        torch.save(model.state_dict(), save_model_path)

                bar.set_postfix(
                    loss=loss,
                    bval=best_val,
                    train=train_metric_dict[early_metric],
                    val=metric_dict[early_metric],
                )
                if writer:
                    for k, v in metric_dict.items():
                        writer.add_scalar(f"val_{k}", v, epoch)
                    for k, v in train_metric_dict.items():
                        writer.add_scalar(f"train_{k}", v, epoch)

                if earlystop.step(val_metric):
                    break

        print(f"{best_metric_dict}")
        return best_metric_dict


class Tester:
    def __init__(self, model, metrics, args=None):
        self.model = model
        self.args = args
        self.metrics = metrics

    @torch.no_grad()
    def pred(self, dataset, device, prob=False):
        model, args, metrics = self.model, self.args, self.metrics
        batch_size = args.batch_size
        loader = DataLoader(dataset, batch_size, shuffle=False)

        model.eval()
        preds = []
        for data in tqdm(loader):
            data = data.to(device)
            out = model(data.x, data.batch).cpu()
            preds.append(out)
        preds = F.softmax(torch.cat(preds), dim=1).numpy()
        if not prob:
            preds = np.argmax(preds, axis=1)
        return preds

    @torch.no_grad()
    def test(self, dataset, device):
        model, args, metrics = self.model, self.args, self.metrics
        batch_size = args.batch_size
        loader = DataLoader(dataset, batch_size, shuffle=False)

        ys = torch.cat([data.y for data in loader]).numpy()
        preds = self.pred(dataset, device, prob=False)
        metric_dict = cal_metric_dict(preds, ys, metrics)
        return metric_dict
