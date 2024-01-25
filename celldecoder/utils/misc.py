from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print("func:%r took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap


import random
import numpy as np
import torch
from tqdm import tqdm
import argparse
import torch
import os
import shutil
import json


def get_arg_dict(args):
    info_dict = args.__dict__
    ks = list(info_dict.keys())
    arg_dict = {}
    for k in ks:
        v = info_dict[k]
        for t in [int, float, str, bool, torch.Tensor]:
            if isinstance(v, t):
                arg_dict[k] = v
                break
    return arg_dict


import pandas as pd


def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in PyTorch, numpy and
    Python.

    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class EarlyStopping:
    """EarlyStopping class to keep NN from overfitting. copied from nni"""

    def __init__(self, mode="min", min_delta=0, patience=20, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        """EarlyStopping step on each epoch
        @params metrics: metric value
        @return : True if stop
        """

        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def reset(self):
        self.best = None

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


def sorteddict(x, min=True, dim=1):
    """return dict sorted by values
    @params x: a dict
    @params min : whether from small to large.
    """
    if min:
        return dict(sorted(x.items(), key=lambda item: item[dim]))
    else:
        return dict(sorted(x.items(), key=lambda item: item[dim])[::-1])
