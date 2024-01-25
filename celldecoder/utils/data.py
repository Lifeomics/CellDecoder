from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np


def get_fold(sample_num, n_splits=5, val_ratio=0.1, show_index=False):
    kf = KFold(n_splits=n_splits)
    folds = []
    for i, (train_index, test_index) in enumerate(kf.split(range(sample_num))):
        train_num = len(train_index)
        if val_ratio > 0:
            val_num = max(1, int(val_ratio * train_num))
            val_index = train_index[-val_num:]
            train_index = train_index[:-val_num]
        else:
            # dummy, do not use val
            val_index = train_index[-1:]
        folds.append([train_index, val_index, test_index])
        print(
            f"fold {i} train: {len(train_index)} val:{len(val_index)} test: {len(test_index)}"
        )
        if show_index:
            print(train_index)
            print(val_index)
            print(test_index)
    return folds


def get_stratified_fold(
    embeddings,
    labels,
    n_splits=10,
    val_ratio=0.1,
    show_index=False,
    shuffle=True,
    random_state=0,
):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    folds = []
    for i, (train_index, test_index) in enumerate(kf.split(embeddings, labels)):
        train_num = len(train_index)
        if val_ratio > 0:
            val_num = max(1, int(val_ratio * train_num))
            val_index = train_index[-val_num:]
            train_index = train_index[:-val_num]
        else:
            # dummy, do not use val
            val_index = train_index[-1:]
        folds.append([train_index, val_index, test_index])
        if show_index:
            print(
                f"fold {i} train: {len(train_index)} val:{len(val_index)} test: {len(test_index)}"
            )
            print(train_index)
            print(val_index)
            print(test_index)
    return folds


def get_stratified_split(y, test_size=0.3, random_state=0, shuffle=True):
    from sklearn.model_selection import train_test_split

    X = np.arange(len(y))
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=y,
    )
    fold = (X_train, X_test)
    return fold


from sklearn.utils import resample
from collections import Counter


def get_stratified_split_bootstrap(
    y, test_size=0.3, random_state=0, shuffle=True, max_num=-1
):
    train_idx, test_idx = get_stratified_split(y, test_size, random_state, shuffle)
    trainy = y[train_idx]

    yset, cate_nums = np.unique(trainy, return_counts=True)
    cate_max_num = max(cate_nums)
    max_num = cate_max_num if max_num < 0 else min(cate_max_num, max_num)

    print(cate_nums)
    print(f"cate_max_num : {cate_max_num}, max_num : {max_num}")
    over_train_idx = []
    for label in yset:
        over_train_idx.extend(
            resample(
                train_idx[trainy == label], n_samples=max_num, random_state=random_state
            )
        )
    print(
        f"before bootstrap : {len(train_idx)}, after bootstrap : {len(over_train_idx)}"
    )
    print(Counter(y[over_train_idx]))
    return over_train_idx, test_idx
