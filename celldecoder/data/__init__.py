from .PPI import *
import os
from ..utils import get_stratified_split_bootstrap, get_stratified_split

CURDIR = os.path.abspath(os.path.dirname(__file__))
DATADIR = os.path.join(CURDIR, "../../data/")


class BenchLoader:
    def __init__(self):
        self.bench_data = "hAdipose hBlood hBone hBRCA hBreast hEye hKidney hLiver hNose hPancreas mAtlas mBone mBrain mHeart mIntestine mKidney mLiver mLung mPancreas mSpleen".split()
        self.data_names = self.bench_data + [x + "-test" for x in self.bench_data]

    def load_data(self, name):
        if "test" in name:
            name = name.split("-")[0]
            dataroot = os.path.join(DATADIR, f"{name}")
            cls2id = MPPIDatasetApp(
                dataroot,
                path_dict={"fn_h5ad": f"{name}_ref_adata.h5ad"},
                fn_process=f"processed-train",
                cache=True,
            ).raw_data["cls2id"]
            dataset = MPPIDatasetApp(
                dataroot,
                path_dict={"fn_h5ad": f"{name}_query_adata.h5ad", "cls2id": cls2id},
                fn_process=f"processed-test",
                cache=True,
            )
        else:
            dataroot = os.path.join(DATADIR, f"{name}")
            dataset = MPPIDatasetApp(
                dataroot,
                path_dict={"fn_h5ad": f"{name}_ref_adata.h5ad"},
                fn_process=f"processed-train",
                cache=True,
            )
        return dataset

def load_train_split(args, dataset, y, random_state=0):
    if args.bootstrap_num > 0:
        fold = get_stratified_split_bootstrap(
            y,
            test_size=args.val_ratio,
            random_state=random_state,
            shuffle=True,
            max_num=args.bootstrap_num,
        )
    elif args.bootstrap_num == -2:
        fold = get_stratified_split_bootstrap(
            y, test_size=args.val_ratio, random_state=random_state, shuffle=True, max_num=-1
        )
    else:
        fold = get_stratified_split(
            y, test_size=args.val_ratio, random_state=random_state, shuffle=True
        )
    dataset.set_split(fold)
    return dataset

def load_data_path(path, fn_process = 'processed-train'):
    paths = os.path.split(path)
    file = paths[-1]
    dir = os.path.join(*paths[:-1])
    dataset = MPPIDatasetApp(
        dir,
        path_dict={"fn_h5ad": file},
        fn_process=fn_process,
        cache=True,
    )
    return dataset

def load_data(args):
    benchloader = BenchLoader()
    if args.dataset == "Tumor":
        dataroot = os.path.join(DATADIR, "PPI")
        dataset = TumourDataset(dataroot=dataroot, hierarchy=args.hierarchy)
    elif args.dataset == "KIPAN":
        dataroot = os.path.join(DATADIR, "KIPAN")
        dataset = KIPANDataset(dataroot=dataroot, hierarchy=args.hierarchy)
    elif args.dataset == "PRCA":
        dataroot = os.path.join(DATADIR, "PRCA")
        dataset = PRCADataset(dataroot=dataroot, hierarchy=args.hierarchy)
    elif args.dataset == "HCC":
        dataroot = os.path.join(DATADIR, "CELL_HCC")
        dataset = HCCDataset(dataroot=dataroot, hierarchy=args.hierarchy)
    elif args.dataset == "BRCA":
        dataroot = os.path.join(DATADIR, "TCGA_BRCA")
        dataset = BRCADataset(dataroot=dataroot, hierarchy=args.hierarchy)
    elif args.dataset == "lung":
        dataroot = os.path.join(DATADIR, "lung_sc")
        dataset = LungDataset(dataroot=dataroot, hierarchy=args.hierarchy)
    elif args.dataset in benchloader.data_names:
        dataset = benchloader.load_data(args.dataset)
    else:
        dataset = load_data_path(args.dataset, args.fn_process)
        # raise NotImplementedError(f" {dataset} Dataset not implemented.")
        
    args.num_features = dataset.num_features
    args.num_classes = dataset.num_classes
    args.num_samples = len(dataset)
    x = [data.x for data in dataset.datas]
    y = [data.y for data in dataset.datas]
    x = torch.stack(x).squeeze(-1).numpy()  # (511, 1138)
    y = np.array(y)  # (511,)
    num_nodes = x.shape[1]
    args.num_nodes = num_nodes
    info = {"x": x, "y": y}
    return dataset, args, info
