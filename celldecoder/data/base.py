from torch.utils.data.dataset import Dataset
import torch
from .utils import *
from .process import *
from torch_geometric.data import Data
from torch.nn.functional import one_hot


class MPPIDatasetLight(Dataset):
    def __init__(self, datas):
        super(MPPIDatasetLight, self).__init__()
        self.datas = datas

    def __getitem__(self, idx):
        return self.datas[idx]

    @property
    def num_features(self):
        return self.datas[0].x.shape[1]

    @property
    def num_classes(self):
        return max([d.y for d in self.datas]) + 1

    def __len__(self):
        return len(self.datas)

    def __repr__(self) -> str:
        return f"#{len(self.datas)} like " + str(self.datas[0])


class MPPIDataset(MPPIDatasetLight):
    def __init__(self, datas, cache=True):
        super(MPPIDatasetLight, self).__init__()
        self.datas = datas
        self.splits = [range(len(datas)) for i in range(3)]
        self.is_cache = cache
        self.cache = {}

    def set_split(self, splits):
        self.splits = splits
        # if len(splits) < 3:
        #     print("Warning : split length < 3")

    @property
    def train_dataset(self):
        return MPPIDatasetLight([self.datas[i] for i in self.splits[0]])

    @property
    def val_dataset(self):
        return MPPIDatasetLight([self.datas[i] for i in self.splits[1]])

    @property
    def test_dataset(self):
        return MPPIDatasetLight([self.datas[i] for i in self.splits[2]])

    def construct_cross_edge(
        self, edge_index, batch_size, num_nodes1, num_nodes2, index
    ):
        if self.is_cache:
            cindex = ("c0", batch_size, index)
            if cindex not in self.cache:
                edge_index = construct_cross_edge(
                    edge_index, batch_size, num_nodes1, num_nodes2
                )
                self.cache[cindex] = edge_index
            else:
                edge_index = self.cache[cindex]
        else:
            edge_index = construct_cross_edge(
                edge_index, batch_size, num_nodes1, num_nodes2
            )
        return edge_index

    # @timing
    def construct_cross_edge_both(
        self, edge_index, batch_size, num_nodes1, num_nodes2, index
    ):
        if self.is_cache:
            cindex = ("c1", batch_size, index)
            if cindex not in self.cache:
                edge_index = construct_cross_edge_both(
                    edge_index, batch_size, num_nodes1, num_nodes2
                )
                self.cache[cindex] = edge_index
            else:
                edge_index = self.cache[cindex]
        else:
            edge_index = construct_cross_edge_both(
                edge_index, batch_size, num_nodes1, num_nodes2
            )
        return edge_index

    # @timing
    def construct_inner_edge(self, edge_index, batch_size, num_nodes, index):
        if self.is_cache:
            cindex = ("i", batch_size, index)
            if cindex not in self.cache:
                edge_index = construct_inner_edge(edge_index, batch_size, num_nodes)
                self.cache[cindex] = edge_index
            else:
                edge_index = self.cache[cindex]
        else:
            edge_index = construct_inner_edge(edge_index, batch_size, num_nodes)
        return edge_index


class MPPIDatasetApp(MPPIDataset):
    def __init__(
        self, dataroot, path_dict, fn_process="processed", cache=True, **kwargs
    ):
        """
        path_dict:(fn_feature='features.csv',
                    fn_edges='filter_ppi.txt',
                    fn_label='sampleid.csv',
                    fn_hierarchy='hierarchy_graph.json')
        """
        processed_file = osp.abspath(osp.join(dataroot, fn_process))
        if osp.exists(processed_file):
            print(f"loading {processed_file}")
            data = torch.load(processed_file)
        else:
            print(f"processing to {processed_file}")
            data = load_data(dataroot, **path_dict)
            torch.save(data, processed_file)

        x = data["x"]
        # add_one_hot=kwargs.get('add_one_hot',0)
        # if add_one_hot:
        #     enc=one_hot(torch.arange(x.shape[1])).expand(x.shape[0],x.shape[1],x.shape[1])
        #     x=torch.cat([x,enc],dim=-1)

        y = data["y"]
        edge_index = data["ei"]
        # edge_weight=data['ef']
        inner_edge_index = data["inner_links"]
        cross_edge_index = data["cross_links"]
        num_nodes = [
            len(_) for _ in data["hid2ids"]
        ]  # num of nodes from 1 to last layer
        self.raw_data = data
        metadata = [edge_index, inner_edge_index, cross_edge_index, num_nodes]
        description = data["description"]
        # print(description)
        datas = [Data(x=x[i], y=int(y[i])) for i in range(x.shape[0])]
        super(MPPIDatasetApp, self).__init__(datas, cache=cache)

        self.description = description
        self.metadata = metadata

    @property
    def num_classes(self):
        if "cls2id" not in self.raw_data:
            return max([d.y for d in self.datas]) + 1
        else:
            return len(self.raw_data["cls2id"])
