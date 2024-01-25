from .base import *
from .process import *
import pandas as pd
import os.path as osp


def make_dict(**kwargs):
    return dict(**kwargs)


class TumourDataset(MPPIDatasetApp):
    def __init__(self, dataroot, hierarchy=3, cache=True, **kwargs):
        assert hierarchy == 3 or hierarchy == 5
        path_dict = make_dict(
            fn_feature="features.csv",
            fn_edges="filter_ppi.txt",
            fn_label="sampleid.csv",
            fn_hierarchy="hierarchy_graph.json"
            if hierarchy == 3
            else "hierarchy_graph5layer.json",
        )
        super(TumourDataset, self).__init__(
            dataroot,
            path_dict=path_dict,
            fn_process=f"processed-{hierarchy}",
            cache=cache,
            **kwargs,
        )


class KIPANDataset(MPPIDatasetApp):
    def __init__(self, dataroot, hierarchy=3, cache=True, **kwargs):
        assert hierarchy == 3 or hierarchy == 5
        path_dict = make_dict(
            fn_feature="KIPANfpkm.csv",
            fn_edges="KIPANPPI.csv",
            fn_label="KIPANlabel.csv",
            fn_hierarchy="hierarchy_graph.json"
            if hierarchy == 3
            else "hierarchy_graph5layer.json",
        )
        super(KIPANDataset, self).__init__(
            dataroot,
            path_dict=path_dict,
            fn_process=f"processed-{hierarchy}",
            cache=cache,
            **kwargs,
        )


class HCCDataset(MPPIDatasetApp):
    def __init__(self, dataroot, hierarchy=3, cache=True, **kwargs):
        assert hierarchy == 3 or hierarchy == 5
        path_dict = make_dict(
            fn_feature="Cellhcc_data_filter.csv",
            fn_edges="HCCPPI.csv",
            fn_label="cellhcc_filtered_labels.csv",
            fn_hierarchy="hierarchy_graph.json"
            if hierarchy == 3
            else "hierarchy_graph5layer.json",
        )
        super(HCCDataset, self).__init__(
            dataroot,
            path_dict=path_dict,
            fn_process=f"processed-{hierarchy}",
            cache=cache,
            **kwargs,
        )


class PRCADataset(MPPIDatasetApp):
    def __init__(self, dataroot, hierarchy=3, cache=True, **kwargs):
        assert hierarchy == 3 or hierarchy == 5
        path_dict = make_dict(
            fn_feature="PRCAfpkm.csv",
            fn_edges="PRCAPPI.csv",
            fn_label="PRCAlabel.csv",
            fn_hierarchy="hierarchy_graph.json"
            if hierarchy == 3
            else "hierarchy_graph5layer.json",
        )
        super(PRCADataset, self).__init__(
            dataroot,
            path_dict=path_dict,
            fn_process=f"processed-{hierarchy}",
            cache=cache,
            **kwargs,
        )


class BRCADataset(MPPIDatasetApp):
    def __init__(self, dataroot, hierarchy=3, cache=True, **kwargs):
        assert hierarchy == 3 or hierarchy == 5
        path_dict = make_dict(
            fn_feature="BRCAfpkm_filter.csv",
            fn_edges="BRCAPPI_filter.csv",
            fn_label="BRCAlabel.csv",
            fn_hierarchy="hierarchy_graph.json"
            if hierarchy == 3
            else "hierarchy_graph5layer.json",
        )
        super(BRCADataset, self).__init__(
            dataroot,
            path_dict=path_dict,
            fn_process=f"processed-{hierarchy}",
            cache=cache,
            **kwargs,
        )


class LungDataset(MPPIDatasetApp):
    def __init__(self, dataroot, hierarchy=3, cache=True, **kwargs):
        assert hierarchy == 3 or hierarchy == 5
        path_dict = make_dict(
            fn_feature="lung_sc_data.csv",
            fn_edges="lung_sc_ppi.csv",
            fn_label="lung_sc_labels.csv",
            fn_hierarchy="hierarchy_graph_layer3.json"
            if hierarchy == 3
            else "hierarchy_graph_layer5.json",
        )
        super(LungDataset, self).__init__(
            dataroot,
            path_dict=path_dict,
            fn_process=f"processed-{hierarchy}",
            cache=cache,
            **kwargs,
        )
