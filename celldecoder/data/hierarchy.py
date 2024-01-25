import pandas as pd
import numpy as np
import itertools
import networkx as nx
import os
from os.path import join
import re

# knowledge files
reactome_dir = "../data/reactome"  #'../reactome/pathway'
interaction_dir = "../data/string"  # ppi data
relations = join(reactome_dir, "ReactomePathwaysRelation.txt")
pathways = join(reactome_dir, "ReactomePathways.txt")
genes = join(reactome_dir, "ReactomePathways.gmt")


def load_gmt_data(filename, genes_col=1, pathway_col=0):
    data_dict_list = []
    with open(filename) as f:
        data = f.readlines()
        for r in data:
            genes = r.strip().split("\t")
            genes = [re.sub("_copy.*", "", g) for g in genes]
            genes = [re.sub("\\n.*", "", g) for g in genes]
            for gene in genes[genes_col:]:
                pathway = genes[pathway_col]
                dict = {"pathway": pathway, "gene": gene}
                data_dict_list.append(dict)
    df = pd.DataFrame(data_dict_list)
    return df


class hierarchy_layer:
    def __init__(self, species) -> None:
        # load data
        df_pathway = pd.read_csv(pathways, sep="\t")
        df_pathway.columns = ["reactome_id", "pathway_name", "species"]

        df_genes = load_gmt_data(genes, pathway_col=1, genes_col=3)  # genes-pathways

        df_relations = pd.read_csv(relations, sep="\t")
        df_relations.columns = ["child", "parent"]

        self.pathway_names = df_pathway
        self.hierarchy = df_relations
        self.pathway_genes = df_genes

        self.net = self.get_network(species=species)

        return

    def get_network(self, species="HSA"):
        if hasattr(self, "net"):
            return self.net
        # filter species
        hierarchy = self.hierarchy
        filtered_hierarchy = hierarchy[hierarchy["child"].str.contains(species)]
        graph = nx.from_pandas_edgelist(
            filtered_hierarchy, "child", "parent", create_using=nx.DiGraph()
        )
        graph.name = "reactome"
        # root
        roots = [n for n, d in graph.in_degree() if d == 0]
        root_node = "root"
        edges = []
        for n in roots:
            edges.append((root_node, n))
        graph.add_edges_from(edges)

        return graph

    def info(self):
        return nx.info(self.net)

    def get_sub_network(self, n_levels=4):
        sub_graph = nx.ego_graph(self.net, "root", radius=n_levels)
        nodes = [n for n, d in sub_graph.out_degree() if d == 0]  # terminal
        for node in nodes:
            distance = len(nx.shortest_path(sub_graph, source="root", target=node))
            if distance <= n_levels:
                d = n_levels - distance + 1
                sub_graph = add_edges(sub_graph, node, d)
        return sub_graph

    def get_layers(self, n_levels, direction="top"):
        if direction == "top":
            net = self.get_sub_network(n_levels)
            layers = get_layers_from_net(net, n_levels)
        else:
            net = self.get_sub_network(5)
            layers = get_layers_from_net(net, 5)
            layers = layers[5 - n_levels : 5]
        terminal_nodes = [n for n, d in net.out_degree() if d == 0]
        genes_df = self.pathway_genes

        dict = {}
        missing_path = []
        for p in terminal_nodes:
            pathway_name = re.sub("_copy.*", "", p)
            genes = genes_df[genes_df["pathway"] == pathway_name]["gene"].unique()
            if len(genes) == 0:
                missing_path.append(pathway_name)
            dict[pathway_name] = genes.tolist()

        layers.append(dict)
        return layers


def add_edges(G, node, n_levels):
    edges = []
    source = node
    for l in range(n_levels):
        target = node + "_copy" + str(l + 1)
        edge = (source, target)
        source = target
        edges.append(edge)

    G.add_edges_from(edges)
    return G


def get_layers_from_net(net, n_levels):
    layers = []
    for i in range(n_levels):
        nodes = get_nodes(net, i)
        layer_dict = {}
        for n in nodes:
            n_name = re.sub("_copy.*", "", n)
            next = net.successors(n)
            layer_dict[n_name] = [re.sub("_copy.*", "", nex) for nex in next]
        layers.append(layer_dict)
    return layers


def get_nodes(G, r):
    nodes = set(nx.ego_graph(G, "root", radius=r))
    if r >= 1:
        nodes -= set(nx.ego_graph(G, "root", radius=r - 1))
    return list(nodes)
