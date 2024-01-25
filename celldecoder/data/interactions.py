import pandas as pd
import mygene
import numpy as np
import scanpy as sc

# knowledge files
reactome_dir = "./data/reactome"  #'../reactome/pathway'
interaction_dir = "./data/string"  # ppi data

string_ppi_path = "10090.protein.links.v11.5.txt"  # mus ppi


def get_gene_symbols_from_protein(ensembl_ids, species):
    mg = mygene.MyGeneInfo()
    res = mg.querymany(
        ensembl_ids,
        scopes="ensembl.protein",
        fields="symbol",
        species=species,
        returnall=True,
    )

    def get_symbol_and_ensembl(d):
        if "symbol" in d:
            return [d["query"], d["symbol"]]
        else:
            return [d["query"], None]

    node_names = [get_symbol_and_ensembl(d) for d in res["out"]]
    # now, retrieve the names and IDs from a dictionary and put in DF
    node_names = pd.DataFrame(node_names, columns=["Ensembl_ID", "Symbol"]).set_index(
        "Ensembl_ID"
    )
    node_names.dropna(axis=0, inplace=True)
    return node_names


def mapping_ens_to_symbol(ppi: pd.DataFrame, species) -> pd.DataFrame:
    """
    @description  : mapping ens_id to symbol id
    ---------
    @param  :ppi:  protein1, protein2,score
    -------
    @Returns  : ppi(symbol id)
    -------
    """
    ens_names = ppi.iloc[:, 0].append(ppi.iloc[:, 1]).unique()
    ens2symbol = get_gene_symbols_from_protein(ens_names, species)
    # join to mapping ens to symbol
    ppi.index = ppi.iloc[:, 0]
    p1_incl = ppi.join(ens2symbol, how="inner", rsuffix="_p1")
    p1_incl.index = p1_incl.iloc[:, 1]
    both_incl = p1_incl.join(ens2symbol, how="inner", rsuffix="_p2")
    both_incl = both_incl.reset_index()
    both_incl = both_incl.iloc[:, 3:]
    both_incl.columns = ["confidence", "partner1", "partner2"]
    ppi_final = both_incl[["partner1", "partner2", "confidence"]]

    return ppi_final


def data_mapping_ppi(data, ppi: pd.DataFrame, top_genes=3000):
    """
    @description  : data features mapping to ppi network
    ---------
    @param  : data: adata.var_names
              ppi:  protein1, protein2
              top_genes: the number of hvg
    -------
    @Returns  : Anndata
    -------
    """
    protein_names = np.unique(ppi.iloc[:, 0])
    merge_feature = set(protein_names).intersection(set(data.var_names))
    data = data[:, list(merge_feature)]
    if 'highly_variable' not in data.var:
        sc.pp.highly_variable_genes(data, n_top_genes=top_genes)
        data = data[:, data.var.highly_variable]
    ppi = ppi[ppi.iloc[:, 0].isin(list(data.var_names))]
    ppi = ppi[ppi.iloc[:, 1].isin(list(data.var_names))]
    data.uns["ppi"] = ppi

    return data
