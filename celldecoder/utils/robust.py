import torch
import random
import torch_geometric
import time
import numpy as np


def feature_attack(x):
    noises = torch.randn(x.size())
    x += noises
    x = x.clip(min=0)
    return x


def feature_attack_distribution(x, strength=0.1):
    # noises = torch.randn(x.size())
    mu = x.mean(0).reshape(-1, x.size(1)).expand(x.size(0), x.size(1))
    std = x.std(0).reshape(-1, x.size(1)).expand(x.size(0), x.size(1))
    noises = torch.normal(mu, std)
    x += strength * noises
    x[x < 0] = 0
    return x


def random_attack(
    edge_index, direct=False, p=0.05, ptb_type="both", num_nodes=None, seed=0
):
    """

    Args:
        type: 0,add;1,delete;2,add+delete

    """
    random.seed(seed)
    assert ptb_type in ["both", "add", "delete"]

    if not num_nodes:
        num_nodes = max(max(edge_index[0]), max(edge_index[1]))

    edge_list = edge_index.cpu().numpy().tolist()
    if direct:
        edge_list = [
            (edge_index[0][i], edge_index[1][i]) for i in range(len(edge_list[0]))
        ]
    else:
        edge_list = [
            (edge_index[0][i], edge_index[1][i])
            for i in range(len(edge_list[0]))
            if edge_list[0][i] <= edge_list[1][i]
        ]

    num_edges = len(edge_list) if direct else len(edge_list) // 2
    num_perturbs = int(num_edges * p)

    if ptb_type == "add":
        if direct:
            candidate_edges = [
                (a, b) for a in range(num_nodes) for b in range(num_nodes) if a != b
            ]
        else:
            candidate_edges = [
                (a, b) for a in range(num_nodes) for b in range(num_nodes) if a < b
            ]
        candidate_edges = list(set(candidate_edges) - set(edge_list))
        perturb_edges = set(random.sample(candidate_edges, num_perturbs))
        # perturb_edges = [candidate_edges[i] for i in range(len(candidate_edges)) if i in add_edge_ind]

    elif ptb_type == "delete":
        perturb_edges = random.sample(edge_list, num_perturbs)
        # perturb_edges = [edge_list[i] for i in range(len(edge_list)) if i in delete_edge_ind]

    elif ptb_type == "both":
        if direct:
            candidate_edges = [
                (a, b) for a in range(num_nodes) for b in range(num_nodes) if a != b
            ]
        else:
            candidate_edges = [
                (a, b) for a in range(num_nodes) for b in range(num_nodes) if a < b
            ]

        perturb_edges = random.sample(candidate_edges, num_perturbs)

    print("perturb_edges:", len(perturb_edges), num_perturbs)
    edge_list = set(edge_list)
    for edge in perturb_edges:
        if edge in edge_list:
            edge_list.remove(edge)
        else:
            edge_list.add(edge)
    edge_list = list(edge_list)

    new_edge_index = torch.LongTensor(
        [[edge_list[i][0], edge_list[i][1]] for i in range(len(edge_list))]
    ).T
    if not direct:
        new_edge_index_verse = torch.stack([new_edge_index[1], new_edge_index[0]], 0)
        new_edge_index = torch.cat([new_edge_index, new_edge_index_verse], 1)

    print(new_edge_index.size(), edge_index.size(), num_perturbs)

    return new_edge_index


def random_attack_cross(
    cross_edge_index, p=0.05, ptb_type="both", num_nodes1=None, num_nodes2=None, seed=0
):
    """

    Args:
        type: 0,add;1,delete;2,add+delete

    """
    random.seed(seed)
    assert ptb_type in ["both", "add", "delete"]

    edge_list = cross_edge_index.cpu().numpy().tolist()
    edge_list = [
        (cross_edge_index[0][i], cross_edge_index[1][i])
        for i in range(len(edge_list[0]))
    ]

    num_edges = len(edge_list)
    num_perturbs = int(num_edges * p)

    if ptb_type == "add":
        candidate_edges = [
            (a, b) for a in range(num_nodes1) for b in range(num_nodes2) if a != b
        ]
        candidate_edges = list(set(candidate_edges) - set(edge_list))
        perturb_edges = set(random.sample(candidate_edges, num_perturbs))

    elif ptb_type == "delete":
        perturb_edges = random.sample(edge_list, num_perturbs)
        # perturb_edges = [edge_list[i] for i in range(len(edge_list)) if i in delete_edge_ind]

    elif ptb_type == "both":
        candidate_edges = [(a, b) for a in range(num_nodes1) for b in range(num_nodes2)]
        perturb_edges = random.sample(candidate_edges, num_perturbs)
    # print(f'attacking, num_edges:{num_edges}, num_perturbs:{num_perturbs}')
    edge_list = set(edge_list)
    for edge in perturb_edges:
        if edge in edge_list:
            edge_list.remove(edge)
        else:
            edge_list.add(edge)
    edge_list = list(edge_list)

    new_edge_index = torch.LongTensor(
        [[edge_list[i][0], edge_list[i][1]] for i in range(len(edge_list))]
    ).T

    return new_edge_index


def generate_random_graph(dataset, rand_type="rand"):
    print("===generate random graph===")
    import networkx as nx

    edge_index, inner_edge_indexs, cross_edge_indexs, num_nodes = dataset.metadata

    # ===随机生成inner edge index
    if rand_type == "rand":
        num_edges = int(edge_index.size(1) // 2)
        e_id = np.random.choice(num_nodes[0] * (num_nodes[0] - 1) // 2, num_edges)
        new_edge_index0 = torch.LongTensor(
            [[i, j] for i in range(num_nodes[0]) for j in range(num_nodes[0]) if i < j]
        )[e_id].T
        new_edge_index1 = torch.stack((new_edge_index0[1], new_edge_index0[0]), 0)
        new_edge_index = torch.cat((new_edge_index0, new_edge_index1), 1)
        assert num_edges == int(new_edge_index.size(1) // 2)
        new_inner_edge_indexs = []
        for i, inner_edge_index in enumerate(inner_edge_indexs):
            num_edges = int(inner_edge_index.size(1) // 2)
            if num_edges <= 0:
                new_inner_edge_indexs.append(inner_edge_index)
                continue
            e_id = np.random.choice(
                num_nodes[i + 1] * (num_nodes[i + 1] - 1) // 2, num_edges
            )
            new_inner_edge_index0 = torch.LongTensor(
                [
                    [p, q]
                    for p in range(num_nodes[i + 1])
                    for q in range(num_nodes[i + 1])
                    if p < q
                ]
            )[e_id].T
            new_inner_edge_index1 = torch.stack(
                (new_inner_edge_index0[1], new_inner_edge_index0[0]), 0
            )
            new_inner_edge_index = torch.cat(
                (new_inner_edge_index0, new_inner_edge_index1), 1
            )

            assert num_edges == int(new_inner_edge_index.size(1) // 2)
            new_inner_edge_indexs.append(new_inner_edge_index)

    elif rand_type == "avg_degree_keep":
        _D = torch_geometric.utils.degree(edge_index[0])
        t0 = time.time()
        # nx_graph = nx.random_degree_sequence_graph(list(_D.cpu()), seed=42)
        d_avg = int(_D.mean())
        if d_avg * num_nodes[0] % 2 != 0:
            d_avg -= 1
        nx_graph = nx.random_regular_graph(d_avg, num_nodes[0], seed=42)
        new_edge_index = torch_geometric.utils.from_networkx(nx_graph).edge_index

        new_inner_edge_indexs = []
        for i, inner_edge_index in enumerate(inner_edge_indexs):
            _D = torch_geometric.utils.degree(inner_edge_index[0])
            d_avg = int(_D.mean())
            if d_avg * num_nodes[i + 1] % 2 != 0:
                d_avg -= 1
            nx_graph = nx.random_regular_graph(d_avg, num_nodes[i + 1], seed=42)
            new_inner_edge_index = torch_geometric.utils.from_networkx(
                nx_graph
            ).edge_index
            new_inner_edge_indexs.append(new_inner_edge_index)

    # ===随机生成cross edge index
    new_cross_edge_indexs = []
    for i, cross_edge_index in enumerate(cross_edge_indexs):
        e1, e2 = cross_edge_index
        new_e1 = torch.randint(low=min(e1), high=max(e1) + 1, size=(len(e1),))
        new_e2 = torch.randint(low=min(e2), high=max(e2) + 1, size=(len(e2),))
        new_cross_edge_index = torch.stack((new_e1, new_e2), 0)
        new_cross_edge_indexs.append(new_cross_edge_index)

    dataset.metadata = (
        new_edge_index,
        new_inner_edge_indexs,
        new_cross_edge_indexs,
        num_nodes,
    )

    return dataset


def get_ptb_dataset(
    dataset, ptb_type="randgraph", ptb_rate=None, seed=2022, target_edge="inner"
):
    """

    Args:
        target_edge: for structure attack
    """
    edge_index, inner_edge_indexs, cross_edge_indexs, num_nodes = dataset.metadata
    # attack
    if ptb_type in ["both", "add", "delete"]:
        if ptb_rate > 0:
            if target_edge == "inner":
                new_edge_index = random_attack(
                    edge_index,
                    direct=False,
                    p=ptb_rate,
                    ptb_type=ptb_type,
                    num_nodes=num_nodes[0],
                    seed=seed,
                )
                # new_inner_edge_indexs = []
                # for l,inner_edge_index in enumerate(inner_edge_indexs):
                #     new_inner_edge_index = random_attack(
                #         inner_edge_index, p=ptb_rate, ptb_type=ptb_type,
                #         num_nodes1=num_nodes[l], num_nodes2=num_nodes[l+1], seed=seed)
                #     new_inner_edge_indexs.append(new_inner_edge_index)
                # dataset.metadata = new_edge_index,new_inner_edge_indexs,cross_edge_indexs,num_nodes
                dataset.metadata = (
                    new_edge_index,
                    inner_edge_indexs,
                    cross_edge_indexs,
                    num_nodes,
                )

            elif target_edge == "cross":
                new_cross_edge_indexs = []
                for l, cross_edge_index in enumerate(cross_edge_indexs):
                    new_edge_index = random_attack_cross(
                        cross_edge_index,
                        p=ptb_rate,
                        ptb_type=ptb_type,
                        num_nodes1=num_nodes[l],
                        num_nodes2=num_nodes[l + 1],
                        seed=seed,
                    )
                    new_cross_edge_indexs.append(new_edge_index)
                dataset.metadata = (
                    edge_index,
                    inner_edge_indexs,
                    new_cross_edge_indexs,
                    num_nodes,
                )

    elif ptb_type == "feature":
        new_datas = []
        for d in dataset.datas:
            # d.x = feature_attack(d.x)
            d.x = feature_attack_distribution(d.x, ptb_rate)
            new_datas.append(d)
        dataset.datas = new_datas
    elif ptb_type == "randgraph":
        dataset = generate_random_graph(dataset)

    return dataset
