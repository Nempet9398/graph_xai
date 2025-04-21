import torch
import random
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset

def generate_house_motif(start_idx=0):
    edges = [(0,1), (1,2), (2,3), (3,0), (0,4), (2,4)]
    edges = [(u+start_idx, v+start_idx) for u, v in edges]
    return edges, list(range(start_idx, start_idx + 5))

def generate_cycle5_motif(start_idx=0):
    edges = [(0,1), (1,2), (2,3), (3,4), (4,0)]
    edges = [(u+start_idx, v+start_idx) for u, v in edges]
    return edges, list(range(start_idx, start_idx + 5))

def create_ba_graph_with_motif(motif_type: str):
    n = 20
    m = 2
    G = nx.barabasi_albert_graph(n=n, m=m)
    node_offset = G.number_of_nodes()

    if motif_type == 'house':
        motif_edges, motif_nodes = generate_house_motif(start_idx=node_offset)
        label = 0
    elif motif_type == 'cycle':
        motif_edges, motif_nodes = generate_cycle5_motif(start_idx=node_offset)
        label = 1
    else:
        raise ValueError("Motif type must be 'house' or 'cycle'.")

    G.add_nodes_from(motif_nodes)
    G.add_edges_from(motif_edges)

    # motif를 BA 그래프에 연결
    attach_node = random.choice(range(node_offset))
    G.add_edge(attach_node, motif_nodes[0])

    # PyG 변환
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)

    x = torch.randn(G.number_of_nodes(), 10)  # 랜덤 피처
    y = torch.tensor([label], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)

def get_ba2motif_2class(num_graphs=800, seed=0):
    random.seed(seed)
    torch.manual_seed(seed)

    graph_list = []

    for _ in range(num_graphs // 2):
        g0 = create_ba_graph_with_motif('house')   # label 0
        g1 = create_ba_graph_with_motif('cycle')   # label 1
        graph_list.append(g0)
        graph_list.append(g1)

    return graph_list



def get_protein_dataset(root='./data', seed=42):
    """
    Loads the PROTEINS dataset and returns a shuffled list of graphs.

    Args:
        root (str): Path to download/store the dataset.
        seed (int): Random seed for shuffling.

    Returns:
        List of Data objects (graph list)
    """
    dataset = TUDataset(root=root, name='PROTEINS')
    dataset = dataset.shuffle()
    return list(dataset)


def get_imdb_binary_dataset(root='./data', max_degree=10):
    """
    Loads the IMDB-BINARY dataset and assigns degree-based node features.
    
    Args:
        root (str): Path to store the dataset.
        max_degree (int): Max degree for one-hot encoding (default 10).
    
    Returns:
        dataset (List[Data]): List of processed PyG Data objects with x.
    """
    dataset = TUDataset(root=root, name='IMDB-BINARY')
    
    for data in dataset:
        deg = degree(data.edge_index[0], data.num_nodes, dtype=torch.long)
        deg = deg.clamp(max=max_degree - 1)  # truncate to prevent oversize
        data.x = F.one_hot(deg, num_classes=max_degree).float()
    
    return list(dataset)