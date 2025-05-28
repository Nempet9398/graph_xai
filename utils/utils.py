import synthetic_data.featgen as featgen
import synthetic_data.gengraph as gengraph
import os
import random
from torch_geometric.utils import from_networkx
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils.convert import to_networkx
from sklearn.model_selection import StratifiedKFold
import pickle

from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt



def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch_geometric.seed_everything(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)
    
class GeneralizedCELoss(torch.nn.Module):

    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q
             
    def forward(self, logits, targets):
        p = F.sigmoid(logits)
        p = torch.cat([p, 1-p], dim=1)
        if np.isnan(p.mean().item()):
            raise NameError('GCE_p')
        Yg = torch.gather(p, 1, targets.to(torch.long))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach()**self.q)*self.q
        if np.isnan(Yg.mean().item()):
            raise NameError('GCE_Yg')

        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none') * loss_weight
        return loss


# Helper function to save/load importance
def load_or_compute_importance(file_name, compute_func, importance_save_path,test=False):
    file_path = os.path.join(importance_save_path, file_name)
    if os.path.exists(file_path) and not test:
        print(f"Loading {file_name} from {file_path}")
        with open(file_path, 'rb') as f:
            importance = pickle.load(f)
    else:
        importance = compute_func()
        with open(file_path, 'wb') as f:
            pickle.dump(importance, f)
        print(f"Saved {file_name} to {file_path}")
    return importance


from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.metrics import average_precision_score

def compute_auc_list(importance_list, dataset):
    auc_list = []
    for imp, data in zip(importance_list, dataset):
        if hasattr(data, 'ground_truth'):
            y_true = data.ground_truth.detach().cpu().numpy()
            y_score = imp.detach().cpu().numpy()
            # Min-max normalization
            if y_score.max() > y_score.min():
                y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())
            if (y_true.sum() == 0) or (y_true.sum() == len(y_true)):
                auc_list.append(float('nan'))
            else:
                auc_list.append(roc_auc_score(y_true, y_score))
    return auc_list

def compute_pr_auc_list(importance_list, dataset):
    pr_auc_list = []
    for imp, data in zip(importance_list, dataset):
        if hasattr(data, 'ground_truth'):
            y_true = data.ground_truth.detach().cpu().numpy()
            y_score = imp.detach().cpu().numpy()
            # Min-max normalization
            if y_score.max() > y_score.min():
                y_score = (y_score - y_score.min()) / (y_score.max() - y_score.min())
            if (y_true.sum() == 0) or (y_true.sum() == len(y_true)):
                pr_auc_list.append(float('nan'))
            else:
                pr_auc_list.append(average_precision_score(y_true, y_score))
    return pr_auc_list


def creat_one_pyg_graph(context, shape, label, feature_dim, shape_num, settings_dict, args=None):
    if args is None:
        noise = 0
    else:
        noise = args.noise
    if feature_dim == -1:
        # use degree as feature
        feature = featgen.ConstFeatureGen(None, max_degree=args.max_degree)
    else:
        feature = featgen.ConstFeatureGen(np.random.uniform(0, 1, feature_dim))
    G, node_label = gengraph.generate_graph(basis_type=context,
                                            shape=shape,
                                            nb_shapes=shape_num,
                                            width_basis=settings_dict[context]["width_basis"],
                                            feature_generator=feature,
                                            m=settings_dict[context]["m"],
                                            random_edges=noise) 
    pyg_G = from_networkx(G)
    pyg_G.y = torch.tensor([label])
    return pyg_G, node_label

def graph_dataset_generate(args, save_path):

    class_list = ["house", "cycle", "grid", "diamond"]
    settings_dict = {"ba": {"width_basis": args.node_num ** 2, "m": 2},
                     "tree": {"width_basis":2, "m": args.node_num}}

    feature_dim = args.feature_dim
    shape_num = args.shape_num
    # class_num = class_list.__len__()
    dataset = {}
    dataset['tree'] = {}
    dataset['ba'] = {}

    for label, shape in enumerate(class_list):
        tr_list = []
        ba_list = []
        print("create shape:{}".format(shape))
        for i in tqdm(range(args.data_num)):
            tr_g, label1 = creat_one_pyg_graph(context="tree", shape=shape, label=label, feature_dim=feature_dim, 
                                               shape_num=shape_num, settings_dict=settings_dict, args=args)
            ba_g, label2 = creat_one_pyg_graph(context="ba", shape=shape, label=label, feature_dim=feature_dim, 
                                               shape_num=shape_num, settings_dict=settings_dict, args=args)
            tr_list.append(tr_g)
            ba_list.append(ba_g)
        dataset['tree'][shape] = tr_list
        dataset['ba'][shape] = ba_list

    save_path += f"/syn_dataset{args.node_num}.pt"
    torch.save(dataset, save_path)
    print("save at:{}".format(save_path))
    return dataset



def dataset_bias_split(dataset, class_list, bias=None, split=None, total=20000):
    
    bias_dict = {}
    for i, shape in enumerate(class_list):
        bias_dict[shape] = bias if (i == 0) else 1 - bias
    
    ba_dataset = dataset['ba']
    tr_dataset = dataset['tree']
    
    train_split, val_split, test_split = float(split[0]) / 10, float(split[1]) / 10, float(split[2]) / 10
    assert train_split + val_split + test_split == 1
    train_num, val_num, test_num = total * train_split, total * val_split, total * test_split
    # blance class
    class_num = len(class_list)
    train_class_num, val_class_num, test_class_num = train_num / class_num, val_num / class_num, test_num / class_num
    train_list, val_list, test_list  = [], [], []
    edges_num = 0
    
    for shape in class_list:
        bias = bias_dict[shape]
        train_tr_num = int(train_class_num * bias)
        train_ba_num = int(train_class_num * (1 - bias))
        val_tr_num = int(val_class_num * bias)
        val_ba_num = int(val_class_num * (1 - bias))
        test_tr_num = int(test_class_num * 0.5)
        test_ba_num = int(test_class_num * 0.5)
        train_list += tr_dataset[shape][:train_tr_num] + ba_dataset[shape][:train_ba_num]
        val_list += tr_dataset[shape][train_tr_num:train_tr_num + val_tr_num] + ba_dataset[shape][train_ba_num:train_ba_num + val_ba_num]
        test_list += tr_dataset[shape][train_tr_num + val_tr_num:train_tr_num + val_tr_num + test_tr_num] + ba_dataset[shape][train_ba_num + val_ba_num:train_ba_num + val_ba_num + test_ba_num]
        _, e1 = print_graph_info(tr_dataset[shape][0], "Tree", shape)
        _, e2 = print_graph_info(ba_dataset[shape][0], "BA", shape)
        
        edges_num += e1 + e2
    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)
    the = float(edges_num) / (class_num * 2)
    return train_list, val_list, test_list, the

def print_graph_info(G, c, o):
    print('-' * 100)
    print("| graph: {}-{} | nodes num:{} | edges num:{} |".format(c, o, G.num_nodes, G.num_edges))
    print('-' * 100)
    return G.num_nodes, G.num_edges

def print_dataset_info(train_set, val_set, test_set, the):
    class_list = ["house", "cycle", "grid", "diamond"]
    dataset_group_dict = {}
    dataset_group_dict["Train"] = dataset_context_object_info(train_set, "Train", class_list, the)
    dataset_group_dict["Val"] = dataset_context_object_info(val_set, "Val   ", class_list, the)
    dataset_group_dict["Test"] = dataset_context_object_info(test_set, "Test  ", class_list, the)
    return dataset_group_dict

def dataset_context_object_info(dataset, title, class_list, the):

    class_num = len(class_list)
    tr_list = [0] * class_num
    ba_list = [0] * class_num
    for g in dataset:
        if g.num_edges > the: # ba
            ba_list[g.y.item()] += 1
        else: # tree
            tr_list[g.y.item()] += 1
    total = sum(tr_list) + sum(ba_list)
    info = "{} Total:{}\n| Tree: House:{:<5d}, Cycle:{:<5d}, Grids:{:<5d}, Diams:{:<5d} \n" +\
                        "| BA  : House:{:<5d}, Cycle:{:<5d}, Grids:{:<5d}, Diams:{:<5d} \n" +\
                        "| All : House:{:<5d}, Cycle:{:<5d}, Grids:{:<5d}, Diams:{:<5d} \n" +\
                        "| BIAS: House:{:.1f}%, Cycle:{:.1f}%, Grids:{:.1f}%, Diams:{:.1f}%"
    print("-" * 150)
    print(info.format(title, total, tr_list[0], tr_list[1], tr_list[2], tr_list[3],
                                    ba_list[0], ba_list[1], ba_list[2], ba_list[3],
                                    tr_list[0] +  ba_list[0],    
                                    tr_list[1] +  ba_list[1], 
                                    tr_list[2] +  ba_list[2], 
                                    tr_list[3] +  ba_list[3],
                                    100 *float(tr_list[0]) / (tr_list[0] +  ba_list[0]),
                                    100 *float(tr_list[1]) / (tr_list[1] +  ba_list[1]),
                                    100 *float(tr_list[2]) / (tr_list[2] +  ba_list[2]),
                                    100 *float(tr_list[3]) / (tr_list[3] +  ba_list[3]),
                     ))
    print("-" * 150)
    total_list = ba_list + tr_list
    group_counts = torch.tensor(total_list).float()
    return group_counts




# Visualize the dataset with heatmap
def visualize_data_with_heatmap(data, heatmap):

    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(10, 10))
    pos = nx.kamada_kawai_layout(G)
    
    # Normalize heatmap values for coloring
    heatmap = heatmap.cpu().detach().numpy()
    node_colors = [heatmap[i] for i in range(len(G.nodes))]
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=1000, font_size=10, cmap=plt.cm.BuPu)
    plt.title("Graph Visualization with Heatmap")
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.BuPu), ax=plt.gca(), label='Heatmap Intensity')
    plt.show()



def add_fragment_labels_to_data(data):
    """
    Takes a molecular graph `data`, decomposes it using BRICS, 
    and assigns each node (atom) a fragment label.

    Parameters:
    - data: A PyG Data object with a `.smiles` attribute.

    Returns:
    - data: Updated PyG Data object with `data.fragment_labels` added.
    """
    smiles = data.smiles  
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES string")


    fragments = BRICS.BreakBRICSBonds(mol)
    frag_mols = Chem.GetMolFrags(fragments, asMols=True)


    fragment_labels = torch.full((data.num_nodes,), -1, dtype=torch.long)

    fragment_id_map = {}  # {fragment_SMILES: fragment_ID}
    fragment_counter = 0
    assigned_atoms = set()

    for frag in frag_mols:
        frag_smiles = Chem.MolToSmiles(frag)
        frag_atoms = [atom.GetIdx() for atom in frag.GetAtoms()]

        if frag_smiles not in fragment_id_map:
            fragment_id_map[frag_smiles] = fragment_counter
            fragment_counter += 1

        frag_id = fragment_id_map[frag_smiles]

        for atom_idx in frag_atoms:
            if atom_idx < data.num_nodes:
                fragment_labels[atom_idx] = frag_id
                assigned_atoms.add(atom_idx)

    for atom_idx in range(data.num_nodes):
        if atom_idx not in assigned_atoms:
            fragment_labels[atom_idx] = fragment_counter
            fragment_counter += 1

    data.fragment_labels = fragment_labels
    return data


