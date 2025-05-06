
import tempfile
import pandas as pd
import itertools
import torch
import random
from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_smiles

from ogb.graphproppred import PygGraphPropPredDataset

from rdkit import Chem
from rdkit.Chem import AllChem, FragmentMatcher,rdmolops
from rdkit import RDLogger

from graphxai.datasets import AlkaneCarbonyl,FluorideCarbonyl,Benzene


from torch_geometric.datasets import UPFD,BA2MotifDataset, BAMultiShapesDataset
from torch_geometric.transforms import ToUndirected


from tdc.single_pred import Tox
from tqdm import tqdm
import sys
import os
# 프로젝트 루트 경로를 path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.chemutils import brics_decomp,recap_decomp,get_mol, tree_decomp, tree_decomp_no_overlap,tree_decomp_no_overlap_clean
from torch.nn.functional import one_hot

RDLogger.DisableLog('rdApp.*')
#%%
def get_random_split(length, mutag = False):
    split_idx = {}
    if mutag:
        test_ratio = round(length * 0.2)
        valid_ratio = round(length * 0.1)
    
    else:
        test_ratio = round(length * 0.1)
        valid_ratio = round(length * 0.1)
    train_idx = torch.randperm(length)
    split_idx["test"] = train_idx[:test_ratio]
    split_idx["valid"] = train_idx[test_ratio:test_ratio + valid_ratio]
    split_idx["train"] = train_idx[test_ratio + valid_ratio:]
    return split_idx
#%%
class MoleculeDataset(InMemoryDataset):
    def __init__(self, root, data_list, transform=None):
        self.data_list = data_list
        self._temp_dir = tempfile.TemporaryDirectory()
        super().__init__(self._temp_dir.name, transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'data.pt'

    def process(self):
        self.save(self.data_list, self.processed_paths[0])



def get_UPFD_GossipCop():
    dataset = UPFD('./dataset/FacebookPagePage', 'gossipcop', 'profile', 'train', ToUndirected())
    split_idx = get_random_split(len(dataset))
    return dataset, split_idx

def get_TUD(dataset_name):
    raw_dataset = TUDataset(root='./dataset', name=dataset_name)
    split_idx = get_random_split(len(raw_dataset))

    # 전체 그래프 중 최대 degree 값을 먼저 계산
    max_degree = 0
    for data in raw_dataset:
        degrees = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
        if degrees.numel() > 0:
            max_degree = max(max_degree, degrees.max().item())

    datalist = []
    for data in raw_dataset:
        if not hasattr(data, 'x') or data.x is None:
            degrees = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
            data.x = one_hot(degrees, num_classes=max_degree + 1).float()
        datalist.append(data)

    return MoleculeDataset('./', datalist), split_idx


def get_syndata(type):
    if type == 'motif':
        dataset = BA2MotifDataset(root = './dataset')
    else:
        dataset = BAMultiShapesDataset(root = './dataset')

    # Set data.x as one-hot degree vectors for each graph in the dataset
    max_degree = 0
    for data in dataset:
        degrees = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
        if degrees.numel() > 0:
            max_degree = max(max_degree, degrees.max().item())

    datalist = []
    for data in dataset:
        degrees = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
        data.x = one_hot(degrees, num_classes=max_degree + 1).float()
        datalist.append(data)

    dataset = MoleculeDataset('./', datalist)

    split_idx = get_random_split(len(dataset))

    return dataset, split_idx



def get_ames():
        # 1. 데이터 로드
    saved_dataset_path = './dataset/saved_ames_dataset.pt'
    if os.path.exists(saved_dataset_path):
        data = torch.load(saved_dataset_path)
    else:
        data = Tox(name='AMES')
        torch.save(data, saved_dataset_path)
    df = data.get_data().iloc[:2000]
    split = data.get_split()
    datalist = []
    for _, row in tqdm(df.iterrows() ,total = len(df)):#, total=2000): #min(len(df), 2000)):
        smiles = row['Drug']
        label = row['Y']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        # GNN input용 그래프
        data = from_smiles(smiles)
        data.smiles = smiles  # <- zinc_ids처럼 사용 가능
        data.y = torch.tensor([label], dtype=torch.long)
        data.ground_truth = None  # Ames에는 GT explanation 없음
        
        # motif (clique) decomposition
        clique, edges = brics_decomp(mol)
        if len(edges) == 0:
            clique, edges = tree_decomp_no_overlap_clean(mol)

        data.clique = clique
        data.num_cliques = len(clique)
        datalist.append(data)
    # GraphXAI용 MoleculeDataset으로 래핑
    return MoleculeDataset('./', datalist), get_random_split(len(datalist)) # split
# dataset, index = get_ames()

def get_alkane():
    dataset_path = './GraphXAI/graphxai/datasets/real_world/alkane_carbonyl/alkane_carbonyl.npz'
    saved_dataset_path = './GraphXAI/graphxai/datasets/real_world/alkane_carbonyl/saved_alkane_carbonyl.pt'

    if os.path.exists(saved_dataset_path):
        dataset = torch.load(saved_dataset_path)
    else:
        dataset = AlkaneCarbonyl(split_sizes=(1.0, 0, 0), seed=42, data_path=dataset_path)
        torch.save(dataset, saved_dataset_path)


    data_list, explanation = dataset.get_data_list([i for i in range(len(dataset))])
    datalist = []
    for idx, data in enumerate(data_list):
        ground_truth = data.node_imp.clone()
        mol = get_mol(data.zinc_ids)
        y = data.y.squeeze()
        data = from_smiles(data.zinc_ids)
        data.y = y
        data.ground_truth = ground_truth


        if mol is None: continue
        clique, edges = brics_decomp(mol)
        if len(edges)==0:
            clique, edges = tree_decomp_no_overlap_clean(mol)

        data.clique = clique

        
        data.num_cliques = len(clique)
        
        datalist.append(data)
    return MoleculeDataset('./', datalist), get_random_split(len(datalist))


def get_fluoride():
    dataset_path = './GraphXAI/graphxai/datasets/real_world/fluoride_carbonyl/fluoride_carbonyl.npz'
    saved_dataset_path = './GraphXAI/graphxai/datasets/real_world/fluoride_carbonyl/saved_fluoride_carbonyl.pt'

    if os.path.exists(saved_dataset_path):
        dataset = torch.load(saved_dataset_path)
    else:
        dataset = FluorideCarbonyl(split_sizes=(1.0, 0, 0), seed=42, data_path=dataset_path)
        torch.save(dataset, saved_dataset_path)

    data_list, explanation = dataset.get_data_list([i for i in range(len(dataset))])
    datalist = []
    for idx, data in enumerate(data_list[:2000]):
        ground_truth = data.node_imp.clone()
        mol = get_mol(data.zinc_ids)
        y = data.y.squeeze()
        data = from_smiles(data.zinc_ids)
        data.y = y
        data.ground_truth = ground_truth


        if mol is None: continue
        clique, edges = brics_decomp(mol)
        if len(edges)==0:
            clique, edges = tree_decomp_no_overlap_clean(mol)

        data.clique = clique

        
        data.num_cliques = len(clique)
        
        datalist.append(data)

    return MoleculeDataset('./', datalist), get_random_split(len(datalist))


def get_benzene():

    dataset_path = './GraphXAI/graphxai/datasets/real_world/benzene/benzene.npz'

    dataset = Benzene(split_sizes=(1.0,0,0), seed=42, data_path=dataset_path)
    data_list, explanation = dataset.get_data_list([i for i in range(len(dataset))])
    

    datalist = []
    for idx, data in enumerate(data_list[:2000]):
        ground_truth = data.node_imp.clone()
        mol = get_mol(data.zinc_ids)
        y = data.y.squeeze()
        data = from_smiles(data.zinc_ids)
        data.y = y
        data.ground_truth = ground_truth


        if mol is None: continue
        clique, edges = brics_decomp(mol)
        if len(edges)==0:
            clique, edges = tree_decomp_no_overlap_clean(mol)

        data.clique = clique

        
        data.num_cliques = len(clique)
        
        datalist.append(data)

    return MoleculeDataset('./', datalist), get_random_split(len(datalist))





#%%

# def get_random_split(length, dataset):
#     split_idx = dataset.get_idx_split()
#     return {
#         "train": split_idx["train"],
#         "valid": split_idx["valid"],
#         "test": split_idx["test"]
#     }


def get_MUTAG_smiles():
    node_label_map = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
    edge_label_map = {0: Chem.BondType.AROMATIC, 1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    dataset = TUDataset(root='./dataset', name='MUTAG')
    smiles_list = []
    for idx, data in enumerate(dataset):

        edge_index = data.edge_index
        edge_attr = data.edge_attr
        node_labels = data.x.squeeze().tolist()

        if (idx in [82,187]): edge_attr[22] = torch.tensor([0.,1.,0.,0.])
        # 빈 분자 객체 생성
        mol = Chem.RWMol()
        
        # 노드 추가
        for node_label in node_labels:
            atom = Chem.Atom(node_label_map[node_label.index(1.0)])
            mol.AddAtom(atom)

        # 엣지 추가
        for k, (i, j) in enumerate(edge_index.t().tolist()):
            bond_type = edge_label_map[torch.argmax(edge_attr[k]).item()]
            try: mol.AddBond(int(i), int(j), bond_type)
            except: pass
            atom_i = mol.GetAtomWithIdx(int(i))
            atom_j = mol.GetAtomWithIdx(int(j))
            
            if bond_type == Chem.BondType.SINGLE:
                if (atom_i.GetSymbol() == 'N' and atom_j.GetSymbol() == 'O'):
                    atom_i.SetFormalCharge(1)
                    atom_j.SetFormalCharge(-1)
                elif (atom_i.GetSymbol() == 'O' and atom_j.GetSymbol() == 'N'):
                    atom_i.SetFormalCharge(-1)
                    atom_j.SetFormalCharge(1)
        
        if (idx in [13,41,88,119,137,177]): mol.GetAtomWithIdx(1).SetFormalCharge(1)
        if (idx==149): mol.GetAtomWithIdx(5).SetFormalCharge(1)
                    
        AllChem.SanitizeMol(mol)  
        smiles = Chem.MolToSmiles(mol)
        smiles_list.append(smiles)
    with open('./dataset/MUTAG/raw/MUTAG_smiles.txt', '+w') as f:
        f.write('\n'.join(smiles_list))

def get_MUTAG(hibi=None):
    dataset = TUDataset(root='./dataset', name='MUTAG')
    smiles_list = []
    try:
        with open('./dataset/MUTAG/raw/MUTAG_smiles.txt', 'r') as f:
            for line in f:
                smiles_list.append(line.strip())
    except:
        get_MUTAG_smiles()
        with open('./dataset/MUTAG/raw/MUTAG_smiles.txt', 'r') as f:
            for line in f:
                smiles_list.append(line.strip())

    datalist = []
    for idx, data in enumerate(dataset):
        mol = get_mol(smiles_list[idx])
        y = data.y.squeeze()
        data = from_smiles(smiles_list[idx])
        data.y = y
        
        if mol is None: continue
        clique, edges = brics_decomp(mol)
        if len(edges)==0:
            clique, edges = tree_decomp_no_overlap_clean(mol)

        data.clique = clique

        
        data.num_cliques = len(clique)
        
        datalist.append(data)
          
    return MoleculeDataset('./', datalist), get_random_split(len(datalist),mutag=True)

def get_MolculeNetData(name, target_col=None, hibi=False):
    name = name.lower()
    dataset = PygGraphPropPredDataset(name='ogbg-mol'+name)
    smiles = pd.read_csv(f'./dataset/ogbg_mol{name}/mapping/mol.csv.gz', compression = 'gzip')
    smiles = smiles['smiles'].to_list()

    datalist = []
    for idx, data in enumerate(dataset):
        data.smiles = smiles[idx]
        if target_col is None:
            data.y = data.y.squeeze()
        else:
            data.y = data.y.squeeze()[target_col]
        
        mol = get_mol(data.smiles)
        if mol is None: continue
        clique, edges = brics_decomp(mol)

        if len(edges)==0:
            clique, edges = tree_decomp_no_overlap_clean(mol)

        data.clique = clique

        
        data.num_cliques = len(clique)
        
        datalist.append(data)
   
    return MoleculeDataset('./', datalist), get_random_split(len(datalist)) #dataset.get_idx_split()


def get_Tox21Data(target_col, hibi):
    dataset = pd.read_csv('../dataset/tox21_v2.csv')
    if target_col == 12:
        dataset = dataset.iloc[:,[0,1,2,3,4,5,6,-1]]
    elif target_col == 13:
        dataset = dataset.iloc[:,[7,8,9,10,11,-1]]
    elif target_col == 14:
        pass
    else:
        dataset = dataset.iloc[:,[target_col, 12,13]].dropna()
    dataset = dataset.to_numpy()
    
    datalist = []
    for datapoint in dataset:
        data = from_smiles(datapoint[-1])
        if data.x.size()[0] == 0: continue
        if len(datapoint) == 3:
            data.y = torch.tensor(datapoint[0], dtype=torch.long)
        else:
            data.y = torch.tensor(datapoint[:-2].astype(float))
        data.id = datapoint[-2]
        
        mol = get_mol(data.smiles)
        if mol is None: continue
        clique, edges = brics_decomp(mol)
        if len(edges)==0:
            clique, edges = tree_decomp(mol)

        pool_edge_index = torch.tensor([])
        if len(edges)!=0:
            pool_edge_index = torch.tensor(edges).mT
            pool_edge_index = torch.cat([pool_edge_index, pool_edge_index[[1,0]]], dim=-1)
            
        data.clique = clique
        data.pool_edge_index = pool_edge_index.to(torch.long)
        data.is_clique = torch.cat([torch.zeros(data.num_nodes), 
                                    torch.ones(len(clique))]).to(torch.long)
        
        node2clique = []
        clique2node = torch.zeros(data.num_nodes, dtype=torch.long)
        for i, sublist in enumerate(clique):
            node2clique.extend([i] * len(sublist))
            clique2node[sublist] = i
        data.node2clique = torch.tensor(node2clique, dtype=torch.long)
        data.clique2node = clique2node
        # data.clique2node = torch.cat([clique2node, torch.arange(len(clique))])
        
        hi_edge = torch.tensor([list(itertools.chain(*clique)),data.node2clique+data.num_nodes])
        if hibi:
            hi_edge = torch.cat([hi_edge, hi_edge[[1,0]]], dim=-1)
        
        data.hi_edge_index = torch.cat(
            [data.edge_index, data.pool_edge_index+data.num_nodes, hi_edge],
            dim=-1)
        
        data.num_cliques = len(clique)
        
        datalist.append(data)
        
    binding =  pd.read_csv('../binding.csv')
    binding = binding.dropna(subset=['binding_node'])
    idx = binding['idx'].tolist()
    
    split_idx = {}
    ratio = round(len(datalist)*0.1)
    train_idx = list(set(range(len(datalist))) - set(idx))
    random.shuffle(train_idx)
    train_idx = torch.tensor(train_idx)
    
    split_idx["test"] = torch.cat([torch.tensor(idx), train_idx[:ratio]])
    split_idx["valid"] = train_idx[ratio:2*ratio]
    split_idx["train"] = train_idx[2*ratio:]
   
    return MoleculeDataset('./', datalist), split_idx
