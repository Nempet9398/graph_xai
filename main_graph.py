#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import importlib
import torch
import wandb
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from utils.utils import set_seed
from torch_geometric.utils import to_networkx
from torch_geometric.explain import Explainer, GNNExplainer
#%%

#%%
import pandas as pd
import networkx as nx
import seaborn as sns
dataset_module = importlib.import_module('graph_dataset.dataset')
importlib.reload(dataset_module)
model_module = importlib.import_module('modules.model')
importlib.reload(model_module)
train_module = importlib.import_module('modules.train')
importlib.reload(train_module)
inference = importlib.import_module('modules.inference')
importlib.reload(inference)
optimize = importlib.import_module('modules.optmizer')
importlib.reload(optimize)
utils = importlib.import_module('utils.utils')
importlib.reload(utils)
gradcam = importlib.import_module('xai_test.gradcam')
importlib.reload(gradcam)
test_module = importlib.import_module('xai_test.noise_test')
importlib.reload(test_module)

#%%
#python main.py  --epochs 300 --model GCN --dataset MUTAG --top_motif 1 --layers 3 --group mean  
parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--hidden', type=int, default=256)

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0)

parser.add_argument('--model', type=str, default="GCN", help="GCN, GIN, GAT")

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='MUTAG',
                        help='name of dataset. For now, only classification.')
parser.add_argument('--eval_metric', type=str, default='auc')

parser.add_argument('--reg_1', type=float, default=0.1)
parser.add_argument('--reg_2', type=float, default=0.1)
parser.add_argument('--reg_3', type=float, default=0.1)
parser.add_argument('--group', type=str, default='mean')


# parser.add_argument('--percent', type=float, default=0.1)
parser.add_argument('--top_motif', type=int, default=1)

parser.add_argument('--target_col', type=int, default=14)

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
args.device = device

name = f'{args.dataset}_{args.model}'
#%%
import os
print(os.getcwd())


#%%
import torch
from graphxai.datasets import Benzene
data_path = './GraphXAI/graphxai/datasets/real_world/benzene/benzene.npz'
dataset = Benzene(split_sizes = (0.75, 0.05, 0.2), data_path = data_path)#%%
#%%
#%%
import matplotlib.pyplot as plt
import networkx as nx
# Get the graph as a NetworkX object
nx_graph = dataset.get_graph_as_networkx(2)
%matplotlib inline
# Draw the graph
plt.figure(figsize=(8, 6))
nx.draw(nx_graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
plt.title("Graph Visualization")
plt.show()

#%%
# Train a model from scratch on the data:
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN_3layer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN_3layer, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x
from graphxai.gnn_models.graph_classification import train #, test_edit

model = GCN_3layer(dataset[0][0].x.shape[1], 64, 2)
train_dataset, train_exp = dataset.get_train_loader()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.001)
criterion = torch.nn.CrossEntropyLoss()


train(model,optimizer, criterion , train_dataset)
#%%
model(dataset[0][0].x, dataset[0][0].edge_index, dataset[0][0].batch).argmax(dim=1)
#%%
f1, precision,recall,auprc,auroc = test(model, train_dataset)
#%%
print(f1, precision, recall, auprc, auroc)
#%%
#%%
#%%

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn import metrics
import numpy as np
import torch
from torch_geometric.loader import DataLoader

def test(model: torch.nn.Module, data_loader: DataLoader):
    model.eval()
    GT, preds, probas = [], [], []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(next(model.parameters()).device)
            out = model(data.x, data.edge_index, data.batch)  # (batch_size, num_classes)
            pred = out.argmax(dim=1).cpu().numpy()
            prob = out.softmax(dim=1)[:, 1].cpu().numpy()  # 클래스 1에 대한 확률

            GT.extend(data.y.cpu().numpy())
            preds.extend(pred)
            probas.extend(prob)

    f1 = f1_score(GT, preds)
    precision = precision_score(GT, preds)
    recall = recall_score(GT, preds)
    auprc = metrics.average_precision_score(GT, probas)
    auroc = metrics.roc_auc_score(GT, probas)

    return f1, precision, recall, auprc, auroc

#%%
subgraphx = SubgraphX(model=model) #, num_classes=2)


_, explanation_results, related_preds = subgraphx(dataset[0][0].x, dataset[0][0].edge_index, dataset[0][0].batch, dataset[0][0].y)
#%%
data = dataset[0][0]
node_idx = torch.arange(data.x.shape[0])
pred_class = model(data.x, data.edge_index, data.batch).argmax(dim=1)
#%%
data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
#%%
model = model.to(device)
model
#%%
from graphxai.explainers import *
exp_method = SubgraphX(model, reward_method = 'gnn_score',  rollout=5)
forward_kwargs={'node_idx': node_idx,
                'x': data.x,
                'edge_index': data.edge_index,
                'label': pred_class,
                'max_nodes': 10}

exp_method.get_explanation_node(**forward_kwargs)
#%%
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.001)
criterion = torch.nn.CrossEntropyLoss()

# Train model:
for _ in range(300):
    loss = train(model, optimizer, criterion, data)

# Final testing performance:
f1, acc, prec, rec, auprc, auroc = test(model, data, num_classes = 2, get_auc = True)

print('Test F1 score: {:.4f}'.format(f1))
print('Test AUROC: {:.4f}'.format(auroc))

#%%
dataset
#%%
from torch_geometric.nn import GINConv

class GNN(torch.nn.Module):
    def __init__(self,input_feat, hidden_channels, classes = 2):
        super(GNN, self).__init__()
        self.mlp_gin1 = torch.nn.Linear(input_feat, hidden_channels)
        self.gin1 = GINConv(self.mlp_gin1)
        self.mlp_gin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.gin2 = GINConv(self.mlp_gin2)
        self.mlp_gin3 = torch.nn.Linear(hidden_channels, classes)
        self.gin3 = GINConv(self.mlp_gin3)

    def forward(self, x, edge_index):
        # NOTE: our provided testing function assumes no softmax
        #   output from the forward call.
        x = self.gin1(x, edge_index)
        x = x.relu()
        x = self.gin2(x, edge_index)
        x = x.relu()
        x = self.gin3(x, edge_index)
        return x
    
from graphxai.gnn_models.node_classification import train, test

model = GNN(dataset.n_features, 64)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.0001)
criterion = torch.nn.CrossEntropyLoss()

# Train model:
for _ in range(1500):
    loss = train(model, optimizer, criterion, data)

# Final testing performance:
f1, acc, prec, rec, auprc, auroc = test(model, data, num_classes = 2, get_auc = True)

print('Test F1 score: {:.4f}'.format(f1))
print('Test AUROC: {:.4f}'.format(auroc))
#%%
#%%
inv_dataset_module = importlib.import_module('graph_dataset.invid_dataset')
importlib.reload(inv_dataset_module)

dataset = inv_dataset_module.get_protein_dataset()
dataset[0]
#%%
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.generator import BAGraph, HouseMotif

# BA 그래프 + house motif 구조를 가진 'syn1'과 동일한 구성
graph_generator = BAGraph(num_graphs=100)
motif_generator = HouseMotif()
num_motifs = 5

dataset = ExplainerDataset(
    graph_generator=graph_generator,
    motif_generator=motif_generator,
    num_motifs=num_motifs
)

#%%
def main(args):

    for code_seed in range(5):
        set_seed(code_seed)
        #%%
        args.dataset ='MUTAG'

        if args.dataset.upper()=='MUTAG':
            dataset, split_idx = dataset_module.get_MUTAG()
        elif args.dataset.upper()=='TOX21V2':
            dataset, split_idx = dataset_module.get_Tox21Data()    
        else:
            dataset, split_idx = dataset_module.get_MolculeNetData(args.dataset)       
        args.num_task = dataset[0].y.view(1,-1).shape[1]
        args.num_classes = 2
        model_module = importlib.import_module('modules.model')
        importlib.reload(model_module)

        atom_encoder = model_module.AtomEncoder(args.hidden).to(device)
        model = model_module.BasicGNN(args).to(device)
        
        test_auc, best_model, best_encoder = train_module.train_function(
            dataset, split_idx, model, atom_encoder, args, device)
        
        test_dataset = dataset[split_idx['test']]

        #%%
#%%

from graphxai.datasets import AlkaneCarbonyl


dataset = AlkaneCarbonyl(
    split_sizes=(0.7, 0.2, 0.1),     # train / val / test 비율
    seed=42,
    downsample=True                  # 클래스 불균형 보정 여부
)
#%%

from graphxai.datasets import ShapeGGen
dataset = ShapeGGen(
    model_layers = 3,
    num_subgraphs = 100,
    subgraph_size = 12,
    prob_connection = 0.1,
    add_sensitive_feature = False
)
#%%

import torch
from torch_geometric.data import Data
import networkx as nx
import random

def generate_house_motif(start_idx=0):
    # 5 nodes, 6 edges
    edges = [
        (0,1), (1,2), (2,3), (3,0), # square
        (0,4), (2,4)               # roof
    ]
    edges = [(u+start_idx, v+start_idx) for u, v in edges]
    return edges, list(range(start_idx, start_idx+5))

def create_graph_with_or_without_motif(include_motif: bool, motif_start_idx: int = 0):
    # 1. BA graph 생성
    G = nx.barabasi_albert_graph(n=20, m=2)
    label = 1 if include_motif else 0
    node_offset = G.number_of_nodes()

    if include_motif:
        motif_edges, motif_nodes = generate_house_motif(start_idx=node_offset)
        G.add_nodes_from(motif_nodes)
        G.add_edges_from(motif_edges)
        # motif를 BA 그래프에 연결
        attach_to = random.randint(0, node_offset - 1)
        G.add_edge(attach_to, motif_nodes[0])  # motif와 BA 연결

    # PyG edge_index
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)  # 양방향

    x = torch.eye(G.number_of_nodes())  # one-hot feature
    y = torch.tensor([label], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    return data
graph_list = []

# 50 positive (motif 있음), 50 negative (motif 없음)
for _ in range(50):
    graph_list.append(create_graph_with_or_without_motif(include_motif=True))

for _ in range(50):
    graph_list.append(create_graph_with_or_without_motif(include_motif=False))

print(f'총 그래프 수: {len(graph_list)}')  # → 100
print(graph_list[3].x)  # Data(x=..., edge_index=..., y=...)

#%%
import torch
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


# 예시 그래프
x = torch.randn(4, 8)  # 4 nodes, 8-dim features
edge_index = torch.tensor([
    [0, 1, 2, 3, 0],
    [1, 0, 1, 2, 3]
], dtype=torch.long)

data = Data(x=x, edge_index=edge_index)

# GATConv 정의
conv = GATConv(in_channels=8, out_channels=4, heads=1, concat=True)

# forward 시 attention weight도 같이 반환
out, (edge_idx_out, attn_weights) = conv(data.x, data.edge_index, return_attention_weights=True)

#%%
attn_weights

#%%
        ratio = []  # 비율을 저장할 리스트

        node_number = []
        for n, data in enumerate(test_dataset):
            # 각 데이터의 노드 개수 계산
            num_nodes = data.x.size(0)
            number = int(num_nodes * 0.3)  # 노드 개수의 30% 계산
            node_number.append(number)
            
            # 비율 계산 및 저장
            ratio.append(number / num_nodes)

        # 비율의 평균 계산
        ratio_mean = sum(ratio) / len(ratio)
        print(ratio_mean)
        '''
        ================ Lasso with Graph Embedding =======================
        '''
        best_alpha_drop = -float('inf')
        best_reg_1 = args.reg_1
        best_alpha_importance = None
        ''' ======= Lasso ========'''
        for reg_1 in [0.01, 0.1,1,5, 10,30]:
            args.reg_1 = reg_1
            alpha_importance = optimize.compute_alpha_importance(test_dataset, atom_encoder, best_model, device, args)
            print(f'Seed {code_seed}, Reg_1 {reg_1}, Alpha Importance Computed')
            
            alpha_base, alpha_avg, alpha_drop = test_module.test_model_with_noise(
            best_model, atom_encoder, test_dataset, device, node_number, alpha_importance
            )
            
            if alpha_drop > best_alpha_drop:
                best_alpha_drop = alpha_drop
                best_reg_1 = reg_1
                best_alpha_importance = alpha_importance.copy()

        alpha_importance = best_alpha_importance

        args.reg_1 = best_reg_1
        print(f'Best Reg_1: {best_reg_1} with Alpha Drop: {best_alpha_drop}')

        ''' ======= Group Lasso ========'''
        best_group_alpha_drop = -float('inf')
        best_reg_2 = args.reg_2
        best_group_alpha_importance = None
        for reg_2 in [1 ,5, 10,30,50,100]:
            args.reg_2 = reg_2
            group_alpha_importance = optimize.compute_group_alpha_importance(
            test_dataset, atom_encoder, best_model, device, args
            )
            print(f'Seed {code_seed}, Reg_2 {reg_2}, Group Alpha Importance Computed')
            
            group_alpha_base, group_alpha_avg, group_alpha_drop = test_module.test_model_with_noise(
            best_model, atom_encoder, test_dataset, device, node_number, group_alpha_importance
            )
            
            if group_alpha_drop > best_group_alpha_drop:
                best_group_alpha_drop = group_alpha_drop
                best_reg_2 = reg_2
                best_group_alpha_importance = group_alpha_importance.copy()
        group_alpha_importance = best_group_alpha_importance

        args.reg_2 = best_reg_2
        print(f'Best Reg_2: {best_reg_2} with Group Alpha Drop: {best_group_alpha_drop}')
        print('Seed', code_seed, 'Group Alpha Importance Clear')

        ''' ======= Equalized Group Lasso ========'''
        ## 이거 기준으로 노드 수를 정하고, 그 노드 수만큼 노이즈를 추가해서 성능을 측정
        if args.group == 'mean':
            node_level_equalized_importance = optimize.equalize_group_alpha(group_alpha_importance, test_dataset)

            top_k = args.top_motif  # 원하는 top-k 값 설정
            ratio = []  # 비율을 저장할 리스트

            node_number = []
            for n, importance in enumerate(node_level_equalized_importance):
                # 중요도 리스트에 대해서
                unique_importance = torch.unique(importance)
                top_k_values = torch.topk(unique_importance, top_k).values
                number = torch.sum(torch.isin(importance, top_k_values))
                node_number.append(number)
                
                # 비율 계산 및 저장
                ratio.append(number.item() / len(importance))

            # 비율의 평균 계산
            ratio_mean = sum(ratio) / len(ratio)
            print(ratio_mean)


        #%%




        ''' ======= Node Importance ========'''
        node_importance = optimize.compute_explainer_importance(test_dataset, best_model, atom_encoder, device)

        print('Seed', code_seed, 'Node Importance Clear')
        #%%
        ''' ======= Grad-CAM ========'''        
        gradcam_importance = optimize.compute_gradcam_importance(test_dataset, best_model, atom_encoder,  device)

        #%%


        alpha_base, alpha_avg, alpha_drop = test_module.test_model_with_noise(best_model, atom_encoder,test_dataset, device, node_number, alpha_importance) #, percent=motif_number)

        grad_base, grad_avg, grad_drop = test_module.test_model_with_noise(best_model, atom_encoder, test_dataset, device, node_number, gradcam_importance)#, percent=motif_number)
        #%%
        group_alpha_base, group_alpha_avg, group_alpha_drop = test_module.test_model_with_noise(best_model,atom_encoder, test_dataset, device,node_number,  group_alpha_importance)        
    
        if args.group == 'mean':
            group_mean_base , group_mean_avg, group_mean_drop = test_module.test_model_with_noise(best_model,atom_encoder, test_dataset, device, node_number, node_level_equalized_importance) 
    
        # if args.group == 'group':    
        #     group_mean_base, group_mean_avg, group_mean_drop = test_module.test_model_with_noise(best_model,atom_encoder, test_dataset, device, node_number, group_importance)#, percent=motif_number)

        nodel_base, node_avg, node_drop = test_module.test_model_with_noise(best_model,atom_encoder, test_dataset, device, node_number, node_importance)

        print('Seed', code_seed, 'Test Clear')


        # 단일 값 결과 → 리스트로 감싸서 1-row DataFrame 만들기

        results = {
            'base': [alpha_base],
            'grad_drop': [grad_drop],
            'alpha_drop': [alpha_drop],
            'group_alpha_drop': [group_alpha_drop],
            'node_drop': [node_drop],
            'group_mean_drop': [group_mean_drop],
        }



        # 중요한 설정값들
        important_args = {
            'seed': code_seed,
            'reg_1': args.reg_1,
            'reg_2': args.reg_2,
            'dataset': args.dataset,
            'epochs': args.epochs,
            'layers': args.layers,
        }
        
        # 각 설정값도 한 줄짜리 DataFrame으로
        for key, value in important_args.items():
            results[key] = [value]

        # 결과를 DataFrame으로 생성
        df = pd.DataFrame(results)

        # 파일 경로
        if args.group == 'mean':
            file_path = f"./results/{args.dataset}_mean_test_select.csv"

        if args.group == 'group':
            file_path = f"./results/{args.dataset}_group_test_select.csv"

        # 파일이 이미 존재하면 append, 아니면 새로 저장
        if os.path.exists(file_path):
            df.to_csv(file_path, mode='a', header=False, index=False)
            print("✅ Appended result to results.csv")
        else:
            df.to_csv(file_path, index=False)
            print("✅ Created new results.csv")


#%%

if __name__ == "__main__":
    main(args)

#%%

import torch
from torch.nn import ReLU, Softmax
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, Sequential
from torch_geometric.utils import to_networkx, k_hop_subgraph
import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Set, Union, Tuple, Dict, Callable
from enum import Enum
from collections import defaultdict

class Task(Enum):
    GRAPH_CLASSIFICATION = 1
    NODE_CLASSIFICATION = 2
    LINK_PREDICTION = 3

class Experiment(Enum):
    DEFAULT = 1
    GREEDY = 2
@torch.no_grad()
def _aggregate_scores(loader, model, class_idx, task: Task = Task.GRAPH_CLASSIFICATION,
                      nodes_to_keep: List[int] = None) -> torch.Tensor:
    result = torch.tensor([]).float()
    for data in iter(loader):
        if task == Task.GRAPH_CLASSIFICATION:
            scores = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device)).detach().cpu()
            score_of_class = scores[:, class_idx]
        elif task == Task.NODE_CLASSIFICATION:
            scores = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device)).detach().cpu()
            node_index = nodes_to_keep[0]
            idx_in_batch = data.ptr[:-1] + node_index  # ignore last pointer and find nodes in batched graphs
            score_of_node = torch.index_select(scores, dim=0, index=idx_in_batch)
            score_of_class = score_of_node[:, class_idx]  # get scores of predicted class
        else:  # link prediction
            x1 = torch.tensor([nodes_to_keep[0]]).long().to(device)  # node 1 of edge
            x2 = torch.tensor([nodes_to_keep[1]]).long().to(device)  # node 2 of edge
            scores = model(x=data.x.to(device), edge_index=data.edge_index.to(device), x1=x1, x2=x2,
                           ptr=data.ptr.to(device)).detach().cpu()
            score_of_class = scores[:, class_idx]
        result = torch.cat((result, score_of_class), dim=0)

    return result


@torch.no_grad()
def _compute_marginal_contribution(include_list, exclude_list, model, class_idx, task: Task = Task.GRAPH_CLASSIFICATION,
                                   nodes_to_keep: List[int] = None) -> float:
    include_loader = DataLoader(include_list, batch_size=4, shuffle=False, num_workers=0)
    exclude_loader = DataLoader(exclude_list, batch_size=4, shuffle=False, num_workers=0)

    include_scores = _aggregate_scores(include_loader, model, class_idx, task, nodes_to_keep)
    exclude_scores = _aggregate_scores(exclude_loader, model, class_idx, task, nodes_to_keep)

    contribution = torch.mean(include_scores - exclude_scores).item()
    return contribution


@torch.no_grad()
def mc_l_shapley(model: torch.nn.Module, graph: Data, subgraph: Set[int], t: int, num_layers: int,
                 task: Task = Task.GRAPH_CLASSIFICATION, nodes_to_keep: List[int] = None) -> float:
    """
    Shapley computation by monte carlo approximation in local neighborhood
    """
    # initialize coalition space
    subgraph_list = list(subgraph)  # v1 to vk
    if task == Task.GRAPH_CLASSIFICATION:
        node_tensor, edge_index, mapping, _ = k_hop_subgraph(subgraph_list, num_layers, graph.edge_index,
                                                             relabel_nodes=False, num_nodes=graph.x.shape[0],
                                                             flow='source_to_target')  # source to target is important
    elif task == Task.NODE_CLASSIFICATION:
        node_index: int = nodes_to_keep[0]
        node_tensor, edge_index, mapping, _ = k_hop_subgraph(nodes_to_keep, num_layers, graph.edge_index,
                                                             relabel_nodes=False, num_nodes=graph.x.shape[0],
                                                             flow='source_to_target')  # source to target is important
    else:
        node_tensor, edge_index, mapping, _ = k_hop_subgraph(nodes_to_keep, num_layers, graph.edge_index,
                                                             relabel_nodes=False, num_nodes=graph.x.shape[0],
                                                             flow='source_to_target')  # source to target is important

    reachable_list = node_tensor.tolist()  # v1 to vr
    p_prime = list(set(reachable_list) - set(subgraph_list))

    placeholder = graph.x.shape[0]
    p = p_prime + [placeholder]

    if task == Task.GRAPH_CLASSIFICATION or task == Task.NODE_CLASSIFICATION:
        scores = model(x=graph.x.to(device), edge_index=graph.edge_index.to(device),
                       batch=torch.zeros(graph.x.shape[0]).long().to(device)).detach().cpu()
    else:
        x1 = torch.tensor([nodes_to_keep[0]]).long().to(device)  # node 1 of edge
        x2 = torch.tensor([nodes_to_keep[1]]).long().to(device)  # node 2 of edge
        scores = model(x=graph.x.to(device), edge_index=graph.edge_index.to(device), x1=x1, x2=x2, ptr=None).detach().cpu()

    if len(scores.shape) > 1:
        scores = scores.squeeze()

    if task == Task.NODE_CLASSIFICATION:
        scores = scores[node_index]  # get score for target node

    predicted_class = torch.argmax(scores, dim=0).item()

    exclude_data_list = []
    include_data_list = []

    for i in range(t):
        perm = np.random.permutation(p)
        split_idx = np.asarray(perm == placeholder).nonzero()[0][0]

        selected = perm[:split_idx]  # nodes selected for coalition

        include_mask = np.zeros(graph.x.shape[0])
        include_mask[selected] = 1
        include_mask[subgraph_list] = 1  # subgraph list already includes target nodes
        masked_x_include = graph.x * torch.tensor(include_mask).unsqueeze(1)  # zero padding
        include_data = Data(masked_x_include.float(), graph.edge_index)
        include_data_list.append(include_data)

        exclude_mask = np.zeros(graph.num_nodes)
        exclude_mask[selected] = 1
        if task == Task.NODE_CLASSIFICATION or task == Task.LINK_PREDICTION:  # exclude target nodes from zero padding
            exclude_mask[nodes_to_keep] = 1
        masked_x_exclude = graph.x * torch.tensor(exclude_mask).unsqueeze(1)  # zero padding
        exclude_data = Data(masked_x_exclude.float(), graph.edge_index)
        exclude_data_list.append(exclude_data)

    score = _compute_marginal_contribution(include_data_list, exclude_data_list, model, predicted_class, task,
                                           nodes_to_keep)
    return score

class MCTSNode:
    def __init__(self, graph: Data, n_min: int, node_set: Set[int], score: Union[None, float] = None):
        self.graph = graph
        self.n_min = n_min
        self.node_set = node_set
        self.score = score  # just for debugging display
        self._hash = self.compute_hash()

    def compute_hash(self):
        l = list(self.node_set)
        l.sort()
        result = 98767 - len(l) * 555
        for i, el in enumerate(l):
            result = result + (hash(el) % 9999999) * 1001 + i
        return result

    def is_terminal(self) -> bool:
        return len(self.node_set) <= self.n_min

    def __hash__(self) -> int:
        return self._hash

    def get_pruned_nodes(self) -> Set[int]:  # inverse of node set
        return set(range(self.graph.x.shape[0])) - self.node_set

    def __str__(self) -> str:
        return f'{sorted(list(self.node_set))}: {self.score}'

    def __eq__(self, node2) -> bool:
        return hash(self) == hash(node2) and self.node_set == node2.node_set


class MCTS:
    def __init__(self, graph: Data, exp_weight: float, n_min: int, score_func: Callable, model: torch.nn.Module,
                 t: int, num_layers: int, high2low: bool = False, max_children: int = -1,
                 task: Task = Task.GRAPH_CLASSIFICATION, nodes_to_keep: List[int] = None,
                 skip_to_leaves: bool = True, experiment: Experiment = None):
        self.W = defaultdict(float)  # total reward of each node
        self.C = defaultdict(int)  # total visit count for each node
        self.children: Dict[MCTSNode, List[MCTSNode]] = {}  # nodes and their children
        self.leaves = []  # all terminal nodes
        self.R: Dict[MCTSNode, float] = {}  # immediate reward for nodes

        self.exp_weight = exp_weight  # lambda
        self.n_min = n_min
        self.score_func = score_func
        self.graph = graph
        self.model = model
        self.t = t
        self.num_layers = num_layers

        self.high2low = high2low
        self.max_children = max_children  # negative number means consider all nodes

        self.nodes_to_keep = nodes_to_keep if nodes_to_keep is not None else []
        self.task = task
        self.skip_to_leaves = skip_to_leaves  # hastens computation, but only offers explanations of size n_min
        self.experiment = experiment  # for reproducible changes in the algorithm

        if experiment == Experiment.GREEDY:
            self.C = defaultdict(lambda:1)

        if self.task == Task.GRAPH_CLASSIFICATION:
            self.root = MCTSNode(graph, n_min, set(range(graph.x.shape[0])))
            self.root.score = self._r(self.root)
        else:  # for node and link classification, only consider k-hop subgraph
            node_tensor, edge_index, mapping, _ = k_hop_subgraph(nodes_to_keep, num_layers, graph.edge_index,
                                                                 relabel_nodes=False, num_nodes=graph.x.shape[0],
                                                                 flow='source_to_target')
            self.root = MCTSNode(graph, n_min, set(node_tensor.tolist()))
            self.root.score = self._r(self.root)

        self.paths = []  # for visualization only

    def _q(self, mcts_node) -> float:
        if self.C[mcts_node] == 0:
            return 0  # avoid unseen moves
        return self.W[mcts_node] / self.C[mcts_node]  # average reward

    def _r(self, mcts_node) -> float:
        if mcts_node in self.R.keys():
            mcts_node.score = self.R[mcts_node]
            return self.R[mcts_node]
        else:
            if self.task == Task.GRAPH_CLASSIFICATION:
                score = self.score_func(self.model, self.graph, mcts_node.node_set, self.t, self.num_layers)
            else:  # node classification and link prediction need node indices
                score = self.score_func(self.model, self.graph, mcts_node.node_set, self.t, self.num_layers,
                                        task=self.task, nodes_to_keep=self.nodes_to_keep)
            self.R[mcts_node] = score
            mcts_node.score = score
            return score

    def _u(self, mcts_node, parent) -> float:  # utility from paper
        children = self.children[parent]
        parent_count = 0
        for c in children:
            parent_count += self.C[c]
        #counts = [self.C[n] for n in children]
        #parent_count = sum(counts)
        if parent_count == 0 and self.skip_to_leaves:  # for computational efficiency: all nodes unexplored
            return 0
        u = self.exp_weight * self._r(mcts_node) * math.sqrt(parent_count) / (1 + self.C[mcts_node])
        return u

    def _ucb(self, node, parent) -> float:  # upper confidence bound
        return self._q(node) + self._u(node, parent)

    def _select_path_by_ucb(self) -> List[MCTSNode]:  # choose best leaf by ucb, training
        mcts_node = self.root
        path = [mcts_node]
        while not mcts_node.is_terminal():
            mcts_node = self._best_child_by_ucb(mcts_node)
            path.append(mcts_node)
        return path

    def _best_child_by_ucb(self, mcts_node: MCTSNode):
        def _score_helper(child):
            return self._ucb(child, mcts_node)

        if mcts_node in self.children.keys():
            children = self.children[mcts_node]
        else:
            children = self._expand_node(mcts_node)

        # this computation matters for the random experiment
        children_scores = [_score_helper(child) for child in children]

        return max(children, key=_score_helper)

    def _expand_node(self, mcts_node) -> List[MCTSNode]:
        if mcts_node in self.children.keys():
            raise Exception(f'Node is already expanded: {mcts_node}')

        if mcts_node.is_terminal():
            raise Exception(f'terminal node cannot be expanded: {mcts_node}')

        children = []
        nx_graph = to_networkx(self.graph, to_undirected=True)  # connected components only works for undirected graphs

        # sort nodes according to pruning strategy and only consider first k nodes
        nodes_to_prune = list(mcts_node.node_set.copy())
        if self.task == Task.NODE_CLASSIFICATION or self.task == Task.LINK_PREDICTION:
            nodes_to_prune = [n for n in nodes_to_prune if n not in self.nodes_to_keep]

        nodes_to_prune.sort(key=lambda x: nx_graph.degree(x), reverse=self.high2low)
        if self.max_children >= 0:
            nodes_to_prune = nodes_to_prune[:self.max_children]

        for node in nodes_to_prune:
            subgraph = nx_graph.subgraph(mcts_node.node_set - {node})
            components = nx.connected_components(subgraph)

            if self.task == Task.GRAPH_CLASSIFICATION:  # only keep largest connected component
                child_set = max(components, key=lambda x: len(x))
            elif self.task == Task.NODE_CLASSIFICATION:  # keep component with target node
                node_to_keep = self.nodes_to_keep[0]
                child_set = set()
                for c in components:
                    if node_to_keep in c:
                        child_set = c
                        break
                if len(child_set) == 0:
                    raise Exception('Target node not in children, which should never happen. You found a bug.')

            else:  # Link prediction: keep two components which include both nodes
                node1 = self.nodes_to_keep[0]
                node2 = self.nodes_to_keep[1]
                child_set = set()
                for c in components:
                    if node1 in c or node2 in c:
                        child_set = child_set | c
                if len(child_set) == 0:
                    raise Exception('Target node not in children, which should never happen. You found a bug.')

            child = MCTSNode(self.graph, self.n_min, child_set)
            children.append(child)

        self.children[mcts_node] = children
        return children

    def _backpropagate(self, path):
        score = self._r(path[-1])  # score is reward of leaf node
        for mcts_node in path:
            self.C[mcts_node] += 1
            self.W[mcts_node] += score
        pass  # put debug breakpoint here, very useful

    def search_one_iteration(self):  # train for one iteration
        path = self._select_path_by_ucb()
        leaf = path[-1]
        if leaf not in self.leaves:
            self.leaves.append(leaf)
        self._backpropagate(path)
        self.paths.append(path)  # for later visualization

    def best_leaf_node(self) -> MCTSNode:  # choose best leaf by reward only
        return max(self.leaves, key=self._r)

    def best_node(self, size: int) -> MCTSNode:
        if self.skip_to_leaves and size != self.n_min:
            print('Warning: Some scores were skipped in the exploration phase. Set skip_to_leaves to False!')

        if size >= len(self.root.node_set):
            print('Warning: The requested explanation-set is too large.')
            return self.root
        elif size <= 0:
            raise RecursionError('There is no subgraph of the requested size')

        candidates = [k for k in self.R.keys() if len(k.node_set) == size]

        if candidates:
            return max(candidates, key=self._r)
        else:
            return self.best_node(size-1)
        
class SubgraphX:
    def __init__(self, model: torch.nn.Module, num_layers: int, exp_weight: float, m: int, t: int,
                 high2low: bool = False, max_children: int = -1,
                 task: Task = Task.GRAPH_CLASSIFICATION,
                 value_func: Callable = mc_l_shapley, experiment: Experiment = None):
        """
        Subgraph-X Implementation from the Paper "On Explainability of Graph Neural Networks via Subgraph Explorations"
        :param model: The model to explain. Output of the model have to be normalized probabilities for each class.
        :param num_layers: The number of convolutional layers in the model
        :param exp_weight: the lambda from formula (3) in the paper. Balances exploration and exploitation
        :param m: Number of MCTS iterations
        :param t: Number of Monte-Carlo sampling steps for shapley approximation
        :param high2low: ordering of nodes when considering pruning action by their node degree
        :param max_children: Maximum number of nodes to consider for pruning actions. -1 means consider all nodes.
        :param task: Graph, Node Classification or Link prediction
        :param value_func: Function used to score subgraph explanation
        """
        self.model = model
        self.num_layers = num_layers
        self.exp_weight = exp_weight
        self.m = m
        self.t = t
        self.value_func = value_func

        self.high2low = high2low
        self.max_children = max_children

        self.task = task
        self.experiment = experiment

    def _get_mcts(self, graph: Data, n_min: int, nodes_to_keep: List[int],
                  exhaustive: bool):
        return MCTS(graph, self.exp_weight, n_min, self.value_func, self.model, self.t,
                    self.num_layers, self.high2low, self.max_children, self.task, nodes_to_keep,
                    skip_to_leaves=(not exhaustive), experiment=self.experiment)

    def __call__(self, graph: Data, n_min: int, nodes_to_keep: List[int] = None, exhaustive: bool = False):
        """
        Obtain explanation for a single instance
        :param graph: The graph to explain
        :param n_min: Maximum number of nodes in the explanation (upper bound, may not be exact)
        :param nodes_to_keep: Task dependent important nodes: None for graph classification, for node classification
        should be index of node. For link prediction both adjacent nodes to edge to explain.
        :param exhaustive: If exhaustive search is enabled, explanations of different sizes may be requested from mcts.
        :return: Set of nodes explaining the prediction of the model on the given instance
        """
        nodes_to_keep = nodes_to_keep if nodes_to_keep is not None else []
        mcts = self._get_mcts(graph, n_min, nodes_to_keep, exhaustive)

        for iteration in tqdm(range(self.m)):
            mcts.search_one_iteration()

        explanation = mcts.best_leaf_node()

        return explanation.node_set, mcts
    
#%%
data = dataset[0][0].to('cpu')
#%%
model = model.to(device)
#%%
sx_params = {
    "model": model,
    "num_layers": 3,
    "exp_weight": 5,
    "m": 10,
    "t": 10,
    "task": Task.GRAPH_CLASSIFICATION,
    "max_children": 8,
    "experiment": None,
    "value_func": mc_l_shapley,
}
subgraphx = SubgraphX(**sx_params)
explaining_subgraph, mcts = subgraphx(data, n_min=4, nodes_to_keep= None, exhaustive=True)
#%%
mcts.best_node(8).node_set
#%%
def get_embedding_model(graph):
    embedding_dim = 10
    walk_length = 80
    context_size = 10
    walks_per_node = 10
    return torch_geometric.nn.Node2Vec(graph.edge_index, embedding_dim=embedding_dim, walk_length=walk_length,
                                        context_size=context_size, walks_per_node=walks_per_node,
                                        num_nodes=graph.x.shape[0]).to(device)

emb_model = get_embedding_model(dataset[0][0])

#%%
