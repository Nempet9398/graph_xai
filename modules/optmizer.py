#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import torch
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from enum import Enum
from typing import List, Set, Union, Tuple, Dict, Callable
from collections import defaultdict
from joblib import Parallel, delayed
from skglm import GroupLasso

import torch.nn.functional as F
from torch.nn import ReLU, Softmax, Parameter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, Sequential
from torch_geometric.utils import to_networkx, k_hop_subgraph
from torch_geometric.explain import Explainer, CaptumExplainer
from torch_geometric.explain.algorithm import GNNExplainer, PGExplainer, GraphMaskExplainer
from torch_geometric.explain.config import ModelConfig, ExplanationType, MaskType

import importlib
gradcam = importlib.import_module('xai_test.gradcam')
importlib.reload(gradcam)

# Logging setup
logging.basicConfig(level=logging.INFO)

# Constants
sns.set(style="whitegrid")

def optimize_normal_graph(data, test_data, device, args):
    if test_data.y.isnan().any():
        return None

    data = data.to(device)
    embedding = data  

    n = embedding.shape[0]

    if args.pooling == 'mean':
        uniform = torch.mean(embedding, dim=0)
    elif args.pooling == 'max':
        uniform = torch.max(embedding, dim=0).values
    elif args.pooling == 'sum':
        uniform = torch.sum(embedding, dim=0)
    else:
        raise ValueError(f"Unsupported pooling method: {args.pooling}")

    alpha = Parameter(torch.ones(n, device=device))
    optimizer = torch.optim.Adam([alpha], lr=0.01)
    lambda_reg = args.reg_1

    for _ in range(300):
        optimizer.zero_grad()
        weighted_sum = torch.sum(alpha.unsqueeze(1) * embedding, dim=0)
        loss = torch.norm(weighted_sum - uniform, p=2) ** 2 / embedding.size(1) + lambda_reg * torch.norm(alpha, p=1)
        loss.backward()
        optimizer.step()

    return alpha.detach().cpu()

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def compute_normal_importance_parallel(embedding_list,  test_dataset, device, args, n_jobs=20):
    results = [None] * len(test_dataset)

    def worker(i):
        embedding_data = embedding_list[i].to(device)
        test_data = test_dataset[i]
        return i, optimize_normal_graph(embedding_data, test_data, device, args)

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(worker, i) for i in range(len(test_dataset))]
        for f in tqdm(as_completed(futures), total=len(futures), desc="GPU L1 "):
            i, res = f.result()
            results[i] = res

    return [r for r in results if r is not None]


def find_best_normal_importance(
        test_dataset,
        embedding_list,
        best_model,
        device,
        args,
        node_number,
        test_func,
        reg_1_list=[0.01, 0.1, 1, 5]
    ):
    best_alpha_drop = -float('inf')
    best_reg_1 = None
    best_alpha_importance = None
    best_sparsity = None 
    max_value = len(reg_1_list)
    for n, reg_1 in enumerate(reg_1_list):
        args.reg_1 = reg_1

        alpha_importance = compute_normal_importance_parallel(
            embedding_list = embedding_list,
            test_dataset=test_dataset,
            device=device,
            args=args
        )
        # Compute sparsity statistics
        sparsity_list = []
        for alpha in alpha_importance:
            if alpha is not None:
                sparsity = (alpha.abs() < 1e-4).float().mean().item()
                sparsity_list.append(sparsity)

        if sparsity_list:
            avg_sparsity = sum(sparsity_list) / len(sparsity_list)
            print(f"[reg_1={reg_1}] Average sparsity: {avg_sparsity:.4f}")
        else:
            print(f"[reg_1={reg_1}] No valid alpha_importance to compute sparsity.")
        print(f'[reg_1={reg_1}] Alpha importance computed.')

        alpha_base, alpha_avg, alpha_drop = test_func(  
            model=best_model,
            dataset=test_dataset,
            device=device,
            number_list=node_number,
            importance_list=alpha_importance,
            args=args
        )
        if avg_sparsity > 1e-5:
            if alpha_drop > best_alpha_drop:

                best_alpha_drop = alpha_drop
                best_reg_1 = reg_1
                best_alpha_base = alpha_base 
                best_alpha_avg = alpha_avg 
                best_alpha_importance = alpha_importance.copy()
                best_sparsity = avg_sparsity
            print(f'Drop: {alpha_drop:.4f} (reg_1: {reg_1})')

        else:
            if n == max_value - 1:
                best_alpha_drop = alpha_drop
                best_reg_1 = reg_1
                best_alpha_base = alpha_base 
                best_alpha_avg = alpha_avg 
                best_alpha_drop = alpha_drop
                best_alpha_importance = alpha_importance.copy()
                best_sparsity = avg_sparsity
            print(f'Drop: {alpha_drop:.4f} (reg_1: {reg_1})')
    args.reg_1 = best_reg_1

    print(f'[Best Reg_1] {best_reg_1} (Drop: {best_alpha_drop:.4f})')
    return best_alpha_importance, best_reg_1, best_alpha_base, best_alpha_avg, best_alpha_drop,best_sparsity


def optimize_one_graph(data, test_data, device, args):
    if test_data.y.isnan().any():
        return None

    data = data.to(device)
    embedding = data  

    n = embedding.shape[0]

    if args.pooling == 'mean':
        uniform = torch.mean(embedding, dim=0)
    elif args.pooling == 'max':
        uniform = torch.max(embedding, dim=0).values
    elif args.pooling == 'sum':
        uniform = torch.sum(embedding, dim=0)
    else:
        raise ValueError(f"Unsupported pooling method: {args.pooling}")

    alpha = Parameter(torch.ones(n, device=device))
    optimizer = torch.optim.Adam([alpha], lr=0.01)
    lambda_reg = args.reg_1

    for _ in range(300):
        optimizer.zero_grad()
        weighted_sum = torch.sum(alpha.unsqueeze(1) * embedding, dim=0)
        loss = torch.norm(weighted_sum - uniform, p=2) ** 2 / embedding.size(1) + lambda_reg * torch.norm(alpha, p=1)
        loss.backward()
        optimizer.step()

    return alpha.detach().cpu()

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def compute_alpha_importance_parallel(embedding_list,  test_dataset, device, args, n_jobs=20):
    results = [None] * len(test_dataset)

    def worker(i):
        embedding_data = embedding_list[i].to(device)
        test_data = test_dataset[i]
        return i, optimize_one_graph(embedding_data, test_data, device, args)

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(worker, i) for i in range(len(test_dataset))]
        for f in tqdm(as_completed(futures), total=len(futures), desc="GPU L1"):
            i, res = f.result()
            results[i] = res

    return [r for r in results if r is not None]

def find_best_alpha_importance(
        test_dataset,
        best_model,
        atom_encoder,
        embedding_list,
        device,
        args,
        test_func,
        node_number,
        reg_1_list=[0.01, 0.1, 1, 5]
    ):
    best_alpha_drop = -float('inf')
    best_reg_1 = None
    best_alpha_importance = None
    best_node_number = None
    for reg_1 in reg_1_list:
        args.reg_1 = reg_1

        alpha_importance = compute_alpha_importance_parallel(
            embedding_list = embedding_list,
            test_dataset=test_dataset,

            device=device,
            args=args
        )
        print(f'[reg_1={reg_1}] Alpha importance computed.')

        node_level_equalized_importance = equalize_group_alpha(alpha_importance, test_dataset)
        node_number = select_motifs_by_node_ratio(node_level_equalized_importance,test_dataset, sparsity_target=0.3)

        alpha_base, alpha_avg, alpha_drop = test_func(  
            model=best_model,
            atom_encoder=atom_encoder,
            dataset=test_dataset,
            device=device,
            number_list=node_number,
            importance_list=node_level_equalized_importance,
            args=args
        )
        # wandb.log({"l1_drop_log": alpha_drop, "reg_1": reg_1})

        if alpha_drop > best_alpha_drop:
            best_alpha_drop = alpha_drop
            best_reg_1 = reg_1
            best_alpha_base = alpha_base
            best_alpha_avg = alpha_avg 
            best_alpha_drop = alpha_drop
            best_node_number = node_number.copy()
            best_alpha_importance = node_level_equalized_importance.copy()
        print(f'Drop: {alpha_drop:.4f} (reg_1: {reg_1})')

    args.reg_1 = best_reg_1

    print(f'[Best Reg_1] {best_reg_1} (Drop: {best_alpha_drop:.4f})')
    return best_alpha_importance, best_reg_1, best_alpha_base, best_alpha_avg, best_alpha_drop, best_node_number


def optimize_one_graph_connectivity(data, test_data, device, args):
    if test_data.y.isnan().any():
        return None

    data = data.to(device)
    embedding = data

    n = embedding.shape[0]

    if args.pooling == 'mean':
        uniform = torch.mean(embedding, dim=0)
    elif args.pooling == 'max':
        uniform = torch.max(embedding, dim=0).values
    elif args.pooling == 'sum':
        uniform = torch.sum(embedding, dim=0)
    else:
        raise ValueError(f"Unsupported pooling method: {args.pooling}")

    alpha = Parameter(torch.ones(n, device=device))
    optimizer = torch.optim.Adam([alpha], lr=0.01)
    lambda_reg = args.reg_1
    lambda_conn = args.reg_conn

    edge_index = test_data.edge_index.to(device)

    for _ in range(300):
        optimizer.zero_grad()
        weighted_sum = torch.sum(alpha.unsqueeze(1) * embedding, dim=0)
        recon_loss = torch.norm(weighted_sum - uniform, p=2) ** 2 / embedding.size(1)
        l1_loss = lambda_reg * torch.norm(alpha, p=1)
        conn_loss = torch.sum((alpha[edge_index[0]] - alpha[edge_index[1]]) ** 2) / edge_index.shape[1]
        loss = recon_loss + l1_loss + lambda_conn * conn_loss
        loss.backward()
        optimizer.step()

    return alpha.detach().cpu()

def compute_alpha_connectivity_parallel(embedding_list, test_dataset, device, args, n_jobs=20):
    results = [None] * len(test_dataset)

    def worker(i):
        embedding_data = embedding_list[i].to(device)
        test_data = test_dataset[i]
        return i, optimize_one_graph_connectivity(embedding_data, test_data, device, args)

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(worker, i) for i in range(len(test_dataset))]
        for f in tqdm(as_completed(futures), total=len(futures), desc="GPU "):
            i, res = f.result()
            results[i] = res

    return [r for r in results if r is not None]

def find_best_alpha_connectivity(
    test_dataset,
    best_model,
    atom_encoder,
    embedding_list,
    device,
    args,
    test_func,
    node_number,
    reg_1_list=[0.01, 0.1, 1, 5]
):
    best_alpha_drop = -float('inf')
    best_reg_1 = None
    best_reg_conn = None
    best_alpha_importance = None

    for reg_1 in reg_1_list:
        args.reg_1 = reg_1
        for lambda_conn in [0.1, 1.0, 5.0]:  
            args.reg_conn = lambda_conn

            alpha_importance = compute_alpha_connectivity_parallel(
                embedding_list = embedding_list,
                test_dataset = test_dataset,
                device = device,
                args = args
            )
            print(f'[reg_1={reg_1}] Connectivity alpha importance computed.')

            alpha_base, alpha_avg, alpha_drop = test_func(
                model = best_model,
                atom_encoder = atom_encoder,
                dataset = test_dataset,
                device = device,
                number_list = node_number,
                importance_list = alpha_importance,
                args = args
            )

            if alpha_drop > best_alpha_drop:
                best_alpha_drop = alpha_drop
                best_reg_1 = reg_1
                best_reg_conn = lambda_conn
                best_alpha_base = alpha_base
                best_alpha_avg = alpha_avg 
                best_alpha_importance = alpha_importance.copy()
            print(f'Drop: {alpha_drop:.4f} (reg_1: {reg_1}, reg_conn: {lambda_conn})')

    args.reg_1 = best_reg_1
    print(f'[Best Reg_1 - Connectivity] {best_reg_1} (reg_conn: {best_reg_conn}, Drop: {best_alpha_drop:.4f})')
    return best_alpha_importance, best_reg_1, best_reg_conn, best_alpha_base, best_alpha_avg, best_alpha_drop



def to_group_tensor_list(group_list, device):
    return [torch.tensor(grp, device=device) for grp in group_list if len(grp) > 0]

def group_lasso_regularization(group_alpha, group_tensor_list):
    reg = 0.0
    for grp_tensor in group_tensor_list:
        v = group_alpha.index_select(0, grp_tensor)
        reg += torch.norm(v, p=2) / (grp_tensor.numel() ** 0.5)
    return reg


def optimize_one_group_graph(data, test, device, args):
    if test.y.isnan().any():
        return None

    group_tensor_list = to_group_tensor_list(test.clique, device)
    data = data.to(device)
    embedding = data
    n = embedding.shape[0]

    if args.pooling == 'mean':
        uniform = torch.mean(data, dim=0)
    elif args.pooling == 'max':
        uniform = torch.max(data, dim=0).values
    elif args.pooling == 'sum':
        uniform = torch.sum(data, dim=0)
    else:
        raise ValueError(f"Unsupported pooling method: {args.pooling}")

    group_alpha = torch.nn.Parameter(torch.ones(n, device=device))
    optimizer = torch.optim.Adam([group_alpha], lr=0.01)
    lambda_reg = args.reg_2
    
    for epoch in range(300):
        optimizer.zero_grad()
        weighted_sum = torch.sum(group_alpha.unsqueeze(1) * embedding, dim=0)
        loss = torch.norm(weighted_sum - uniform, p=2) ** 2  / embedding.size(1)
        loss += lambda_reg * group_lasso_regularization(group_alpha, group_tensor_list)
        loss.backward()
        optimizer.step()

    return group_alpha.detach().cpu()



def compute_group_alpha_importance_parallel(embedding_list, test_dataset, device, args, n_jobs=20):
    results = [None] * len(test_dataset)

    def worker(i):
        embedding_data = embedding_list[i].to(device)
        test_data = test_dataset[i]
        return i, optimize_one_group_graph(embedding_data, test_data, device, args)

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(worker, i) for i in range(len(test_dataset))]
        for f in tqdm(as_completed(futures), total=len(futures), desc="GPU"):
            i, res = f.result()
            results[i] = res

    return [r for r in results if r is not None]

import warnings
import torch

def find_best_group_alpha_importance(
    test_dataset,
    best_model,
    atom_encoder,
    embedding_list,
    device,
    args,
    test_func,
    reg_2_list=[100, 150, 200, 250]
):
    warnings.filterwarnings("ignore") 

    best_group_alpha_drop = -float('inf')
    best_reg_2 = None
    best_group_alpha_importance = None
    best_node_number = None

    for reg_2 in reg_2_list:
        args.reg_2 = reg_2

        group_alpha_importance = compute_group_alpha_importance_parallel(
            embedding_list = embedding_list,
            test_dataset = test_dataset,
            device=device,
            args=args
        )
        print(f'[reg_2={reg_2}] Group alpha importance computed.')
        node_level_equalized_importance = equalize_group_alpha(group_alpha_importance, test_dataset)

        node_number = select_motifs_by_node_ratio(node_level_equalized_importance,test_dataset, sparsity_target=0.3)
        # top_k = args.top_motif
        # node_number = []
        # for importance in node_level_equalized_importance:
        #     unique_vals = torch.unique(importance)
        #     top_k_vals = torch.topk(unique_vals, top_k).values
        #     number = torch.sum(torch.isin(importance, top_k_vals))
        #     node_number.append(number)

        base,average , diff = test_func(
            model=best_model,
            atom_encoder=atom_encoder,
            dataset=test_dataset,
            device=device,
            number_list=node_number,
            importance_list=node_level_equalized_importance, args=args
        )




        if diff > best_group_alpha_drop:
            best_group_alpha_drop = diff
            best_reg_2 = reg_2
            best_base = base
            best_average = average
            best_drop = diff
            best_group_alpha_importance = group_alpha_importance.copy()
            best_node_number = node_number.copy()
        print(f'Drop: {diff:.4f} (reg_2: {reg_2})')
    args.reg_2 = best_reg_2

    print(f'[Best Reg_2] {best_reg_2} (Drop: {best_group_alpha_drop:.4f})')
    return best_group_alpha_importance, best_reg_2, best_base, best_average, best_drop, best_node_number



def equalize_group_alpha(group_alpha_importance, test_dataset):
    node_level_equalized_importance = []
    for data_index in range(len(test_dataset)):

        if test_dataset[data_index].y.isnan().any():
            continue

        group_level = group_alpha_importance[data_index]
        group = test_dataset[data_index].clique  # List[List[int]]
        n = test_dataset[data_index].x.shape[0]  

        node_alpha_equalized = torch.zeros(n)
        for node_indices in group:
            if len(node_indices) > 0:

                mean_value = group_level[torch.tensor(node_indices)].mean()
                node_alpha_equalized[torch.tensor(node_indices)] = mean_value

        node_level_equalized_importance.append(node_alpha_equalized)
    return node_level_equalized_importance

def select_motifs_by_node_ratio(node_level_equalized_importance, test_dataset, sparsity_target=0.3):
    
    node_number = []
    for data_index in range(len(test_dataset)):
        importance = node_level_equalized_importance[data_index]
        group = test_dataset[data_index].clique  # List[List[int]]
        total_nodes = len(importance)
        node_limit = int(total_nodes * sparsity_target)

        motif_scores = [importance[torch.tensor(m)].mean().item() for m in group]
        sorted_indices = sorted(range(len(group)), key=lambda i: -motif_scores[i])


        selected_nodes = set()
        for idx in sorted_indices:
            selected_nodes.update(group[idx])
            if len(selected_nodes) >= node_limit:
                break
        node_number.append(len(selected_nodes))

    return node_number


def compute_gradcam_importance(test_dataset, best_model,   device, atom_encoder = None):
    gradcam_importance = []
    for i in range(len(test_dataset)):
        if test_dataset[i].y.isnan().any():
            print(f"Skipping data index {i} due to NaN in target.")
            continue
        if atom_encoder is not None:
            grad_node_importance = gradcam.grad_cam(best_model, test_dataset[i], device, atom_encoder=atom_encoder)
        else: 
            grad_node_importance = gradcam.grad_cam(best_model, test_dataset[i], device)
        
        gradcam_importance.append(grad_node_importance.detach().to('cpu'))
    return gradcam_importance


def compute_gnnexplainer_importance(test_dataset, model,  device,atom_encoder = None):
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(lr=0.001),
        explanation_type=ExplanationType.model,
        model_config=ModelConfig(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
        node_mask_type=MaskType.object,
        edge_mask_type=None
    )

    node_importance = []
    for data in tqdm(test_dataset):
        data = data.to(device)
        if data.y.isnan():
            continue
        with torch.no_grad():
            if atom_encoder is not None:
                x = atom_encoder(data.x)
            else:
                x = data.x

            model(x, data.edge_index, batch=data.batch)
        explanation = explainer(x, data.edge_index, batch=data.batch)
        node_importance.append(explanation.node_mask.view(-1).detach().cpu())
    return node_importance



def compute_node_number(node_level_equalized_importance, top_k=2):
    node_number = []
    for importance in node_level_equalized_importance:
        unique_importance = torch.unique(importance)
        top_k_values = torch.topk(unique_importance, top_k).values
        number = torch.sum(torch.isin(importance, top_k_values))
        node_number.append(number)
    return node_number



def compute_pgexplainer_importance(test_dataset, model, atom_encoder=None):
    explainer = Explainer(
        model=model,
        algorithm=PGExplainer( lr=0.003),
        explanation_type='phenomenon',
        model_config=ModelConfig(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
        edge_mask_type='object'
    )

    loader = DataLoader(test_dataset, batch_size=32, shuffle=True) #, atom_encoder=None)


    model.to('cpu').eval()
    if atom_encoder is not None:\
        atom_encoder.to('cpu').eval()

    for epoch in range(100):
        for data in loader:
            data = data.to('cpu')
            if atom_encoder is not None:
                x = atom_encoder(data.x)
            else:
                x = data.x
            loss = explainer.algorithm.train(epoch, model, x, data.edge_index,
                                                target=data.y, batch=data.batch)

    def edge_mask_to_node_importance(edge_index, edge_mask, num_nodes):
        node_importance = torch.zeros(num_nodes, device=edge_mask.device)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i]
            node_importance[src] += edge_mask[i]
            node_importance[dst] += edge_mask[i]
        return node_importance


    pg_node_importances = []

    for data in tqdm(test_dataset, desc='Generating Explanations'):
        data = data.to('cpu')
        with torch.no_grad():
            if atom_encoder is not None:
                x = atom_encoder(data.x)
            else:
                x = data.x
            explanation = explainer.algorithm.forward(
                model=model,
                x=x,
                edge_index=data.edge_index,
                target=data.y,
                index=None,
                batch=data.batch if hasattr(data, 'batch') else None
            )

        edge_mask = explanation.edge_mask
        node_importance = edge_mask_to_node_importance(
            data.edge_index, edge_mask, data.num_nodes
        )

        pg_node_importances.append(node_importance)

    return pg_node_importances


def compute_graphmask_importance(test_dataset, model, atom_encoder=None):

    for module in model.modules():
        if isinstance(module, GINConv):
            if not hasattr(module, 'in_channels'):
                module.in_channels = module.nn[0].in_features  
            if not hasattr(module, 'out_channels'):
                module.out_channels = module.nn[-1].out_features  

    explainer = Explainer(
        model=model,
        algorithm=GraphMaskExplainer(epochs=30, num_layers=3, lr=0.003, log=False),
        explanation_type=ExplanationType.model,
        model_config=ModelConfig(
            mode='multiclass_classification',
            task_level='graph',
            return_type='raw',
        ),
        node_mask_type=MaskType.object,
        edge_mask_type=None
    )

    model.to('cpu').eval()
    if atom_encoder is not None:
        atom_encoder.to('cpu').eval()

    node_importance = []
    for data in tqdm(test_dataset, desc='GraphMask Explanation'):
        data = data.to('cpu')
        if data.y.isnan():
            continue
        with torch.no_grad():
            if atom_encoder is not None:
                x = atom_encoder(data.x)
            else:
                x = data.x
            model(x, data.edge_index, batch=data.batch)

        explanation = explainer(x, data.edge_index, batch=data.batch)
        node_importance.append(explanation.node_mask.view(-1).detach().cpu())

    return node_importance


def compute_captum_importance(test_dataset, model,  model_name, atom_encoder=None):
    explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer(model_name),  # or IntegratedGradients
        explanation_type='model',
        model_config=ModelConfig(
            mode='multiclass_classification',
            task_level='graph',
            return_type='probs',
        ),
        node_mask_type=MaskType.attributes,
    )

    model.to('cpu').eval()
    if atom_encoder is not None:
        atom_encoder.to('cpu').eval()

    node_importance = []
    for data in tqdm(test_dataset, desc='Captum Explanation'):
        data = data.to('cpu')
        if data.y.isnan():
            continue
        with torch.no_grad():
            if atom_encoder is not None:
                x = atom_encoder(data.x)
            else:
                x = data.x
            data.x = x

        explanation = explainer(data.x, data.edge_index, batch=data.batch)
        node_importance.append(explanation.node_mask.sum(1).detach().cpu())

    return node_importance



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
            print('')
            # print('Warning: Some scores were skipped in the exploration phase. Set skip_to_leaves to False!')

        if size >= len(self.root.node_set):
            print('')
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

        for iteration in range(self.m):
            mcts.search_one_iteration()

        explanation = mcts.best_leaf_node()

        return explanation.node_set, mcts
    
def compute_subgraphx_importance(test_dataset, model,  node_number, device_parm, atom_encoder=None):


    global device

    device = device_parm

    sx_params = {
        "model": model.to('cpu'),
        "num_layers": 3,
        "exp_weight": 5,
        "m": 100,
        "t": 20,
        "task": Task.GRAPH_CLASSIFICATION,
        "max_children": 10,
        "experiment": None,
        "value_func": mc_l_shapley,
    }
    subgraphx = SubgraphX(**sx_params)
    model.to(device).eval()
    if atom_encoder is not None:
        atom_encoder.to('cpu').eval()

    node_importance = []
    for n, data in enumerate(test_dataset):
        data = data.to('cpu')
        if data.y.isnan():
            continue
        with torch.no_grad():
            if atom_encoder is not None:
                x = atom_encoder(data.x)
            else:
                x = data.x

            data.x = x.float()


        explaining_subgraph, mcts = subgraphx(data, n_min=node_number[n] , nodes_to_keep=None, exhaustive=False)
        explanation_set = mcts.best_node(node_number[n]).node_set

        importance = torch.zeros(data.num_nodes)
        importance[list(explanation_set)] = 1  # Mark the nodes in the explanation set as important
        node_importance.append(importance)

    return node_importance
