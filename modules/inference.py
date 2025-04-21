#%%
import os
import wandb
from copy import copy
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
import random
# import matplotlib
# matplotlib.use('module://matplotlib_inline.backend_inline')
#%%
def fidelity(model, dataset, sparsity, args):
    #%%
    data_loader = DataLoader(dataset, 256, shuffle=False)
    fidelity_plus = 0
    fidelity_minus = 0
    model.eval()
    with torch.no_grad():
        for it, data in enumerate(data_loader):
            data = data.to(args.device)
            original_pred = model(data, silence_node=[])

            att_nodes = []
            st = 0
            for i in range(data.num_graphs):
                k = round(torch.sum(data.batch==i).item()*(1-sparsity))
                num_nodes = torch.sum(data.batch==i).item()
                att_nodes.extend(random.sample(range(st, st+num_nodes), k))
                st += torch.sum(data.batch==i).item()
            if len(att_nodes)==0: continue
           
            trivial_node = list(set(range(len(data.x))) - set(att_nodes))
            
            perturbed_pred = model(data, silence_node=[att_nodes])
            fidelity_plus += (data.y.view(-1) == original_pred.view(-1, args.num_classes).max(dim=1)[1]).sum().item()
            fidelity_plus -= (data.y.view(-1) == perturbed_pred.view(-1, args.num_classes).max(dim=1)[1]).sum().item()
            perturbed_pred = model(data, silence_node=[trivial_node])
            fidelity_minus += (data.y.view(-1) == original_pred.view(-1, args.num_classes).max(dim=1)[1]).sum().item()
            fidelity_minus -= (data.y.view(-1) == perturbed_pred.view(-1, args.num_classes).max(dim=1)[1]).sum().item()
    #%%
    return fidelity_plus / len(dataset)/args.num_task, fidelity_minus / len(dataset)/args.num_task
#%%

atom_dic = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne',
    11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar',
    19: 'K', 20: 'Ca'}

def visualize(graph, path=None, color=None, edge_color=None):
    if color==None:
        color=torch.arange(len(graph.x))
    G = to_networkx(graph, to_undirected=True)
    fig = plt.figure()
    nx.draw_networkx(G, pos=nx.kamada_kawai_layout(G), alpha=0.7, node_size=200 ,with_labels=False,
                         node_color=color, edge_color=edge_color)
    
    label_list = [atom_dic.get(a.item(), 'X') for a in graph.x[:,0]]
    labels = {node: label for node, label in zip(G.nodes(), label_list)}
    nx.draw_networkx_labels(G, nx.kamada_kawai_layout(G), labels, font_size=12, font_color='black')
    
    if path is not None:
        plt.savefig(path)
        try: wandb.log({'image': wandb.Image(path)})
        except: pass
        plt.close()
    else:
        plt.show()
        # plt.close()
    return fig
# visualize(graph, './')
#%%
def mol_visualize(model, dataset, vis_num, path, args):
    # loader = DataLoader(dataset, 1, shuffle=False)
    if isinstance(vis_num, tuple):
        st, end = vis_num
        vis_num = range(st, end)
    model.eval()
    for j in vis_num:
        # if j == vis_num: break
        graph = Batch.from_data_list([dataset[j]])

        att_score, edge_score = model(graph.to(args.device), infer=True)
        color = ['blue' if item<0.5 else 'yellow' for item in att_score[:,1]]
        visualize(graph, path + f'{j}.png', color)
        
def mol_visualize(model, dataset, vis_num, path, args):
    if isinstance(vis_num, tuple):
        st, end = vis_num
        vis_num = range(st, end)
    model.eval()
    for j in vis_num:
        graph = Batch.from_data_list([dataset[j]])
        ncgc_id = dataset[j].id
        smiles = dataset[j].smiles
        original = copy(graph)
        att_score, _ = model(graph.to(args.device), infer=True)
        _, _, pred = model(graph.to(args.device))
        pred = torch.max(pred, dim=1)[1]
        
        if pred.item() == 1: path_ = path+'True/' 
        else: path_ = path+'False/' 
        
        mol = Chem.MolFromSmiles(smiles)
        Draw.MolToFile(mol, path_+f'{ncgc_id}_mol.png')
        color = ['blue' if item<0.5 else 'yellow' for item in att_score[:,1]]

        visualize(original, path_ + f'{ncgc_id}.png', color)
# %%
def sparsity(model, dataset, args):
    data_loader = DataLoader(dataset, 1, shuffle=False)
    att_node = 0
    num_node = 0
    model.eval()
    with torch.no_grad():
        for it, data in enumerate(data_loader):
            data = data.to(args.device)
            att_score, edge_score = model(data, infer=True)

            att_node += (att_score[:,1]>0.5).sum().item()
            num_node += len(att_score)
            
    return 1 - att_node/num_node
# %%
def unjust_ratio(model, dataset, args):
    data_loader = DataLoader(dataset, 1, shuffle=False)
    model.eval()
    with torch.no_grad():
        right_node = 0
        wrong_node = 0
        for it, data in enumerate(data_loader):
            data = data.to(args.device)
            att_score, edge_score = model(data, infer=True)
            if data.y.item() == 0:
                real_cause = list(range(len(data.x)-5,len(data.x)))
            else:
                real_cause = list(range(len(data.x)-6,len(data.x)))
                
            att_motif = torch.where(att_score[:,1]>0.5)[0]
            for i in att_motif:
                if set(data.clique[0][i]).isdisjoint(set(real_cause)):
                    wrong_node += len(data.clique[0][i])
                else:
                    unjust_node = set(data.clique[0][i])-set(real_cause)
                    right_node += len(unjust_node)
    return right_node/(right_node+wrong_node)
# %%
# model = best_model
# data_loader = DataLoader(dataset, 128, shuffle=False)
# model.eval()
# taus = torch.tensor([]).to(args.device)
# with torch.no_grad():
#     for it, data in enumerate(data_loader):
#         data = data.to(args.device)
#         tau = model(data, tau_check = True)
#         taus = torch.cat((taus, tau), dim=0)
# #%%        
# plt.hist(taus.to('cpu').numpy(), bins=30)
# plt.show()
def make_subgraph(data, att_nodes, args):
    trivial_node = list(set(range(len(data.x))) - set(att_nodes))
    subgraph = data.to_data_list()[0].clone()
    edge_index = subgraph.edge_index
    new_edge_index = torch.tensor([], device=args.device, dtype=torch.long)
    for edge in edge_index.T:
        if torch.all(edge.view(-1,1) != torch.tensor(att_nodes, device=args.device)):
            new_edge_index = torch.cat([new_edge_index, edge], dim=0)
            
    maping = {node: i for i, node in enumerate(trivial_node)}
    subgraph.edge_index = torch.tensor(list(map(lambda x: maping[x.item()], new_edge_index)),
                                    device=args.device).view(-1,2).T
    if subgraph.edge_index.size(1) == 0:
        subgraph.edge_index = torch.tensor([[0],[0]],device=args.device)
    
    new_clique = []
    deleted_clique = []
    for i, clique in enumerate(subgraph.clique):
        clique = [node for node in clique if node not in att_nodes]
        if len(clique) > 0:
            clique = list(map(lambda x: maping[x], clique))
            new_clique.append(clique)
        else:
            deleted_clique.append(i)
    subgraph.clique = new_clique
    
    pool_edge_index = subgraph.pool_edge_index
    new_edge_index = torch.tensor([], device=args.device, dtype=torch.long)
    for edge in pool_edge_index.T:
        if torch.all(edge.view(-1,1) != torch.tensor(deleted_clique, device=args.device)):
            new_edge_index = torch.cat([new_edge_index, edge], dim=0)
            
    live_clique = list(set(range(len(data.clique[0]))) - set(deleted_clique))        
    maping = {node: i for i, node in enumerate(live_clique)}
    subgraph.pool_edge_index = torch.tensor(list(map(lambda x: maping[x.item()], new_edge_index)),
                                    device=args.device).view(-1,2).T
    if subgraph.pool_edge_index.size(1) == 0:
        subgraph.pool_edge_index = torch.tensor([[0],[0]],device=args.device)
    
    subgraph.x = subgraph.x[trivial_node]
    subgraph.num_nodes = len(subgraph.x)
    subgraph = Batch.from_data_list([subgraph])
    return subgraph
# %%
# def fidelity(model, dataset, sparsity, args):
#     data_loader = DataLoader(dataset, 1, shuffle=False)
#     fidelity_plus = 0
#     fidelity_minus = 0
#     model.eval()
#     with torch.no_grad():
#         for it, data in enumerate(data_loader):
#             if data.num_nodes < 2 : continue
#             data = data.to(args.device)
#             att_score, original_pred = model(data, silence_node=[])
            
#             top_k = round(data.num_nodes*(1-sparsity))
#             if top_k==data.num_nodes or top_k==0: continue
#             att_nodes = []
#             if args.pool == 'noMotif':
#                 att_nodes = torch.topk(att_score[:,1], top_k)[1].tolist()
#             else:
#                 temp_att = torch.zeros(data.num_nodes)
#                 for i, motif in enumerate(data.clique[0]):
#                     temp_att[motif] = att_score[i,1]
#                 att_nodes = torch.topk(temp_att, top_k)[1].tolist()
            
#             subgraph = make_subgraph(data, att_nodes, args)
#             #%%
#             _, perturbed_pred = model(subgraph, silence_node=[])
#             #%%
#             fidelity_plus += (data.y.view(args.num_task) == original_pred.view(args.num_task, -1).max(dim=1)[1]).sum().item()
#             fidelity_plus -= (data.y.view(args.num_task) == perturbed_pred.view(args.num_task,-1).max(dim=1)[1]).sum().item()
            
#             trivial_node = list(set(range(len(data.x))) - set(att_nodes))
#             subgraph = make_subgraph(data, trivial_node, args)
            
#             _, perturbed_pred = model(subgraph, silence_node=[])
#             fidelity_minus += (data.y.view(args.num_task) == original_pred.view(args.num_task,-1).max(dim=1)[1]).sum().item()
#             fidelity_minus -= (data.y.view(args.num_task) == perturbed_pred.view(args.num_task,-1).max(dim=1)[1]).sum().item()
#     return fidelity_plus / len(dataset) / args.num_task , fidelity_minus / len(dataset) / args.num_task
#%%
# def fidelity_prob(model, dataset, sparsity, args):
#     data_loader = DataLoader(dataset, 1, shuffle=False)
#     fidelity_plus = 0
#     fidelity_minus = 0
#     model.eval()
#     with torch.no_grad():
#         for it, data in enumerate(data_loader):
#             data = data.to(args.device)
#             att_score, original_pred = model(data, silence_node=[])

#             top_k = round(data.num_nodes*(1-sparsity))
#             att_nodes = []
#             if args.pool == 'noMotif':
#                 att_nodes = torch.topk(att_score[:,1], top_k)[1].tolist()
#             else:
#                 temp_att = torch.zeros(data.num_nodes)
#                 for i, motif in enumerate(data.clique[0]):
#                     temp_att[motif] = att_score[i,1]
#                 att_nodes = torch.topk(temp_att, top_k)[1].tolist()
#             trivial_node = set(range(len(data.x))) - set(att_nodes)
#             trivial_node = torch.tensor(list(trivial_node), dtype=torch.long).to(args.device)
            
#             _, perturbed_pred = model(data, silence_node=[att_nodes])
#             fidelity_plus += (F.softmax(original_pred, dim=-1) - F.softmax(perturbed_pred, dim=-1))[:,1].item()
#             _, perturbed_pred = model(data, silence_node=[trivial_node])
#             fidelity_minus += (F.softmax(original_pred, dim=-1) - F.softmax(perturbed_pred, dim=-1))[:,1].item()
#     return fidelity_plus / len(dataset), fidelity_minus / len(dataset)
# %%
