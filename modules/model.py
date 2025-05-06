import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU, Dropout
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn import GCNConv, GINConv, GATConv, SAGEConv

from ogb.utils.features import get_atom_feature_dims

full_atom_feature_dims = get_atom_feature_dims()

class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()
        
        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim+1, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:,i])

        return x_embedding

# class AtomEncoder(torch.nn.Module):

#     def __init__(self, emb_dim):
#         super(AtomEncoder, self).__init__()
        
#         self.atom_mlp_list = torch.nn.ModuleList()
#         for i, dim in enumerate(full_atom_feature_dims):
            
#             mlp = nn.Sequential(
#                 nn.Linear(1, 2 * emb_dim),  # 🔥 dim+1 → 1
#                 nn.ReLU(),
#                 nn.Linear(2 * emb_dim, emb_dim)
#             )
                    
#             torch.nn.init.xavier_uniform_(mlp[0].weight)  # 첫 번째 레이어 초기화
#             torch.nn.init.xavier_uniform_(mlp[2].weight)  # 두 번째 레이어 초기화
#             self.atom_mlp_list.append(mlp)

#     def forward(self, x):
#         x_embedding = 0
#         for i in range(x.shape[1]):
#             x_embedding += self.atom_mlp_list[i](x[:, i].unsqueeze(-1).float())  # MLP에 입력 전달

#         return x_embedding


class MLP_layer(nn.Module):
    def __init__(self, hidden, hidden_out):
        super().__init__()
        self.fc1_bn = BatchNorm1d(hidden)
        self.fc1 = Linear(hidden, hidden)
        self.fc2_bn = BatchNorm1d(hidden)
        self.fc2 = Linear(hidden, hidden_out)
        
    def forward(self, x):
        # x = self.fc1_bn(x)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.fc2_bn(x)
        x = self.fc2(x)
        return x

class GNN_layer(nn.Module):
    def __init__(self, model, hidden, dropout):
        super().__init__()    
        self.bn = BatchNorm1d(hidden)
        if model == 'GCN':
            self.conv = GCNConv(hidden, hidden)
        elif model == 'GIN':
            self.conv = GINConv(
                Sequential(
                    Linear(hidden, hidden),
                    ReLU(),
                    Linear(hidden, hidden)
                )
            )
        elif model == 'GAT':
            self.conv = GATConv(hidden, hidden, heads=4, concat=False)
        elif model == 'SAGE':
            self.conv = SAGEConv(hidden, hidden)
            
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        # x = self.bn(x)
        x = self.conv(x, edge_index)
        # x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x

class BasicGNN(nn.Module):
    def __init__(self, args, dropout=0.5):
        super().__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args

        if args.pooling == 'max':
            self.global_pool = global_max_pool
        elif args.pooling == 'mean':
            self.global_pool = global_mean_pool
        elif args.pooling == 'sum':
            self.global_pool = global_add_pool
        else:
            raise ValueError(f"Unsupported pooling type: {args.pooling}")
        
        self.dropout = dropout
        # self.atom_encoder = AtomEncoder(hidden)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_conv_layers):
            self.convs.append(
                GNN_layer(args.model, hidden, dropout))
            
        self.readout_layer = MLP_layer(hidden, args.num_classes * args.num_task)
        self.flatten_out_shape = True
        
    def forward(self, x, edge_index, batch=None, infer=False, silence_node=None):
        # x = self.atom_encoder(x)
        
        if silence_node is not None: 
            x[silence_node] = torch.zeros_like(x[silence_node])
        
        # 그래프 전체 representation 생성
        for conv in self.convs:
            x = conv(x, edge_index)        

        if infer:
            return x
        
        if batch is not None:
            x = self.global_pool(x, batch)
        else:
            if self.args.pooling == 'max':
                x = x.max(dim=0, keepdim=True)[0]  # Batch가 없을 경우 max pooling
            elif self.args.pooling == 'mean':
                x = x.mean(dim=0, keepdim=True)  # Batch가 없을 경우 mean pooling
            elif self.args.pooling == 'sum':
                x = x.sum(dim=0, keepdim=True)  # Batch가 없을 경우 sum pooling
        
        logit = self.readout_layer(x)
        if self.flatten_out_shape:
            return logit
        else:
            return logit.reshape(-1, self.args.num_task, self.args.num_classes)
        

        

# BasicGNN_MLP for float features (e.g., UPFD dataset)
class BasicGNN_MLP(nn.Module):
    def __init__(self, args, dropout=0.5):
        super().__init__()
        num_conv_layers = args.layers
        hidden = args.hidden
        self.args = args

        if args.pooling == 'max':
            self.global_pool = global_max_pool
        elif args.pooling == 'mean':
            self.global_pool = global_mean_pool
        elif args.pooling == 'sum':
            self.global_pool = global_add_pool
        else:
            raise ValueError(f"Unsupported pooling type: {args.pooling}")
        
        self.dropout = dropout
        self.mlp_proj = nn.Sequential(
            nn.Linear(args.input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden)
        )
        self.convs = torch.nn.ModuleList()
        for _ in range(num_conv_layers):
            self.convs.append(
                GNN_layer(args.model, hidden, dropout))
        
        self.readout_layer = MLP_layer(hidden, args.num_classes * args.num_task)
        self.flatten_out_shape = True

    def forward(self, x, edge_index, batch=None, infer=False, silence_node=None):
        x = self.mlp_proj(x)

        if silence_node is not None: 
            x[silence_node] = torch.zeros_like(x[silence_node])
        
        for conv in self.convs:
            x = conv(x, edge_index)

        if infer:
            return x
        
        if batch is not None:
            x = self.global_pool(x, batch)
        else:
            if self.args.pooling == 'max':
                x = x.max(dim=0, keepdim=True)[0]
            elif self.args.pooling == 'mean':
                x = x.mean(dim=0, keepdim=True)
            elif self.args.pooling == 'sum':
                x = x.sum(dim=0, keepdim=True)

        logit = self.readout_layer(x)
        if self.flatten_out_shape:
            return logit
        else:
            return logit.reshape(-1, self.args.num_task, self.args.num_classes)