import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np

def grad_cam(model, data,atom_encoder, device):
        def backward_hook(module, grad_input, grad_output):
            global gradients
            gradients = grad_output[0]

        def forward_hook(module, input, output):
            global activations
            activations = output

        target_layer = model.convs[-1]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        atom_encoder = atom_encoder.to(device)
        model = model.to(device)
        data = data.to(device)

        model.eval()
        x = atom_encoder(data.x)
        output= model(x, data.edge_index, batch=data.batch)
        target_class = output.argmax(dim=1)
        assert target_class.max().item() < output.shape[1]
        loss = torch.nn.functional.cross_entropy(output, target_class)
        model.zero_grad()
        loss.backward()

        # Convert gradients to numpy for visualization
        pooled_gradients = torch.mean(gradients, dim=0)
        for i in range(activations.size(0)):
            activations[i, :] *= pooled_gradients

        # Sum only positive activations
        heatmap = torch.sum(torch.relu(activations), dim=1)
        heatmap -= heatmap.min()
        heatmap /= heatmap.max() + 1e-10

        # heatmap = torch.mean(activations, dim=1)
        # heatmap = torch.nn.functional.relu(heatmap)
        # heatmap /= torch.max(heatmap) + 1e-10
        # Compute the L2 norm of gradients for each node

        # node_importance = torch.norm(activations, p=2, dim=1)
        # # Normalize the node importance scores
        # node_importance -= node_importance.min()
        # node_importance /= node_importance.max() + 1e-10

        # Assign the node importance scores to the heatmap
       

        return heatmap


def motif_grad_cam(model, data, device):
    def backward_hook(module, grad_input, grad_output):
        global gradients
        gradients = grad_output[0]

    def forward_hook(module, input, output):
        global activations
        activations = output

    target_layer = model.convs[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    model.eval()
    data = data.to(device)
    output = model(data)
    target_class = output.argmax(dim=1)
    loss = torch.nn.functional.cross_entropy(output, target_class)
    model.zero_grad()
    loss.backward()

    pooled_gradients = torch.mean(gradients, dim=0)  # F 개 계수
    for i in range(activations.size(0)):
        activations[i, :] *= pooled_gradients  # 각 노드에 계수 곱함

    cliques = data.clique
    clique_importance = []
    for clique in cliques:
        clique_activation = activations[clique, :]

        importance = torch.mean(torch.max(clique_activation, dim=0)[0])
        clique_importance.append(max(0, importance))

    data.clique_importance = [float(importance) for importance in clique_importance]
    node_importance = torch.zeros(activations.size(0))
    for i in range(activations.size(0)):
        importance_sum = 0
        count = 0
        for j, clique in enumerate(cliques):
            if i in clique:
                importance_sum += clique_importance[j]
                count += 1
        if count > 0:
            node_importance[i] = importance_sum / count

    data.node_importance = node_importance
    
    return  data

def score_motif_cam(model, data,device):
    def backward_hook(module, grad_input, grad_output):
        global gradients
        gradients = grad_output[0]

    def forward_hook(module, input, output):
        global activations
        activations = output

    target_layer = model.convs[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    model.eval()
    data = data.to(device)
    output = model(data)
    target_class = output.argmax(dim=1)
    loss = torch.nn.functional.cross_entropy(output, target_class)
    model.zero_grad()
    loss.backward()


    original_x = data.x if data.x is not None else data.feat
    edge_index = data.edge_index
    original_x, edge_index = original_x.to(device), edge_index.to(device)
    original_x = model.atom_encoder(original_x)

    for conv in model.convs:
        original_x = conv(original_x, edge_index)
    original_x = original_x.sum(dim=0, keepdim=True)
    original_logit = model.readout_layer(original_x)
    original_class = original_logit.argmax(dim=1)

    clique_importance = []
    for clique in data.clique:
        modified_x = original_x.clone()
        modified_x[clique, :] = 0  # Set the features of nodes in the clique to 0

        for conv in model.convs:
            modified_x = conv(modified_x, edge_index)
        modified_x = modified_x.sum(dim=0, keepdim=True)
        modified_logit = model.readout_layer(modified_x)

        importance = torch.abs(original_logit[0, original_class] - modified_logit[0, original_class])
        clique_importance.append(importance.item())

    clique_importance = torch.tensor(clique_importance)
    clique_importance = torch.softmax(clique_importance, dim=0).tolist()

    for i, clique in enumerate(data.clique):
        for node in clique:
            activations[node, :] *= clique_importance[i]


    node_importance = torch.zeros(activations.size(0))
    for i in range(activations.size(0)):
        importance_sum = 0
        count = 0
        for j, clique in enumerate(data.clique):
            if i in clique:
                importance_sum += clique_importance[j]
                count += 1
        if count > 0:
            node_importance[i] = importance_sum / count

    data.node_importance = node_importance
    
    return data
