import os
import torch
from sklearn.metrics import f1_score, accuracy_score

def forward_with_noise(model, atom_encoder, data, important_nodes, device):
    atom_encoder = atom_encoder.to(device)
    x = atom_encoder(data.x.to(device))

    if important_nodes is not None:
        important_nodes = important_nodes.clone().detach().to(device)
        x[important_nodes] = 0 # x[important_nodes] + 10 * torch.randn_like(x[important_nodes])
        print("I erase " , len(important_nodes), " nodes")

    for conv in model.convs:
        x = conv(x, data.edge_index.to(device))



    x = x.mean(dim=0, keepdim=True)
    co_logs = model.readout_layer(x)
    return co_logs

def forward_with_import_noise(model, atom_encoder, data, important_nodes , device, args):
    atom_encoder = atom_encoder.to(device)
    x = atom_encoder(data.x.to(device))
    model = model.to(device)
    if important_nodes is not None:
        important_nodes = important_nodes.clone().detach().to(device)
        x[important_nodes] = 0 # x[important_nodes] + 10 * torch.randn_like(x[important_nodes])


    for conv in model.convs:
        x = conv(x, data.edge_index.to(device))



    if args.pooling == 'mean':
        x = x.mean(dim=0, keepdim=True)
    elif args.pooling == 'sum':
        x = x.sum(dim=0, keepdim=True)
    elif args.pooling == 'max':
        x = x.max(dim=0, keepdim=True)[0]
    else:
        raise ValueError(f"Unsupported pooling method: {args.pooling}")


    co_logs = model.readout_layer(x)
    return co_logs


def eval_f1_dataset(model, atom_encoder, dataset, device):
    model.eval()
    atom_encoder = atom_encoder.to(device)
    model = model.to(device)
    correct = 0
    pred, y = torch.tensor([]), torch.tensor([])
    for i, data in enumerate(dataset):


        target = data.y.view(-1)
        mask = ~target.isnan()
        target = target[mask].to(torch.long)
        y = torch.cat([y, target.to('cpu')], dim=0)
        with torch.no_grad():
            x = atom_encoder(data.x.to(device))
            for conv in model.convs:
                x = conv(x, data.edge_index.to(device))
            x = x.mean(dim=0, keepdim=True)
            co_logs = model.readout_layer(x)
            pred = torch.cat([pred, co_logs.detach().to('cpu')], dim=0)
    for i in range(1):  # Assuming num_task is 1 since it's not provided
        mask = ~y.isnan()
        y_pred = pred.max(1)[1]
        correct += f1_score(y[mask], y_pred[mask])
    return correct   # Assuming num_task is 1 since it's not provided


def eval_acc_dataset(model, atom_encoder, dataset, device):
    model.eval()
    atom_encoder = atom_encoder.to(device)
    model = model.to(device)
    correct = 0
    total = 0
    for i, data in enumerate(dataset):
        target = data.y.view(-1)
        mask = ~target.isnan()
        target = target[mask].to(torch.long)
        total += target.size(0)
        with torch.no_grad():
            x = atom_encoder(data.x.to(device))
            for conv in model.convs:
                x = conv(x, data.edge_index.to(device))
            x = x.mean(dim=0, keepdim=True)
            co_logs = model.readout_layer(x)
            y_pred = co_logs.max(1)[1]
            correct += (y_pred[mask].to(device) == target.to(device)).sum().item()
    accuracy = correct / total if total > 0 else 0
    return accuracy



def test_model_with_noise(model, atom_encoder , dataset, device, number_list, importance_list, args):
    baseline_accuracy = eval_acc_dataset(model,atom_encoder,  dataset, device)
    baseline_f1_score = eval_f1_dataset(model,atom_encoder,  dataset, device)
    all_y = torch.tensor([]).to('cpu')
    all_pred = torch.tensor([]).to('cpu')

    for i, data in enumerate(dataset):
        data = data.to(device)
        number= number_list[i]
        important_nodes = importance_list[i]
        noise_nodes = torch.topk(important_nodes, number, largest=True).indices
        with torch.no_grad():

            model_logits = forward_with_import_noise(model, atom_encoder, data, noise_nodes, device,args)

        all_y = torch.cat([all_y, data.y.to('cpu')], dim=0)
        all_pred = torch.cat([all_pred, model_logits.detach().to('cpu')], dim=0)
    
    y_pred = all_pred.max(1)[1]


    acc_score = accuracy_score(all_y, y_pred)
    f1_result = f1_score(all_y, y_pred)



    return baseline_f1_score, f1_result, baseline_f1_score - f1_result 

# Example usage:
# model = ...  # Your model
# dataset = ...  # Your dataset
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# permutation_test(model, dataset, device)
