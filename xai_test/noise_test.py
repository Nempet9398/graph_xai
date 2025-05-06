import os
import torch
from sklearn.metrics import f1_score, accuracy_score

def forward_with_noise(model, data, important_nodes):
    
    co_logs = model(data.x, data.edge_index, silence_node = important_nodes)


    return co_logs

def forward_with_atom_noise(model, atom_encoder, data, important_nodes , device, args):

    x = atom_encoder(data.x.to(device))
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


def eval_f1_dataset(model,  dataset, device, args,atom_encoder=None):
    model.eval()
    
    if atom_encoder is not None:
        atom_encoder.eval()
        atom_encoder = atom_encoder.to(device)
    model = model.to(device)
    correct = 0
    pred, y = torch.tensor([]), torch.tensor([])
    for i, data in enumerate(dataset):

        data = data.to(device)
        target = data.y.view(-1)
        mask = ~target.isnan()
        target = target[mask].to(torch.long)
        y = torch.cat([y, target.to('cpu')], dim=0)
        with torch.no_grad():
            if atom_encoder is not None:
                x = atom_encoder(data.x)

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
                pred = torch.cat([pred, co_logs.detach().to('cpu')], dim=0)

            else:
                co_logs = model(data.x, data.edge_index)
                pred = torch.cat([pred, co_logs.detach().to('cpu')], dim=0)

    for i in range(1):  # Assuming num_task is 1 since it's not provided
        mask = ~y.isnan()
        y_pred = pred.max(1)[1]
        if args.dataset.upper() =='ENZYME':
            correct += f1_score(y[mask],y_pred[mask],average='macro')
        else:
            correct += f1_score(y[mask], y_pred[mask])
    return correct  # Assuming num_task is 1 since it's not provided


def eval_acc_dataset(model, atom_encoder, dataset, device, args):
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
            if args.pooling == 'mean':
                x = x.mean(dim=0, keepdim=True)
            elif args.pooling == 'sum':
                x = x.sum(dim=0, keepdim=True)
            elif args.pooling == 'max':
                x = x.max(dim=0, keepdim=True)[0]
            else:
                raise ValueError(f"Unsupported pooling method: {args.pooling}")
            co_logs = model.readout_layer(x)
            y_pred = co_logs.max(1)[1]
            correct += (y_pred[mask].to(device) == target.to(device)).sum().item()
    accuracy = correct / total if total > 0 else 0
    return accuracy



def test_model_with_noise(model,   dataset, device, number_list, importance_list, args, atom_encoder=None):
    # baseline_accuracy = eval_acc_dataset(model,  dataset, device,args,atom_encoder=None)
    baseline_f1_score = eval_f1_dataset(model,  dataset, device,args, atom_encoder=atom_encoder)
    all_y = torch.tensor([]).to('cpu')
    all_pred = torch.tensor([]).to('cpu')

    if atom_encoder is not None:
        atom_encoder = atom_encoder.to(device)
    
    model = model.to(device)


    for i, data in enumerate(dataset):
        data = data.to(device)
        number= number_list[i]
        important_nodes = importance_list[i]
        noise_nodes = torch.topk(important_nodes, number, largest=True).indices
        with torch.no_grad():
            if atom_encoder is not None:
                model_logits = forward_with_atom_noise(model, atom_encoder, data, noise_nodes, device,args)
            else:
                model_logits = forward_with_noise(model, data, noise_nodes)
                # print(model_logits)
        all_y = torch.cat([all_y, data.y.to('cpu')], dim=0)
        all_pred = torch.cat([all_pred, model_logits.detach().to('cpu')], dim=0)
    
    y_pred = all_pred.max(1)[1]

    if args.dataset.upper() == 'ENZYME':
        f1_result = f1_score(all_y, y_pred, average='macro')
    else:
        f1_result = f1_score(all_y, y_pred)



    return baseline_f1_score, f1_result, baseline_f1_score - f1_result 

