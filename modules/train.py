from tqdm import tqdm
import copy
import torch
import torch.nn.functional as F
import wandb
from torch_geometric.loader import DataLoader

from sklearn.metrics import roc_auc_score, f1_score
import sys
import os
# 프로젝트 루트 경로를 path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import num_graphs

def train_function(dataset, split_idx, model, atom_encoder, args, device):
    best_valid_auc, best_epoch = 0, 0
    best_model = copy.deepcopy(model)
    best_encoder = copy.deepcopy(atom_encoder)
    
    train_dataset = dataset[split_idx["train"]]
    valid_dataset = dataset[split_idx["valid"]]
    test_dataset = dataset[split_idx["test"]]
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(
        list(model.parameters())+list(atom_encoder.parameters()), 
        lr=args.lr, 
        weight_decay=args.weight_decay)
    
    for epoch in tqdm(range(1, args.epochs + 1)):

        train_loss = train_causal_epoch(
            model, atom_encoder, optimizer, train_loader, device, args)
        if args.eval_metric == 'auc':
            valid_auc = eval_auc_causal(
                model, atom_encoder, valid_loader, device, args)
        elif args.eval_metric == 'acc':
            valid_auc = eval_acc_causal(
                model, valid_loader, device, args)
        else:
            raise Exception("metric option not valid")
        
        try:
            wandb.log({
                'train_loss': train_loss,
                'valid_auc': valid_auc,
            })
        except: pass

        if (valid_auc > best_valid_auc)&(epoch>=10):
            best_valid_auc = valid_auc
            best_epoch = epoch
            best_model = copy.deepcopy(model)
            best_encoder = copy.deepcopy(atom_encoder)
        

    print("syd: Causal | Dataset:[{}] Model:[{}] | Best Valid:[{:.2f}] at epoch [{}]"
            .format(args.dataset,
                    args.model,
                    best_valid_auc * 100, 
                    best_epoch))
    if args.eval_metric == 'auc':
        last_test_auc = eval_auc_causal(
            model, atom_encoder, test_loader, device, args)
        test_auc = eval_auc_causal(
            best_model, best_encoder, test_loader, device, args)
    elif args.eval_metric == 'acc':
        last_test_auc = eval_acc_causal(
            model, test_loader, device, args)
        test_auc = eval_auc_causal(
            best_model, test_loader, device, args)
    else:
        raise Exception("metric option not valid")
    
    print("Dataset:[{}] Model:[{}] | Best Test:[{:.2f}]"
            .format(args.dataset,
                    args.model,
                    test_auc * 100))
    print("Dataset:[{}] Model:[{}] | Final Test:[{:.2f}]"
            .format(args.dataset,
                    args.model,
                    last_test_auc * 100))
    try:
        wandb.log({
            'test_auc': test_auc,
            'last_test_auc': last_test_auc,
        })
    except: pass
    
    return test_auc, best_model, best_encoder

def  train_causal_epoch(model, atom_encoder, optimizer, train_loader, device, args):
    
    model.train()
    total_loss = 0
    for it, data in enumerate(train_loader):
        
        optimizer.zero_grad()
        data = data.to(device)
        target = data.y.view(-1)
        mask = ~target.isnan()
        target = target[mask].to(torch.long)
        x = atom_encoder(data.x)
        logit = model(x, data.edge_index, data.batch)
        
        loss = torch.nn.CrossEntropyLoss()(logit.view(-1,args.num_classes)[mask], target)

        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    
    num = len(train_loader.dataset)
    total_loss = total_loss / num
    return total_loss

def eval_auc_causal(model, atom_encoder, loader, device, args):
    
    model.eval()
    correct = 0
    pred, y = torch.tensor([]), torch.tensor([])
    for data in loader:
        data = data.to(device)
        y = torch.cat([y, data.y.view(-1, args.num_task).to('cpu')], dim=0)
        with torch.no_grad():
            x = atom_encoder(data.x)
            co_logs = model(x, data.edge_index, data.batch)
            pred = torch.cat([pred, co_logs.detach().to('cpu')], dim=0)
    for i in range(args.num_task):
        mask = ~y[:,i].isnan()
        if args.num_classes == 2:
            y_score = F.softmax(pred[:,i*2:i*2+2], dim=-1)
            correct += roc_auc_score(y[mask,i], y_score[mask,1])
        else:
            y_score = F.softmax(pred[:, i*args.num_classes:(i+1)*args.num_classes], dim=-1)
            correct += roc_auc_score(y[mask,i], y_score, multi_class='ovr')
            
    return correct / args.num_task

def eval_acc_causal(model, loader, device, args):
    
    model.eval()
    pred, y = torch.tensor([]), torch.tensor([])
    for data in loader:
        data = data.to(device)
        y = torch.cat([y, data.y.view(-1).to('cpu')], dim=0)
        with torch.no_grad():
            co_logs = model(data)
            co_logs = co_logs.view(-1, args.num_classes).max(1)[1]
            pred = torch.cat([pred, co_logs.detach().to('cpu')], dim=0)

    correct = pred.eq(y).sum().item()

    acc_co = correct /len(loader.dataset) / args.num_task
    return acc_co

def eval_f1(model, loader, device, args):
    
    model.eval()
    correct = 0
    pred, y = torch.tensor([]), torch.tensor([])
    for data in loader:
        data = data.to(device)
        y = torch.cat([y, data.y.view(-1, args.num_task).to('cpu')], dim=0)
        with torch.no_grad():
            co_logs = model(data)
            pred = torch.cat([pred, co_logs.detach().to('cpu')], dim=0)
    for i in range(args.num_task):
        mask = ~y[:,i].isnan()
        y_pred = pred[:,i*2:i*2+2].max(1)[1]
        correct += f1_score(y[mask,i], y_pred[mask])
            
    return correct / args.num_task

###############


def train_function_normal(dataset, split_idx, model, args, device):
    best_valid_auc, best_epoch = 0, 0
    best_model = copy.deepcopy(model)
    
    train_dataset = dataset[split_idx["train"]]
    valid_dataset = dataset[split_idx["valid"]]
    test_dataset = dataset[split_idx["test"]]
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    for epoch in tqdm(range(1, args.epochs + 1)):

        train_loss = train_epoch_normal(model, optimizer, train_loader, device, args)
        if args.eval_metric == 'auc':
            valid_auc = eval_auc_normal(model, valid_loader, device, args)
        elif args.eval_metric == 'acc':
            valid_auc = eval_acc_normal(model, valid_loader, device, args)
        else:
            raise Exception("metric option not valid")
        
        try:
            wandb.log({
                'train_loss': train_loss,
                'valid_auc': valid_auc,
            })
        except: pass

        if (valid_auc > best_valid_auc) and (epoch >= 10):
            best_valid_auc = valid_auc
            best_epoch = epoch
            best_model = copy.deepcopy(model)

    print("Dataset:[{}] Model:[{}] | Best Valid:[{:.2f}] at epoch [{}]"
          .format(args.dataset, args.model, best_valid_auc * 100, best_epoch))

    if args.eval_metric == 'auc':
        last_test_auc = eval_auc_normal(model, test_loader, device, args)
        test_auc = eval_auc_normal(best_model, test_loader, device, args)
    elif args.eval_metric == 'acc':
        last_test_auc = eval_acc_normal(model, test_loader, device, args)
        test_auc = eval_acc_normal(best_model, test_loader, device, args)
    else:
        raise Exception("metric option not valid")

    print("Dataset:[{}] Model:[{}] | Best Test:[{:.2f}]".format(args.dataset, args.model, test_auc * 100))
    print("Dataset:[{}] Model:[{}] | Final Test:[{:.2f}]".format(args.dataset, args.model, last_test_auc * 100))
    try:
        wandb.log({
            'test_auc': test_auc,
            'last_test_auc': last_test_auc,
        })
    except: pass

    return test_auc, best_model

# Functions for datasets that do not require atom encoder and use float features directly (e.g., UPFD)
def train_epoch_normal(model, optimizer, train_loader, device, args):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        target = data.y.view(-1)
        mask = ~target.isnan()
        target = target[mask].to(torch.long)
        logit = model(data.x, data.edge_index, data.batch)
        loss = torch.nn.CrossEntropyLoss()(logit.view(-1, args.num_classes)[mask], target)
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    num = len(train_loader.dataset)
    total_loss = total_loss / num
    return total_loss

def eval_auc_normal(model, loader, device, args):
    model.eval()
    pred, y = torch.tensor([]), torch.tensor([])
    for data in loader:
        data = data.to(device)
        y = torch.cat([y, data.y.view(-1, args.num_task).to('cpu')], dim=0)
        with torch.no_grad():
            co_logs = model(data.x, data.edge_index, data.batch)
            pred = torch.cat([pred, co_logs.detach().to('cpu')], dim=0)
    correct = 0
    for i in range(args.num_task):
        mask = ~y[:, i].isnan()
        if args.num_classes == 2:
            y_score = F.softmax(pred[:, i*2:i*2+2], dim=-1)
            correct += roc_auc_score(y[mask, i], y_score[mask, 1])
        else:
            y_score = F.softmax(pred[:, i*args.num_classes:(i+1)*args.num_classes], dim=-1)
            correct += roc_auc_score(y[mask, i], y_score, multi_class='ovr')
    return correct / args.num_task

def eval_acc_normal(model, loader, device, args):
    model.eval()
    pred, y = torch.tensor([]), torch.tensor([])
    for data in loader:
        data = data.to(device)
        y = torch.cat([y, data.y.view(-1).to('cpu')], dim=0)
        with torch.no_grad():
            co_logs = model(data.x, data.edge_index, data.batch)
            co_logs = co_logs.view(-1, args.num_classes).max(1)[1]
            pred = torch.cat([pred, co_logs.detach().to('cpu')], dim=0)
    correct = pred.eq(y).sum().item()
    acc = correct / len(loader.dataset) / args.num_task
    return acc
