#%%
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import importlib
import torch
import wandb
from torch_geometric.loader import DataLoader
from graph.utils.utils import set_seed
dataset_module = importlib.import_module('dataset')
importlib.reload(dataset_module)
model_module = importlib.import_module('model')
importlib.reload(model_module)
train_module = importlib.import_module('train')
importlib.reload(train_module)
inference = importlib.import_module('inference')
importlib.reload(inference)
#%%
parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--hidden', type=int, default=256)

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0)

parser.add_argument('--model', type=str, default="GIN", help="GCN, GIN, GAT")

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--dataset', type=str, default='tox21',
                        help='name of dataset. For now, only classification.')
parser.add_argument('--eval_metric', type=str, default='auc')


parser.add_argument('--target_col', type=int, default=14)

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device

name = f'{args.dataset}_{args.model}'
wandb.init(project = f'Basic_GNN')
wandb.run.name = name
wandb.run.save()
wandb.config.update(args)

#%%
def main(args):
    set_seed(args.seed)

    if args.dataset.upper()=='MUTAG':
        dataset, split_idx = dataset_module.get_MUTAG()
    # elif args.dataset.upper()=='TOX21':
    #     dataset, split_idx = dataset_module.get_Tox21Data(args.target_col)    
    else:
        dataset, split_idx = dataset_module.get_MolculeNetData(args.dataset) 
    # else:
    #     if args.dataset.upper()=='SYN':
    #         dataset, split_idx = dataset_module.get_SynData(args)
    #     else: 
    #         dataset, split_idx = dataset_module.get_GeneralPoolData(args.dataset)
    args.num_task = dataset[0].y.view(1,-1).shape[1]
    # args.num_classes = dataset.num_classes
    args.num_classes = 2
    
    model = model_module.BasicGNN(args).to(device)
    
    test_acc, best_model = train_module.train_function(
        dataset, split_idx, model, args, device)
    
    test_dataset = dataset[split_idx["test"]]
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)
    f1 = train_module.eval_f1(model, test_loader, device, args)
    
    model_dir = f"./assets/model/{args.dataset}/target_{args.target_col}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_name = f"{name}_{args.seed}"
    torch.save(best_model.state_dict(), f"./{model_dir}/{model_name}.pt")
    try:
        artifact = wandb.Artifact(
            f'GCN_{args.dataset}', 
            type='model',
            metadata=vars(args))
        artifact.add_file(f'./{model_dir}/{model_name}.pt')
        wandb.log_artifact(artifact)
    except: pass
    
    return test_acc, f1
#%%
result_name = ['AUC', 'F1']

result = []
for i in range(10):
    args.seed = i
    res = main(args)
    result.append([*res])
result = torch.tensor(result)*100
res_mean = result.mean(dim=0)
res_std = result.std(dim=0)
print("=" * 150)
print(f'{args.dataset}')
for i in range(len(res_mean)):    
    print(f'& ${res_mean[i]:.2f}_{{\\pm {res_std[i]:.2f}}}$ ', end='')
print('')
print("=" * 150)   
try:
    for i in range(len(res_mean)): 
        wandb.log({
            f'{result_name[i]}_mean': res_mean[i],
            f'{result_name[i]}_std': res_std[i]})
except:pass 
#%%
wandb.run.finish()
#%%
