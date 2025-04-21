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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import networkx as nx
import seaborn as sns
import numpy as np

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
#python main.py  --epochs 300 --model GCN --dataset MUTAG --top_motif 1 --layers 3 --pooling mean

# ============================ #
#       Argument Parsing       #
# ============================ #

parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--hidden', type=int, default=256)

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0)

parser.add_argument('--model', type=str, default="GCN", help="GCN, GIN, GAT")
parser.add_argument('--pooling', type=str, default="mean", help="mean, max, sum")

parser.add_argument('--dataset', type=str, default='MUTAG', help='name of dataset. For now, only classification.')
parser.add_argument('--eval_metric', type=str, default='auc')

parser.add_argument('--group_param', 
                    type=lambda s: [int(item) for item in s.split(',')], 
                    default=[1, 10, 50, 100, 150, 200, 250],
                    help='Comma-separated list of group regularization parameters')

parser.add_argument('--lasso_param', 
                    type=lambda s: [float(item) for item in s.split(',')], 
                    default=[0.01, 0.1, 1, 3, 5, 10, 50],
                    help='Comma-separated list of lasso regularization parameters')

parser.add_argument('--top_motif', type=int, default=1)


try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])
#%%
# ============================ #
#      Device Configuration    #
# ============================ #

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
args.device = device

name = f'{args.dataset}_{args.model}'

# ============================ #
#         Main Function        #
# ============================ #

def main(args):

    for code_seed in range(5):
        
        set_seed(code_seed)

        # ----- Dataset Loading and Preparation -----
        if args.dataset.upper() == 'MUTAG':
            dataset, split_idx = dataset_module.get_MUTAG()
        elif args.dataset.upper() == 'TOX21V2':
            dataset, split_idx = dataset_module.get_Tox21Data()    
        elif args.dataset.lower() == 'alkane':
            dataset, split_idx = dataset_module.get_alkane()
        elif args.dataset.lower() == 'benzene':
            dataset, split_idx = dataset_module.get_benzene()
        elif args.dataset.lower() == 'ames':
            dataset, split_idx = dataset_module.get_ames()
        elif args.dataset.lower() == 'fluoride':
            dataset, split_idx = dataset_module.get_fluoride()
        else:
            dataset, split_idx = dataset_module.get_MolculeNetData(args.dataset)       

        args.num_task = dataset[0].y.view(1, -1).shape[1]
        args.num_classes = 2

        # ----- Model and Atom Encoder Loading or Training -----
        model_name = f'{args.model}_{args.dataset}_{args.pooling}_{code_seed}_model.pt'
        atom_encoder_name = f'{args.model}_{args.dataset}_{code_seed}_atom_encoder.pt'
        model_path = os.path.join('model', 'graph', model_name)
        atom_encoder_path = os.path.join('model', 'atom_encoder', atom_encoder_name)

        # Initialize wandb
        wandb.init(
            project="GRAPH_XAI_GROUP",
            name=f"{args.dataset}_{args.model}_{args.pooling}_seed_{code_seed}",
            config={
                "dataset": args.dataset,
                "model": args.model,
                "pooling": args.pooling,
                "batch_size": args.batch_size,
                "layers": args.layers,
                "hidden": args.hidden,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "eval_metric": args.eval_metric,
                "top_motif": args.top_motif,
                "seed": code_seed,
            }
        )

        # Check if the model and atom encoder exist
        if os.path.exists(model_path) and os.path.exists(atom_encoder_path):
            print(f"Loading existing model and atom encoder from {model_path} and {atom_encoder_path}")
            atom_encoder = torch.load(atom_encoder_path).to(device)
            best_model = torch.load(model_path).to(device)
        else:
            print("Model and atom encoder not found. Creating new ones.")
            atom_encoder = model_module.AtomEncoder(args.hidden).to(device)
            model = model_module.BasicGNN(args).to(device)

            # Train the model
            test_auc, best_model, best_encoder = train_module.train_function(
                dataset, split_idx, model, atom_encoder, args, device
            )

            # Save the trained model and atom encoder
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.makedirs(os.path.dirname(atom_encoder_path), exist_ok=True)

            torch.save(best_model, model_path)
            torch.save(best_encoder, atom_encoder_path)
            print(f"Model and atom encoder saved to {model_path} and {atom_encoder_path}")

        train_dataset = dataset[split_idx['train']]
        valid_dataset = dataset[split_idx['valid']]
        test_dataset = dataset[split_idx['test']]

        # ----- Node Embedding Computation -----
        embedding_list = [
            best_model(atom_encoder(data.x.to(device)).to(device), data.edge_index.to(device), infer=True).detach() 
            for data in test_dataset
        ]

        # ----- Group Lasso and Lasso Importance -----
        group_alpha_importance, best_reg_2, group_base, group_average, group_drop, best_node_number = optimize.find_best_group_alpha_importance(
            test_dataset=test_dataset,
            best_model=best_model,
            atom_encoder=atom_encoder,
            embedding_list=embedding_list,
            device=device,
            args=args,
            test_func=test_module.test_model_with_noise,
            reg_2_list= args.group_param,
        )

        # ----- Alpha Importance -----
        alpha_importance, best_reg_1, alpha_base, alpha_avg, alpha_drop = optimize.find_best_alpha_importance(
            test_dataset=test_dataset,
            best_model=best_model,
            atom_encoder=atom_encoder,
            embedding_list=embedding_list,
            device=device,
            args=args,
            node_number=best_node_number,
            test_func=test_module.test_model_with_noise,
            reg_1_list=args.lasso_param,
        )

        # ----- Explanation Method Importance Calculation -----
        pg_node_importances = optimize.compute_pgexplainer_importance(test_dataset, best_model, atom_encoder)

        graph_mask_importance = optimize.compute_graphmask_importance(test_dataset, best_model, atom_encoder)

        sa_importance = optimize.compute_captum_importance(test_dataset, best_model, atom_encoder, 'Saliency')

        gbp_importance = optimize.compute_captum_importance(test_dataset, best_model, atom_encoder, 'GuidedBackprop')

        subgraphx_importance = optimize.compute_subgraphx_importance(test_dataset, best_model, atom_encoder, best_node_number, device)

        atom_encoder = atom_encoder.to(device)
        best_model = best_model.to(device)
        gnn_explainer_importance = optimize.compute_gnnexplainer_importance(test_dataset, best_model, atom_encoder, device)
    
        gradcam_importance = optimize.compute_gradcam_importance(test_dataset, best_model, atom_encoder, device)


        # ----- Robustness Testing with Noise Injection -----      
        grad_base, grad_avg, grad_drop = test_module.test_model_with_noise(best_model, atom_encoder, test_dataset, device, best_node_number, gradcam_importance, args)
        gnn_base, gnn_avg, gnn_drop = test_module.test_model_with_noise(best_model, atom_encoder, test_dataset, device, best_node_number, gnn_explainer_importance, args)
        sub_base, sub_avg, sub_drop = test_module.test_model_with_noise(best_model, atom_encoder, test_dataset, device, best_node_number, subgraphx_importance, args)
        gbp_base, gbp_avg, gbp_drop = test_module.test_model_with_noise(best_model, atom_encoder, test_dataset, device, best_node_number, gbp_importance, args)
        sa_base, sa_avg, sa_drop = test_module.test_model_with_noise(best_model, atom_encoder, test_dataset, device, best_node_number, sa_importance, args)
        mask_base, mask_avg, mask_drop = test_module.test_model_with_noise(best_model, atom_encoder, test_dataset, device, best_node_number, graph_mask_importance, args)
        pg_base, pg_avg, pg_drop = test_module.test_model_with_noise(best_model, atom_encoder, test_dataset, device, best_node_number, pg_node_importances, args)

        print('Seed', code_seed, 'Test with Noise')
        print('Group Lasso', group_base, group_average, group_drop)
        print('Lasso', alpha_base, alpha_avg, alpha_drop)
        print('Grad-CAM', grad_base, grad_avg, grad_drop)
        print('GNNExplainer', gnn_base, gnn_avg, gnn_drop)
        print('SubgraphX', sub_base, sub_avg, sub_drop)
        print('Guided Backprop', gbp_base, gbp_avg, gbp_drop)
        print('Saliency', sa_base, sa_avg, sa_drop)
        print('Graph Mask', mask_base, mask_avg, mask_drop)
        print('PGExplainer', pg_base, pg_avg, pg_drop)

        # ----- Logging Results to Weights & Biases -----
        wandb.config.update({
            "reg_lasso": best_reg_1,
            "reg_group_lasso": best_reg_2,
        })

        # Log results to wandb
        wandb.log({
            "Group Lasso": {"Base": group_base, "Average": group_average, "Drop": group_drop},
            "Lasso": {"Base": alpha_base, "Average": alpha_avg, "Drop": alpha_drop},
            "Grad-CAM": {"Base": grad_base, "Average": grad_avg, "Drop": grad_drop},
            "GNNExplainer": {"Base": gnn_base, "Average": gnn_avg, "Drop": gnn_drop},
            "SubgraphX": {"Base": sub_base, "Average": sub_avg, "Drop": sub_drop},
            "Guided Backprop": {"Base": gbp_base, "Average": gbp_avg, "Drop": gbp_drop},
            "Saliency": {"Base": sa_base, "Average": sa_avg, "Drop": sa_drop},
            "Graph Mask": {"Base": mask_base, "Average": mask_avg, "Drop": mask_drop},
            "PGExplainer": {"Base": pg_base, "Average": pg_avg, "Drop": pg_drop},
        })

        wandb.finish()


if __name__ == "__main__":
    main(args)
