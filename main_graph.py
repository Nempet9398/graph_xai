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

#python main.py  --epochs 300 --model GCN --dataset MUTAG --top_motif 1 --layers 3 --pooling mean

# ============================ #
#       Argument Parsing       #
# ============================ #

parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--hidden', type=int, default=256)

parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0)

parser.add_argument('--model', type=str, default="GCN", help="GCN, GIN, GAT")
parser.add_argument('--pooling', type=str, default="mean", help="mean, max, sum")

parser.add_argument('--dataset', type=str, default='MUTAG', help='name of dataset. For now, only classification.')
parser.add_argument('--eval_metric', type=str, default='auc')

parser.add_argument('--lasso_param', 
                    type=lambda s: [float(item) for item in s.split(',')], 
                    default=[ 0.01,0.05,0.1,0.5,1,3,5,10,30,50,70,100],
                    help='Comma-separated list of lasso regularization parameters')
# [ 1, 10, 100, 500,1000]

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


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

#%%

# ============================ #
#         Main Function        #
# ============================ #

def main(args):

    for code_seed in range(5):
        #%%
        code_seed = 1
        args.dataset= 'MOTIF'
        print(f'Start experiment with seed {code_seed}, Dataset {args.dataset}, Model {args.model}, Pooling {args.pooling}')

        set_seed(code_seed)

        if args.dataset.upper() == 'UPFD':
            dataset, split_idx = dataset_module.get_UPFD_GossipCop()
        elif args.dataset.upper() == 'IMDB':
            dataset, split_idx = dataset_module.get_TUD('IMDB-BINARY')
        elif args.dataset.upper() == 'ENZYME':
            dataset, split_idx = dataset_module.get_TUD('ENZYMES')
        elif args.dataset.upper() == 'PROTEINS':
            dataset, split_idx = dataset_module.get_TUD('PROTEINS')
        elif args.dataset.upper() == 'MOTIF':
            dataset, split_idx = dataset_module.get_syndata('motif')
        elif args.dataset.upper() == 'MULTI':
            dataset, split_idx = dataset_module.get_syndata('multi')

        args.num_task = 1
        args.num_classes = len(np.unique(dataset.y))
        args.input_dim = dataset[0].x.shape[1]

        #%%
        # # ----- Model and Atom Encoder Loading or Training -----
        # model_name = f'{args.model}_{args.dataset}_{args.pooling}_{code_seed}_model.pt'
        # model_path = os.path.join('model', 'graph_normal', model_name)

        # # ----- Model and Atom Encoder Loading or Training -----
        model_name = f'model_{code_seed}.pt'
        model_path = os.path.join('model', 'graph_normal', args.dataset,args.model, args.pooling,model_name)


        #%%
        # Initialize wandb
        wandb.init(
            project="GRAPH_XAI_NORMAL",
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
                "seed": code_seed,
            }
        )
        #%%
        # Check if the model and atom encoder exist
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")

            best_model = torch.load(model_path).to(device)
        else:
            print("Model not found. Creating new ones.")
            model = model_module.BasicGNN_MLP(args).to(device)

            test_auc, best_model = train_module.train_function_normal(
                dataset, split_idx, model, args, device
            )

            # Save the trained model and atom encoder
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(best_model, model_path)

            print(f"Model saved to {model_path} ")

        # ----- Data Preparation -----
        train_dataset = dataset[split_idx['train']]
        valid_dataset = dataset[split_idx['valid']]
        test_dataset = dataset[split_idx['test']]

        best_node_number = [max(2, int(i.x.shape[0] * 0.2)) for i in test_dataset]
        embedding_list = [
            best_model(data.x.to(device), data.edge_index.to(device), infer=True).detach() 
            for data in test_dataset
        ]



        # ----- Importance Calculation -----
        importance_save_path = os.path.join(os.path.dirname(model_path), 'importance')
        os.makedirs(importance_save_path, exist_ok=True)
        importance_base = model_name.replace('_model.pt', '')

        # Alpha Importance
        alpha_result = utils.load_or_compute_importance(
            f'{importance_base}_alpha_importance.pkl',
            lambda: optimize.find_best_normal_importance(
                test_dataset=test_dataset,
                embedding_list=embedding_list,
                best_model=best_model,
                device=device,
                args=args,
                node_number=best_node_number,
                test_func=test_module.test_model_with_noise,
                reg_1_list=args.lasso_param,
            ),
            importance_save_path
        )

        alpha_importance, best_reg_1, alpha_base, alpha_avg, alpha_drop , alpha_sparsity = alpha_result
        #%%
        # Grad-CAM
        gradcam_importance = utils.load_or_compute_importance(
            f'{importance_base}_gradcam_importance.pkl',
            lambda: optimize.compute_gradcam_importance(test_dataset, best_model, device),
            importance_save_path
        )

        # PGExplainer
        pg_node_importances = utils.load_or_compute_importance(
            f'{importance_base}_pgexplainer_importance.pkl',
            lambda: optimize.compute_pgexplainer_importance(test_dataset, best_model),
            importance_save_path
        )

        # Graph Mask
        graph_mask_importance = utils.load_or_compute_importance(
            f'{importance_base}_graphmask_importance.pkl',
            lambda: optimize.compute_graphmask_importance(test_dataset, best_model),
            importance_save_path
        )

        # Saliency
        sa_importance = utils.load_or_compute_importance(
            f'{importance_base}_saliency_importance.pkl',
            lambda: optimize.compute_captum_importance(test_dataset, best_model, 'Saliency'),
            importance_save_path
        )

        # GuidedBackprop
        gbp_importance = utils.load_or_compute_importance(
            f'{importance_base}_guidedbackprop_importance.pkl',
            lambda: optimize.compute_captum_importance(test_dataset, best_model, 'GuidedBackprop'),
            importance_save_path
        )

        # SubgraphX
        subgraphx_importance = utils.load_or_compute_importance(
            f'{importance_base}_subgraphx_importance.pkl',
            lambda: optimize.compute_subgraphx_importance(test_dataset, best_model, best_node_number, device),
            importance_save_path
        )

        best_model = best_model.to(device)

        # GNNExplainer
        gnn_explainer_importance = utils.load_or_compute_importance(
            f'{importance_base}_gnnexplainer_importance.pkl',
            lambda: optimize.compute_gnnexplainer_importance(test_dataset, best_model, device),
            importance_save_path
        )


        # ----- Robustness Testing with Noise Injection -----      
        grad_base, grad_avg, grad_drop = test_module.test_model_with_noise(best_model,  test_dataset, device, best_node_number, gradcam_importance, args)
        gnn_base, gnn_avg, gnn_drop = test_module.test_model_with_noise(best_model,  test_dataset, device, best_node_number, gnn_explainer_importance, args)
        sub_base, sub_avg, sub_drop = test_module.test_model_with_noise(best_model, test_dataset, device, best_node_number, subgraphx_importance, args)
        gbp_base, gbp_avg, gbp_drop = test_module.test_model_with_noise(best_model,  test_dataset, device, best_node_number, gbp_importance, args)
        sa_base, sa_avg, sa_drop = test_module.test_model_with_noise(best_model, test_dataset, device, best_node_number, sa_importance, args)
        mask_base, mask_avg, mask_drop = test_module.test_model_with_noise(best_model, test_dataset, device, best_node_number, graph_mask_importance, args)
        pg_base, pg_avg, pg_drop = test_module.test_model_with_noise(best_model,  test_dataset, device, best_node_number, pg_node_importances, args)


        print('Seed', code_seed, 'Test with Noise')
        # print('Group Lasso', group_base, group_average, group_drop)
        print('Lasso', alpha_base, alpha_avg, alpha_drop)
        print('Grad-CAM', grad_base, grad_avg, grad_drop)
        print('GNNExplainer', gnn_base, gnn_avg, gnn_drop)
        print('SubgraphX', sub_base, sub_avg, sub_drop)
        print('Guided Backprop', gbp_base, gbp_avg, gbp_drop)
        print('Saliency', sa_base, sa_avg, sa_drop)
        print('Graph Mask', mask_base, mask_avg, mask_drop)
        print('PGExplainer', pg_base, pg_avg, pg_drop)

        #%%
        # ----- Logging Results to Weights & Biases -----
        wandb.config.update({
            "reg_lasso": best_reg_1,

        })

        # Log results to wandb
        wandb.log({
            "Lasso": {"Base": alpha_base, "Average": alpha_avg, "Drop": alpha_drop, "Sparsity": alpha_sparsity},
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
