#%%
import os
import torch
import pickle
import argparse
from utils.utils import set_seed
from graph_dataset import dataset as dataset_module
from modules import model as model_module
from xai_test import noise_test
from modules import optmizer as optimize

#%%
def load_importance(path, name):
    with open(os.path.join(path, name), 'rb') as f:
        return pickle.load(f)

#%%
def main(args):
    accepted_seeds = []
    kept_indices = []

    # Stage 1: Filter seeds based on drop score
    for seed in range(5):
        #%%
        set_seed(seed)
        model_path = os.path.join("model", "v3", "group", args.dataset, args.pooling, "model", f"model_{seed}.pt")
        atom_encoder_path = os.path.join("model", "v3", "group", args.dataset, args.pooling, "atom_encoder", f"atom_encoder_{seed}.pt")
        imp_path = os.path.join("importance", "v3", args.dataset, args.model, args.pooling, f"model_{seed}_alpha_importance.pkl")
        #%%
        if not os.path.exists(model_path) or not os.path.exists(atom_encoder_path) or not os.path.exists(imp_path):
            continue

        model = torch.load(model_path, map_location=args.device).to(args.device).eval()
        atom_encoder = torch.load(atom_encoder_path, map_location=args.device).to(args.device).eval()

        # Dataset selection logic
        if args.dataset.lower() == 'alkane':
            dataset, split_idx = dataset_module.get_alkane()
        elif args.dataset.lower() == 'benzene':
            dataset, split_idx = dataset_module.get_benzene()
        elif args.dataset.lower() == 'fluoride':
            dataset, split_idx = dataset_module.get_fluoride()
        elif args.dataset.lower() == 'ames':
            dataset, split_idx = dataset_module.get_ames()
        else:
            dataset, split_idx = dataset_module.get_MolculeNetData(args.dataset)

        test_dataset = dataset[split_idx["test"]]

        with open(imp_path, 'rb') as f:
            alpha_importance = pickle.load(f)

        best_node_number = []
        filtered_dataset = []
        for i, data in enumerate(test_dataset):
            if hasattr(data, 'groundtruth') and data.groundtruth.sum() > 0:
                filtered_dataset.append(data)
                best_node_number.append(int(data.groundtruth.sum().item()))
        if not filtered_dataset:
            continue

        base, avg, drop = noise_test.test_model_with_noise(
            model, filtered_dataset, args.device, best_node_number, alpha_importance, args, atom_encoder
        )
        if drop >= 0.05:
            accepted_seeds.append(seed)
            kept_indices.append([i for i, d in enumerate(test_dataset) if hasattr(d, 'groundtruth') and d.groundtruth.sum() > 0])

    # Stage 2: Evaluate IoU
    results = {k: [] for k in ["alpha", "gradcam", "pgexplainer", "graphmask", "saliency", "gbp", "gnnexplainer", "group"]}

    for seed, valid_indices in zip(accepted_seeds, kept_indices):
        imp_path = os.path.join("importance", "v3", args.dataset, args.model, args.pooling)

        importances = {
            "alpha": load_importance(imp_path, f"model_{seed}_alpha_importance.pkl"),
            "group": load_importance(imp_path, f"model_{seed}_group_alpha_importance.pkl"),
            "gradcam": load_importance(imp_path, f"model_{seed}_gradcam_importance.pkl"),
            "pgexplainer": load_importance(imp_path, f"model_{seed}_pg_node_importances.pkl"),
            "graphmask": load_importance(imp_path, f"model_{seed}_graph_mask_importance.pkl"),
            "saliency": load_importance(imp_path, f"model_{seed}_sa_importance.pkl"),
            "gbp": load_importance(imp_path, f"model_{seed}_gbp_importance.pkl"),
            "gnnexplainer": load_importance(imp_path, f"model_{seed}_gnn_explainer_importance.pkl"),
        }

        # Dataset selection logic
        if args.dataset.lower() == 'alkane':
            dataset, split_idx = dataset_module.get_alkane()
        elif args.dataset.lower() == 'benzene':
            dataset, split_idx = dataset_module.get_benzene()
        elif args.dataset.lower() == 'fluoride':
            dataset, split_idx = dataset_module.get_fluoride()
        elif args.dataset.lower() == 'ames':
            dataset, split_idx = dataset_module.get_ames()
        else:
            dataset, split_idx = dataset_module.get_MolculeNetData(args.dataset)

        test_dataset = dataset[split_idx["test"]]

        for idx in valid_indices:
            data = test_dataset[idx]
            gt_mask = data.groundtruth.bool()
            k = gt_mask.sum().item()

            for method, imp_dict in importances.items():
                scores = imp_dict[idx]
                topk_idx = scores.topk(k).indices
                pred_mask = torch.zeros_like(gt_mask)
                pred_mask[topk_idx] = 1
                iou = (pred_mask & gt_mask).sum().item() / (pred_mask | gt_mask).sum().item()
                results[method].append(iou)

    for method, scores in results.items():
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"[{method}] Average IoU over accepted seeds: {avg:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='fluoride')
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--pooling', type=str, default='mean')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    main(args)