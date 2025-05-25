#%%
import os
import importlib
import torch
import pickle
import argparse
from utils.utils import set_seed
import numpy as np
import wandb

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

def load_importance(path, name):
    with open(os.path.join(path, name), 'rb') as f:
        return pickle.load(f)

#%%
def main(args):


    # Initialize wandb
    wandb.init(
        project="GRAPH_GROUND_TRUTH",
        name=f"{args.dataset}_{args.model}_{args.pooling}",
        config={
            "dataset": args.dataset,
            "model": args.model,
            "pooling": args.pooling,
        }, tags=["GROUND_TRUTH","PR"]
    )
    all_auc_results = {
        "alpha": [],
        # "group_alpha": [],
        "gradcam": [],
        "gnn_explainer": [],
        "graph_mask": [],
        "pg": [],
        "saliency": [],
        "gbp": []
    }

    # 각 시드에 대해 반복
    for seed in range(5):

        set_seed(seed)

        # 모델과 atom encoder 로드
        model_path = os.path.join("model", "v4", "group", args.dataset, args.pooling, "model",args.model, f"model_{seed}.pt")
        atom_encoder_path = os.path.join("model", "v4", "group", args.dataset, args.pooling, "atom_encoder", args.model, f"atom_encoder_{seed}.pt")
        best_model = torch.load(model_path, map_location=args.device).to(args.device).eval()
        atom_encoder = torch.load(atom_encoder_path, map_location=args.device).to(args.device).eval()

        # 데이터셋 로드 - 세가지 종류
        if args.dataset.lower() == 'alkane':
            dataset, split_idx = dataset_module.get_alkane()
        elif args.dataset.lower() == 'benzene':
            dataset, split_idx = dataset_module.get_benzene()
        elif args.dataset.lower() == 'fluoride':
            dataset, split_idx = dataset_module.get_fluoride()
        else:
            dataset, split_idx = dataset_module.get_MolculeNetData(args.dataset)

        # 테스트 데이터에 대해 진행
        test_dataset = dataset[split_idx["test"]]

        # 테스트 데이터 중 ground_truth가 있는 데이터만 필터링
        filtered_idx = [i for i, data in enumerate(test_dataset)
                        if hasattr(data, 'ground_truth')
                        and 0 < data.ground_truth.sum() < data.num_nodes]
        filtered_dataset = test_dataset[filtered_idx]
        best_node_number = [int(test_dataset[i].ground_truth.sum().item()) for i in filtered_idx]
        best_importance = [filtered_dataset[i].ground_truth for i in range(len(filtered_dataset))]

        base, avg, drop = test_module.test_model_with_noise(
            best_model, filtered_dataset, args.device, best_node_number, best_importance, args, atom_encoder
        )
        print(seed, '-->',drop)

        if drop < 0.5:
            print("skip")
            continue
        
        device = args.device
        args.lasso_param = [0.01,0.05,0.1,0.5,1,5,10,15,20,25,30,40,50]
        embedding_list = [
            best_model(atom_encoder(data.x.to(device)).to(device), data.edge_index.to(device), infer=True).detach() 
            for data in filtered_dataset
        ]

        # ----- Explanation Method Importance Calculation -----
                # ----- Alpha Importance -----
        alpha_importance, best_reg_1, alpha_base, alpha_avg, alpha_drop = optimize.find_best_alpha_importance(
            test_dataset=filtered_dataset,
            best_model=best_model,
            atom_encoder=atom_encoder,
            embedding_list=embedding_list,
            device=device,
            args=args,
            node_number=best_node_number,
            test_func=test_module.test_model_with_noise,
            reg_1_list=args.lasso_param,
        )

        # # ----- Group Lasso and Lasso Importance -----
        # args.top_motif = 1
        # group_alpha_importance, best_reg_2, group_base, group_average, group_drop, best_node_number = optimize.find_best_group_alpha_importance(
        #     test_dataset=filtered_dataset,
        #     best_model=best_model,
        #     atom_encoder=atom_encoder,
        #     embedding_list=embedding_list,
        #     device=device,
        #     args=args,
        #     test_func=test_module.test_model_with_noise,
        #     reg_2_list=args.group_param,
        # )

        # ----- Other Importance Methods -----
        pg_node_importances = optimize.compute_pgexplainer_importance(filtered_dataset, best_model, atom_encoder)

        graph_mask_importance = optimize.compute_graphmask_importance(filtered_dataset, best_model, atom_encoder)

        sa_importance = optimize.compute_captum_importance(filtered_dataset, best_model, 'Saliency', atom_encoder)
 
        gbp_importance = optimize.compute_captum_importance(filtered_dataset, best_model, 'GuidedBackprop', atom_encoder)

        # subgraphx_importance = optimize.compute_subgraphx_importance(filtered_dataset, best_model, best_node_number, device, atom_encoder)

        atom_encoder = atom_encoder.to(device)
        best_model = best_model.to(device)

        gnn_explainer_importance = optimize.compute_gnnexplainer_importance(filtered_dataset, best_model, device, atom_encoder)

        gradcam_importance = optimize.compute_gradcam_importance(filtered_dataset, best_model, device, atom_encoder)

        # ============================ #
        #    AUC List Calculation     #
        # ============================ #


        # Compute AUC-ROC and AUC-PR for each method
        auc_alpha_roc = utils.compute_auc_list(alpha_importance, filtered_dataset)
        auc_alpha_pr = utils.compute_pr_auc_list(alpha_importance, filtered_dataset)
        print(np.mean(auc_alpha_roc), 'alpha auc-roc')
        print(np.mean(auc_alpha_pr), 'alpha auc-pr')

        # auc_group_alpha_roc = utils.compute_auc_list(group_alpha_importance, filtered_dataset)
        # auc_group_alpha_pr = utils.compute_pr_auc_list(group_alpha_importance, filtered_dataset)
        # print(np.mean(auc_group_alpha_roc), 'group_alpha auc-roc')
        # print(np.mean(auc_group_alpha_pr), 'group_alpha auc-pr')

        auc_gradcam_roc = utils.compute_auc_list(gradcam_importance, filtered_dataset)
        auc_gradcam_pr = utils.compute_pr_auc_list(gradcam_importance, filtered_dataset)
        print(np.mean(auc_gradcam_roc), 'gradcam auc-roc')
        print(np.mean(auc_gradcam_pr), 'gradcam auc-pr')

        auc_gnn_explainer_roc = utils.compute_auc_list(gnn_explainer_importance, filtered_dataset)
        auc_gnn_explainer_pr = utils.compute_pr_auc_list(gnn_explainer_importance, filtered_dataset)
        print(np.mean(auc_gnn_explainer_roc), 'gnn_explainer auc-roc')
        print(np.mean(auc_gnn_explainer_pr), 'gnn_explainer auc-pr')

        auc_graphmask_roc = utils.compute_auc_list(graph_mask_importance, filtered_dataset)
        auc_graphmask_pr = utils.compute_pr_auc_list(graph_mask_importance, filtered_dataset)
        print(np.mean(auc_graphmask_roc), 'graph_mask auc-roc')
        print(np.mean(auc_graphmask_pr), 'graph_mask auc-pr')

        auc_pg_roc = utils.compute_auc_list(pg_node_importances, filtered_dataset)
        auc_pg_pr = utils.compute_pr_auc_list(pg_node_importances, filtered_dataset)
        print(np.mean(auc_pg_roc), 'pg auc-roc')
        print(np.mean(auc_pg_pr), 'pg auc-pr')

        auc_saliency_roc = utils.compute_auc_list(sa_importance, filtered_dataset)
        auc_saliency_pr = utils.compute_pr_auc_list(sa_importance, filtered_dataset)
        print(np.mean(auc_saliency_roc), 'saliency auc-roc')
        print(np.mean(auc_saliency_pr), 'saliency auc-pr')

        auc_gbp_roc = utils.compute_auc_list(gbp_importance, filtered_dataset)
        auc_gbp_pr = utils.compute_pr_auc_list(gbp_importance, filtered_dataset)
        print(np.mean(auc_gbp_roc), 'gbp auc-roc')
        print(np.mean(auc_gbp_pr), 'gbp auc-pr')

        # Store AUCs for this seed
        all_auc_results["alpha"].append((np.mean(auc_alpha_roc), np.mean(auc_alpha_pr)))
        # all_auc_results["group_alpha"].append((np.mean(auc_group_alpha_roc), np.mean(auc_group_alpha_pr)))
        all_auc_results["gradcam"].append((np.mean(auc_gradcam_roc), np.mean(auc_gradcam_pr)))
        all_auc_results["gnn_explainer"].append((np.mean(auc_gnn_explainer_roc), np.mean(auc_gnn_explainer_pr)))
        all_auc_results["graph_mask"].append((np.mean(auc_graphmask_roc), np.mean(auc_graphmask_pr)))
        all_auc_results["pg"].append((np.mean(auc_pg_roc), np.mean(auc_pg_pr)))
        all_auc_results["saliency"].append((np.mean(auc_saliency_roc), np.mean(auc_saliency_pr)))
        all_auc_results["gbp"].append((np.mean(auc_gbp_roc), np.mean(auc_gbp_pr)))

        print("\n===== AUC Summary Across Seeds =====")
        for method, values in all_auc_results.items():
            mean_auc_roc = np.mean([v[0] for v in values])
            std_auc_roc = np.std([v[0] for v in values])
            mean_auc_pr = np.mean([v[1] for v in values])
            std_auc_pr = np.std([v[1] for v in values])
            print(f"{method} AUC-ROC: {mean_auc_roc:.4f} ± {std_auc_roc:.4f}")
            print(f"{method} AUC-PR: {mean_auc_pr:.4f} ± {std_auc_pr:.4f}")
            wandb.log({
                f"{method}_mean_auc_roc": mean_auc_roc, f"{method}_std_auc_roc": std_auc_roc,
                f"{method}_mean_auc_pr": mean_auc_pr, f"{method}_std_auc_pr": std_auc_pr
            })


    # Add this to avoid wandb errors when running multiple times in Jupyter or scripts
    wandb.finish()


if __name__ == "__main__":

    debug = False  # Set to False when running from command line
    if debug:
        class Args:
            dataset = 'fluoride'
            model = 'GCN'
            pooling = 'mean'
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args = Args()
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='fluoride')
        parser.add_argument('--model', type=str, default='GCN')
        parser.add_argument('--pooling', type=str, default='mean')
        parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
        args = parser.parse_args()

    main(args)

#python gt_test.py --dataset fluoride --model GCN --pooling mean --device cuda
# #%%
# %matplotlib inline
# import torch
# import networkx as nx
# import matplotlib.pyplot as plt
# from torch_geometric.utils import to_networkx
# i = 9
# # 데이터 예시: PyG Data 객체
# data = filtered_dataset[i]
# # importance_ind_alpha = alpha_importance[i]
# # importance_ind_group = group_alpha_importance[i]
# # Alpha importance 색상 설정 (연속값)
# # imp_norm = (importance_ind_alpha - importance_ind_alpha.min()) / (importance_ind_alpha.max() - importance_ind_alpha.min() + 1e-6)
# # imp_colors = [plt.cm.Reds(val.item()) for val in imp_norm]
# # Group Alpha importance 색상 설정 (연속값)
# # imp_norm_group = (importance_ind_group - importance_ind_group.min()) / (importance_ind_group.max() - importance_ind_group.min() + 1e-6)
# # imp_colors_group = [plt.cm.Blues(val.item()) for val in imp_norm_group]
# # NetworkX 변환
# G = to_networkx(data, to_undirected=True)

# # 레이아웃 고정
# pos = nx.kamada_kawai_layout(G)

# # Ground Truth 색 지정
# ground_truth = data.ground_truth.bool().tolist()
# ground_truth_colors = ['red' if gt else 'lightgrey' for gt in ground_truth]

# # Clique 색 지정 (각기 다른 색)
# cliques = data.clique  # ex: [[1, 2], [4, 5, 6]]
# clique_colors = [''] * data.num_nodes
# cmap = plt.get_cmap('tab20')  # 최대 20가지 색

# for idx, clique in enumerate(cliques):
#     for node in clique:
#         clique_colors[node] = cmap(idx % 20)

# # 시각화
# plt.figure(figsize=(20, 4))

# # (1) Clique Subgraphs 시각화
# plt.subplot(1, 4, 1)
# nx.draw(G, pos, node_color=clique_colors, with_labels=True, node_size=500)
# plt.title("Clique Subgraphs")

# # (2) Ground Truth 시각화
# plt.subplot(1, 4, 2)
# nx.draw(G, pos, node_color=ground_truth_colors, with_labels=True, node_size=500)
# plt.title("Ground Truth Explanation")

# # # (3) Alpha Importance Heatmap
# # plt.subplot(1, 4, 3)
# # nx.draw(G, pos, node_color=imp_colors, with_labels=True, node_size=500)
# # plt.title("Alpha Importance")

# # # (4) Group Alpha Importance Heatmap
# # plt.subplot(1, 4, 4)
# # nx.draw(G, pos, node_color=imp_colors_group, with_labels=True, node_size=500)
# # plt.title("Group Alpha Importance")

# plt.tight_layout()
# plt.show()
# #%%