# Estimating Subgraph Importance with Structural Prior Domain Knowledge

This repository provides the implementation for evaluating node, subgraph-level explanations in graph classification tasks. We propose a **(Group-) Lasso-based explanation method** and benchmark it against existing XAI methods such as **Grad-CAM**, **GNNExplainer**, **PGExplainer**, **SubgraphX**, **GraphMask**, **Saliency**, and **Guided Backpropagation**.

## Overview

- **Datasets**: MUTAG, TOX21, alkane, benzene, fluoride, ames, etc.
- **Models**: GCN, GIN, GAT with configurable pooling (mean/sum/max)
- **Explanation Methods**:
  - Ours: Group Lasso, Lasso-based α importance
  - Baselines: Grad-CAM, GNNExplainer, PGExplainer, SubgraphX, GraphMask, Saliency, GBP
- **Evaluation**:
  - Node-level importance
  - Equalization for fair comparison (node-ratio matched)
  - **Robustness test** with noise injection:

## Usage

```bash
python main.py --dataset MUTAG --model GCN --pooling mean 
```

main.py for subgraph level, main_graph.py for node level and gt_test.py for ground-truth test
Key arguments:
- `--group_param`: Group Lasso regularization values
- `--lasso_param`: L1 regularization values

## Output

Results and importance scores are stored in:
```
/<DATASET>/<MODEL>/<POOLING>/<SEED>/
```

Each method's importance `.pkl` file and robustness metrics (Base, Average, Drop) are logged to **Weights & Biases (wandb)** for tracking and comparison.

## Dependencies

- PyTorch
- PyTorch Geometric
- Captum
- wandb
- NetworkX
- Seaborn
- RDKit (for molecular datasets)

## Folder Structure

```
.
├── main.py              # Main pipeline
├── graph_dataset/       # Dataset loading
├── modules/             # Model, training, optimizer
├── xai_test/            # Explanation & robustness test
├── utils/               # Helper functions
└── importance/          # Saved importance scores
```

