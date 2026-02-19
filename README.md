# Estimating Subgraph Importance with Structural Prior Domain Knowledge (PAKDD 2026)

This repository provides the official implementation of our PAKDD 2026 paper: **Estimating Subgraph Importance with Structural Prior Domain Knowledge** (Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD), 2026).

> **Abstract**
> We introduce a structural-prior-aware explanation method for graph classification. Unlike conventional node-level attribution methods that treat nodes independently, our approach leverages *group-wise sparsity regularization (Group Lasso)* to estimate subgraph-level importance in a principled manner.

---

### Experimental Settings

* Datasets: MUTAG, TOX21, alkane, benzene, fluoride, ames  
* Backbone Models: GCN, GIN, GAT  
* Pooling strategies: mean, sum, max
* Baselines: Grad-CAM, GNNExplainer, PGExplainer, SubgraphX, GraphMask, Saliency, Guided Backpropagation (GBP)
* Evaluation: Node-level, Subgraph-level, Ground-truth validation

### Usage

* Subgraph-level explanation
```bash
python main.py --dataset MUTAG --model GCN --pooling mean
```
* Node-level explanation
```bash
python main_graph.py --dataset MUTAG --model GCN --pooling mean
```
* Ground-truth validation
```bash
python gt_test.py --dataset MUTAG
```
* Key Hyperparameters
    - `--group_param`: Group Lasso regularization strength
    - `--lasso_param`: L1 regularization strength

### Dependencies
- PyTorch, PyTorch Geometric, Captum, wandb, NetworkX, Seaborn, RDKit (for molecular datasets)

### Repository Structure

```
.
├── main.py              # Subgraph-level explanation pipeline
├── main_graph.py        # Node-level explanation pipeline
├── gt_test.py           # Ground-truth validation
├── graph_dataset/       # Dataset loaders
├── modules/             # Models and training
├── xai_test/            # Explanation & robustness evaluation
├── utils/               # Utility functions
└── importance/          # Saved importance scores
```

---

### Output Structure

Results are stored in:

```
/<DATASET>/<MODEL>/<POOLING>/<SEED>/
```

Each directory contains:

- Importance scores (.pkl)  
- Robustness evaluation metrics  
- Visualization files (if enabled)  

> All experiment logs are tracked using *Weights & Biases (wandb)* for reproducibility and comparison.


### Reproducibility

- Random seed control supported  
- Deterministic training option available  
- Full experiment logs tracked via wandb  

---

### Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{kim2026estimating,
  title={Estimating Subgraph Importance with Structural Prior Domain Knowledge},
  author={Changhyun Kim, Seunghwan An, and Jong-June Jeon},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)},
  year={2026}
}
```

### Contact

For questions, please open an issue or contact the authors.
