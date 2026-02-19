# Estimating Subgraph Importance with Structural Prior Domain Knowledge  
### Accepted at Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD 2026)

This repository provides the official implementation of our PAKDD 2026 paper:

> **Estimating Subgraph Importance with Structural Prior Domain Knowledge**  
> Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD), 2026

We propose a **Group-Lasso-based explanation framework** for graph neural networks (GNNs), incorporating structural prior knowledge to estimate node- and subgraph-level importance.  
Our method is systematically benchmarked against state-of-the-art graph explanation approaches.

---

## Abstract

We introduce a structural-prior-aware explanation method for graph classification.  
Unlike conventional node-level attribution methods that treat nodes independently, our approach leverages **group-wise sparsity regularization (Group Lasso)** to estimate subgraph-level importance in a principled manner.

### Key Contributions

- Group-Lasso-based importance estimation framework
- Unified node- and subgraph-level explanation
- Fair comparison protocol via node-ratio equalization
- Robustness evaluation through structured noise injection
- Extensive benchmarking against multiple XAI baselines

---

## Benchmark Setup

### Datasets

- MUTAG  
- TOX21  
- alkane  
- benzene  
- fluoride  
- ames  

### Backbone Models

- GCN  
- GIN  
- GAT  

Pooling strategies:
- mean
- sum
- max

### Compared Explanation Methods

**Ours**
- Group Lasso
- Lasso-based α importance

**Baselines**
- Grad-CAM
- GNNExplainer
- PGExplainer
- SubgraphX
- GraphMask
- Saliency
- Guided Backpropagation (GBP)

---

## Evaluation Protocol

We evaluate explanations at both:

- Node-level  
- Subgraph-level  

Evaluation includes:

- Ground-truth motif alignment  
- Node-ratio matched equalization for fair comparison  
- Robustness analysis via noise injection  

Robustness metrics:

- Base score  
- Average score  
- Drop score  

---

## Usage

### Subgraph-level explanation

```bash
python main.py --dataset MUTAG --model GCN --pooling mean
```

### Node-level explanation

```bash
python main_graph.py --dataset MUTAG --model GCN --pooling mean
```

### Ground-truth validation

```bash
python gt_test.py --dataset MUTAG
```

### Key Hyperparameters

- `--group_param`: Group Lasso regularization strength  
- `--lasso_param`: L1 regularization strength  

---

## Output Structure

Results are stored in:

```
/<DATASET>/<MODEL>/<POOLING>/<SEED>/
```

Each directory contains:

- Importance scores (.pkl)  
- Robustness evaluation metrics  
- Visualization files (if enabled)  

All experiment logs are tracked using **Weights & Biases (wandb)** for reproducibility and comparison.

---

## Reproducibility

- Random seed control supported  
- Deterministic training option available  
- Full experiment logs tracked via wandb  

---

## Dependencies

- PyTorch  
- PyTorch Geometric  
- Captum  
- wandb  
- NetworkX  
- Seaborn  
- RDKit (for molecular datasets)  

---

## Project Structure

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

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{yourname2026subgraph,
  title={Estimating Subgraph Importance with Structural Prior Domain Knowledge},
  author={Author1 and Author2 and Author3},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD)},
  year={2026}
}
```

---

## Contact

For questions, please open an issue or contact the authors.
