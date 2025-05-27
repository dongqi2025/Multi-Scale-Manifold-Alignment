# Multi-Scale Manifold Alignment: A Unified Framework for Enhanced Explainability of Large Language Models

![Framework Overview](docs/figures/framework.png)  

## 🔍 Overview
This repository presents **Multi-Scale Manifold Alignment (MSMA)**, a novel framework for interpreting and controlling Large Language Models (LLMs) by decomposing their latent spaces into hierarchically organized semantic manifolds. Our method bridges geometric and information-theoretic approaches to reveal how LLMs process language across scales—from word-level details to global discourse.

Key innovations:
- **Hierarchical decomposition** of LLM representations into global, intermediate, and local semantic manifolds
- **Cross-scale alignment** combining Procrustes analysis with mutual information constraints
- **Theoretical guarantees** on alignment quality via KL divergence bounds
- **Practical applications** in bias detection, robustness enhancement, and controlled generation




## 🎯 Key Features
- **Multi-Scale Interpretation**: Identifies three semantic levels in LLMs:
  - **Global**: Document-level themes and discourse
  - **Intermediate**: Sentence-level relationships
  - **Local**: Word-level syntax and lexicon

- **Unified Alignment**:
  - Geometric preservation via Procrustes analysis
  - Information retention through mutual information
  - Curvature regularization for stable optimization

- **Model-Agnostic**: Works with GPT-2, BERT, RoBERTa, and T5 architectures

## 🚀 Quick Start
*(This section will be populated with installation and usage instructions)*

## 📊 Experimental Results
Our framework demonstrates strong empirical performance:

### Semantic Layer Distribution
| Model   | Local Layers | Intermediate Layers | Global Layers |
|---------|-------------|---------------------|--------------|
| GPT-2   | 0-2 (25%)   | 3-8 (50%)           | 9-12 (25%)   |
| BERT    | 0-4 (42%)   | 5-8 (29%)           | 9-12 (29%)   |

### Alignment Quality (BERT)
| Method       | KL (G→I) | MI (G→I) | DC (G→I) |
|--------------|---------|---------|---------|
| Baseline     | 403     | 0.06    | 0.87    |
| Full MSMA    | **0.51**| **2.89**| **1.00**|


## 📂 Repository Structure
```
.
├── configs/            # Experiment configurations
├── data/               # Sample datasets
├── docs/               # Supplementary materials
├── models/             # Pretrained model checkpoints
├── scripts/            # Training/evaluation scripts
├── src/
│   ├── alignment/      # Cross-scale mapping modules
│   ├── analysis/       # Interpretation tools
│   ├── geometry/       # Manifold operations
│   └── utils/          # Helper functions
├── LICENSE
└── README.md
```


## 🤝 Contributing
We welcome contributions! Please open an issue or submit a PR for:
- Bug fixes
- New alignment methods
- Additional model support
- Documentation improvements


## 📜 Citation
If you use this work, please cite our paper:
```bibtex

```

## ✉️ Contact
For questions, please contact:
- Yukun Zhang: 215010026@link.cuhk.edu.cn
- Qi Dong: 19210980065@fudan.edu.cn
