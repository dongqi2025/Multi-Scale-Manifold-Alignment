# Multi-Scale Manifold Alignment: A Unified Framework for Enhanced Explainability of Large Language Models[![arXiv](https://img.shields.io/badge/arXiv-2505.20340-b31b1b.svg)](https://arxiv.org/abs/2505.20333)

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


### Dependency Installation
Ensure that all necessary dependency libraries are installed, such as `torch`, `numpy`, `matplotlib`, `seaborn`, etc. You can use the following command to install them:
```bash
pip install -r requirements.txt
```

### Experiment Startup Method
#### Configure Experiment Parameters
Open the `configs/default_config.yaml` file and modify the experiment parameters as needed, such as learning rate, number of training epochs, and device.

#### Run the Experiment Script
Run the following command in the terminal to start the experiment:
```bash
python experiments/run_experiment.py --config configs/default_config.yaml
```

## 📊 Experimental Results
Our framework demonstrates strong empirical performance:

### Semantic Layer Distribution
| Model    | Local Layers      | Intermediate Layers | Global Layers    |
|----------|-------------------|---------------------|------------------|
| **GPT-2**   | 0-2 (25%)        | 3-8 (50%)           | 9-12 (25%)       |
| **BERT**    | 0-4 (42%)        | 5-8 (29%)           | 9-12 (29%)       |
| **RoBERTa** | 0-4 (42%)        | 5-8 (29%)           | 9-12 (29%)       |
| **T5**      | 0-2 (50%)        | 3-4 (33%)           | 5-6 (17%)        |

Key observations:
- Autoregressive models (GPT-2) allocate 50% layers to intermediate semantics
- Bidirectional encoders (BERT/RoBERTa) emphasize local processing (>40%)
- Encoder-decoder (T5) shows compressed hierarchy with 50% local layers

#### Code Repository  
Details and code for this experiment are available at:  
[`https://github.com/dongqi2025/Multi-Scale-Probabilistic-Generation-Theory.git`](https://github.com/dongqi2025/Multi-Scale-Probabilistic-Generation-Theory.git)  
(Includes layer definition scripts, activation analysis, and visualization code.)

### 🔬 Intervention 

#### Experimental Design
We conducted systematic interventions across semantic scales to validate functional specialization:
- **Intervention Types**:
  1. Translation: $\mathbf{h'} = \mathbf{h} + \Delta$
  2. Scaling: $\mathbf{h'} = \alpha\mathbf{h}$ 
  3. Gaussian Noise: $\mathbf{h'} = \mathbf{h} + \epsilon$
  4. Attention Modification
- **Evaluation Metrics**:
  - Lexical diversity
  - Sentence count
  - Mean sentence length
  - Max dependency depth  
  - Discourse coherence
  - Sentiment consistency
#### Key Findings
| Model  | Scale       | Intervention | Key Impact                     | Effect Size (δ) |
|--------|-------------|--------------|--------------------------------|-----------------|
| GPT-2  | **Global**  | Amplify      | +7.4% Lexical Diversity        | +0.23           |
|        |             |              | -24% Coherence                 | -0.24           |
|        | **Intermediate** | Scale   | +25% Sentence Count            | +0.24           |
|        |             |              | -19% Mean Sentence Length      | -0.27           |
|        | **Local**   | Amplify      | +34% Lexical Variation         | +0.34           |

| Model  | Scale       | Intervention | Stability Observation          |
|--------|-------------|--------------|---------------------------------|
| BERT   | Intermediate| Attention    | Robust structural preservation  |
| XLM-R  | Global      | Noise        | Sentiment resilience (-14% Δ)   |


#### Code Repository  
Experimental code and data are available at:  
[`https://github.com/dongqi2025/Multi-Scale-Probabilistic-Generation-Theory.git`](https://github.com/dongqi2025/Multi-Scale-Probabilistic-Generation-Theory.git)  
(Includes hidden-state perturbation scripts, metric calculations, and configuration files.)  

 
### 🛠️ Alignment
We validate MSMA through ablation studies with three key components:

**Ablation Settings**:
| Configuration | Geometry | Information | Curvature | λ_geo | λ_info | λ_curv |
|---------------|:--------:|:-----------:|:---------:|:-----:|:------:|:------:|
| Full MSMA     |    ✓     |      ✓      |     ✓     |  0.1  |   0.1  |  0.01  |
| No Geometry   |    ✗     |      ✓      |     ✓     |  0.0  |   0.1  |  0.01  |  
| No Information|    ✓     |      ✗      |     ✓     |  0.1  |   0.0  |  0.01  |

**Alignment Performance**:
| Model  | Method       | KL Divergence ↓ | Mutual Info ↑ | Distance Corr →1 |
|--------|--------------|----------------:|--------------:|-----------------:|
| GPT-2  | Baseline     |          6,955  |          0.23 |             0.97 |
|        | **Full MSMA**|             33  |          1.25 |             1.00 |
| BERT   | Baseline     |            403  |          0.06 |             0.87 |
|        | **Full MSMA**|           0.51  |          2.89 |             1.00 |

Critical insights:
1. **Multi-component Synergy**: Full MSMA achieves 99.5% KL reduction vs baseline
2. **Architecture Matters**: BERT shows better alignability (KL=0.51 vs GPT-2's 33)
3. **Curvature Regularization**: Prevents distortion (no-curv KL increases 18-32%)
4. **Geometric Foundation**: Removing geometry degrades MI by 43-68%

## 📂 Repository Structure
This project adopts a structured code repository layout, primarily consisting of the following components:
- `data/`: Used to store both raw data and processed data.
- `models/`: Contains definitions of different models, such as model components related to BERT and GPT2.
- `utils/`: Holds utility functions, including data loading, training record - keeping, and visualization functionalities.
- `configs/`: Stores configuration files for experiments, allowing you to modify experimental parameters as needed.
- `experiments/`: Contains the main scripts for experiments, used to initiate experiments.
- `results/`: Used to save experiment results, including training metrics, validation metrics, and interpretability metrics.
- `logs/`: Stores log files for experiments.
-


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
