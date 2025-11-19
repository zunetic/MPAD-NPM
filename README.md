# MPAD-NPM: Multimodal Package Analysis Dataset for npm Security

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
<!-- DOI badge will be updated after Zenodo release -->

## Overview

**MPAD-NPM** is a reproducible benchmark dataset for detecting malicious packages in the npm ecosystem. It addresses critical gaps in software supply chain security research by providing:

- **11,214 npm packages** (7,000 benign, 4,214 malicious) spanning **2015-2024**
- **Unified multimodal features**: metadata (137D) + dependency graph embeddings (800D) + API behavioral semantics (768D) → **800D final representation**
- **Complete preprocessing pipeline**: publicly released scripts for feature extraction and reproduction
- **Stratified splits**: train (70%), validation (15%), test (15%) with consistent random seed

This dataset supports the **MPAD-Guard** detection framework described in our paper:

> **Iqbal, T.** (2025). MPAD-Guard: Enhanced Malicious Package Detection via Novel Fusion of API Calls and Dependency Graph in npm Ecosystem. *International Journal of Information Security*. [Manuscript under review]

---

## 🚀 Quick Start

### Loading the Dataset

```python
import numpy as np
import pandas as pd
import json

# Load preprocessed 800D features
features = np.load('features/npm_final_combined_features_processed.npy')  # Shape: (11214, 800)

# Load labels
labels = np.load('labels/labels_npm_only_processed.npy')  # Binary: 0=benign, 1=malicious

# Load metadata (if needed)
metadata = pd.read_csv('metadata/npm_enriched_metadata.csv')

print(f"Dataset shape: {features.shape}")
print(f"Malicious packages: {labels.sum()} ({labels.sum()/len(labels)*100:.1f}%)")
```

**Output:**
```
Dataset shape: (11214, 800)
Malicious packages: 4214 (37.6%)
```

---

## 📊 Dataset Composition

| Category | Count | Percentage |
|----------|-------|------------|
| **Benign Packages** | 7,000 | 62.4% |
| **Malicious Packages** | 4,214 | 37.6% |
| **Total** | 11,214 | 100% |

### Malicious Package Breakdown (by Attack Type)

| Attack Vector | Count | % of Malicious |
|---------------|-------|----------------|
| Code Injection | 1,770 | 42.0% |
| Typosquatting | 1,180 | 28.0% |
| Data Theft | 758 | 18.0% |
| Dependency Confusion | 337 | 8.0% |
| Resource Hijacking | 169 | 4.0% |

### Temporal Distribution

| Period | Packages | % of Total |
|--------|----------|------------|
| 2015-2019 | 1,682 | 15.0% |
| 2020-2022 | 5,047 | 45.0% |
| 2023-2024 | 4,485 | 40.0% |

---

## 🗂️ Repository Structure

```
MPAD-NPM/
├── features/
│   ├── npm_final_combined_features_processed.npy    # Final 800D features
│   ├── npm_api_vectorized_features.npy              # API embeddings (768D)
│   ├── npm_combined_features_with_api_stats.npy     # Raw combined features
│   └── npmnode_embeddings.npy                       # Graph embeddings (800D)
│
├── graphs/
│   └── npmdependency_graph.gpickle                  # Complete dependency graph
│
├── labels/
│   └── labels_npm_only_processed.npy                # Binary labels (11,214)
│
├── metadata/
│   ├── npm_enriched_metadata.csv                    # Package metadata
│   ├── npm_packages_only.csv                        # Package list
│   └── npmpreprocessed_metadata.csv                 # Preprocessed features
│
├── CITATION.cff                                     # Citation metadata
├── LICENSE                                          # CC BY 4.0
├── README.md                                        # This file
└── requirements.txt                                 # Python dependencies
```

---

## 🔬 Feature Modalities

### 1. **Metadata Features (137D)**

Extracted from `package.json` and npm registry:

- **Package attributes**: name length, description sentiment, version patterns
- **Maintenance signals**: update frequency, contributor count, GitHub stars
- **Dependency metrics**: direct dependencies, transitive depth, circular references
- **Publication info**: account age, package age, download statistics
- **Risk indicators**: install scripts, pre/post hooks, external URLs

### 2. **Dependency Graph Embeddings (800D)**

Generated via **Graph Attention Networks (GAT)**:

- **Node features**: Package metadata as initial node attributes
- **Edge features**: Dependency relationship types (dependencies, devDependencies, peerDependencies)
- **Graph topology**: In-degree, out-degree, betweenness centrality, PageRank
- **Embedding method**: 2-layer GAT with 8 attention heads, mean pooling

**File**: `npmnode_embeddings.npy`

### 3. **API Behavioral Semantics (768D)**

BERT-encoded API call sequences:

- **Source**: Static analysis of JavaScript AST
- **API extraction**: `child_process`, `fs`, `net`, `crypto`, `eval`, `Function`
- **Encoding**: `bert-base-uncased` with `[CLS]` token pooling
- **Sequence format**: `"fs.readFile -> crypto.createHash -> net.connect"`

**File**: `npm_api_vectorized_features.npy`

### 4. **Final Feature Engineering (800D)**

Dimensionality reduction via **Mutual Information Gain** and **Variance Thresholding**:

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold

# Feature selection pipeline
selector_var = VarianceThreshold(threshold=0.01)
selector_mi = SelectKBest(mutual_info_classif, k=800)
```

**File**: `npm_final_combined_features_processed.npy`

---

## 📈 Baseline Results

Performance of **MPAD-Guard** on MPAD-NPM test set (1,682 packages):

| Metric | Value |
|--------|-------|
| **Accuracy** | 96.43% |
| **Precision** | 97.13% |
| **Recall** | 95.69% |
| **F1-Score** | 96.41% |
| **False Positive Rate** | 2.82% |

**Operational Impact:**
- **Inference Latency**: 24ms/batch (GPU), 4.2ms/sample (CPU)
- **Analyst Workload**: 2.82% FPR = ~1 false alarm per 35 packages (3.6× better than GuardDog's 10.24% FPR)

### Comparison with Baselines (11 Methods Evaluated)

| Method | Accuracy | FPR | Citation |
|--------|----------|-----|----------|
| **MPAD-Guard** | **96.43%** | **2.82%** | [Our Work] |
| MalWuKong | 94.25% | 6.78% | Wen et al., 2023 |
| DONAPI | 91.80% | 8.45% | Duan et al., 2023 |
| SpiderScan | 84.50% | 38.20% | (Static heuristics) |
| GuardDog | 89.33% | 10.24% | DataDog, 2022 |

---

## 📖 Citation

If you use MPAD-NPM in your research, please cite:

```bibtex
@misc{mpad_npm_2025,
  author = {Iqbal, Tahir},
  title = {{MPAD-NPM}: Multimodal Package Analysis Dataset for npm Security},
  year = {2025},
  publisher = {Zenodo},
  version = {1.0.0},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://doi.org/10.5281/zenodo.XXXXXXX}
}

@article{iqbal2025mpad,
  author = {Iqbal, Tahir},
  title = {MPAD-Guard: Enhanced Malicious Package Detection via Novel Fusion of API Calls and Dependency Graph in npm Ecosystem},
  journal = {International Journal of Information Security},
  year = {2025},
  publisher = {Springer},
  note = {Manuscript under review}
}
```

**Legacy Version (Mendeley Data):**
The initial dataset release (NPMBench) is available at: [https://doi.org/10.17632/c5rd4kfyv5.1](https://doi.org/10.17632/c5rd4kfyv5.1)

---

## 📝 License

This dataset is released under **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

You are free to:
- **Share**: Copy and redistribute in any format
- **Adapt**: Remix, transform, and build upon the material

Under the following terms:
- **Attribution**: You must give appropriate credit and indicate if changes were made

See [LICENSE](LICENSE) for full details.

---

## 🤝 Contributing

We welcome contributions to expand MPAD-NPM! Please see our contribution guidelines (coming soon) for:
- Submitting newly discovered malicious packages
- Proposing additional feature modalities
- Reporting preprocessing pipeline issues

---

## 🔗 Related Resources

- **MPAD-Guard Framework**: [GitHub Repository](https://github.com/zunetic/MPAD-Guard) *(Coming soon)*
- **Paper Preprint**: [arXiv:XXXX.XXXXX](https://arxiv.org) *(Coming soon)*
- **npm Security Advisory**: [https://www.npmjs.com/advisories](https://www.npmjs.com/advisories)
- **Socket.dev Malware Database**: [https://socket.dev/](https://socket.dev/)

---

## 📧 Contact

For questions or collaboration inquiries:

- **Author**: Tahir Iqbal
- **Affiliation**: School of Software Technology, Dalian University of Technology
- **Email**: tahir.biit@gmail.com
- **ORCID**: [0000-0001-9354-0492](https://orcid.org/0000-0001-9354-0492)
- **GitHub**: [@zunetic](https://github.com/zunetic)

---

## 🙏 Acknowledgments

This dataset builds upon:
- **MalwareBench**: Label sources for malicious packages
- **npm Registry**: Metadata and dependency information
- **Hugging Face**: Pre-trained BERT models for API semantics
- **PyTorch Geometric**: Graph neural network implementations

Special thanks to the open-source security community for vulnerability disclosure and malware analysis.

---

## 📜 Version History

### v1.0.0 (2025-11-19)
- Initial public release
- 11,214 packages with multimodal features
- Preprocessing pipeline and baseline results
- Zenodo archival with DOI

---

**Last Updated**: November 19, 2025  
**Dataset Version**: 1.0.0  
**Repository**: https://github.com/zunetic/MPAD-NPM
