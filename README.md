# MPAD-NPM: Multimodal Package Analysis Dataset for npm Security

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17645147.svg)](https://doi.org/10.5281/zenodo.17645147)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

A reproducible benchmark dataset for detecting malicious packages in the npm ecosystem.

---

## Dataset Overview

**11,214 npm packages** (7,000 benign, 4,214 malicious) from 2015-2024 with multimodal features:
- Metadata (137D): Package characteristics, maintenance signals, dependency metrics, risk indicators
- Graph embeddings (800D): GAT-based dependency structure analysis
- API semantics (768D): BERT-encoded behavioral patterns
- **Final features**: 800D via mutual information feature selection

---

## Quick Start

```python
import numpy as np

# Load data
features = np.load('features/npm_final_combined_features_processed.npy')  # (11214, 800)
labels = np.load('labels/labels_npm_only_processed.npy')  # (11214,)

print(f"Dataset: {features.shape[0]} packages, {features.shape[1]} features")
print(f"Malicious: {labels.sum()} ({labels.sum()/len(labels)*100:.1f}%)")
```

---

## Dataset Statistics

| Category | Benign | Malicious | Total |
|----------|--------|-----------|-------|
| Train (70%) | 4,900 | 2,950 | 7,850 |
| Validation (15%) | 1,050 | 632 | 1,682 |
| Test (15%) | 1,050 | 632 | 1,682 |
| **Total** | **7,000** | **4,214** | **11,214** |

**Attack types**: Code injection (42%), typosquatting (28%), data theft (18%), dependency confusion (8%), resource hijacking (4%)

---

## Repository Files

```
MPAD-NPM/
├── features/
│   ├── npm_final_combined_features_processed.npy    # Final 800D features
│   ├── npmnode_embeddings.npy                       # Graph embeddings
│   └── npm_api_vectorized_features.npy              # API embeddings
├── graphs/
│   └── npmdependency_graph.gpickle                  # Dependency graph
├── labels/
│   └── labels_npm_only_processed.npy                # Binary labels
└── metadata/
    ├── npm_enriched_metadata.csv                    # Package metadata
    └── npm_packages_only.csv                        # Package identifiers
```

---

## Feature Modalities

**Metadata (137D)**: Extracted from package.json and npm registry  
**File**: `metadata/npm_enriched_metadata.csv`

**Graph Embeddings (800D)**: 2-layer GAT with 8 attention heads, global mean pooling  
**File**: `npmnode_embeddings.npy`

**API Semantics (768D)**: BERT-encoded (`bert-base-uncased`) API call sequences tracking `child_process`, `fs`, `net`, `crypto`, `eval`  
**File**: `npm_api_vectorized_features.npy`

**Final Features (800D)**: Mutual information feature selection + variance thresholding  
**File**: `npm_final_combined_features_processed.npy`

---

## Baseline Performance

Evaluation on test set (1,682 packages):

| Metric | Value |
|--------|-------|
| Accuracy | 96.43% |
| Precision | 97.13% |
| Recall | 95.69% |
| F1-Score | 96.41% |
| False Positive Rate | 2.82% |

---

## Installation

```bash
git clone https://github.com/zunetic/MPAD-NPM.git
cd MPAD-NPM
pip install -r requirements.txt
```

**Requirements**: Python 3.8+, NumPy, PyTorch, scikit-learn

---

## Usage Example

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
X = np.load('features/npm_final_combined_features_processed.npy')
y = np.load('labels/labels_npm_only_processed.npy')

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# Train
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious']))
```

---

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{iqbal_2025_mpad_npm,
  author       = {Iqbal, Tahir},
  title        = {{MPAD-NPM: Multimodal Package Analysis Dataset 
                   for npm Security}},
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.17645147},
  url          = {https://doi.org/10.5281/zenodo.17645147}
}
```

---

## License

CC BY 4.0 - Free to share and adapt with attribution

---

## Contact

**Tahir Iqbal**  
School of Software Technology, Dalian University of Technology  
Email: tahir.biit@gmail.com | ORCID: [0000-0001-9354-0492](https://orcid.org/0000-0001-9354-0492)

---

**Version 1.0.0** | November 2025 | DOI: [10.5281/zenodo.17645147](https://doi.org/10.5281/zenodo.17645147)
