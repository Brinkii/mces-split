# MCES-Split: Reliable evaluation for small molecule machine learning

This repository contains the code accompanying the ICLR 2027 submission **"Reliable evaluation for small molecule machine learning"**.

It provides a splitting procedure that keeps structurally similar molecules out of the train/test boundary. Molecules are clustered by agglomerative clustering on precomputed pairwise **MCES (Maximum Common Edge Subgraph) distances**. Whole clusters are then assigned to folds or partitions, so no cluster spans train and test.

> **Note:** This is an anonymized version for peer review. A full release will follow after the review process.

---

## Repository structure

```
.
├── main.py                 # Entry point: runs splitting on the provided datasets
├── split_mces.py           # MCESSplit class (clustering + fold/partition assignment)
├── utils/
│   ├── chem.py             # Loading MCES distance matrices from HDF5
│   └── customAgglomerativeClustering.py  # Agglomerative clustering with a max cluster-size constraint
├── datasets/               # CSVs / precomputed distance files 
└── pyproject.toml
```

## Installation

Requires Python ≥ 3.11.

```bash
cd mces-split
uv sync
```

## Data

The splitter expects, for each dataset:

1. A cleaned CSV of molecules (SMILES column: `smiles_std`).
2. An HDF5 file with the precomputed pairwise MCES distance matrix and the corresponding SMILES.

Bace and FreeSolv are examplatroy and addional files will be downloadable after de-anonymization.

If the HDF5 contains SMILES that are missing from the cleaned CSV, they are dropped automatically with a warning (`MCESSplit.restrict_to_smiles`).

## Usage

### Command line

```bash
python main.py -d dataset -l single -t 10 -a 0.3
```

### Python API

```python
from split_mces import MCESSplit

splitter = MCESSplit(
    n_splits=5,                  # number of CV folds
    cluster_mces_threshold=10,   # MCES distance threshold for merging clusters
    linkage="single",            # agglomerative linkage: single / complete / average
    dists_lookup_hdf5="path/to/dists.h5",
    alpha=None,                  # optional max cluster size ratio (see below)
)

# Optionally align the distance matrix with your cleaned dataset
splitter.restrict_to_smiles(df["smiles"], dataset_name="my_dataset")

# (a) k-fold cross-validation: GroupKFold over MCES clusters
folds = splitter.split()                 # list of (train_idx, test_idx)

# (b) single train / test / val partition
train_idx, test_idx, val_idx = splitter.fractional_split(train=0.8, test=0.1, val=0.1)
```

**Key parameters**

| Parameter | Description |
|---|---|
| `cluster_mces_threshold` | Molecules within this MCES distance are merged into the same cluster. Higher values give stricter, more out-of-distribution splits. |
| `linkage` | Linkage criterion for agglomerative clustering. `single` guarantees no cross-split pair falls below the threshold. |
| `alpha` (`max_size_ratio`) | If set, uses a constrained clustering that caps cluster size relative to the dataset, preventing one giant cluster from making balanced splits impossible. |

`fractional_split` Produces `train`, `val` and `test` splits for given fraction (e.g. 80/10/10). |

## License

License to be added upon de-anonymization.
