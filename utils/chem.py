import h5py
import numpy as np
from scipy.spatial.distance import squareform


def resolve_duplicates(df, smiles_key, dataset_name=""):
    label_cols = [c for c in df.columns if c != smiles_key]
    n_total_before = len(df)

    dup_mask = df[smiles_key].duplicated(keep=False)
    dup_smiles = df.loc[dup_mask, smiles_key].unique()

    if len(dup_smiles) == 0:
        return df.reset_index(drop=True)

    conflicting = []
    true_dupes = []
    for smi in dup_smiles:
        sub = df.loc[df[smiles_key] == smi, label_cols]
        if sub.drop_duplicates().shape[0] > 1:
            conflicting.append(smi)
        else:
            true_dupes.append(smi)

    df_clean = df[~df[smiles_key].isin(conflicting)]
    df_clean = df_clean.drop_duplicates(subset=[smiles_key], keep="first").reset_index(drop=True)

    return df_clean

def load_mces_distance_matrix(dists_lookup_hdf5, no_filter=False):
    with h5py.File(dists_lookup_hdf5, 'r') as f:
        if 'mces' not in f:
            raise KeyError(
                f"HDF5 file '{dists_lookup_hdf5}' is missing required dataset 'mces'"
            )
        if 'mces_smiles_order' not in f:
            raise KeyError(
                f"HDF5 file '{dists_lookup_hdf5}' is missing required dataset "
                "'mces_smiles_order'"
            )

        raw_mces = f['mces'][:]
        raw_smiles = [
            x.decode() if isinstance(x, bytes) else str(x)
            for x in f['mces_smiles_order']
        ]

    if raw_mces.ndim not in (1, 2):
        raise ValueError(
            "'mces' must be a 1-D condensed or 2-D square distance matrix, "
            f"got shape {raw_mces.shape}"
        )
    if raw_mces.ndim == 1:
        n = int(round((1 + (1 + 8 * len(raw_mces)) ** 0.5) / 2))
        if n * (n - 1) // 2 != len(raw_mces):
            raise ValueError(
                f"Condensed 'mces' vector length {len(raw_mces)} does not "
                "correspond to a valid square matrix"
            )
        full_matrix = squareform(raw_mces)
    else:
        if raw_mces.shape[0] != raw_mces.shape[1]:
            raise ValueError(
                f"'mces' must be a square matrix; got shape {raw_mces.shape}"
            )
        if not np.all(np.isfinite(raw_mces)):
            raise ValueError("Distance matrix must contain only finite values")
        if not np.allclose(raw_mces, raw_mces.T, rtol=1e-5, atol=1e-8):
            raise ValueError("'mces' square matrix must be symmetric")
        full_matrix = raw_mces

    if full_matrix.shape[0] != len(raw_smiles):
        raise ValueError(
            f"Distance matrix size ({full_matrix.shape[0]}) does not match "
            f"number of SMILES ({len(raw_smiles)})"
        )
    if not np.all(np.isfinite(full_matrix)):
        raise ValueError("Distance matrix must contain only finite values")
    if np.any(np.diag(full_matrix) != 0):
        raise ValueError(
            "Distance matrix diagonal contains non-zero values; "
            "self-distances must be 0"
        )
    if no_filter:
        print("WARNING: Skipping '-1' Filter, this will also skip any checks for invalid distances!")
    else:
        invalid_mask = np.any(full_matrix == -1, axis=1)
        n_invalid = int(invalid_mask.sum())
        if n_invalid > 0:
            keep = ~invalid_mask
            full_matrix = full_matrix[np.ix_(keep, keep)]
            raw_smiles = [s for s, k in zip(raw_smiles, keep) if k]

            if np.any(full_matrix == -1):
                raise ValueError(
                    "Distance matrix still contains -1 values after removing flagged molecules. "
                    "The -1 entries may not be symmetric — please inspect the source data."
                )
        if not np.all(full_matrix >= 0):
            raise ValueError(
                "Distance matrix contains negative values other than -1 after filtering"
            )

    dists = full_matrix
    dists_smiles = raw_smiles
    dists_smiles_lookup = {s: i for i, s in enumerate(dists_smiles)}
    return dists,dists_smiles_lookup,dists_smiles