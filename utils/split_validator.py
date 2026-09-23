import warnings
from collections import Counter
from pathlib import Path
from typing import Optional, Union
from scipy.spatial.distance import squareform
import h5py
import numpy as np
import pandas as pd

from utils.chem import load_mces_distance_matrix

# add to main for val:
# validate_dirs = []
        # validate_dirs.append((dataset, os.path.join(output_path, save_path)))
# from utils.split_validator import validate_splits
# for dataset, fold_dir in validate_dirs:
#     validate_splits(
#         dataset=dataset,
#         fold_dir=fold_dir,
#         mces_path=f"{args.dataset_path}/{dataset}/{dataset}_mces.hdf5",
#         n_splits=args.folds,
#         dataset_path=args.dataset_path,
#     )


def inspect_hdf5(hdf5_path: Union[str, Path]) -> None:
    """
    Print a summary of all datasets in an HDF5 file.

    Useful for checking key names and data shapes before calling
    load_mces_distance_matrix.

    Parameters
    ----------
    hdf5_path:
        Path to the HDF5 file to inspect.
    """
    with h5py.File(hdf5_path, "r") as f:
        print(f"Keys in {hdf5_path}:")
        for key in f.keys():
            ds = f[key]
            print(f"  {key!r:<25} shape={ds.shape}   dtype={ds.dtype}")
        smiles_candidates = [k for k in f.keys() if "smiles" in k.lower()]
        if smiles_candidates:
            key = smiles_candidates[0]
            preview = [
                x.decode() if isinstance(x, bytes) else str(x)
                for x in f[key][:3]
            ]
            print(f"First 3 entries in {key!r}: {preview}")
        if "mces" in f.keys():
            key = "mces"
            print(f"MCES matrix: {key!r}")
            raw = f[key][:]
            print(f"  raw shape   : {raw.shape}   dtype: {raw.dtype}")

            # squareform accepts either a condensed 1-D vector or a full square matrix
            M = squareform(raw)
            n = M.shape[0]
            print(f"  square shape: {M.shape}")

            # --- diagonal: should be all 0 (squareform guarantees this, but verify raw data) ---
            diag = np.diag(M)
            nonzero_diag = np.count_nonzero(diag)
            print(f"  Diagonal")
            print(f"    all zero  : {nonzero_diag == 0}")
            if nonzero_diag:
                print(f"    non-zero entries : {nonzero_diag}  (first indices: {np.where(diag != 0)[0][:5].tolist()})")

            # --- off-diagonal: should be > 0 ---
            # squareform output is symmetric, so upper triangle == lower triangle transposed;
            # work only on the condensed vector to avoid double-counting
            condensed = squareform(M)  # back to 1-D, no diagonal
            neg       = np.sum(condensed < 0)
            zero_off  = np.sum(condensed == 0)
            print(f"  Off-diagonal values  (n={len(condensed)} pairs)")
            print(f"    negative  : {neg}")
            print(f"    == 0      : {zero_off}{'  ⚠ unexpected zeros' if zero_off else ''}")
            print(f"    min       : {condensed.min():.4f}")
            print(f"    max       : {condensed.max():.4f}")
            print(f"    mean      : {condensed.mean():.4f}")
            print(f"    median    : {np.median(condensed):.4f}")

            # --- symmetry sanity-check on the raw stored data (if it was already square) ---
            if raw.ndim == 2:
                sym_diff = np.abs(raw - raw.T)
                print(f"  Symmetry of raw stored matrix")
                print(f"    max |M - M^T|         : {sym_diff.max():.2e}")
                print(f"    asymmetric pairs (>1e-6): {np.sum(sym_diff > 1e-6)}")

            # --- text histogram of pairwise distances ---
            print("  Distribution of pairwise distances:")
            labels = ["<0", "0", ">0-5", ">5-10",">10-20", ">20-29", "30+"]
            counts = [
                np.sum(condensed < 0),
                np.sum(condensed == 0),
                np.sum((condensed > 0) & (condensed <= 5)),
                np.sum((condensed > 5) & (condensed <= 10)),
                np.sum((condensed > 10) & (condensed <= 20)),
                np.sum((condensed > 20) & (condensed <= 29)),
                np.sum(condensed > 29)
            ]

            bar_max = 30
            max_count = max(counts)
            scale = bar_max / max_count if max_count > 0 else 1

            for label, c in zip(labels, counts):
                bar = "█" * int(c * scale)
                print(f"    {label:>7} {bar:<{bar_max}} {c}")
        else:
            print("No MCES key found in file.")

def fold_summary(
    folds: dict[int, list[int]],
    smiles: list[str],
    distance_matrix: np.ndarray,
) -> None:
    """
    Print fold sizes, unique scaffold counts, and inter-fold min-MCES distances.

    Parameters
    ----------
    folds:
        Mapping of fold_id → list of test-set molecule indices.
    smiles:
        Full dataset SMILES list (index-aligned with distance_matrix).
    distance_matrix:
        Square NxN distance matrix.
    """
    n = len(smiles)
    n_splits = len(folds)

    print(f"{'Fold':>5}  {'N mols':>8}  {'% total':>8}")
    print("-" * 42)
    for fold_id in sorted(folds):
        indices = folds[fold_id]
        n_mols = len(indices)
        print(f"{fold_id:>5}  {n_mols:>8}  {100 * n_mols / n:>7.1f}% ")
    print("Test → Train MCES (higher = harder split):")
    print(f"{'Fold':>6}  {'mean-min':>10}  {'min':>10}")
    print("-" * 30)

    fold_indices = {fold_id: np.array(indices) for fold_id, indices in folds.items()}
    all_indices = np.arange(n)

    for fold_id in sorted(folds):
        test_idx  = fold_indices[fold_id]
        train_idx = np.setdiff1d(all_indices, test_idx)

        sub = distance_matrix[np.ix_(test_idx, train_idx)]
        per_mol_min = sub.min(axis=1)
        mean_min = per_mol_min.mean()
        min_mces = per_mol_min.min()
        print(f"{fold_id:>6}  {mean_min:>10.3f}  {min_mces:>10.3f}")

def load_folds_from_dir(
    fold_dir: Union[str, Path],
    smiles: list[str],
    distance_matrix: Union[np.ndarray, str, Path],
    dataset: Optional[str] = None,
    n_splits: int = 5,
    hdf5_smiles_key: str = "mces_smiles_order",
    hdf5_distances_key: str = "mces",
) -> tuple[dict[int, list[int]], np.ndarray]:
    """
    Load pre-computed fold assignments from a directory and run fold_summary.

    Supports two directory layouts (auto-detected):

    Layout A — split into train/test per fold (e.g. DeepChem / your pipeline)::

        fold_dir/
            {dataset}_fold_0_train.csv
            {dataset}_fold_0_test.csv
            {dataset}_fold_1_train.csv
            {dataset}_fold_1_test.csv
            ...

        Pass `dataset` (e.g. "tox21") to enable this layout.
        The test indices for each fold become that fold's held-out set;
        the union of all molecules across all files is used to verify coverage.

    Layout B — one file per fold::

        fold_dir/
            fold_0.csv   fold_1.csv  ...   # CSV with smiles/index column or plain list
            fold_0.txt   fold_1.txt  ...

    For CSV files the loader looks for a column named 'smiles', 'smiles_std',
    or 'index' (case-insensitive). If none found, assumes first column is SMILES.

    Parameters
    ----------
    fold_dir          : directory containing the fold files
    smiles            : list of N SMILES strings for the full dataset
    distance_matrix   : NxN numpy array or path to HDF5
    dataset           : dataset name prefix (e.g. "tox21") — required for Layout A
    n_splits          : number of folds (default 5)
    hdf5_smiles_key   : HDF5 SMILES key
    hdf5_distances_key: HDF5 distances key

    Returns
    -------
    folds : dict[int, list[int]]
        Keys are fold IDs, values are test-set molecule indices (0-based,
        aligned to `smiles`).
    """
    fold_dir = Path(fold_dir)
    # smiles_lookup = {s: i for i, s in enumerate(smiles)}

    # Resolve distance matrix if path given
    if isinstance(distance_matrix, (str, Path)):
        distance_matrix,smiles_lookup,dists_smiles = load_mces_distance_matrix(dists_lookup_hdf5=distance_matrix)
        #This block is needed since we removed conflicting SMILES pairs while we still computed the mces for all of them 
        # (eg If there are 22 duplicates we calculated mces for 11 (kept first) but the csv removed all 22 resulting in a missmatch)
        kept_smiles = [s for s in smiles if s in smiles_lookup]
        missing     = [s for s in smiles if s not in smiles_lookup]
        if missing:
            print(f"{len(missing)} smiles not in distance matrix, e.g. {missing[:5]}")

        idx = np.asarray([smiles_lookup[s] for s in kept_smiles], dtype=int)

        distance_matrix = distance_matrix[np.ix_(idx, idx)]          # (n, n)
        smiles_lookup = {s: i for i, s in enumerate(kept_smiles)}
        dists_smiles = smiles

    def _parse_csv(path: Path) -> list[int]:
        """Parse a CSV (or txt) fold file into a list of 0-based dataset indices."""
        import csv
        if path.suffix.lower() != ".csv":
            lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
            try:
                return [int(l) for l in lines]
            except ValueError:
                return [smiles_lookup[l] for l in lines]

        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            fieldnames = reader.fieldnames or []
            fl = [f.lower() for f in fieldnames]
            smiles_col = next(
                (fieldnames[i] for i, f in enumerate(fl)
                 if f in ("smiles", "smiles_std", "smiles_standardized")), None)
            index_col = next(
                (fieldnames[i] for i, f in enumerate(fl)
                 if f in ("index", "idx", "mol_index")), None)
            rows = list(reader)

        if index_col:
            return [int(r[index_col]) for r in rows]
        elif smiles_col:
            missing = [r[smiles_col] for r in rows if r[smiles_col] not in smiles_lookup]
            if missing:
                raise KeyError(
                    f"{len(missing)} SMILES in {path} not found in dataset. "
                    f"First few: {missing[:3]}"
                )
            return [smiles_lookup[r[smiles_col]] for r in rows]
        else:
            # No recognised column — assume first column is SMILES
            first_col = fieldnames[0]
            return [smiles_lookup[r[first_col]] for r in rows]

    folds: dict[int, list[int]] = {}

    if dataset is not None:
        # Layout A: {dataset}_fold_{N}_test.csv / {dataset}_fold_{N}_train.csv
        for fold_id in range(n_splits):
            test_path  = fold_dir / f"{dataset}_fold_{fold_id}_test.csv"
            train_path = fold_dir / f"{dataset}_fold_{fold_id}_train.csv"
            if not test_path.exists():
                raise FileNotFoundError(
                    f"Expected {test_path} — not found. "
                    f"Check fold_dir and dataset name."
                )
            folds[fold_id] = _parse_csv(test_path)
            n_train = len(_parse_csv(train_path)) if train_path.exists() else "?"
            print(f"  Fold {fold_id}: {n_train} train  |  {len(folds[fold_id])} test  "
                  f"({test_path.name})")
    else:
        # Layout B: fold_0.csv / fold_0.txt
        for fold_id in range(n_splits):
            found = None
            for pattern in [f"fold_{fold_id}.csv", f"fold_{fold_id}.txt",
                            f"fold{fold_id}.csv",  f"fold{fold_id}.txt"]:
                p = fold_dir / pattern
                if p.exists():
                    found = p
                    break
            if found is None:
                raise FileNotFoundError(
                    f"Could not find fold {fold_id} in {fold_dir}. "
                    f"Pass dataset= to use the train/test naming convention."
                )
            folds[fold_id] = _parse_csv(found)
            print(f"  Loaded fold {fold_id} from {found.name}  ({len(folds[fold_id])} molecules)")

    # Sanity: check for overlap between test folds
    seen: set[int] = set()
    for fold_id, indices in folds.items():
        overlap = seen & set(indices)
        if overlap:
            warnings.warn(
                f"Fold {fold_id} shares {len(overlap)} indices with earlier folds.",
                UserWarning,
            )
        seen.update(indices)

    print("")
    fold_summary(folds, dists_smiles, distance_matrix)
    return folds,distance_matrix


def validate_splits(
    dataset: str,
    fold_dir: Union[str, Path],
    mces_path: Union[str, Path],
    n_splits: int = 5,
    hdf5_smiles_key: str = "mces_smiles_order",
    hdf5_distances_key: str = "mces",
    dataset_path: str = "datasets",
) -> None:
    """
    Run the full validation suite on a saved split directory.

    Checks performed
    ----------------
    1. CSV sanity — fold sizes and columns present.
    2. Cross-fold test duplicate detection (SMILES-based).
    3. HDF5 inspection via inspect_hdf5.
    4. Fold loading via load_folds_from_dir.
    5. Zero-distance pair assertion — two distinct post-dedup molecules
       should never share distance 0.
    6. Unique test index coverage across all folds.
    7. fold_summary — sizes, scaffold counts, inter-fold min-MCES distances.

    Parameters
    ----------
    dataset:
        Dataset name (e.g. ``"hiv"``).
    fold_dir:
        Directory containing the split CSVs.
    mces_path:
        Path to the HDF5 file with precomputed MCES distances.
    n_splits:
        Number of folds (default 5).
    hdf5_smiles_key:
        HDF5 key for the SMILES array.
    hdf5_distances_key:
        HDF5 key for the distance matrix.
    """
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    fold_dir = Path(fold_dir)
    input_csv = f"{dataset_path}/{dataset}/{dataset}_cleaned.csv"

    assert Path(input_csv).exists(), f"Dataset CSV not found: {input_csv!r}"
    df = pd.read_csv(input_csv)
    smiles_col = next(
        (c for c in df.columns if "smiles" in c.lower()),
        df.columns[0],
    )
    df_filtered = df

    n_total = len(df_filtered)
    assert n_total >= n_splits, (
        f"[{dataset}] Only {n_total} samples but requested {n_splits} folds. "
        f"Cannot create non-empty folds."
    )
    smiles = df_filtered[smiles_col].tolist()
    print("[%s] Loaded %d molecules from %s", dataset, len(smiles), input_csv)

    print(f"{'='*60}")
    print(f"Validating: {dataset}  |  {fold_dir}")
    print(f"{'='*60}")

    print(">>> [1/5] CSV sanity check")
    for fold_i in range(n_splits):
        totallen = 0
        for split_label in ("train", "test"):
            path = fold_dir / f"{dataset}_fold_{fold_i}_{split_label}.csv"
            assert path.exists(), f"Missing split file: {path}"
            split_df = pd.read_csv(path)
            assert len(split_df) > 0, f"Empty file: {path}"
            assert smiles_col in split_df.columns, (
                f"{path} is missing expected SMILES column {smiles_col!r}. "
                f"Found: {list(split_df.columns)}"
            )
            print("  %s — %d rows", path.name, len(split_df))
            totallen += len(split_df)
        print("  %s — %d rows", f"{dataset}_fold_{fold_i}_total", totallen)
    print("  OK")

    print(">>> [2/5] Cross-fold test duplicate check (SMILES)")
    all_test_smiles: dict[str, int] = {}
    duplicates_found = False
    for fold_i in range(n_splits):
        test_df = pd.read_csv(fold_dir / f"{dataset}_fold_{fold_i}_test.csv")
        for smi in test_df[smiles_col]:
            if smi in all_test_smiles:
                print(
                    "Duplicate SMILES in fold %d and fold %d: %s",
                    all_test_smiles[smi], fold_i, smi,
                )
                duplicates_found = True
            else:
                all_test_smiles[smi] = fold_i
    assert not duplicates_found, (
        "Cross-fold SMILES duplicates detected — see warnings above."
    )
    print("  OK")

    print(">>> [3/5] HDF5 inspection")
    inspect_hdf5(mces_path)
    print("")

    print(">>> [4/5] Loading folds and running fold_summary")
    folds, D = load_folds_from_dir(
        fold_dir=fold_dir,
        smiles=smiles,
        distance_matrix=mces_path,
        dataset=dataset,
        n_splits=n_splits,
        hdf5_smiles_key=hdf5_smiles_key,
        hdf5_distances_key=hdf5_distances_key,
    )
    print("")

    print(">>> [5/5] Zero-distance pair assertion")
    zero_pairs = [(i, j) for i, j in zip(*np.where(D == 0)) if i < j]
    real_duplicates = [(i, j) for i, j in zero_pairs if smiles[i] == smiles[j]]
    different_mols = [(i, j) for i, j in zero_pairs if smiles[i] != smiles[j]]
    if different_mols:
        # limit = len(different_mols) if is_logging_to_file(logger) else 5 #TODO mb add another args to print full list but for now just one
        limit = 1
        print(
            f"{len(different_mols)} zero-distance pair(s) found between different molecules. Showing {limit}:" + 
            "".join(f"  [{i}] {smiles[i]}  |  [{j}] {smiles[j]}" for i, j in different_mols[:limit])
        )
    else:
        print(f"  OK — no zero-distance pairs found")
    assert len(real_duplicates) == 0, (
        f"{len(real_duplicates)} exact duplicate(s) found. First few:" + 
        "".join(f"  [{i}] {smiles[i]}  |  [{j}] {smiles[j]}" for i, j in real_duplicates[:5])
    )
    

    idx_to_fold = {}
    for fold_id, test_indices in folds.items():
        for idx in test_indices:
            idx_to_fold[idx] = fold_id

    cross_fold_zeros = [
        (i, j) for i, j in zero_pairs
        if idx_to_fold.get(i) != idx_to_fold.get(j)
    ]
    same_fold_zeros = [
        (i, j) for i, j in zero_pairs
        if idx_to_fold.get(i) == idx_to_fold.get(j)
    ]

    if cross_fold_zeros:
        limit = 5
        print(
            f"{len(cross_fold_zeros)} zero-distance pair(s) span DIFFERENT folds "
            f"({len(same_fold_zeros)} are within the same fold). "
            f"This risks data leakage. Showing up to {limit}:" +
            "".join(
                f"  [{i}] fold={idx_to_fold.get(i)} {smiles[i]}  |  "
                f"[{j}] fold={idx_to_fold.get(j)} {smiles[j]}"
                for i, j in cross_fold_zeros[:limit]
            )
        )
    else:
        print(f"  OK — all {len(same_fold_zeros)} zero-distance pair(s) are within the same fold")

    all_test_indices = [idx for indices in folds.values() for idx in indices]
    unique_test = set(all_test_indices)
    print(
        "Total test assignments: %d  |  Unique: %d  |  Dataset size: %d",
        len(all_test_indices), len(unique_test), len(smiles),
    )

    counts = Counter(all_test_indices)
    duplicated_indices = {idx: [] for idx, c in counts.items() if c > 1}
    if duplicated_indices:
        for fold_id, indices in folds.items():
            for idx in indices:
                if idx in duplicated_indices:
                    duplicated_indices[idx].append(fold_id)
        print(
            "%d index/indices appear in multiple test folds:",
            len(duplicated_indices),
        )
        for idx, fold_ids in list(duplicated_indices.items())[:10]:
            print("  [%d] in folds %s: %s", idx, fold_ids, smiles[idx])

    print(f"{'='*60}")
    print(f"Validation complete: {dataset}")
    print(f"{'='*60}")
