
from argparse import ArgumentParser
from utils.chem import resolve_duplicates
from split_mces import MCESSplit
import os
import pandas as pd

#TODO adjust pathing and use std instead of _new

ALL_DATASETS_REG = ["delaney","freesolv","lipo"]
ALL_DATASETS_CLASS = ["tox21", "sider", "toxcast", "hiv","clintox", "bbbp", "bace"]

def resolve_datasets(dataset_arg):
    if dataset_arg == "all" or dataset_arg == ["all"]:
        return ALL_DATASETS_CLASS + ALL_DATASETS_REG
    return dataset_arg

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d',  '--dataset',    choices=ALL_DATASETS_CLASS+ ALL_DATASETS_REG + ["all"], required=True,nargs='+')
    parser.add_argument('-dp',  '--dataset_path', default="datasets", help="Choose a base folder where your datasets are stored, this will also be used as output folder")
    parser.add_argument(
    '-sf', '--save_fractional_split',
    nargs=3,
    type=float,
    metavar=('TRAIN', 'VAL', 'TEST'),
    default=None,
    help="Save a fractional split with the given percentages, e.g. -sf 80 10 10."
)

    ### MCES Split Args
    parser.add_argument('-l',  '--linkage',    choices=['average', 'complete', 'single'], default='single')
    parser.add_argument('-t',  '--threshold',  type=int, default=10)
    parser.add_argument('-a', '--alpha', type=float, default=0.3)
    parser.add_argument('-f',  '--folds',      default=5, type=int)
    

    args = parser.parse_args()
    try:
        datasets = resolve_datasets(args.dataset)
    except ValueError as exc:
        parser.error(str(exc))
    for dataset in datasets:
        output_path = f"{args.dataset_path}/{dataset}/splits"
        sep = ","
        input_csv = f'{args.dataset_path}/{dataset}/{dataset}_std.csv'
        df = pd.read_csv(input_csv, sep=sep)
        extension = 'tsv' if sep == '\t' else 'csv'
        smiles_key = df.keys()[0]

        df = resolve_duplicates(df, smiles_key, dataset_name=dataset)
        ms = MCESSplit(
            n_splits=args.folds,
            cluster_mces_threshold=args.threshold,
            linkage=args.linkage,
            alpha=args.alpha,
            dists_lookup_hdf5=f"{args.dataset_path}/{dataset}/{dataset}_mces.hdf5",
        )
        ms.restrict_to_smiles(df[smiles_key].tolist(), dataset_name=dataset)

        df_filtered = (
            df[df[smiles_key].isin(set(ms.dists_smiles))]
            .set_index(smiles_key)
            .loc[ms.dists_smiles]
            .reset_index()
        )

        assert len(df_filtered) == len(ms.dists_smiles), (
            f"df_filtered ({len(df_filtered)}) != dists_smiles ({len(ms.dists_smiles)}) — "
            "some SMILES in the HDF5 are missing from the CSV"
        )
        assert list(df_filtered[smiles_key]) == ms.dists_smiles, \
            "SMILES order mismatch between df_filtered and dists_smiles"

        tags = []
        if args.alpha:
            aggStr = str(args.alpha).replace("0.","")
            tags.append(f"A{aggStr}")
        save_path = os.path.join("mces", *tags, args.linkage, f"T{args.threshold}")
        os.makedirs(os.path.join(output_path, save_path), exist_ok=True)

        if args.save_fractional_split:
            train, val, test = args.save_fractional_split
            if abs(train + val + test - 100) > 1e-6:
                parser.error("Split percentages must sum to 100.")
            train_frac, val_frac, test_frac = train / 100, val / 100, test / 100
            splits_frac = ms.fractional_split(train=train_frac,test=test_frac,val=val_frac,linkage=args.linkage,cluster_mces_threshold=args.threshold,max_size_ratio=args.alpha)
            frac_dir = os.path.join(output_path, save_path, 'fractionalSplits')
            os.makedirs(frac_dir, exist_ok=True)

            for name, indices in zip(('train', 'test', 'val'), splits_frac):
                df_filtered.iloc[indices].to_csv(
                    os.path.join(frac_dir, f'{dataset}_{name}.{extension}'),
                    index=False, sep=sep,
                )
            continue
        
        splits_mces = ms.split(linkage=args.linkage,cluster_mces_threshold=args.threshold,max_size_ratio=args.alpha)
   
        for i, (train_idx, test_idx) in enumerate(splits_mces):
            test_smiles = df_filtered.iloc[test_idx][smiles_key]
            dups = test_smiles[test_smiles.duplicated()]
            assert len(dups) == 0, (
                f"Fold {i} has {len(dups)} duplicate SMILES after iloc: {dups.values[:3]}"
            )
            assert len(set(train_idx) & set(test_idx)) == 0, f"Fold {i} has train/test overlap"
        missing = set(ms.dists_smiles) - set(df[smiles_key])
        if missing:
            raise ValueError(f"{len(missing)} SMILES in HDF5 not found in CSV: {list(missing)[:3]}")


        for i, (train_indices, test_indices) in enumerate(splits_mces):
            df_filtered.iloc[train_indices].to_csv(
                os.path.join(output_path, save_path, f'{dataset}_fold_{i}_train.{extension}'),
                index=False, sep=sep,
            )
            df_filtered.iloc[test_indices].to_csv(
                os.path.join(output_path, save_path, f'{dataset}_fold_{i}_test.{extension}'),
                index=False, sep=sep,
            )
