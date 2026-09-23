import numpy as np
from sklearn.model_selection import GroupKFold
from utils.customAgglomerativeClustering import CustomAgglomerativeClustering
from utils.chem import load_mces_distance_matrix

class   MCESSplit:
    def __init__(self, n_splits=5, cluster_mces_threshold=10, linkage="single",
                 dists_lookup_hdf5=None, alpha=None):
        self.max_size_ratio = alpha
        self.n_splits = n_splits
        self.linkage = linkage
        self.cluster_mces_threshold = cluster_mces_threshold
        if dists_lookup_hdf5 is not None:
            self.dists_lookup_hdf5 = dists_lookup_hdf5
            self._load_dists()
        else:
            self.dists = self.dists_smiles = self.dists_smiles_lookup = None

    def _load_dists(self):
        dists,dists_smiles_lookup,dists_smiles = load_mces_distance_matrix(self.dists_lookup_hdf5)
        self.dists = dists
        self.dists_smiles = dists_smiles
        self.dists_smiles_lookup = dists_smiles_lookup

    def restrict_to_smiles(self, keep_smiles, dataset_name=""):
        keep_set = set(keep_smiles)
        keep_idx = [i for i, s in enumerate(self.dists_smiles) if s in keep_set]

        if len(keep_idx) == len(self.dists_smiles):
            return 
        
        if len(keep_idx) == 0:
            raise ValueError(
                f"{dataset_name}: none of the HDF5 SMILES are present in the cleaned CSV — "
                "cannot proceed with split."
            )

        self.dists = self.dists[np.ix_(keep_idx, keep_idx)]
        self.dists_smiles = [self.dists_smiles[i] for i in keep_idx]
        self.dists_smiles_lookup = {smi: i for i, smi in enumerate(self.dists_smiles)}
    
    def split(self,linkage=None,cluster_mces_threshold=None,max_size_ratio=None):
        gkf = GroupKFold(n_splits=self.n_splits)
        cluster_ids = self._cluster(linkage,cluster_mces_threshold,max_size_ratio)
        try:
            splits =  list(gkf.split(self.dists_smiles, groups=cluster_ids))
        except Exception as e:
            print(f"Skipping {self.linkage}_{self.cluster_mces_threshold} due to {e}")
            return None
        return splits

    def fractional_split(self, train=0.8, test=0.1, val=0.1,
                     linkage=None, cluster_mces_threshold=None,
                     max_size_ratio=None):
        fracs = [train, test] + ([] if val in (None, 0) else [val])
        total = float(sum(fracs))
        fracs = [f / total for f in fracs]
        names = ["train", "test", "val"][:len(fracs)]

        cluster_ids = np.asarray(self._cluster(linkage, cluster_mces_threshold,
                                            max_size_ratio))
        n_total = len(cluster_ids)
        tag = (f"{self.linkage}_{self.cluster_mces_threshold}_{max_size_ratio}_"
            + "/".join(f"{f:.3g}" for f in fracs))

        clusters = [np.where(cluster_ids == c)[0] for c in np.unique(cluster_ids)]
        clusters.sort(key=len, reverse=True)
        parts = [[] for _ in fracs]
        filled = [0] * len(fracs)
        for idx in clusters:
            b = max(range(len(fracs)), key=lambda b: fracs[b] * n_total - filled[b])
            parts[b].append(idx)
            filled[b] += len(idx)

        empty = [names[i] for i, p in enumerate(parts) if not p]
        if empty:
            print(f"Skipping {tag}: empty partition(s) {empty}")
            return None
        parts = [np.sort(np.concatenate(p)) for p in parts]

        part_of = np.empty(n_total, dtype=int)
        for i, p in enumerate(parts):
            part_of[p] = i
        for c in np.unique(cluster_ids):
            idx = np.where(cluster_ids == c)[0]
            if len(np.unique(part_of[idx])) != 1:
                raise AssertionError(
                    f"cluster {c} spans partitions {np.unique(part_of[idx])} -- "
                    f"cluster_ids and split indices are misaligned")
        return tuple(parts)

    def _cluster(self,linkage=None,cluster_mces_threshold=None,max_size_ratio=None):
        if linkage is not None:
            self.linkage = linkage
        if cluster_mces_threshold is not None:
            self.cluster_mces_threshold = cluster_mces_threshold
        if max_size_ratio is not None:
            self.max_size_ratio = max_size_ratio
        
        clustering = CustomAgglomerativeClustering(
                metric='precomputed',
                linkage=self.linkage,
                distance_threshold=self.cluster_mces_threshold,
                max_size_ratio=self.max_size_ratio
            ).fit(self.dists)                    
        return np.array(clustering.labels_)
