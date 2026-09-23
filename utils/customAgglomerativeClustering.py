import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import _hierarchical_fast

class CustomAgglomerativeClustering(AgglomerativeClustering):
    def __init__(self, distance_threshold, max_size_ratio=None, **kwargs):
        super().__init__(n_clusters=None, distance_threshold=0, **kwargs)

        self.max_size_ratio = max_size_ratio
        self.new_distance_threshold = distance_threshold

    def _custom_hc_cut(self, n_clusters, children, n_leaves, max_cluster_size):
        node_sizes = np.ones(2 * n_leaves - 1, dtype=int)

        for i, (left, right) in enumerate(children):
            node_sizes[n_leaves + i] = node_sizes[left] + node_sizes[right]

        root = n_leaves + len(children) - 1

        active = {root}

        def split(node):
            active.discard(node)

            left, right = children[node - n_leaves]

            active.add(left)
            active.add(right)

        while len(active) < n_clusters:

            splittable = [n for n in active if n >= n_leaves]

            if not splittable:
                break

            biggest = max(splittable, key=lambda n: n)
            split(biggest)

        if max_cluster_size is not None:
            for _ in range(2 * n_leaves):

                oversized = [
                    n for n in active
                    if node_sizes[n] > max_cluster_size and n >= n_leaves
                ]

                if not oversized:
                    break

                split(max(oversized, key=lambda n: node_sizes[n]))

        label = np.zeros(n_leaves, dtype=np.intp)

        for i, node in enumerate(sorted(active)):
            label[_hierarchical_fast._hc_get_descendent(node, children, n_leaves)] = i

        return label

    def fit(self, X, y=None):
        super().fit(X, y)

        self.n_clusters_ = (
            np.count_nonzero(self.distances_ >= self.new_distance_threshold) + 1
        )

        max_cluster_size = (
            int(self.n_leaves_ * self.max_size_ratio)
            if self.max_size_ratio is not None else None
        )

        self.labels_ = self._custom_hc_cut(
            self.n_clusters_,
            self.children_,
            self.n_leaves_,
            max_cluster_size,
        )

        self.n_clusters_ = len(np.unique(self.labels_))
        return self
