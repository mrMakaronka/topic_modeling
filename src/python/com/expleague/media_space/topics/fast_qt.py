import faiss
import numpy as np
from numpy import ma


class FastQt:
    def __init__(self, threshold, cluster_min_size):
        self.threshold = threshold
        self.cluster_min_size = cluster_min_size

    # noinspection PyArgumentList
    def fit(self, X, callback):
        # noinspection PyAttributeOutsideInit
        labels = np.full(len(X), -1)

        index = faiss.IndexFlatL2(X.shape[1])
        index.add(X)
        lims, distances, indices = index.range_search(X, self.threshold)

        lims = np.array([0] + lims, dtype=np.int64)
        counters = np.int64(np.diff(lims))
        left = np.repeat(np.arange(0, len(counters), dtype=np.int64), counters)
        pairs = np.left_shift(left, 32) + np.array(indices, dtype=np.int64)

        cluster_index = 0
        mask = np.zeros(len(counters), dtype=np.bool)
        mask[:] = True
        while True:
            best = np.argmax(counters)
            if counters[best] < self.cluster_min_size:
                break

            cluster_mask = ma.masked_where(((pairs >> 32) != best), pairs, True)
            cluster = cluster_mask.compressed() & np.int64(0xFFFFFFFF)
            counters[cluster] = 0

            labels[cluster] = cluster_index
            callback(cluster, ma.array(distances, mask=cluster_mask.mask).compressed())

            mask[cluster] = False
            pairs = ma.masked_where(~mask[pairs & 0xFFFFFFFF], pairs, False)
            (indices, removes) = np.unique(ma.masked_where(mask[pairs >> 32], pairs, True).compressed() & 0xFFFFFFFF,
                                           return_counts=True)
            counters[indices] -= removes
            mask[cluster] = True

            cluster_index += 1

        return labels
