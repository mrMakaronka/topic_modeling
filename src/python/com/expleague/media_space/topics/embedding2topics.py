from abc import abstractmethod
from typing import Optional

import faiss
import numpy as np

from com.expleague.media_space.topics.file_read_util import FileReadUtil


class Embedding2Topics:
    @abstractmethod
    def convert(self, embeddings: np.ndarray) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def topic_names(self) -> np.ndarray:
        pass


class Embedding2TopicsClustering(Embedding2Topics):
    def topic_names(self) -> np.ndarray:
        return self.names

    def centroid_by_index(self, index: int) -> np.ndarray:
        return self.centroids[index]

    def convert(self, embeddings: np.ndarray) -> Optional[np.ndarray]:
        result_vec = np.zeros(self.index.ntotal)
        for embedding in embeddings:
            norm_embedding = embedding / np.linalg.norm(embedding)
            norm_embedding = norm_embedding.reshape((1, norm_embedding.shape[0]))
            norm_embedding = norm_embedding.astype(np.float32)

            if len(embeddings) > 1:
                # noinspection PyArgumentList
                lims, dist, ind = self.index.range_search(norm_embedding, self.threshold)
                if len(ind) == 0:
                    continue
                result_vec[ind] += 1 / len(ind)
            else:
                # noinspection PyArgumentList
                dist, ind = self.index.search(norm_embedding, 2)
                np_sum = np.sum(np.exp(-self.scale_dist * dist))
                if np_sum != 0:
                    dist = np.exp(-self.scale_dist * dist) / np_sum
                else:
                    dist = 0
                result_vec[ind] = dist

        if np.count_nonzero(result_vec) == 0:
            return None
        result_vec = result_vec / np.linalg.norm(result_vec, 1)
        return result_vec

    def __init__(self, centroids_path: str, names_path: str, threshold: float, scale_dist: int):
        self.threshold = threshold
        self.centroids = FileReadUtil.load_cluster_centroids(centroids_path)
        self.names = FileReadUtil.load_clusters_names(names_path)
        self.index = faiss.IndexFlatIP(self.centroids.shape[1])
        self.scale_dist = scale_dist
        # noinspection PyArgumentList
        self.index.add(self.centroids)
