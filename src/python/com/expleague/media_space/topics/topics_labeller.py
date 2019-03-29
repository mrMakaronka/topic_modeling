from abc import abstractmethod
from collections import Counter
from typing import List

from com.expleague.media_space.topics.embedding_model import TextNormalizer
from com.expleague.media_space.topics.vec_doc import VecDoc

import numpy as np
# noinspection PyPackageRequirements
from scipy.spatial import distance


class TopicsLabeller:
    @abstractmethod
    def label(self, vec_docs: List[VecDoc]) -> str:
        pass


class NgramTopicsLabeller(TopicsLabeller):
    def __init__(self, ngram: int, normalizer: TextNormalizer):
        self.normalizer = normalizer
        self.ngram = ngram

    def label(self, vec_docs: List[VecDoc]) -> str:
        texts = [doc.article().title for doc in vec_docs]
        freq = Counter()
        for txt in texts:
            for sentence in self.normalizer.normalized_sentences(txt):
                grams = [sentence[i:i + self.ngram] for i in range(len(sentence) - self.ngram + 1)]
                for gram in grams:
                    freq[' '.join(gram)] += 1

        common = freq.most_common(1)
        if len(common) == 0:
            return texts[0].lower()
        return common[0][0]


class EmbeddingTopicsLabeller(TopicsLabeller):
    def __init__(self, cos_threshold=0.3, min_title_len=2):
        self.min_title_len = min_title_len
        self.cos_threshold = cos_threshold

    def label(self, vec_docs: List[VecDoc]) -> str:
        embeddings = np.array([doc.embedding() for doc in vec_docs])
        centroid = np.sum(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        min_len = 1000
        min_len_title = None

        min_dist = 1
        min_dist_title = None

        for doc in vec_docs:
            norm_title = doc.title_vec / np.linalg.norm(doc.title_vec)
            dist = distance.cosine(centroid, norm_title)
            if doc.title_words_len() < min_len and dist < self.cos_threshold:
                min_len = doc.title_words_len()
                min_len_title = doc.article().title.lower()

            if dist < min_dist:
                min_dist = dist
                min_dist_title = doc.article().title.lower()

        if min_len_title is not None:
            return min_len_title
        elif min_dist_title is not None:
            return min_dist_title
        else:
            return vec_docs[0].article().title.lower()
