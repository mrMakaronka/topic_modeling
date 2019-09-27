import re
from abc import abstractmethod
from collections import defaultdict
from typing import Optional, List, Tuple

import faiss
import emoji
import numpy as np
from sklearn.preprocessing import normalize

from com.expleague.media_space.topics.file_read_util import FileReadUtil


class TextNormalizer:
    @abstractmethod
    def normalized_sentences(self, text: str):
        pass


class SimpleTextNormalizer(TextNormalizer):
    def __init__(self):
        self.pattern = re.compile(r'\s+')
        self.punctuation = ['!', '\"', '&', '(', ')', '[', ']', ',', '.', '?', '{', '}']

    def normalized_sentences(self, text: str):
        text = emoji.get_emoji_regexp().sub(r'.', text)
        text = text.replace('?', '.').replace('!', '.')
        sentences = text.split('.')
        for s in sentences:
            s = s.strip().lower().replace('\n', ' ').replace('\t', ' ')
            s = s.translate(str.maketrans({key: None for key in self.punctuation}))
            yield re.sub(self.pattern, ' ', s).split(' ')


class EmbeddingModel:
    @abstractmethod
    def word2vec(self, word: str) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def normalizer(self) -> TextNormalizer:
        pass

    @abstractmethod
    def lexis_distribution(self, all_words: np.ndarray, vec: np.ndarray, result_word_num) -> List[Tuple[str, float]]:
        pass


class FastTextModel(EmbeddingModel):
    def normalizer(self) -> TextNormalizer:
        return self.text_normalizer

    def dimension(self) -> int:
        return self.embeddings.shape[1]

    def lexis_distribution(self, all_words: np.ndarray, vec: np.ndarray, result_word_num) -> List[Tuple[str, float]]:
        unique_indices = np.unique(all_words, return_index=True)[1]
        all_words = all_words[unique_indices]
        all_vecs = np.array([self.word2vec(word) for word in all_words])
        lexis_index = faiss.IndexFlatL2(self.dimension())
        # noinspection PyArgumentList
        lexis_index.add(all_vecs)
        vec = vec.reshape((1, vec.shape[0]))
        vec = vec.astype(np.float32)
        # noinspection PyArgumentList
        dist, ind = lexis_index.search(vec, result_word_num)
        ind = np.concatenate(ind)
        dist = np.concatenate(dist)
        dist = dist / 1000000
        np_sum = np.sum(np.exp(-dist))
        dist = np.exp(-dist) / np_sum
        result = []
        for i in range(len(ind)):
            result.append((all_words[ind[i]], float(dist[i])))
        return result

    def lexis_distribution_for_vec(self, vec: np.ndarray, count: int):
        vec = vec.reshape((1, vec.shape[0]))
        vec = vec.astype(np.float32)
        # noinspection PyArgumentList
        dist, ind = self.complete_index.search(vec, count)
        ind = np.concatenate(ind)
        dist = np.concatenate(dist)
        dist = dist / 1000000
        np_sum = np.sum(np.exp(-dist))
        dist = np.exp(-dist) / np_sum
        result = []
        for i in range(len(ind)):
            result.append((self.words[ind[i]], float(dist[i])))
        return result

    def word2vec(self, word: str) -> Optional[np.ndarray]:
        index = self.words_to_index[word]
        if index == -1:
            return None
        embedding = self.embeddings[index]
        if not embedding.any():
            return None
        return embedding

    def __init__(self, file_path, idf_path):
        self.embeddings, self.words = FileReadUtil.load_fasttext(file_path)
        self.text_normalizer = SimpleTextNormalizer()
        idf = FileReadUtil.load_idf(idf_path)
        self.embeddings = self.embeddings * idf[:, None]
        self.words_to_index = defaultdict(lambda: -1)
        for i in range(len(self.words)):
            self.words_to_index[self.words[i]] = i

        self.complete_index = faiss.IndexFlatL2(self.dimension())
        normalized = normalize(self.embeddings)
        # noinspection PyArgumentList
        self.complete_index.add(normalized)
