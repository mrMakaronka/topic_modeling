from datetime import datetime
from typing import List

import dateutil
import numpy as np

from com.expleague.media_space.publisher_parser import PublisherParser
from com.expleague.media_space.article import Article
from com.expleague.media_space.topics.date_utils import DateUtils
from com.expleague.media_space.topics.embedding_model import EmbeddingModel


class VecDoc:
    def __init__(self, article: Article, embedding_model: EmbeddingModel, min_sentence_length: int):
        self.min_sentence_len = min_sentence_length
        self.embedding_model = embedding_model
        self.doc_article = article
        self.publisher = PublisherParser.parse(article.publisher)
        self.dt = DateUtils.normalize(dateutil.parser.parse(article.pub_datetime))

        self.built = False
        self.embedding_vec = None
        self.unique_words = None
        self.sentence_vecs = None
        self.title_vec = None
        self.title_len = None

    def article(self) -> Article:
        return self.doc_article

    def parsed_publisher(self) -> str:
        return self.publisher

    def date(self) -> datetime:
        return self.dt

    def valid(self) -> bool:
        self._build()
        return (self.embedding_vec is not None) and (self.title_vec is not None)

    def embedding(self) -> np.ndarray:
        self._build()
        return self.embedding_vec

    def title_embedding(self) -> np.ndarray:
        self._build()
        return self.title_vec

    def title_words_len(self) -> int:
        self._build()
        return self.title_len

    def embedding_sentences(self) -> List[np.ndarray]:
        self._build()
        return self.sentence_vecs

    def words(self) -> np.ndarray:
        self._build()
        return self.unique_words

    def _build(self) -> None:
        if self.built:
            return
        self.built = True

        sentence_vecs, unique_words = self._text2embedding(self.doc_article.text)
        article_vec = np.sum(sentence_vecs, axis=0, dtype=np.float32)
        if np.count_nonzero(article_vec) == 0:
            return

        title_vecs, title_words = self._text2embedding(self.doc_article.title)
        title_vec = np.sum(title_vecs, axis=0, dtype=np.float32) / len(title_vecs) if len(title_vecs) != 0 else 0
        if np.count_nonzero(title_vec) == 0:
            return

        self.title_len = len(title_words)
        self.title_vec = title_vec
        self.embedding_vec = article_vec
        self.unique_words = np.unique(unique_words)
        self.sentence_vecs = sentence_vecs

    def _text2embedding(self, text: str):
        sentence_vecs = []
        unique_words = []
        for sentence in self.embedding_model.normalizer().normalized_sentences(text):
            if len(sentence) < self.min_sentence_len:
                continue
            word_vecs = [
                (word, self.embedding_model.word2vec(word) / np.linalg.norm(self.embedding_model.word2vec(word)))
                for word in sentence if
                self.embedding_model.word2vec(word) is not None]
            if len(word_vecs) == 0:
                continue
            words, vecs = zip(*word_vecs)
            unique_words.extend(words)
            sentence_vecs.append(np.sum(vecs, axis=0))
        return sentence_vecs, unique_words
