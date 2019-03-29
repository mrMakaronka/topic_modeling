import uuid
from datetime import datetime
from typing import List, Tuple

import numpy as np
from sklearn.preprocessing import normalize

from com.expleague.media_space.topics.embedding2topics import Embedding2Topics
from com.expleague.media_space.topics.embedding_model import EmbeddingModel
from com.expleague.media_space.topics.fast_qt import FastQt
from com.expleague.media_space.topics.items import StoryItem, NewsItem
from com.expleague.media_space.topics.topics_labeller import TopicsLabeller
from com.expleague.media_space.topics.topics_matching import TopicsMatching
from com.expleague.media_space.topics.vec_doc import VecDoc


class NewsCluster(NewsItem):
    def __init__(self, item_id: str, name: str, dt: datetime, story: StoryItem, vec_sum: np.ndarray, vec_num: int,
                 topics: np.ndarray,
                 cluster_docs: List[VecDoc]):
        super().__init__(item_id, name, dt, story, vec_sum, vec_num, topics)
        self.cluster_docs = cluster_docs

    def docs(self) -> List[VecDoc]:
        return self.cluster_docs

    def set_story(self, story: StoryItem):
        self.item_story = story


class NewsClustering:
    def __init__(self, threshold: float, min_size: int, embedding2topics: Embedding2Topics,
                 topics_labeller: TopicsLabeller):
        self.min_size = min_size
        self.threshold = threshold
        self.embedding2topics = embedding2topics
        self.topics_labeller = topics_labeller

    def cluster(self, docs: List[VecDoc]) -> List[NewsCluster]:
        if len(docs) == 0:
            return list()
        vectors_for_clustering = np.zeros((len(docs), len(docs[0].embedding())), dtype=np.float32)
        for j in range(len(docs)):
            vectors_for_clustering[j] = docs[j].embedding()
        result = []

        def news_clustering_callback(indices):
            seen_sources = set()
            unique_source_indices = []
            for ind in indices:
                if docs[ind].parsed_publisher() is not None and docs[ind].parsed_publisher() not in seen_sources:
                    seen_sources.add(docs[ind].parsed_publisher())
                    unique_source_indices.append(ind)
                elif docs[ind].parsed_publisher() is None:
                    unique_source_indices.append(ind)

            unique_source_indices = np.array(unique_source_indices, dtype=np.int)
            if len(unique_source_indices) < self.min_size:
                return

            labelled_news[unique_source_indices] = 0
            all_articles_sentences = np.concatenate([docs[i].embedding_sentences() for i in unique_source_indices])
            topics_vec = self.embedding2topics.convert(all_articles_sentences)
            if topics_vec is None:
                return

            news_id = str(uuid.uuid4())
            vec_docs = [docs[i] for i in unique_source_indices]
            name = self.topics_labeller.label(vec_docs)
            dates = [docs[i].dt for i in unique_source_indices]
            vec_sum = np.sum(vectors_for_clustering[unique_source_indices], axis=0)
            # noinspection PyTypeChecker
            result.append(
                NewsCluster(news_id, name, max(dates), None, vec_sum, len(unique_source_indices),
                            topics_vec, [docs[i] for i in unique_source_indices]))

        labelled_news = np.ones(len(docs), dtype=np.bool)
        FastQt(self.threshold, self.min_size).fit(normalize(vectors_for_clustering, axis=1, norm='l2'),
                                                  lambda indices, distances: news_clustering_callback(indices))
        non_labelled_articles = np.nonzero(labelled_news)
        for j in non_labelled_articles[0]:
            doc = docs[j]
            article_sentences = np.array(doc.embedding_sentences(), dtype=np.object)
            result_vec = self.embedding2topics.convert(article_sentences)
            if result_vec is None:
                continue

            new_id = str(uuid.uuid4())
            label = self.topics_labeller.label([doc])
            # noinspection PyTypeChecker
            result.append(NewsCluster(new_id, label, doc.dt, None, vectors_for_clustering[j], 1, result_vec, [doc]))

        return result


class StoryCluster(StoryItem):
    def __init__(self, story_id: str, name: str, dt: datetime, topics: List[Tuple[str, float]],
                 lexis: List[Tuple[str, float]], vec_sum: np.ndarray, vec_num: int, news_clusters: List[NewsCluster]):
        super().__init__(story_id, name, dt, topics, lexis, vec_sum, vec_num)
        self.clusters = news_clusters

    def news_clusters(self) -> List[NewsCluster]:
        return self.clusters


class StoriesClustering:
    def __init__(self, threshold: float, min_size: int, topics_matching: TopicsMatching,
                 topics_labeller: TopicsLabeller, embedding_model: EmbeddingModel, lexis_len: int):
        self.threshold = threshold
        self.min_size = min_size
        self.topics_matching = topics_matching
        self.topics_labeller = topics_labeller
        self.embedding_model = embedding_model
        self.lexis_len = lexis_len

    def cluster(self, news_clusters: List[NewsCluster]) -> List[StoryCluster]:
        if len(news_clusters) == 0:
            return list()
        vectors_for_clustering = np.zeros((len(news_clusters), len(news_clusters[0].topics_vec())), dtype=np.float32)
        for j in range(len(news_clusters)):
            vectors_for_clustering[j] = news_clusters[j].topics_vec()
        result = []

        def stories_clustering_callback(indices):
            labelled_stories[indices] = 0
            story_id = str(uuid.uuid4())
            center = np.sum(vectors_for_clustering[indices], axis=0)
            topics = self.topics_matching.match(center)
            vec_docs = [doc for l in indices for doc in news_clusters[l].docs()]
            name = self.topics_labeller.label(vec_docs)

            all_words = np.concatenate([doc.words() for l in indices for doc in news_clusters[l].docs()])
            all_vecs = [news_clusters[l].vec_sum for l in indices]
            all_sentence_len = np.sum(
                [len(doc.embedding_sentences()) for l in indices for doc in news_clusters[l].docs()])
            vec_to_search = np.sum(all_vecs, axis=0) / all_sentence_len
            lexis = self.embedding_model.lexis_distribution(all_words, vec_to_search, self.lexis_len)

            dates = [news_clusters[l].date() for l in indices]
            story = StoryCluster(story_id, name, max(dates), topics, lexis, center, len(indices),
                                 [news_clusters[l] for l in indices])
            result.append(story)

        labelled_stories = np.ones(len(news_clusters), dtype=np.bool)
        FastQt(self.threshold, self.min_size).fit(vectors_for_clustering,
                                                  lambda indices, distances: stories_clustering_callback(indices))
        non_labelled_stories = np.nonzero(labelled_stories)[0]
        for i in non_labelled_stories:
            id_story = str(uuid.uuid4())
            doc_vecs = [doc for doc in news_clusters[i].docs()]
            label = self.topics_labeller.label(doc_vecs)
            tops = self.topics_matching.match(vectors_for_clustering[i])

            words_all = np.concatenate([doc.words() for doc in news_clusters[i].docs()])
            sentence_len = np.sum([len(doc.embedding_sentences()) for doc in news_clusters[i].docs()])
            vec_search = news_clusters[i].vec_sum / sentence_len
            lex = self.embedding_model.lexis_distribution(words_all, vec_search, self.lexis_len)

            s = StoryCluster(id_story, label, news_clusters[i].dt, tops, lex, vectors_for_clustering[i], 1,
                             [news_clusters[i]])
            result.append(s)

        return result
