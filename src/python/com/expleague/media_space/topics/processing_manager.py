from typing import List, Callable

from com.expleague.media_space.article import Article
from com.expleague.media_space.topics.clustering import NewsClustering, StoriesClustering
from com.expleague.media_space.topics.embedding2topics import Embedding2TopicsClustering
from com.expleague.media_space.topics.embedding_model import FastTextModel, TextNormalizer
from com.expleague.media_space.topics.items import NewsItem, StoryItem
from com.expleague.media_space.topics.params import ProcessingParams
from com.expleague.media_space.topics.state_handler import StateHandler
from com.expleague.media_space.topics.topics_labeller import EmbeddingTopicsLabeller
from com.expleague.media_space.topics.topics_matching import TopicsMatching
from com.expleague.media_space.topics.vec_doc import VecDoc


class ProcessingManager:
    def __init__(self, params: ProcessingParams, state_handler: StateHandler, text_normalizer: TextNormalizer):
        self.params = params
        self.state_handler = state_handler
        self.embedding_model = FastTextModel(params.embedding_file_path, params.idf_file_path, text_normalizer)

        topics_matching = TopicsMatching(params.topics_matching_file_path)
        topics_labeller = EmbeddingTopicsLabeller()
        embedding2topics = Embedding2TopicsClustering(params.cluster_centroids_file_path,
                                                      params.cluster_names_file_path,
                                                      params.topic_cos_threshold,
                                                      params.scale_dist)
        self.news_clustering = NewsClustering(params.news_clustering_threshold, params.news_clustering_min_cluster_size,
                                              embedding2topics, topics_labeller)
        self.story_clustering = StoriesClustering(params.stories_clustering_threshold,
                                                  params.stories_clustering_min_cluster_size, topics_matching,
                                                  topics_labeller,
                                                  self.embedding_model, params.lexic_result_word_num)

    def process(self, batch: List[Article], update: bool, callback: Callable[[Article, NewsItem], None]):
        vec_docs = []
        for article in batch:
            vec_docs.append(VecDoc(article, self.embedding_model, self.params.min_sentence_len))

        docs_for_clustering = []
        for doc in vec_docs:
            if not doc.valid():
                continue

            news_item = self.state_handler.find_news(doc)
            if news_item is not None:
                if update:
                    self.state_handler.insert_news(
                        NewsItem(news_item.id(), news_item.name(), news_item.date(), news_item.story(),
                                 news_item.vec_sum + doc.embedding(), news_item.vec_num + 1,
                                 news_item.topics))
                callback(doc.article(), news_item)
            else:
                docs_for_clustering.append(doc)

        news_clusters = self.news_clustering.cluster(docs_for_clustering)
        news_for_clustering = []
        for news_cluster in news_clusters:
            story = self.state_handler.find_story(news_cluster)
            if story is not None:
                if story.date() != news_cluster.date():
                    story = StoryItem(story.id(), story.name(), news_cluster.date(), story.topics(),
                                      story.lexis_distribution(),
                                      story.vec_sum + news_cluster.topics_vec(),
                                      story.vec_num + len(news_cluster.docs()))
                    if update:
                        self.state_handler.insert_story(story)

                news_cluster.set_story(story)
                if update:
                    self.state_handler.insert_news(news_cluster)

                for doc in news_cluster.docs():
                    callback(doc.article(), news_cluster)
            else:
                news_for_clustering.append(news_cluster)

        stories_clusters = self.story_clustering.cluster(news_for_clustering)
        for story_cluster in stories_clusters:
            if update:
                self.state_handler.insert_story(story_cluster)
            for news_cluster in story_cluster.news_clusters():
                news_cluster.set_story(story_cluster)
                if update:
                    self.state_handler.insert_news(news_cluster)
                for doc in news_cluster.docs():
                    callback(doc.article(), news_cluster)
            del story_cluster.clusters

        if update:
            self.state_handler.commit()
