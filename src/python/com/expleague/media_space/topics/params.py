from datetime import datetime

from dataclasses import dataclass


@dataclass
class StartupParams:
    start: datetime
    end: datetime


@dataclass
class ProcessingParams:
    embedding_file_path: str
    idf_file_path: str
    cluster_centroids_file_path: str
    cluster_names_file_path: str
    topics_matching_file_path: str
    min_sentence_len: int
    topic_cos_threshold: float
    news_clustering_threshold: float
    news_clustering_min_cluster_size: int
    stories_clustering_threshold: float
    stories_clustering_min_cluster_size: int
    ngrams_for_topics_labelling: int
    stories_connecting_cos_threshold: float
    story_window: int
    lexic_result_word_num: int
    scale_dist: int
