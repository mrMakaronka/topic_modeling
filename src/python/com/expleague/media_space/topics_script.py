import logging
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

from com.expleague.media_space.input import LentaCsvInput
from com.expleague.media_space.article import Article
from com.expleague.media_space.topics.params import ProcessingParams, StartupParams
from com.expleague.media_space.topics.processing_manager import NewsItem, ProcessingManager
from com.expleague.media_space.topics.state_handler import InMemStateHandler


class TopicsScript:
    def __init__(self, startup_params: StartupParams, processing_params: ProcessingParams):
        self.processing_params = processing_params
        self.startup_params = startup_params

    def run(self):
        state_handler = InMemStateHandler(self.processing_params.story_window,
                                          self.processing_params.stories_connecting_cos_threshold)
        processing_manager = ProcessingManager(self.processing_params, state_handler)
        articles_input = LentaCsvInput("../src/resources/lenta_small.csv")

        ranges = pd.date_range(self.startup_params.start, self.startup_params.end, freq='1D')
        topic_news = defaultdict(list)
        topics = {}
        for i in range(len(ranges) - 1):
            logging.info('Day %d/%d: %s-%s', i + 1, len(ranges) - 1, str(ranges[i]), str(ranges[i + 1]))
            logging.info('Start loading news...')
            start_time = time.time()
            articles_iterator = articles_input.iterator(ranges[i], ranges[i + 1])
            all_articles = []
            for article in articles_iterator:
                all_articles.append(article)
            logging.info('End loading news. Total: %s sec', time.time() - start_time)

            def processed_callback(processed: Article, item: NewsItem):
                topic_news[item.story().id()].append(processed)
                topics[item.story().id()] = item.story()

            logging.info('Start processing...')
            start_time = time.time()
            processing_manager.process(all_articles, True, processed_callback)
            logging.info('End processing. Total: %s sec', time.time() - start_time)

        limit = 1000
        it = 0
        for cluster_id in sorted(topic_news, key=lambda k: len(topic_news[k]), reverse=True):
            logging.info('STORY %s %s', cluster_id, topics[cluster_id].name().upper())
            articles = topic_news[cluster_id]
            for article in articles:
                logging.info('%s %s', article.pub_datetime, article.text.replace('\n', ' ')[:300])
            # print(topics[cluster_id].topics())
            # print(topics[cluster_id].lexis_distribution())
            logging.info('###################')

            it += 1
            if it == limit:
                break


# noinspection PyArgumentList
def main():
    logging.getLogger().setLevel(logging.INFO)

    start = os.getenv('FROM_DATE', '07.12.2018')
    start = datetime.strptime(start, '%d.%m.%Y').replace(tzinfo=timezone.utc)
    end = os.getenv('TO_DATE', '15.12.2018')
    end = datetime.strptime(end, '%d.%m.%Y').replace(tzinfo=timezone.utc)

    embedding_file_path = os.getenv('EMBEDDING_FILE_PATH', 'lenta.vec')
    idf_file_path = os.getenv('IDF_FILE_PATH', 'idf.txt')
    cluster_centroids_file_path = os.getenv('CLUSTER_CENTROIDS_FILE_PATH', 'cluster_centroids.txt')
    cluster_names_file_path = os.getenv('CLUSTER_NAMES_FILE_PATH', 'cluster_names.txt')
    topics_matching_file_path = os.getenv('TOPICS_MATCHING_FILE_PATH', 'topics_matching.txt')
    min_sentence_len = int(os.getenv('MIN_SENTENCE_LEN', 3))
    topic_cos_threshold = float(os.getenv('TOPIC_COS_THRESHOLD', 0.5))
    news_clustering_threshold = float(os.getenv('NEWS_CLUSTERING_THRESHOLD', 0.025))
    news_clustering_min_cluster_size = int(os.getenv('NEWS_CLUSTERING_MIN_CLUSTER_SIZE', 2))
    stories_clustering_threshold = float(os.getenv('STORIES_CLUSTERING_THRESHOLD', 0.25))
    stories_clustering_min_cluster_size = int(os.getenv('STORIES_CLUSTERING_MIN_CLUSTER_SIZE', 2))
    ngrams_for_topics_labelling = int(os.getenv('NGRAMS_FOR_TOPICS_LABELLING', 3))
    stories_connecting_cos_threshold = float(os.getenv('STORIES_CONNECTING_COS_THRESHOLD', 0.9))
    story_window = int(os.getenv('STORY_WINDOW', 3))
    lexic_result_word_num = int(os.getenv('LEXIC_RESULT_WORD_NUM', 10))

    processor = TopicsScript(
        StartupParams(start, end),
        ProcessingParams(embedding_file_path, idf_file_path, cluster_centroids_file_path,
                         cluster_names_file_path, topics_matching_file_path, min_sentence_len,
                         topic_cos_threshold,
                         news_clustering_threshold,
                         news_clustering_min_cluster_size, stories_clustering_threshold,
                         stories_clustering_min_cluster_size, ngrams_for_topics_labelling,
                         stories_connecting_cos_threshold, story_window, lexic_result_word_num))
    processor.run()
    # cProfile.runctx('processor.run()', globals(), locals())


if __name__ == "__main__":
    main()
