import logging
import argparse
import os
import time
import yaml

import pandas as pd
import json

from collections import defaultdict, namedtuple
from datetime import datetime, timezone
from enum import Enum

from com.expleague.media_space.input import LentaCsvInput, NewsGasparettiInput
from com.expleague.media_space.article import Article
from com.expleague.media_space.topics.params import ProcessingParams, StartupParams
from com.expleague.media_space.topics.processing_manager import NewsItem, ProcessingManager
from com.expleague.media_space.topics.state_handler import InMemStateHandler
from com.expleague.media_space.topics.embedding_model import SimpleTextNormalizer, GasparettiTextNormalizer

InputInfo = namedtuple('InputInfo', ['input', 'text_normalizer'])


class InputType(Enum):
    lenta = InputInfo(LentaCsvInput, SimpleTextNormalizer)
    gasparetti = InputInfo(NewsGasparettiInput, GasparettiTextNormalizer)


class TopicsScript:
    def __init__(self, startup_params: StartupParams, processing_params: ProcessingParams):
        self.processing_params = processing_params
        self.startup_params = startup_params

    def run(self, articles_input, text_normalizer, verbose=True):
        with open(self.processing_params.cluster_names_file_path, "r") as f:
            number_of_clusters = int(f.readline())

        state_handler = InMemStateHandler(self.processing_params.story_window,
                                          self.processing_params.stories_connecting_cos_threshold,
                                          number_of_clusters)
        processing_manager = ProcessingManager(self.processing_params, state_handler, text_normalizer)
        ranges = pd.date_range(self.startup_params.start, self.startup_params.end, freq='1D')
        topic_news = defaultdict(list)
        topics = {}
        news_items = {}
        articles_processed = []
        for i in range(len(ranges) - 1):
            if verbose:
                logging.info('Day %d/%d: %s-%s', i + 1, len(ranges) - 1, str(ranges[i]), str(ranges[i + 1]))
                logging.info('Start loading news...')
            start_time = time.time()
            articles_iterator = articles_input.iterator(ranges[i], ranges[i + 1])
            all_articles = []
            for article in articles_iterator:
                all_articles.append(article)
            if verbose:
                logging.info('End loading news. Total: %s sec', time.time() - start_time)

            def processed_callback(processed: Article, item: NewsItem):
                topic_news[item.story().id()].append(processed)
                topics[item.story().id()] = item.story()
                news_items[processed.id] = item

            if verbose:
                logging.info('Start processing...')
            start_time = time.time()
            processing_manager.process(all_articles, True, processed_callback)
            if verbose:
                logging.info('End processing. Total: %s sec', time.time() - start_time)

        if verbose:
            for i, cluster_id in enumerate(sorted(topic_news, key=lambda k: len(topic_news[k]), reverse=True)):
                names = processing_manager.news_clustering.embedding2topics.names
                vec_base_names = [(n, abs(val))
                                  for n, val in zip(names, topics[cluster_id].topics_vec())]
                vec_base_names.sort(key=lambda x: x[1], reverse=True)
                logging.info('STORY %s %s %s',
                             cluster_id,
                             topics[cluster_id].name().upper(),
                             vec_base_names)

                articles = topic_news[cluster_id]
                for article in articles:
                    # vector_in_base_space = [(n, val) for n, val in zip(names, article.vec_sum)]
                    vec_base_names_n = [(n, abs(val))
                                        for n, val in zip(names, news_items[article.id].topics_vec())]
                    vec_base_names_n.sort(key=lambda x: x[1], reverse=True)
                    logging.info('%s %s %s %s \n%s',
                                 article.pub_datetime,
                                 article.story_id,
                                 article.publisher,
                                 vec_base_names_n,
                                 article.text.replace('\n', ' ')[:2000])
                    # logging.info('%s %s %s', article.id, article.pub_datetime, vector_in_base_space)
                logging.info(f'\n###################-----STORY-№{i}--#############################')

        return topic_news, news_items


def read_config(path_to_config):
    with open(path_to_config) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


# noinspection PyArgumentList
def main():
    time_now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    parser = argparse.ArgumentParser(description='Run topics matching')
    parser.add_argument('-c', '--config-path', type=str, default="../config_lenta.yml",
                        help='Path to config file')
    parser.add_argument('-l', '--log-file-path', type=str,
                        default=f"topics-script-log-{time_now}.txt",
                        help='Path to log file')
    parser.add_argument('-o', '--out-file-path', type=str,
                        default=f"article_cluster_id.csv",
                        help='Path to output file with cluster id added to each news for gasparetti')
    args = parser.parse_args()
    logging.getLogger()
    logging.basicConfig(filename=args.log_file_path, filemode='w', level=logging.INFO)
    print("Log file is created: " + args.log_file_path)
    logging.info("Output file with clusters: " + args.out_file_path)
    config = read_config(args.config_path)
    input_type = InputType[config["input_type"]].value
    start = datetime.strptime(config["start"], '%d.%m.%Y').replace(tzinfo=timezone.utc)
    end = datetime.strptime(config["end"], '%d.%m.%Y').replace(tzinfo=timezone.utc)

    logging.info('Parameters used:\n' + json.dumps(config, indent=4, sort_keys=True))
    processor = TopicsScript(
        StartupParams(start, end),
        ProcessingParams(config["embedding_file_path"],
                         config["idf_file_path"],
                         config["cluster_centroids_file_path"],
                         config["cluster_names_file_path"],
                         config["topics_matching_file_path"],
                         config["min_sentence_len"],
                         config["topic_cos_threshold"],
                         config["news_clustering_threshold"],
                         config["news_clustering_min_cluster_size"],
                         config["stories_clustering_threshold"],
                         config["stories_clustering_min_cluster_size"],
                         config["ngrams_for_topics_labelling"],
                         config["stories_connecting_cos_threshold"],
                         config["story_window"],
                         config["lexic_result_word_num"],
                         config["sclale_dist"]))
    topic_news = processor.run(input_type.input(config["news_file_path"]), input_type.text_normalizer())
    dict_clusters = dict()
    for cluster_id in topic_news:
        articles = topic_news[cluster_id]
        for article in articles:
            dict_clusters[article.id] = cluster_id

    output_clusters = pd.DataFrame(columns=["url", "story_id_predicted"])
    for index, row in input_type.input(config["news_file_path"]).df.iterrows():
        cluster_id = dict_clusters.get(row["url"], "0")
        output_clusters.loc[index] = [row["url"], cluster_id]
    output_clusters.to_csv(args.out_file_path, index=False)
    # cProfile.runctx('processor.run()', globals(), locals())


if __name__ == "__main__":
    main()
