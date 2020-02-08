#!/usr/bin/env python

import scipy.optimize as optimize
import datetime
import os
import argparse
import pandas as pd
import numpy as np
import logging
import itertools
from sklearn.metrics.cluster import *
from com.expleague.media_space.topics_script import TopicsScript
from com.expleague.media_space.input import NewsGasparettiInput
from com.expleague.media_space.topics.params import ProcessingParams, StartupParams
from com.expleague.media_space.topics.embedding_model import GasparettiTextNormalizer
from com.expleague.media_space.topics.file_read_util import FileReadUtil

DATA_DIR = os.environ.get("DATA_DIR_TM", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODELS_DIR = os.path.join(DATA_DIR, "models", "gasparetti")


class ScoreComputer:
    def __init__(self, story_ids):
        self.story_ids = story_ids

    def compute_score(self, predicted_stories_list, only_in_cluster_values=True):
        predict_list = list()
        real_story_ids = list()
        if only_in_cluster_values:
            for i in range(len(self.story_ids)):
                if predicted_stories_list[i] != "0":
                    predict_list.append(predicted_stories_list[i])
                    real_story_ids.append(self.story_ids[i])
        else:
            predict_list = predicted_stories_list
            real_story_ids = self.story_ids
        return adjusted_rand_score(real_story_ids, predict_list)


def get_df_clusters_predicted(theta, url_list):
    df = pd.DataFrame(columns=['url', 'story_id_predicted'])
    for i in range(len(url_list)):
        df.loc[i] = [url_list[i], np.argmax(np.array(theta[i]))]
    return df


def compute_number_of_topics(file_path, limit):
    """
    Number of topics
    :return:
    """
    texts = pd.read_csv(file_path, chunksize=10000)
    clusters = list()
    url_list = list()
    i = 0
    for chunk in texts:
        for index, row in chunk.iterrows():
            clusters.append(row["story"])
            url_list.append(row["url"])
            i += 1
            if limit and i >= limit:
                break
        else:
            # Continue if the inner loop wasn't broken.
            continue
            # Inner loop was broken, break the outer.
        break
    return clusters, url_list


def compute_score_topic_modeling(score_cmp=None,
                                 min_sentence_len=6,
                                 topic_cos_threshold=0.7,
                                 news_clustering_threshold=0.025,
                                 news_clustering_min_cluster_size=4,
                                 stories_clustering_threshold=0.25,
                                 stories_clustering_min_cluster_size=2,
                                 ngrams_for_topics_labelling=3,
                                 stories_connecting_cos_threshold=0.6,
                                 story_window=4,
                                 lexic_result_word_num=10,
                                 sclale_dist=100,
                                 verbose=False,
                                 input_file_path="gasparetti_small.csv",
                                 start='10.03.2014',
                                 end='26.03.2014'):
    articles_input = NewsGasparettiInput(input_file_path)
    text_normalizer = GasparettiTextNormalizer()

    start = datetime.datetime.strptime(start, '%d.%m.%Y').replace(tzinfo=datetime.timezone.utc)
    end = datetime.datetime.strptime(end, '%d.%m.%Y').replace(tzinfo=datetime.timezone.utc)

    embedding_file_path = os.path.join(MODELS_DIR, "news_dragnet.vec")
    idf_file_path = os.path.join(MODELS_DIR, 'idf_dragnet.txt')
    cluster_centroids_file_path = os.path.join(MODELS_DIR, 'cluster_centroids_filtered.txt')
    cluster_names_file_path = os.path.join(MODELS_DIR, 'cluster_names_filtered.txt')
    topics_matching_file_path = os.path.join(MODELS_DIR, 'topic_matching.txt')

    params_logging_str = f"FROM_DATE: {start}\n" \
                         f"TO_DATE: {end}\n\n" \
                         f"EMBEDDING_FILE_PATH: {embedding_file_path}\n" \
                         f"IDF_FILE_PATH: {idf_file_path}\n" \
                         f"CLUSTER_CENTROIDS_FILE_PATH: {cluster_centroids_file_path}\n\n" \
                         f"MIN_SENTENCE_LEN: {min_sentence_len}\n" \
                         f"TOPIC_COS_THRESHOLD: {topic_cos_threshold}\n" \
                         f"NEWS_CLUSTERING_THRESHOLD: {news_clustering_threshold}\n" \
                         f"NEWS_CLUSTERING_MIN_CLUSTER_SIZE: {news_clustering_min_cluster_size}\n" \
                         f"STORIES_CLUSTERING_THRESHOLD: {stories_clustering_threshold}\n" \
                         f"STORIES_CLUSTERING_MIN_CLUSTER_SIZE: {stories_clustering_min_cluster_size}\n" \
                         f"NGRAMS_FOR_TOPICS_LABELLING: {ngrams_for_topics_labelling}\n" \
                         f"STORIES_CONNECTING_COS_THRESHOLD: {stories_connecting_cos_threshold}\n" \
                         f"STORY_WINDOW: {story_window}\n" \
                         f"LEXIC_RESULT_WORD_NUM: {lexic_result_word_num}\n" \
                         f"SCALE_DIST: {sclale_dist}\n"
    logging.info('Parameters used:\n' + params_logging_str)
    processor = TopicsScript(
        StartupParams(start, end),
        ProcessingParams(embedding_file_path, idf_file_path, cluster_centroids_file_path,
                         cluster_names_file_path, topics_matching_file_path, min_sentence_len,
                         topic_cos_threshold,
                         news_clustering_threshold,
                         news_clustering_min_cluster_size, stories_clustering_threshold,
                         stories_clustering_min_cluster_size, ngrams_for_topics_labelling,
                         stories_connecting_cos_threshold, story_window, lexic_result_word_num, sclale_dist))
    topic_news = processor.run(articles_input, text_normalizer, verbose=verbose)
    names = FileReadUtil.load_clusters_names(cluster_names_file_path)
    dict_clusters = dict()
    for cluster_id in topic_news:
        docs = topic_news[cluster_id]
        for doc in docs:
            article = doc.article()
            dict_clusters[article.publisher] = cluster_id

    output_clusters = pd.DataFrame(columns=["url", "timestamp", "story_id_predicted", "story_id"])
    for index, row in articles_input.df.iterrows():
        cluster_id = dict_clusters.get(row["url"], "0")

        def convert_epoch(ts):
            return datetime.datetime.utcfromtimestamp(int(ts) / 1000).strftime('%Y/%m/%d')

        logging.info(f'\n################################################')
        logging.info('%s %s %s \n%s', convert_epoch(row["timestamp"]), row["story"], "news not in any cluster:",
                     row["text"])
        output_clusters.loc[index] = [row["url"], row["timestamp"], cluster_id, row["story"]]
    score = None
    if score_cmp:
        score = score_cmp.compute_score(output_clusters["story_id_predicted"].to_list())
        logging.info('Score : ' + str(score) + "\n")
    return score


if __name__ == "__main__":
    time_now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    parser = argparse.ArgumentParser(description='Run topics matching')
    parser.add_argument('-i', '--input', type=str, default=os.path.join(DATA_DIR, "src", "resources",
                                                                        "gasparetti_small.csv"),
                        help='Input news source')
    parser.add_argument('-l', '--log-file', type=str,
                        default=f"topics-script-log-{time_now}.txt",
                        help='Path to log file')
    args = parser.parse_args()

    input_file_path = os.path.join(DATA_DIR, args.input)
    logging.getLogger()
    logging.basicConfig(filename=args.log_file, filemode='w', level=logging.INFO)

    clusters, url_list = compute_number_of_topics(input_file_path, None)
    score_computer = ScoreComputer(clusters)
    num_topics = len(set(clusters))
    logging.info("Number of topics initially: " + str(num_topics))


    def f(params):
        topic_cos_threshold, news_clustering_threshold, \
        stories_clustering_threshold, stories_connecting_cos_threshold, scale_dist = params
        scr = compute_score_topic_modeling(
            score_cmp=score_computer,
            min_sentence_len=7,
            topic_cos_threshold=topic_cos_threshold,
            news_clustering_threshold=news_clustering_threshold,
            news_clustering_min_cluster_size=3,
            stories_clustering_threshold=stories_clustering_threshold,
            stories_clustering_min_cluster_size=4,
            stories_connecting_cos_threshold=stories_connecting_cos_threshold,
            story_window=5,
            lexic_result_word_num=5,
            sclale_dist=scale_dist,
            input_file_path=input_file_path,
            verbose=False,
            start='10.03.2014',
            end='25.03.2014')
        print(scr, params)
        return scr


    initial_guess = np.array([0.5, 0.5, 0.5, 0.5, 200])
    compute_score_topic_modeling(
        score_cmp=score_computer,
        min_sentence_len=5,
        topic_cos_threshold=0.5,
        news_clustering_threshold=0.6,
        news_clustering_min_cluster_size=2,
        stories_clustering_threshold=0.7,
        stories_clustering_min_cluster_size=2,
        stories_connecting_cos_threshold=0.7,
        story_window=3,
        lexic_result_word_num=5,
        sclale_dist=200,
        input_file_path=input_file_path,
        verbose=True,
        start='10.03.2014',
        end='25.03.2014')
    # bounds = ((0.2, 0.8), (0.2, 0.8), (0.2, 0.8), (100, 300))
    # result = optimize.minimize(f,
    #                            initial_guess,
    #                            tol=0.1,
    #                            method='nelder-mead',
    #                            options={'xtol': 0.1, 'maxiter': 20},
    #                            )
    # if result.success:
    #     fitted_params = result.x
    #     print(fitted_params)
    # else:
    #     raise ValueError(result.message)
