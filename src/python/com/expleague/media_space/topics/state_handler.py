import logging
from typing import Optional

import faiss
import numpy as np
from sklearn.preprocessing import normalize

from com.expleague.media_space.topics.items import NewsItem, StoryItem
from com.expleague.media_space.topics.vec_doc import VecDoc


class StateHandler:
    def find_news(self, doc: VecDoc) -> Optional[NewsItem]:
        pass

    def find_story(self, news: NewsItem) -> Optional[StoryItem]:
        pass

    def insert_news(self, news: NewsItem) -> None:
        pass

    def insert_story(self, story: StoryItem) -> None:
        pass

    def commit(self):
        pass


class InMemStateHandler(StateHandler):
    def __init__(self, story_window: int, stories_connecting_cos_threshold: float, number_of_clusters: int):
        self.story_window = story_window
        self.stories_connecting_cos_threshold = stories_connecting_cos_threshold
        self.number_of_clusters = number_of_clusters
        self.stories_cache = {}
        self.stories_indices = {}

        self.inserted_news = set()
        self.inserted_stories = set()

    def find_news(self, doc: VecDoc) -> Optional[NewsItem]:
        return None

    def find_story(self, news: NewsItem) -> Optional[StoryItem]:
        for k in sorted(self.stories_indices.keys(), reverse=True):
            vec_to_search = news.topics_vec().reshape((1, news.topics_vec().shape[0]))
            vec_to_search = vec_to_search.astype(np.float32)
            lim, dist, ind = self.stories_indices[k].range_search(vec_to_search, self.stories_connecting_cos_threshold)
            if len(ind) > 0:
                return self.stories_cache[k][ind[0]]
        return None

    def insert_news(self, news: NewsItem) -> None:
        self.inserted_news.add(news)

    def insert_story(self, story: StoryItem) -> None:
        self.inserted_stories.add(story)

    def commit(self):
        logging.info('Start committing')

        # TODO: investigate how this change affects result
        # remove single news-story clusters
        # for news_item in list(self.inserted_news):
        #     if news_item.vec_num == 1 and news_item.story().vec_num == 1:
        #         self.inserted_stories.discard(news_item.story())
        #         self.inserted_news.discard(news_item)

        if len(self.inserted_stories) > 0:
            # update cache (for optimization)
            date = next(iter(self.inserted_stories)).date()
            if date in self.stories_cache:
                self.stories_cache[date].extend(self.inserted_stories)
                self.stories_cache[date] = list(set(self.stories_cache[date]))
            else:
                self.stories_cache[date] = list(self.inserted_stories)
            self.inserted_stories = set()

        if len(self.stories_cache) > self.story_window:
            to_remove = min(self.stories_cache.keys())
            del self.stories_cache[to_remove]
            self.stories_indices.pop(to_remove, None)

        for k, v in self.stories_cache.items():
            if len(v) > 0:
                index = faiss.IndexFlatIP(self.number_of_clusters)
                vecs = np.zeros((len(v), self.number_of_clusters), dtype=np.float32)
                i = 0
                for story in v:
                    if not np.isnan(np.min(story.vec_sum)) and story.vec_num != 0:
                        vecs[i] = story.vec_sum / story.vec_num
                    i += 1
                # noinspection PyArgumentList
                index.add(normalize(vecs, axis=1, norm='l2'))
                self.stories_indices[k] = index
