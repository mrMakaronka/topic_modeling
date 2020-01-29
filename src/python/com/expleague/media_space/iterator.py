import random
from abc import abstractmethod
from datetime import datetime
from uuid import UUID

import diskcache
from pandas import DataFrame
from redis import StrictRedis

from com.expleague.media_space.article import Article


class ArticlesIterator:
    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self) -> Article:
        pass


class GasparettiCsvIterator(ArticlesIterator):
    def __init__(self, df: DataFrame, start: datetime, end: datetime, limit: int):
        self.limit = limit
        self.end = end
        self.start = start
        self.df = df

    def __iter__(self):
        self.iter = self.df[
            (self.df.date >= self.start.strftime("%Y/%m/%d")) & (
                    self.df.date < self.end.strftime("%Y/%m/%d"))].iterrows()
        return self

    def __next__(self) -> Article:
        for row in self.iter:
            return Article(row[1]['url'], row[1]['url'], row[1]['text'],
                           row[1]['date'], row[1]['title'], row[1]['story'])
        raise StopIteration


class LentaCsvIterator(ArticlesIterator):
    def __init__(self, df: DataFrame, start: datetime, end: datetime, limit: int):
        self.limit = limit
        self.end = end
        self.start = start
        self.df = df

    def __iter__(self):
        self.iter = self.df[
            (self.df.date >= self.start.strftime("%Y/%m/%d")) & (
                    self.df.date < self.end.strftime("%Y/%m/%d"))].iterrows()
        return self

    def __next__(self) -> Article:
        for row in self.iter:
            return Article(row[1]['url'], row[1]['url'], row[1]['text'], row[1]['date'], row[1]['title'], "")
        raise StopIteration


class RedisLoadIterator(ArticlesIterator):
    def __init__(self, redis: StrictRedis, prefix: str):
        self.prefix = prefix
        self.redis = redis

    def __iter__(self):
        self.keys = self.redis.keys(self.prefix + "*")
        self.index = 0
        return self

    def __next__(self) -> Article:
        if self.index < len(self.keys):
            article = Article.from_json(self.redis.get(self.keys[self.index]))
            self.index += 1
            return article
        raise StopIteration


class RedisSaveIterator(ArticlesIterator):
    def __init__(self, redis: StrictRedis, prefix: str, articles_iterator: ArticlesIterator):
        self.redis = redis
        self.prefix = prefix
        self.articles_iterator = articles_iterator

    def __iter__(self):
        return self

    def __next__(self) -> Article:
        for article in self.articles_iterator:
            key = self.prefix + str(UUID(int=random.getrandbits(128), version=1))
            value = article.to_json()
            self.redis.set(key, value)
            return article
        raise StopIteration


class DiskCacheLoadIterator(ArticlesIterator):
    def __init__(self, disk_cache: diskcache.Cache):
        self.disk_cache = disk_cache

    def __iter__(self):
        self.iter = self.disk_cache.__iter__()
        return self

    def __next__(self) -> Article:
        for key in self.iter:
            article = Article.from_json(self.disk_cache.get(key))
            return article
        self.disk_cache.close()
        raise StopIteration


class DiskCacheSaveIterator(ArticlesIterator):
    def __init__(self, disk_cache: diskcache.Cache, articles_iterator: ArticlesIterator):
        self.articles_iterator = articles_iterator
        self.disk_cache = disk_cache

    def __iter__(self):
        return self

    def __next__(self) -> Article:
        for article in self.articles_iterator:
            key = str(UUID(int=random.getrandbits(128), version=1))
            value = article.to_json()
            self.disk_cache[key] = value
            return article
        self.disk_cache.close()
        raise StopIteration
