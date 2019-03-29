from abc import abstractmethod
from datetime import datetime

import diskcache
import pandas as pd
import redis

from com.expleague.media_space.iterator import (
    ArticlesIterator,
    RedisSaveIterator,
    RedisLoadIterator,
    DiskCacheLoadIterator,
    DiskCacheSaveIterator,
    LentaCsvIterator)


class ArticlesInput:
    @abstractmethod
    def iterator(self, start: datetime, end: datetime, limit: int = -1) -> ArticlesIterator:
        pass


class LentaCsvInput(ArticlesInput):
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df['date'] = self.df['url'].str[22:32]

    def iterator(self, start: datetime, end: datetime, limit: int = -1) -> ArticlesIterator:
        return LentaCsvIterator(self.df, start, end, limit)


class CachedInput(ArticlesInput):
    def __init__(self, articles_input: ArticlesInput):
        self.articles_input = articles_input

    def iterator(self, start: datetime, end: datetime, limit: int = -1) -> ArticlesIterator:
        prefix = str(type(self.articles_input)) + '|' + str(start.date()) + '|' + str(end.date()) + '|' + str(limit)
        if self.is_empty(prefix):
            return self.save_iterator(prefix, self.articles_input.iterator(start, end, limit))
        else:
            return self.load_iterator(prefix)

    @abstractmethod
    def is_empty(self, prefix: str) -> bool:
        pass

    @abstractmethod
    def save_iterator(self, prefix: str, input_iterator: ArticlesIterator) -> ArticlesIterator:
        pass

    @abstractmethod
    def load_iterator(self, prefix: str) -> ArticlesIterator:
        pass


class RedisCachedArticlesInput(CachedInput):
    def __init__(self, host: str, port: int, articles_input: ArticlesInput):
        super().__init__(articles_input)
        self.redis = redis.StrictRedis(host, port, db=0)

    def is_empty(self, prefix: str) -> bool:
        keys = self.redis.keys(prefix + "*")
        return len(keys) == 0

    def save_iterator(self, prefix: str, input_iterator: ArticlesIterator) -> ArticlesIterator:
        return RedisSaveIterator(self.redis, prefix, input_iterator)

    def load_iterator(self, prefix: str) -> ArticlesIterator:
        return RedisLoadIterator(self.redis, prefix)


class DiskCacheArticlesInput(CachedInput):
    def __init__(self, articles_input: ArticlesInput):
        super().__init__(articles_input)

    def is_empty(self, prefix: str) -> bool:
        cache = diskcache.Cache(prefix)
        result = cache.__len__() == 0
        cache.close()
        return result

    def save_iterator(self, prefix: str, input_iterator: ArticlesIterator) -> ArticlesIterator:
        return DiskCacheSaveIterator(diskcache.Cache(prefix, size_limit=int(4e9)), input_iterator)

    def load_iterator(self, prefix: str) -> ArticlesIterator:
        return DiskCacheLoadIterator(diskcache.Cache(prefix))
