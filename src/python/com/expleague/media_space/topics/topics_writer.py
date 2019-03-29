from abc import abstractmethod

from pymongo import MongoClient

from com.expleague.media_space.article import Article
from com.expleague.media_space.topics.params import DBParams
from com.expleague.media_space.topics.processing_manager import NewsItem
from com.expleague.media_space.topics.result_dict import ResultDict


class TopicsWriter:
    @abstractmethod
    def write(self, article: Article, news_item: NewsItem):
        pass


class MongoDBTopicsWriter(TopicsWriter):
    def __init__(self, db_params: DBParams):
        self.db_params = db_params
        self.result_dict = ResultDict(db_params.topics_max_array_size)

    def __enter__(self):
        uri = 'mongodb://{creds}{host}{port}/{database}'.format(
            creds=(
                f'{self.db_params.out_username}:{self.db_params.out_password}@'
                if self.db_params.out_username
                else ''
            ),
            host=self.db_params.out_host,
            port='' if self.db_params.out_port is None else ':{}'.format(self.db_params.out_port),
            database=self.db_params.out_db_name
        )
        self.client = MongoClient(uri)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.client.__exit__(exc_type, exc_val, exc_tb)

    def write(self, article: Article, news_item: NewsItem):
        result = self.result_dict.generate(news_item)

        self.client[self.db_params.out_db_name][self.db_params.out_collection].update_one(
            {self.db_params.out_id_field: article.id},
            {
                '$set': {
                    'topics': result,
                    'etl_flags.' + self.db_params.set_flag: True,
                }
            },
            upsert=True
        )
