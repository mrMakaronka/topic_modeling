from typing import Dict, List

from com.expleague.media_space.topics.items import NewsItem
from com.expleague.media_space.topics.levels_parser import LevelsParser


class ResultDict:
    def __init__(self, max_array_size: int, include_lexis=True):
        self.include_lexis = include_lexis
        self.max_array_size = max_array_size
        self.level_parser = LevelsParser()

    def generate(self, news_item: NewsItem) -> List[Dict]:
        result = []
        for lvl1_lvl2, proba in news_item.story().topics()[:self.max_array_size]:
            levels = self.level_parser.parse(lvl1_lvl2)
            if self.include_lexis:
                lexis = []
                for t in news_item.story().lexis_distribution():
                    lexis.append({
                        'word': t[0],
                        'prob': t[1]
                    })
                result.append({
                    'lvl1': {
                        'topic_label': levels.level1_name,
                        'id': levels.level1_id,
                        'distribution': lexis
                    },
                    'lvl2': {
                        'topic_label': levels.level2_name,
                        'id': levels.level2_id,
                        'distribution': lexis
                    },
                    'lvl3': {
                        'topic_label': news_item.story().name(),
                        'id': news_item.story().id(),
                        'distribution': []
                    },
                    'lvl4': {
                        'topic_label': news_item.name(),
                        'id': news_item.id(),
                        'distribution': []
                    },
                    'probability': proba
                })
            else:
                result.append({
                    'lvl1': {
                        'topic_label': levels.level1_name,
                        'id': levels.level1_id
                    },
                    'lvl2': {
                        'topic_label': levels.level2_name,
                        'id': levels.level2_id
                    },
                    'lvl3': {
                        'topic_label': news_item.story().name(),
                        'id': news_item.story().id()
                    },
                    'lvl4': {
                        'topic_label': news_item.name(),
                        'id': news_item.id()
                    },
                    'probability': proba
                })

        return result
