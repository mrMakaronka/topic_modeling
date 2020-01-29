from dataclasses import dataclass
from typing import Any


@dataclass
class Article:
    id: Any
    publisher: str
    text: str
    pub_datetime: str
    title: str
    story_id: str


@dataclass
class ArticleBatched(Article):
    __slots__ = [
        'id',
        'publisher',
        'text',
        'pub_datetime',
        'title',
        'mongo_id',
    ]

    mongo_id: str
