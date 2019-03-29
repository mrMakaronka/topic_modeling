from dataclasses import dataclass
from typing import Any

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Article:
    id: Any
    publisher: str
    text: str
    pub_datetime: str
    title: str


@dataclass_json
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
