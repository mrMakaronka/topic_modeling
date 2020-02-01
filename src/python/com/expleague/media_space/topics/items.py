from datetime import datetime
from typing import List, Tuple

import numpy as np
from dataclasses import dataclass

from com.expleague.media_space.topics.date_utils import DateUtils


@dataclass
class StoryState:
    id: str
    name: str
    date: datetime
    topics: List[Tuple[str, float]]
    lexis: List[Tuple[str, float]]
    vec_sum: List[float]
    vec_num: int


class StoryItem:
    def __init__(self, story_id: str, name: str, dt: datetime, topics: List[Tuple[str, float]],
                 lexis: List[Tuple[str, float]], vec_sum: np.ndarray, vec_num: int):
        self.story_id = story_id
        self.story_name = name
        self.dt = DateUtils.normalize(dt)
        self.story_topics = topics
        self.lexis = lexis
        self.vec_sum = vec_sum
        self.vec_num = vec_num

    def id(self) -> str:
        return self.story_id

    def name(self) -> str:
        return self.story_name

    def date(self) -> datetime:
        return self.dt

    def topics(self) -> List[Tuple[str, float]]:
        return self.story_topics

    def lexis_distribution(self) -> List[Tuple[str, float]]:
        return self.lexis

    def topics_vec(self) -> np.ndarray:
        vec = self.vec_sum / self.vec_num
        return vec

    def state(self) -> StoryState:
        # noinspection PyTypeChecker
        return StoryState(self.story_id, self.story_name, self.dt, self.story_topics, self.lexis, self.vec_sum.tolist(),
                          self.vec_num)

    @staticmethod
    def build(state: StoryState):
        return StoryItem(state.id, state.name, state.date, state.topics, state.lexis,
                         np.fromiter(state.vec_sum, dtype=np.float32), state.vec_num)

    def __eq__(self, other):
        return isinstance(other, StoryItem) and self.story_id == other.id()

    def __hash__(self):
        return hash(self.story_id)


@dataclass
class NewsState:
    id: str
    name: str
    date: datetime
    story_id: str
    vec_sum: List[float]
    vec_num: int
    topics: List[float]


class NewsItem:
    def __init__(self, item_id: str, name: str, dt: datetime, story: StoryItem, vec_sum: np.ndarray, vec_num: int,
                 topics: np.ndarray):
        self.topics = topics
        self.item_id = item_id
        self.item_name = name
        self.dt = DateUtils.normalize(dt)
        self.item_story = story
        self.vec_num = vec_num
        self.vec_sum = vec_sum

    def id(self) -> str:
        return self.item_id

    def name(self) -> str:
        return self.item_name

    def date(self) -> datetime:
        return self.dt

    def story(self) -> StoryItem:
        return self.item_story

    def embedding_vec(self) -> np.ndarray:
        return self.vec_sum / self.vec_num

    def topics_vec(self) -> np.ndarray:
        return self.topics

    def state(self) -> NewsState:
        # noinspection PyTypeChecker
        return NewsState(self.item_id, self.item_name, self.dt, self.story().id(), self.vec_sum.tolist(), self.vec_num,
                         self.topics.tolist())

    @staticmethod
    def build(story: StoryItem, state: NewsState):
        return NewsItem(state.id, state.name, state.date, story, np.fromiter(state.vec_sum, dtype=np.float32),
                        state.vec_num,
                        np.fromiter(state.topics, dtype=np.float32))

    def __eq__(self, other):
        return isinstance(other, NewsItem) and self.item_id == other.id()

    def __hash__(self):
        return hash(self.item_id)
