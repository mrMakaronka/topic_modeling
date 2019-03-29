from datetime import datetime
from typing import List, Tuple

from dataclasses import dataclass

from com.expleague.media_space.topics.processing_manager import StoryItem


@dataclass
class StoryState:
    id: str
    name: str
    last_updated: datetime
    topics: List[Tuple[str, float]]
    lexis: List[Tuple[str, float]]
    vec_sum: List[float]
    vec_num: int

    def story(self) -> StoryItem:
        pass
