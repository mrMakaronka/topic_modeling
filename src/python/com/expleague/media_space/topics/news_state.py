from typing import List

from dataclasses import dataclass


@dataclass
class NewsState:
    id: str
    name: str
    vec_sum: List[float]
    vec_num: int
