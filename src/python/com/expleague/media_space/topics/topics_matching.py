from collections import defaultdict
from typing import List, Tuple, Optional

from com.expleague.media_space.topics.file_read_util import FileReadUtil
import numpy as np

from com.expleague.media_space.topics.levels_parser import LevelsParser


class TopicsMatching:
    def __init__(self, file_path):
        self.topics_matching = FileReadUtil.load_topics_matching(file_path)

        self.level1_id_names = {}
        self.level1_names_id = {}
        self.level2_id_names = {}
        self.level2_names_id = {}

        self.levels_parer = LevelsParser()
        for l in self.topics_matching:
            for lvl1_lvl2 in l:
                levels = self.levels_parer.parse(lvl1_lvl2)
                self.level1_id_names[levels.level1_id] = levels.level1_name
                self.level1_names_id[levels.level1_name] = levels.level1_id
                self.level2_id_names[levels.level2_id] = levels.level2_name
                self.level2_names_id[levels.level2_name] = levels.level2_id

    def match(self, vec: np.ndarray) -> List[Tuple[str, float]]:
        weights = defaultdict(float)
        for i in range(len(vec)):
            if vec[i] > 0:
                for t in self.topics_matching[i]:
                    weights[t] += vec[i] / len(self.topics_matching[i])

        s = sum(weights.values())
        if s == 0:
            return list()
        factor = 1.0 / s
        normalised_weights = {k: v * factor for k, v in weights.items()}
        normalised_weights = sorted(normalised_weights.items(), key=lambda kv: kv[1], reverse=True)
        return list(normalised_weights)

    def topics_by_id(self, id_) -> List[Tuple[int, float]]:
        result = []
        for i in range(len(self.topics_matching)):
            for lvl1_lvl2 in self.topics_matching[i]:
                levels = self.levels_parer.parse(lvl1_lvl2)
                if id_ == levels.level1_id or id_ == levels.level2_id:
                    result.append((i, 1.0 / len(self.topics_matching[i])))
        return result

    def level1_name_by_id(self, _id: str) -> Optional[str]:
        return self.level1_id_names.get(_id, None)

    def level1_id_by_name(self, name: str) -> Optional[str]:
        return self.level1_names_id.get(name, None)

    def level2_name_by_id(self, _id: str) -> Optional[str]:
        return self.level2_id_names.get(_id, None)

    def level2_id_by_name(self, name: str) -> Optional[str]:
        return self.level2_names_id.get(name, None)

    def level1_ids(self) -> List[str]:
        return list(self.level1_id_names.keys())

    def level2_ids(self) -> List[str]:
        return list(self.level2_id_names.keys())
