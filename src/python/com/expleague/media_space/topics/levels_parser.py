from dataclasses import dataclass


@dataclass
class Levels:
    level1_id: str
    level1_name: str
    level2_id: str
    level2_name: str


class LevelsParser:
    def __init__(self, delimiter='/'):
        self.delimiter = delimiter

    def parse(self, lvl1_lvl2: str) -> Levels:
        split = lvl1_lvl2.split(self.delimiter)
        lvl1 = split[0]
        lvl2 = split[1]
        lvl1_id = 'lvl1_' + lvl1.replace(' ', '_').replace(',', '_')
        lvl2_id = 'lvl2_' + lvl2.replace(' ', '_').replace(',', '_')
        return Levels(lvl1_id, lvl1, lvl2_id, lvl2)
