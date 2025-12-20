from enum import Enum


class UnitType(Enum):
    '''
    Enum for unit types - provides type safety and prevents typos
    '''
    PASS = "pass"
    RUSH = "rush"
    SPECIAL_TEAMS = "st"

class Side(Enum):
    '''
    Enum for sides - provides type safety and prevents typos
    '''
    OFFENSE = "off"
    DEFENSE = "def"

