from enum import Enum
from typing import TypedDict


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

class TeamGameRecord(TypedDict):
    '''
    Type definition for a single team-game record output by the model.
    '''
    game_id: str
    season: int
    week: int
    team: str
    opponent: str
    is_home: bool
    result: float
    qb_value: float
    qb_adj: float
    coach: str
    ## pre-game unit values ##
    pass_off_value_pre: float
    rush_off_value_pre: float
    st_off_value_pre: float
    pass_def_value_pre: float
    rush_def_value_pre: float
    st_def_value_pre: float
    ## elo values ##
    elo: float
    context_adj: float
    win_prob: float
    ## expected/observed ##
    pass_off_expected: float
    pass_off_observed: float
    pass_def_expected: float
    pass_def_observed: float
    rush_off_expected: float
    rush_off_observed: float
    rush_def_expected: float
    rush_def_observed: float
    st_off_expected: float
    st_off_observed: float
    st_def_expected: float
    st_def_observed: float
    ## value created ##
    pass_off_value_created: float
    pass_def_value_created: float
    rush_off_value_created: float
    rush_def_value_created: float
    st_off_value_created: float
    st_def_value_created: float
    ## league averages ##
    pass_league_avg: float
    rush_league_avg: float
    st_league_avg: float
    ## opponent faced (positive = harder) ##
    pass_off_faced: float
    pass_def_faced: float
    rush_off_faced: float
    rush_def_faced: float
    st_off_faced: float
    st_def_faced: float
    ## post-game unit values ##
    pass_off_value_post: float
    rush_off_value_post: float
    st_off_value_post: float
    pass_def_value_post: float
    rush_def_value_post: float
    st_def_value_post: float

