'''
UnitModelState Dataclass

Data container for model state - holds teams, baselines, and history.
Pure data class with no IO logic.
'''

from dataclasses import dataclass
from typing import Dict, List

from ..Entities import (
    Team, LeagueBaseline, LeagueQb,
    LeaguePace, TeamGameRecord
)

@dataclass
class UnitModelState:
    '''
    Immutable snapshot of model state at a point in time.

    Attributes:
        season: The season this state represents
        week: The week this state is FOR (about to predict)
        teams: Dictionary of team abbreviation -> Team object
        league_baseline: LeagueBaseline object
        league_qb: LeagueQb object
        league_pace: LeaguePace object
        team_game_records: List of TeamGameRecord dicts (historical output)
    '''
    season: int
    week: int
    teams: Dict[str, Team]
    league_baseline: LeagueBaseline
    league_qb: LeagueQb
    league_pace: LeaguePace = None
    team_game_records: List[TeamGameRecord] = None

