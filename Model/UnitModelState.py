'''
UnitModelState Classes

Captures model state at week boundaries for recalibration caching.
'''

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .Team import Team
from .LeagueBaseline import LeagueBaseline
from .LeagueQb import LeagueQb


@dataclass
class UnitModelState:
    '''
    Captures model state at a week boundary.
    
    Attributes:
    * season: The season year
    * for_week: The week this state is FOR (about to predict)
    * teams: Deep copy of teams dict at this point
    * league_baseline: Deep copy of LeagueBaseline
    * league_qb: Deep copy of LeagueQb
    '''
    season: int
    for_week: int
    teams: Dict[str, Team]
    league_baseline: LeagueBaseline
    league_qb: LeagueQb


class UnitModelStateCollector:
    '''
    Observer that collects UnitModelState objects during UnitModel.run().
    
    Called by UnitModel at week boundaries to capture state for later
    recalibration computation.
    '''
    
    def __init__(self):
        '''Initialize empty state collection'''
        self.states: List[UnitModelState] = []
        self._index: Dict[Tuple[int, int], int] = {}  # (season, for_week) -> list index
    
    def on_week_boundary(self,
        season: int,
        for_week: int,
        teams: Dict[str, Team],
        league_baseline: LeagueBaseline,
        league_qb: LeagueQb
    ) -> None:
        '''
        Called by UnitModel at week transitions to capture state.
        
        Parameters:
        * season: Current season
        * for_week: The week we're about to predict (game_week)
        * teams: Current teams dict (will be deep copied)
        * league_baseline: Current LeagueBaseline (will be deep copied)
        * league_qb: Current LeagueQb (will be deep copied)
        '''
        state = UnitModelState(
            season=season,
            for_week=for_week,
            teams=copy.deepcopy(teams),
            league_baseline=copy.deepcopy(league_baseline),
            league_qb=copy.deepcopy(league_qb)
        )
        self._index[(season, for_week)] = len(self.states)
        self.states.append(state)
    
    def get_state(self, season: int, for_week: int) -> Optional[UnitModelState]:
        '''
        Lookup state by season and for_week.
        
        Parameters:
        * season: Season year
        * for_week: Week the state is FOR
        
        Returns:
        * UnitModelState if found, None otherwise
        '''
        idx = self._index.get((season, for_week))
        if idx is not None:
            return self.states[idx]
        return None
    
    def get_all_states(self) -> List[UnitModelState]:
        '''Return all collected states'''
        return self.states
    
    def clear(self) -> None:
        '''Clear all collected states'''
        self.states = []
        self._index = {}
