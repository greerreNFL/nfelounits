'''
TeamQb Class

Tracks Week 1 starter info and calculates QB adjustments.
'''

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class TeamQb:
    '''
    Tracks a team's Week 1 starter QB information
    
    Returns adjustment = 0 when starter plays, otherwise returns
    the difference between current QB and starter (in Elo units).
    '''
    team: str
    starter_name: Optional[str] = None
    starter_value: float = 0.0
    starter_season: Optional[int] = None
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        '''Initialize params if not provided'''
        if self.params is None:
            self.params = {}
    
    def get_adjustment(self, current_qb_name: str, current_qb_value: float, season: int) -> float:
        '''
        Calculate QB adjustment based on starter vs current QB
        
        Logic:
        - First game of season: Set current QB as starter, return 0
        - Starter playing: Update starter value, return 0
        - Backup playing: Return difference (in Elo, Unit.py divides by 25)
        
        Parameters:
        * current_qb_name: Name of QB playing this game
        * current_qb_value: QB's projected value (in Elo units) from qbelo
        * season: Current season
        
        Returns:
        * QB adjustment in Elo units (0 if starter, difference if backup)
        '''
        ## if new season or first game ever, set starter ##
        if self.starter_season is None or season > self.starter_season:
            self.starter_name = current_qb_name
            self.starter_value = current_qb_value
            self.starter_season = season
            return 0.0
        
        ## if current QB is the starter, update value and return 0 ##
        if current_qb_name == self.starter_name:
            self.starter_value = current_qb_value
            return 0.0
        
        ## different QB playing - return adjustment (in Elo) ##
        return current_qb_value - self.starter_value
    
    def as_record(self) -> Dict[str, Any]:
        '''Return team QB state as dictionary'''
        return {
            'team': self.team,
            'starter_name': self.starter_name,
            'starter_value': round(self.starter_value, 3),
            'starter_season': self.starter_season,
        }
