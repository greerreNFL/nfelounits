'''
LeagueQb Class

Tracks league-wide QB value using EWMA (no regression).
'''

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class LeagueQb:
    '''
    Tracks league-wide average QB value (in Elo units) using EWMA
    
    No offseason regression - tracks absolute league-wide QB quality
    '''
    qb_avg: float = 75.0  # Initialize at 75 Elo
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        '''Initialize params if not provided'''
        if self.params is None:
            self.params = {}
    
    def update(self, observed_qb_value: float) -> None:
        '''
        Update league average QB value
        
        Parameters:
        * observed_qb_value: Observed QB value (in Elo units) from qbelo
        '''
        ## get smoothing factor ##
        sf = self.params['unit_config']['league_qb_sf']
        
        ## update average ##
        self.qb_avg = sf * observed_qb_value + (1 - sf) * self.qb_avg
    
    def get_avg(self) -> float:
        '''
        Get league average QB value (no regression applied)
        
        Parameters:
        * current_season: Current season year
        
        Returns:
        * League average QB value in Elo units
        '''
        return self.qb_avg
    
    def as_record(self) -> Dict[str, Any]:
        '''Return league QB state as dictionary'''
        return {
            'qb_avg': round(self.qb_avg, 3),
        }

