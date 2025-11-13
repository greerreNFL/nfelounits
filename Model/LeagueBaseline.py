'''
LeagueBaseline Class

Tracks league-wide EPA averages for each unit type using EWMA.
'''

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class LeagueBaseline:
    '''
    Tracks league-wide average EPA for pass, rush, and special teams
    using exponentially weighted moving average

    Initialize with 1999 values
    '''
    ## main active levels ##
    pass_avg: float = 0.721
    rush_avg: float = -3.911
    st_avg: float = 2.249
    ## long term levels for regression ##
    pass_avg_lt: float = 0.721
    rush_avg_lt: float = -3.911 
    st_avg_lt: float = 2.249
    last_game_season: Optional[int] = None
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        '''Initialize params if not provided'''
        if self.params is None:
            self.params = {}
    
    def update(self, unit_type: str, observed_epa: float, season: int) -> None:
        '''
        Update league average for a given unit type
        
        Parameters:
        * unit_type: 'pass', 'rush', or 'st'
        * observed_epa: Observed EPA value from the game
        * season: Current season
        '''
        ## get smoothing factor ##
        sf_param = f'league_{unit_type}_sf'
        sf = self.params[sf_param]
        
        ## update the appropriate average ##
        if unit_type == 'pass':
            self.pass_avg = sf * observed_epa + (1 - sf) * self.pass_avg
        elif unit_type == 'rush':
            self.rush_avg = sf * observed_epa + (1 - sf) * self.rush_avg
        elif unit_type == 'st':
            self.st_avg = sf * observed_epa + (1 - sf) * self.st_avg
        else:
            raise ValueError(f'Invalid unit_type: {unit_type}')
        
        self.last_game_season = season
    
    def regress(self) -> None:
        '''
        Apply offseason regression to league averages
        
        League averages should regress back toward 0 between seasons
        since EPA is theoretically centered at 0
        '''
        ## get reversion rates ##
        pass_reversion = self.params['league_pass_reversion']
        rush_reversion = self.params['league_rush_reversion']
        st_reversion = self.params['league_st_reversion']
        
        ## regress each average ##
        self.pass_avg = (1 - pass_reversion) * self.pass_avg + pass_reversion * self.pass_avg_lt
        self.rush_avg = (1 - rush_reversion) * self.rush_avg + rush_reversion * self.rush_avg_lt    
        self.st_avg = (1 - st_reversion) * self.st_avg + st_reversion * self.st_avg_lt

        ## update long term averages ##
        self.pass_avg_lt = self.pass_avg
        self.rush_avg_lt = self.rush_avg
        self.st_avg_lt = self.st_avg
        
        ## reset season tracking ##
        self.last_game_season = None
    
    def get_avg(self, unit_type: str, current_season: int) -> float:
        '''
        Get league average for a unit type, applying regression if needed
        
        Parameters:
        * unit_type: 'pass', 'rush', or 'st'
        * current_season: Current season year
        
        Returns:
        * League average EPA for the unit type
        '''
        ## check if we need to regress (new season) ##
        if self.last_game_season is not None and current_season > self.last_game_season:
            self.regress()
        
        ## return the appropriate average ##
        if unit_type == 'pass':
            return self.pass_avg
        elif unit_type == 'rush':
            return self.rush_avg
        elif unit_type == 'st':
            return self.st_avg
        else:
            raise ValueError(f'Invalid unit_type: {unit_type}')
    
    def as_record(self) -> Dict[str, Any]:
        '''Return league baseline state as dictionary'''
        return {
            'pass_avg': round(self.pass_avg, 3),
            'rush_avg': round(self.rush_avg, 3),
            'st_avg': round(self.st_avg, 3),
        }

