'''
LeaguePace Class

Tracks league-wide average play counts per unit type using EWMA.
Provides regression targets for team-level pace and initialization seeds.
'''

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class LeaguePace:
    '''
    Tracks league-wide average play count for pass, rush, and special teams
    using exponentially weighted moving average.

    Initialized with 1999 values from historical data.
    Variance tracks within-team game-to-game play count variance.
    '''
    ## pace means (league avg plays/game by unit) ##
    pass_mean: float = 39.9
    rush_mean: float = 25.0
    st_mean: float = 13.8
    ## pace variances (within-team game-to-game variance) ##
    pass_var: float = 79.2
    rush_var: float = 57.8
    st_var: float = 5.8
    last_game_season: Optional[int] = None
    params: Dict[str, Any] = None

    def __post_init__(self):
        '''Initialize params if not provided'''
        if self.params is None:
            self.params = {}

    def update(self, unit_type: str, plays: int, season: int) -> None:
        '''
        Update league pace for a given unit type

        Parameters:
        * unit_type: 'pass', 'rush', or 'st'
        * plays: Number of plays observed
        * season: Current season
        '''
        ## get smoothing factor (shares team-level pace_sf per unit type) ##
        sf = self.params['unit_config'][f'{unit_type}_pace_sf']
        ## update mean and variance ##
        mean_attr = f'{unit_type}_mean'
        var_attr = f'{unit_type}_var'
        current_mean = getattr(self, mean_attr)
        deviation = plays - current_mean
        new_mean = sf * plays + (1 - sf) * current_mean
        new_var = sf * (deviation ** 2) + (1 - sf) * getattr(self, var_attr)
        setattr(self, mean_attr, new_mean)
        setattr(self, var_attr, new_var)
        self.last_game_season = season

    def regress(self) -> None:
        '''
        Apply offseason regression to league pace.
        League pace persists fully (no reversion) since it tracks
        macro trends (rule changes, era shifts).
        '''
        self.last_game_season = None

    def get_pace(self, unit_type: str, current_season: int) -> tuple:
        '''
        Get league pace mean and variance for a unit type

        Parameters:
        * unit_type: 'pass', 'rush', or 'st'
        * current_season: Current season year

        Returns:
        * Tuple of (mean, variance) for the unit type
        '''
        ## check if we need to regress (new season) ##
        if self.last_game_season is not None and current_season > self.last_game_season:
            self.regress()
        mean = getattr(self, f'{unit_type}_mean')
        var = getattr(self, f'{unit_type}_var')
        return mean, var
