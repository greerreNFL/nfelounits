'''
GameContext Class

Contains state about game conditions (weather, etc.) and calculates contextual adjustments.
'''

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import pandas as pd
from ...Utilities.CurveUtils import s_curve


@dataclass
class GameContext:
    '''
    An object that contains game context information and calculates adjustments
    '''
    ## initing meta ##
    game_id: str
    config: Dict[str, Any]
    location_effect_base: float
    ## optional ##
    temp: Optional[float] = None
    wind: Optional[float] = None
    
    def weather_adj(self, unit_type: str) -> float:
        '''
        Calculate the weather adjustment for a specific unit type.
        
        Returns a negative value for bad weather (high wind, low temp) since
        bad weather reduces offensive EPA. Adjustments should be added directly.
        
        Parameters:
        * unit_type: The unit type ('pass', 'rush', or 'st')
        
        Returns:
        * The total weather adjustment (negative for bad weather)
        '''
        ## hard-coded midpoints ##
        TEMP_MP = 32
        WIND_MP = 18
        
        ## handle values ##
        wind = max(0, min(30, self.wind-5 if not pd.isnull(self.wind) else 0))
        temp = max(0, self.temp if not pd.isnull(self.temp) else 70)
        ## calc adjs using unit-specific height params ##
        wind_adj = s_curve(
            self.config['unit_config'][f'{unit_type}_wind_disc_height'],
            WIND_MP,
            wind,
            'up'
        )
        temp_adj = s_curve(
            self.config['unit_config'][f'{unit_type}_temp_disc_height'],
            TEMP_MP,
            temp,
            'down'
        )
        ## return negative since bad weather hurts offense ##
        return -1 * (temp_adj + wind_adj)
    
    def location_effect_adj(self, unit_type: str, is_home: bool) -> float:
        '''
        Calculate the location effect adjustment for a specific unit type.
        Positive for home teams, negative for away; negative share = discount home expectation.

        Parameters:
        * unit_type: The unit type ('pass', 'rush', or 'st')
        * is_home: Whether the team is home

        Returns:
        * The location effect adjustment (positive for home, negative for away)
        '''
        ## divide by 2 since adjustment is applied to both home and away teams ##
        adj = self.location_effect_base / 2 * self.config['unit_config'][f'{unit_type}_location_effect_share'] * (1 if is_home else -1)
        return adj

