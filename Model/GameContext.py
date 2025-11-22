'''
GameContext Class

Contains state about game conditions (weather, etc.) and calculates contextual adjustments.
'''

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import pandas as pd
from ..Utilities.CurveUtils import s_curve


@dataclass
class GameContext:
    '''
    An object that contains game context information and calculates adjustments
    '''
    ## initing meta ##
    game_id: str
    config: Dict[str, Any]
    hfa_base: float
    ## optional ##
    temp: Optional[float] = None
    wind: Optional[float] = None
    
    def weather_adj(self, unit_type: str) -> float:
        '''
        Calculate the negative adjustment for wind and temp for a specific unit type
        
        Parameters:
        * unit_type: The unit type ('pass', 'rush', or 'st')
        
        Returns:
        * The total weather adjustment (negative value that reduces expected EPA)
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
        ## calc the adjustment ##
        return temp_adj + wind_adj
    
    def hfa_adj(self, unit_type: str, is_home: bool) -> float:
        '''
        Calculate the home field advantage adjustment for a specific unit type
        
        Parameters:
        * unit_type: The unit type ('pass', 'rush', or 'st')
        * is_home: Whether the team is home
        
        Returns:
        * The HFA adjustment (positive for home teams, negative for away teams)
        '''
        ## if unit is home, receive positive HFA, otherwise negative ##
        ## divide by 2 since HFA is applied to both home and away teams ##
        hfa_adj = self.hfa_base / 2 * self.config['unit_config'][f'{unit_type}_hfa_share'] * (1 if is_home else -1)
        return hfa_adj

