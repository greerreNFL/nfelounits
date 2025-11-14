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
            self.config[f'{unit_type}_wind_disc_height'],
            WIND_MP,
            wind,
            'up'
        )
        temp_adj = s_curve(
            self.config[f'{unit_type}_temp_disc_height'],
            TEMP_MP,
            temp,
            'down'
        )
        ## calc the adjustment ##
        return temp_adj + wind_adj

