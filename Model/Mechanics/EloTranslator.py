'''
EloTranslator Class

Translates unit values to elo ratings and calculates contextual adjustments.
'''

from typing import Dict
from ..Entities.Team import Team
from .GameContext import GameContext


class EloTranslator:
    '''
    Translates team unit values to elo ratings with contextual adjustments
    
    Base elo is 1505 + weighted sum of unit values
    Context adjustments capture weather impacts on the elo rating
    '''
    
    def __init__(self, elo_config: Dict[str, float]):
        '''
        Initialize EloTranslator with coefficient values
        
        Parameters:
        * elo_config: Dictionary of coefficient values (not ModelParam objects)
                     Keys: pass_off_coef, rush_off_coef, st_off_coef,
                           pass_def_coef, rush_def_coef, st_def_coef
        '''
        self.pass_off_coef = elo_config['pass_off_coef']
        self.rush_off_coef = elo_config['rush_off_coef']
        self.st_off_coef = elo_config['st_off_coef']
        self.pass_def_coef = elo_config['pass_def_coef']
        self.rush_def_coef = elo_config['rush_def_coef']
        self.st_def_coef = elo_config['st_def_coef']
    
    def translate_to_elo(self, team: Team) -> float:
        '''
        Convert team unit values to an elo rating
        
        Formula: 1505 + sum((unit.value + unit.trend) * coefficient)

        Uses value + trend to include momentum in the elo forecast,
        consistent with how get_expected_epa() forecasts unit performance.

        Parameters:
        * team: Team object with all 6 units

        Returns:
        * Elo rating for the team
        '''
        elo = 1505.0
        elo += (team.pass_off.value + team.pass_off.trend) * self.pass_off_coef
        elo += (team.rush_off.value + team.rush_off.trend) * self.rush_off_coef
        elo += (team.st_off.value + team.st_off.trend) * self.st_off_coef
        elo += (team.pass_def.value + team.pass_def.trend) * self.pass_def_coef
        elo += (team.rush_def.value + team.rush_def.trend) * self.rush_def_coef
        elo += (team.st_def.value + team.st_def.trend) * self.st_def_coef
        return elo
    
    def calculate_context_adj(self, team: Team, game_context: GameContext) -> float:
        '''
        Calculate weather-based elo adjustment for a team
        
        Applies ONLY weather adjustments (no HFA, no QB) to unit values,
        recalculates elo, and returns the delta from base elo
        
        This captures how weather conditions affect the team's effective strength
        relative to their baseline rating
        
        Parameters:
        * team: Team object with all 6 units
        * game_context: GameContext object with weather information
        
        Returns:
        * Context adjustment (elo_with_weather - base_elo)
        '''
        ## calculate base elo ##
        base_elo = self.translate_to_elo(team)
        
        ## get weather adjustments for each unit type ##
        pass_weather_adj = game_context.weather_adj('pass')
        rush_weather_adj = game_context.weather_adj('rush')
        st_weather_adj = game_context.weather_adj('st')
        
        ## calculate adjusted elo with weather impacts ##
        ## weather_adj is negative (reduces EPA) for the offense ##
        adjusted_elo = 1505.0
        adjusted_elo += (team.pass_off.value + team.pass_off.trend + pass_weather_adj) * self.pass_off_coef
        adjusted_elo += (team.rush_off.value + team.rush_off.trend + rush_weather_adj) * self.rush_off_coef
        adjusted_elo += (team.st_off.value + team.st_off.trend + st_weather_adj) * self.st_off_coef
        adjusted_elo += (team.pass_def.value + team.pass_def.trend) * self.pass_def_coef
        adjusted_elo += (team.rush_def.value + team.rush_def.trend) * self.rush_def_coef
        adjusted_elo += (team.st_def.value + team.st_def.trend) * self.st_def_coef
        
        ## return the delta ##
        return adjusted_elo - base_elo

