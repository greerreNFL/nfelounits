'''
Unit Class

Represents a team unit (offensive or defensive) with EWMA-style updates.
'''

from dataclasses import dataclass
from typing import Dict, Any, Optional
from .Types import UnitType, Side


@dataclass
class Unit:
    '''Represents a team unit (offense or defense) with EPA tracking'''
    team: str
    unit_type: UnitType
    side: Side
    value: float = 0.0
    last_game_season: Optional[int] = None
    coach: Optional[str] = None
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        '''Initialize params if not provided'''
        if self.params is None:
            self.params = {}
    
    def update(self,
        ## base values ##
        observed_epa: float, opponent_value: float,
        ## adj values ##
        hfa_adj: float, home_qb_adj: float, away_qb_adj: float,
        weather_adj: float,
        ## state values ##
        season: int, coach: str,
        ## determine usage ##
        is_home: bool,
        league_avg: float,
    ) -> None:
        '''
        Update unit rating using exponentially weighted moving average
        
        Parameters:
        * observed_epa: Actual EPA generated (off) or allowed (def) by unit in this game
        * opponent_value: Opponent unit's pre-game value for adjustment
        * hfa_adj: Home field advantage adjustment (already calculated for this unit)
        * home_qb_adj: Home team QB adjustment
        * away_qb_adj: Away team QB adjustment
        * weather_adj: Weather adjustment (negative value that reduces expected EPA)
        * season: Season year
        * is_home: Whether this unit's team is home
        * league_avg: League-wide average EPA for this unit type
        
        The opponent adjustment:
        - For offense: subtract opponent's defensive value (easier vs bad defense)
        - For defense: flip sign (allowing less EPA = better defense)
        - Subtract league average to center ratings around 0
        
        '''
        ## get smoothing factor ##
        sf_param = f'{self.unit_type.value}_{self.side}_sf'
        sf = self.params['unit_config'][sf_param]
        ## if pass related unit, include the QB adjustment for self and opponent
        if self.unit_type == UnitType.PASS:
            qb_adj = home_qb_adj / 25 if is_home else away_qb_adj / 25
            opp_qb_adj = away_qb_adj / 25 if is_home else home_qb_adj / 25
        else:
            qb_adj = 0
            opp_qb_adj = 0
        ## calculate opponent-adjusted value ##
        if self.side == 'off':
            observed_performance = (
                observed_epa - (qb_adj + hfa_adj - weather_adj) + ## observed value adjusted for QB, HFA, and weather
                opponent_value - ## adjust for opponent difficulty (good defense = positive, makes this harder)
                league_avg ## subtract league average to center around 0
            )
        else:  ## def ##
            observed_performance = (
                opponent_value + league_avg + (opp_qb_adj - hfa_adj - weather_adj) - ## expected absolute EPA = opponent relative value + league avg + adjs
                observed_epa ## subtract observed absolute EPA to get defensive performance relative to league average
            )
        ## update value ##
        self.value = sf * observed_performance + (1 - sf) * self.value
        self.last_game_season = season
        self.coach = coach
    
    def regress(self, coach: str, team_qb_starter_value: float = 0.0, league_qb_avg: float = 75.0) -> None:
        '''
        Offseason regression with optional QB-adjusted target for pass offense
        
        Parameters:
        * coach: Coach name
        * team_qb_starter_value: Week 1 starter's value (in Elo), used for pass offense
        * league_qb_avg: League average QB value (in Elo), used for pass offense
        '''
        ## get reversion rate ##
        reversion_param = f'{self.unit_type.value}_{self.side}_reversion'
        reversion_rate = self.params['unit_config'][reversion_param]
        
        ## for pass offense, also regress toward QB value ##
        if self.unit_type == UnitType.PASS and self.side == 'off':
            qb_reversion_rate = self.params['unit_config'].get('pass_off_qb_reversion', 0.0)
            qb_target = (team_qb_starter_value - league_qb_avg) / 25  # convert to EPA scale
            
            ## normalize weights if they sum > 1 ##
            current_weight = max(0, 1 - reversion_rate - qb_reversion_rate)
            total = current_weight + reversion_rate + qb_reversion_rate
            
            current_weight_norm = current_weight / total
            reversion_rate_norm = reversion_rate / total
            qb_reversion_norm = qb_reversion_rate / total
            
            self.value = (
                current_weight_norm * self.value +
                reversion_rate_norm * 0 +
                qb_reversion_norm * qb_target
            )
        else:
            ## normal regression ##
            self.value = (1 - reversion_rate) * self.value
        
        ## update state ##
        self.last_game_season = None
        self.coach = coach
    
    def get_value(self, current_season: int, coach: str, team_qb_starter_value: float = 0.0, league_qb_avg: float = 75.0) -> float:
        '''
        Gets the value of the unit while handling regression if needed 
        
        Parameters:
        * current_season: Current season year
        * coach: Coach name
        * team_qb_starter_value: Week 1 starter's value (in Elo), used for pass offense regression
        * league_qb_avg: League average QB value (in Elo), used for pass offense regression
        
        Returns:
        * Value of the unit
        '''
        ## check if offseason regression is needed ##
        if self.last_game_season is not None and self.last_game_season < current_season:
            self.regress(coach, team_qb_starter_value, league_qb_avg)
        ## return value ##
        return self.value
    
    def get_expected_epa(self,
        opponent_value: float,
        hfa_adj: float,
        home_qb_adj: float,
        away_qb_adj: float,
        weather_adj: float,
        is_home: bool,
        league_avg: float
    ) -> float:
        '''
        Calculate expected EPA for this unit given game conditions
        
        Mirrors the adjustment logic from update() but returns expected EPA
        instead of updating the unit value
        
        Parameters:
        * opponent_value: Opponent unit's pre-game value
        * hfa_adj: Home field advantage adjustment (already calculated for this unit)
        * home_qb_adj: Home team QB adjustment
        * away_qb_adj: Away team QB adjustment
        * weather_adj: Weather adjustment (negative value that reduces expected EPA)
        * is_home: Whether this unit's team is home
        * league_avg: League-wide average EPA for this unit type
        
        Returns:
        * Expected EPA for this unit
        '''
        ## if pass related unit, include the QB adjustment for self and opponent ##
        if self.unit_type == UnitType.PASS:
            qb_adj = home_qb_adj / 25 if is_home else away_qb_adj / 25
            opp_qb_adj = away_qb_adj / 25 if is_home else home_qb_adj / 25
        else:
            qb_adj = 0
            opp_qb_adj = 0
        ## calculate expected EPA ##
        if self.side == 'off':
            expected = (
                self.value + ## team's unit value (relative to league avg)
                (qb_adj + hfa_adj - weather_adj) - ## add team advantages, subtract weather penalty
                opponent_value + ## subtract opponent defense (good defense = positive, so subtract)
                league_avg ## add back league average since unit value is relative
            )
        else:  ## def ##
            expected = (
                opponent_value + (opp_qb_adj - hfa_adj - weather_adj) + ## opponent's expected EPA given their advantages and weather
                league_avg ## add back league average since opponent unit value is relative
            )
        return expected

    def as_record(self) -> Dict[str, Any]:
        '''Return unit state as dictionary for storage'''
        return {
            'unit_type': self.unit_type.value,
            'team': self.team,
            'side': self.side,
            'value': round(self.value, 3),
        }
