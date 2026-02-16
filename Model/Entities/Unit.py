'''
Unit Class

Represents a team unit (offensive or defensive) with EWMA-style updates.
'''

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from .Types import UnitType, Side
from .Pace import Pace


@dataclass
class Unit:
    '''
    Represents a team unit (offense or defense) with EPA tracking
    '''
    team: str
    unit_type: UnitType
    side: Side
    value: float = 0.0
    trend: float = 0.0
    last_game_season: Optional[int] = None
    coach: Optional[str] = None
    params: Dict[str, Any] = None
    pending_update: bool = False
    pace: Optional[Pace] = None
    
    def __post_init__(self):
        '''Initialize params if not provided'''
        if self.params is None:
            self.params = {}

    ## ==================== Private Helpers ==================== ##

    def _param(self, suffix: str) -> str:
        '''
        Build a config param key for this unit, e.g. "pass_off_sf"
        Params for all units and context is contained in the config that is passed.
        Therefore, the unit specific params are dynamically looked up using these param
        constructors to avoid repeating the same construction throughout the code
        '''
        return f'{self.unit_type.value}_{self.side.value}_{suffix}'

    def _qb_adjs(self,
        home_qb_adj: float,
        away_qb_adj: float,
        is_home: bool
    ) -> Tuple[float, float]:
        '''
        Return (qb_adj, opp_qb_adj) in EPA scale.
        Both are 0 for non-pass units.
        '''
        if self.unit_type == UnitType.PASS:
            qb_adj = home_qb_adj / 25 if is_home else away_qb_adj / 25
            opp_qb_adj = away_qb_adj / 25 if is_home else home_qb_adj / 25
            return qb_adj, opp_qb_adj
        return 0.0, 0.0

    def _observed_performance(self,
        observed_epa: float,
        opponent_value: float,
        qb_adj: float,
        opp_qb_adj: float,
        location_effect_adj: float,
        weather_adj: float,
        league_avg: float,
    ) -> float:
        '''
        Calculate opponent-adjusted observed performance.

        Offense: how much better/worse than expected given context
        Defense: how much less/more EPA allowed than expected
        '''
        if self.side == Side.OFFENSE:
            return (
                observed_epa - (qb_adj + location_effect_adj + weather_adj) +
                opponent_value -
                league_avg
            )
        return (
            opponent_value + league_avg + (opp_qb_adj - location_effect_adj + weather_adj) -
            observed_epa
        )

    def _expected_epa_raw(self,
        unit_forecast: float,
        opponent_value: float,
        qb_adj: float,
        opp_qb_adj: float,
        location_effect_adj: float,
        weather_adj: float,
        league_avg: float,
    ) -> float:
        '''
        Calculate expected EPA from unit forecast and game context.

        Offense: unit value + context advantages - opponent + league avg
        Defense: opponent value + context + league avg
        '''
        if self.side == Side.OFFENSE:
            return (
                unit_forecast +
                (qb_adj + location_effect_adj + weather_adj) -
                opponent_value +
                league_avg
            )
        return (
            opponent_value + (opp_qb_adj - location_effect_adj + weather_adj) +
            league_avg
        )

    ## ==================== Public Methods ==================== ##

    def update(self,
        ## base values ##
        observed_epa: float, opponent_value: float,
        ## adj values ##
        location_effect_adj: float, home_qb_adj: float, away_qb_adj: float,
        weather_adj: float,
        ## state values ##
        season: int, coach: str,
        ## determine usage ##
        is_home: bool,
        league_avg: float,
        plays: Optional[int] = None,
    ) -> None:
        '''
        Update unit rating using exponentially weighted moving average

        Parameters:
        * observed_epa: Actual EPA generated (off) or allowed (def) by unit in this game
        * opponent_value: Opponent unit's pre-game value for adjustment
        * location_effect_adj: Location effect adjustment (already calculated for this unit)
        * home_qb_adj: Home team QB adjustment
        * away_qb_adj: Away team QB adjustment
        * weather_adj: Weather adjustment (negative for bad weather, add directly)
        * season: Season year
        * coach: name of the coach for the team
        * is_home: Whether this unit's team is home
        * league_avg: League-wide average EPA for this unit type
        * plays: Number of plays for pace tracking (optional)
        '''
        ## get smoothing factors ##
        sf = self.params['unit_config'][self._param('sf')]
        trend_sf = self.params['unit_config'][self._param('trend_sf')]
        ## calculate adjustments and observed performance ##
        qb_adj, opp_qb_adj = self._qb_adjs(home_qb_adj, away_qb_adj, is_home)
        observed_performance = self._observed_performance(
            observed_epa, opponent_value,
            qb_adj, opp_qb_adj,
            location_effect_adj, weather_adj, league_avg,
        )
        ## apply pace-based sf discount ##
        ut = self.unit_type.value
        pace_sf = self.params['unit_config'][f'{ut}_pace_sf']
        pace_threshold = self.params['unit_config'][f'{ut}_pace_disc_threshold']
        if plays is not None and pace_threshold > 0:
            ## if we have a pace initialized, apply the discount ##
            if self.pace is not None:
                pace_discount = self.pace.get_sf_discount(plays, pace_threshold)
                sf = sf * pace_discount
                if trend_sf > 0:
                    trend_sf = trend_sf * pace_discount
            ## post discount logic, handle pace update ##
            ## if no pace (ie the above was skipped), initialize) ##
            ## else, update normally
            if self.pace is None:
                self.pace = Pace(mean=float(plays), var=0.0)
            else:
                self.pace.update(plays, pace_sf)
        ## update value using Holt-style exponential smoothing ##
        prev_value = self.value
        self.value = sf * observed_performance + (1 - sf) * (self.value + self.trend)
        ## update trend ##
        if trend_sf > 0:
            self.trend = trend_sf * (self.value - prev_value) + (1 - trend_sf) * self.trend
        ## update state ##
        self.last_game_season = season
        self.coach = coach
        self.pending_update = False
    
    def regress(self,
        coach: str,
        team_qb_starter_value: float = 0.0,
        league_qb_avg: float = 75.0,
        league_pace_mean: float = None,
        league_pace_var: float = None,
    ) -> None:
        '''
        Offseason regression with optional QB-adjusted target for pass offense

        Parameters:
        * coach: Coach name
        * team_qb_starter_value: Week 1 starter's value (in Elo), used for pass offense
        * league_qb_avg: League average QB value (in Elo), used for pass offense
        * league_pace_mean: League average pace for this unit type (regression target)
        * league_pace_var: League average pace variance for this unit type (regression target)
        '''
        ## get reversion rate ##
        reversion_rate = self.params['unit_config'][self._param('reversion')]
        ## for pass offense, also regress toward QB value ##
        if self.unit_type == UnitType.PASS and self.side == Side.OFFENSE:
            qb_reversion_rate = self.params['unit_config']['pass_off_qb_reversion']
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
        ## reset trend at offseason (no momentum carries over) ##
        self.trend = 0.0
        ## regress pace toward league average ##
        pace_reversion = self.params['unit_config'][f'{self.unit_type.value}_pace_reversion']
        if self.pace is not None and league_pace_mean is not None:
            self.pace.regress(league_pace_mean, league_pace_var or 0.0, pace_reversion)
        ## update state ##
        self.last_game_season = None
        self.coach = coach
    
    def get_value(self,
        current_season: int,
        coach: str,
        team_qb_starter_value: float = 0.0,
        league_qb_avg: float = 75.0,
        league_pace_mean: float = None,
        league_pace_var: float = None,
    ) -> float:
        '''
        Gets the value of the unit while handling regression if needed

        Parameters:
        * current_season: Current season year
        * coach: Coach name
        * team_qb_starter_value: Week 1 starter's value (in Elo), used for pass offense regression
        * league_qb_avg: League average QB value (in Elo), used for pass offense regression
        * league_pace_mean: League average pace for this unit type (for pace regression)
        * league_pace_var: League average pace variance for this unit type (for pace regression)

        Returns:
        * Value of the unit

        Raises:
        * RuntimeError: If get_value() is called when a previous game was not updated
          (indicates attempting to process multiple unplayed weeks)
        '''
        ## check if previous game was not updated (unplayed game protection) ##
        if self.pending_update:
            raise RuntimeError(
                f'Unit {self.team} {self.unit_type.value}_{self.side.value} has a pending update. '
                f'Cannot get value again before update() is called. '
                f'This typically means multiple unplayed weeks were passed to the model.'
            )
        ## check if offseason regression is needed ##
        if self.last_game_season is not None and self.last_game_season < current_season:
            self.regress(coach, team_qb_starter_value, league_qb_avg, league_pace_mean, league_pace_var)
        ## initialize pace from league if no history ##
        if self.pace is None and league_pace_mean is not None:
            self.pace = Pace.from_league(league_pace_mean, league_pace_var)
        ## set pending update flag ##
        self.pending_update = True
        ## return value + trend ##
        return self.value + self.trend
    
    def get_expected_epa(self,
        opponent_value: float,
        location_effect_adj: float,
        home_qb_adj: float,
        away_qb_adj: float,
        weather_adj: float,
        is_home: bool,
        league_avg: float
    ) -> float:
        '''
        Calculate expected EPA for this unit given game conditions

        Parameters:
        * opponent_value: Opponent unit's pre-game value
        * location_effect_adj: Location effect adjustment (already calculated for this unit)
        * home_qb_adj: Home team QB adjustment
        * away_qb_adj: Away team QB adjustment
        * weather_adj: Weather adjustment (negative for bad weather, add directly)
        * is_home: Whether this unit's team is home
        * league_avg: League-wide average EPA for this unit type

        Returns:
        * Expected EPA for this unit
        '''
        qb_adj, opp_qb_adj = self._qb_adjs(home_qb_adj, away_qb_adj, is_home)
        return self._expected_epa_raw(
            self.value + self.trend, opponent_value,
            qb_adj, opp_qb_adj,
            location_effect_adj, weather_adj, league_avg,
        )

    def as_record(self) -> Dict[str, Any]:
        '''Return unit state as dictionary for storage'''
        return {
            'unit_type': self.unit_type.value,
            'team': self.team,
            'side': self.side.value,
            'value': round(self.value, 3),
            'trend': round(self.trend, 3),
            'pace_mean': round(self.pace.mean, 1) if self.pace is not None else None,
            'pace_var': round(self.pace.var, 1) if self.pace is not None else None,
        }
