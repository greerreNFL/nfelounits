'''
UnitRecalibrator Class

Optimizes season priors to minimize prediction error, providing ideal values
that can be blended into the main model's sequential ratings.
'''

import math
from typing import Dict, List, Any, Optional
import copy
import time
import pandas as pd
from scipy.optimize import minimize
from .Types import UnitType
from .Unit import Unit
from .Team import Team
from .TeamQb import TeamQb
from .LeagueBaseline import LeagueBaseline
from .LeagueQb import LeagueQb
from ..Performance import UnitGrader


class UnitRecalibrator:
    '''
    Finds ideal season priors by optimizing to minimize MAE.
    
    Created by UnitModel at the start of each week (after min_recal_week).
    Optimizes on initialization, then provides ideal values via get_recalibration_object().
    '''

    ############################################################################
    ## CLASS CONSTANTS
    ############################################################################
    
    ## bounds by unit type based on observed EPA ranges ##
    UNIT_BOUNDS = {
        'pass_off': (-20.0, 20.0),
        'pass_def': (-15.0, 15.0),
        'rush_off': (-12.0, 12.0),
        'rush_def': (-9.0, 9.0),
        'st_off': (-4.5, 4.5),
        'st_def': (-3.5, 3.5)
    }
    
    ## runtime constants for smart maxiter calculation ##
    ## based on analysis, time per 100 rounds grows exponentially with week ##
    TIME_PER_100_ROUNDS_BASE = 0.739  ## seconds at week 4 ##
    TIME_PER_100_ROUNDS_GROWTH = 0.147  ## exponential growth factor per week ##
    TARGET_ROUNDS = 2000  ## rounds needed for ~95% of convergence lift ##
    TIME_LIMIT_SECONDS = 100  ## max time budget per week ##
    
    ## optimizer tolerances ##
    DEFAULT_FTOL = 1e-7
    DEFAULT_EPS = 1e-6

    ############################################################################
    ## INITIALIZATION
    ############################################################################

    def __init__(self,
        games: pd.DataFrame,
        config: Dict[str, Any],
        league_baseline: LeagueBaseline,
        league_qb: LeagueQb,
        season: int,
        week: int,
        teams: Optional[Dict[str, Team]] = None
    ):
        '''
        Initialize and run optimization.
        
        Parameters:
        * games: Full games DataFrame (will be filtered to season/week)
        * config: Model configuration
        * league_baseline: Snapshot of league baseline state
        * league_qb: Snapshot of league QB state
        * season: Current season
        * week: Week we're entering (data through week-1 used for optimization)
        * teams: Current teams dict for better starting guesses (optional)
        '''
        self.season = season
        self.current_teams = teams
        self.week = week
        self.config = config
        ## filter games: this season, up until this week ##
        self.games = games[
            (games['season'] == season) & 
            (games['week'] < week)
        ].copy()
        ## store snapshots (deep copy to avoid mutation during optimization) ##
        self.league_baseline = copy.deepcopy(league_baseline)
        self.league_qb = copy.deepcopy(league_qb)
        ## get teams in data and build feature mapping ##
        if len(self.games) > 0:
            self.teams_in_data = self.get_teams_in_data()
            self.build_feature_map()
            ## bgs, and optimize() results are normalized to 0-1 range, local unit model will denorm ##
            optimal_x = self.optimize()
            self.model = self.create_local_unit_model(optimal_x)
        else:
            self.model = None

    ############################################################################
    ## NORMALIZATION UTILITIES
    ############################################################################

    def normalize(self, value: float, unit_type: str) -> float:
        '''Normalize a value to 0-1 range based on unit bounds'''
        min_val, max_val = self.UNIT_BOUNDS[unit_type]
        return (value - min_val) / (max_val - min_val)

    def denormalize(self, value: float, unit_type: str) -> float:
        '''Denormalize a 0-1 value back to real unit scale'''
        min_val, max_val = self.UNIT_BOUNDS[unit_type]
        return value * (max_val - min_val) + min_val

    ############################################################################
    ## SMART MAXITER CALCULATION
    ############################################################################

    def estimate_time_per_100_rounds(self) -> float:
        '''Estimate time per 100 rounds based on week (exponential growth)'''
        return self.TIME_PER_100_ROUNDS_BASE * math.exp(self.TIME_PER_100_ROUNDS_GROWTH * self.week)

    def calculate_target_rounds(self) -> int:
        '''
        Calculate target rounds based on time budget.
        
        Returns TARGET_ROUNDS if within time limit, otherwise calculates
        how many rounds fit in TIME_LIMIT_SECONDS.
        '''
        time_per_100 = self.estimate_time_per_100_rounds()
        estimated_time_for_target = (self.TARGET_ROUNDS / 100) * time_per_100
        if estimated_time_for_target <= self.TIME_LIMIT_SECONDS:
            return self.TARGET_ROUNDS
        else:
            ## calculate how many rounds fit in time limit ##
            rounds_per_second = 100 / time_per_100
            return int(rounds_per_second * self.TIME_LIMIT_SECONDS)

    def calculate_maxiter(self, target_rounds: int) -> int:
        '''
        Convert target rounds to maxiter for SLSQP.
        
        SLSQP does (n_params + 1) function evals per iteration.
        maxiter = ceil(target_rounds / (n_params + 1))
        '''
        n_params = len(self.features)
        return math.ceil(target_rounds / (n_params + 1))

    ############################################################################
    ## FEATURE MAP AND TEAM BUILDING
    ############################################################################

    def get_teams_in_data(self) -> List[str]:
        '''Get unique teams that appear in the filtered games'''
        home_teams = set(self.games['home_team'].unique())
        away_teams = set(self.games['away_team'].unique())
        return sorted(home_teams | away_teams)

    def build_feature_map(self) -> None:
        '''Build mapping from optimizer index to team/unit (normalized to 0-1)'''
        self.features: List[str] = []
        self.feature_to_idx: Dict[str, int] = {}
        self.feature_unit_type: Dict[str, str] = {}  ## for denormalization ##
        self.bgs: List[float] = []
        self.bounds: List[tuple] = []
        idx = 0
        for team in self.teams_in_data:
            for unit_type in ['pass_off', 'pass_def', 'rush_off', 'rush_def', 'st_off', 'st_def']:
                key = f'{team}_{unit_type}'
                self.features.append(key)
                self.feature_to_idx[key] = idx
                self.feature_unit_type[key] = unit_type
                self.bounds.append((0.0, 1.0))  ## normalized bounds ##
                ## use current team value as starting guess if available ##
                if self.current_teams and team in self.current_teams:
                    unit = getattr(self.current_teams[team], unit_type)
                    raw_value = unit.value
                else:
                    raw_value = 0.0
                ## normalize to 0-1 range ##
                self.bgs.append(self.normalize(raw_value, unit_type))
                idx += 1

    def form_teams(self, x: List[float]) -> Dict[str, Team]:
        '''
        Create Team dictionary from optimizer array.
        
        Parameters:
        * x: Optimizer values (normalized 0-1, one per team/unit combination)
        
        Returns:
        * Dict mapping team abbreviation to Team object with priors set
        '''
        teams: Dict[str, Team] = {}
        for team_abbr in self.teams_in_data:
            team = Team(
                team_abbr=team_abbr,
                pass_off=Unit(unit_type=UnitType.PASS, team=team_abbr, side='off', params=self.config),
                rush_off=Unit(unit_type=UnitType.RUSH, team=team_abbr, side='off', params=self.config),
                st_off=Unit(unit_type=UnitType.SPECIAL_TEAMS, team=team_abbr, side='off', params=self.config),
                pass_def=Unit(unit_type=UnitType.PASS, team=team_abbr, side='def', params=self.config),
                rush_def=Unit(unit_type=UnitType.RUSH, team=team_abbr, side='def', params=self.config),
                st_def=Unit(unit_type=UnitType.SPECIAL_TEAMS, team=team_abbr, side='def', params=self.config),
                qb=TeamQb(team=team_abbr, params=self.config)
            )
            ## set unit values from optimizer (denormalize from 0-1 to real scale) ##
            for unit_type in ['pass_off', 'pass_def', 'rush_off', 'rush_def', 'st_off', 'st_def']:
                key = f'{team_abbr}_{unit_type}'
                idx = self.feature_to_idx[key]
                unit = getattr(team, unit_type)
                unit.value = self.denormalize(x[idx], unit_type)
                ## set the last game to current season so no offseason regression is applied ##
                unit.last_game_season = self.season
            teams[team_abbr] = team
        return teams

    ############################################################################
    ## MODEL CREATION AND OPTIMIZATION
    ############################################################################

    def create_local_unit_model(self, x: List[float]):
        '''
        Create a localized UnitModel with given priors.
        
        This model:
        - Uses only games for this season up to this week
        - Has recalibrate=False to prevent infinite recursion
        - Uses the provided priors as starting unit values
        '''
        from .UnitModel import UnitModel
        teams = self.form_teams(x)
        model = UnitModel(
            games=self.games,
            config=self.config,
            teams=teams,
            league_baseline=copy.deepcopy(self.league_baseline),
            league_qb=copy.deepcopy(self.league_qb)
        )
        model.run()
        return model

    def objective(self, x: List[float]) -> float:
        '''Optimization objective: average MAE across all units'''
        self.rounds += 1
        model = self.create_local_unit_model(x)
        results = model.get_results_df()
        grader = UnitGrader(results)
        grades = grader.grade()
        return grades['overall_mae']

    def optimize(self) -> List[float]:
        '''Run optimization to find ideal priors with smart maxiter'''
        ## calculate smart maxiter ##
        target_rounds = self.calculate_target_rounds()
        maxiter = self.calculate_maxiter(target_rounds)
        time_est = self.estimate_time_per_100_rounds() * (target_rounds / 100)
        print(f"Recalibrating week {self.week}, {self.season} (target={target_rounds} rounds, maxiter={maxiter}, est={time_est:.0f}s)...")
        ## run optimization ##
        self.rounds = 0
        start_time = time.time()
        solution = minimize(
            self.objective,
            self.bgs,
            bounds=self.bounds,
            method='SLSQP',
            options={
                'ftol': self.DEFAULT_FTOL,
                'eps': self.DEFAULT_EPS,
                'maxiter': maxiter
            }
        )
        end_time = time.time()
        print(f"     {self.rounds} rounds, {end_time - start_time:.1f}s, MAE: {solution.fun:.4f}")
        return list(solution.x)

    ############################################################################
    ## PUBLIC API
    ############################################################################

    def get_ideal_value(self, team: str, unit_type: str) -> Optional[float]:
        '''
        Get ideal value for a specific team/unit.
        
        This is a convenience method for extracting just the ideal value
        without creating a RecalibrationObject.
        
        Parameters:
        * team: Team abbreviation
        * unit_type: Unit type ('pass_off', 'rush_def', etc.)
        
        Returns:
        * Ideal value if available, None otherwise
        '''
        if self.model is None:
            return None
        if team not in self.model.teams:
            return None
        team_obj = self.model.teams[team]
        unit = getattr(team_obj, unit_type, None)
        if unit is None:
            return None
        return unit.value
