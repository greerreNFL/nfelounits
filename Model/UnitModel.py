'''
UnitModel Class

Main model class that iterates through games and updates unit ratings.
'''

from typing import Dict, List, Any, Optional
import pandas as pd
import time
from .Types import UnitType, Side
from .Unit import Unit
from .Team import Team
from .TeamQb import TeamQb
from .LeagueBaseline import LeagueBaseline
from .LeagueQb import LeagueQb
from .GameContext import GameContext
from .EloTranslator import EloTranslator
from .RecalibrationObject import RecalibrationObject
from .RecalibrationNormalizer import RecalibrationNormalizer
from ..Utilities import calculate_win_probability, s_curve


class UnitModel:
    '''Main model for tracking unit ratings across games'''
    
    def __init__(self,
        games: pd.DataFrame,
        config: Dict[str, Any],
        ## optional params for injecting state ##
        teams: Optional[Dict[str, Team]] = None,
        league_baseline: Optional[LeagueBaseline] = None,
        league_qb: Optional[LeagueQb] = None,
        min_recal_week: int = 4,
        state_collector: Optional['UnitModelStateCollector'] = None,
        recalibration_set: Optional['RecalibrationSet'] = None
    ):
        '''
        Initialize model
        
        Parameters:
        * games: Flattened team-game DataFrame from DataLoader.flatten_to_team_game()
        * config: Dictionary with nested structure {unit_config: {...}, elo_config: {...}}
        * teams: Optional pre-initialized teams dictionary (for injecting state)
        * league_baseline: Optional pre-initialized LeagueBaseline (for injecting state)
        * league_qb: Optional pre-initialized LeagueQb (for injecting state)
        * min_recal_week: Minimum week before recalibration applies (default 4)
        * state_collector: Optional collector for capturing model state at week boundaries
        * recalibration_set: Optional pre-computed recalibration values (None = no recalibration)
        '''
        self.games = games.sort_values(['season', 'week', 'game_id']).reset_index(drop=True)
        self.config = config
        ## storage - use provided or create fresh ##
        self.teams: Dict[str, Team] = teams if teams is not None else {}
        self.team_game_records: List[Dict[str, Any]] = []
        self.league_baseline: LeagueBaseline = league_baseline if league_baseline is not None else LeagueBaseline(params=config)
        self.league_qb: LeagueQb = league_qb if league_qb is not None else LeagueQb(params=config)
        ## elo translator ##
        self.elo_translator: EloTranslator = EloTranslator(config.get('elo_config', {}))
        ## runtime tracking ##
        self.model_runtime: float = 0.0
        ## recalibration settings ##
        self.min_recal_week = min_recal_week
        ## state collection ##
        self.state_collector = state_collector
        ## recalibration cache for instant lookups ##
        self.recalibration_set = recalibration_set
        ## normalizer for recalibration values ##
        self.recal_normalizer = RecalibrationNormalizer(config)
    
    def get_team(self, team_abbr: str) -> Team:
        '''
        Get existing team or create new one with fresh units
        
        Creates team with 6 units + QB tracker:
        - 3 offensive (pass, rush, st)
        - 3 defensive (pass, rush, st)
        - QB tracker
        '''
        if team_abbr not in self.teams:
            self.teams[team_abbr] = Team(
                team_abbr=team_abbr,
                pass_off=Unit(unit_type=UnitType.PASS, team=team_abbr, side='off', params=self.config),
                rush_off=Unit(unit_type=UnitType.RUSH, team=team_abbr, side='off', params=self.config),
                st_off=Unit(unit_type=UnitType.SPECIAL_TEAMS, team=team_abbr, side='off', params=self.config),
                pass_def=Unit(unit_type=UnitType.PASS, team=team_abbr, side='def', params=self.config),
                rush_def=Unit(unit_type=UnitType.RUSH, team=team_abbr, side='def', params=self.config),
                st_def=Unit(unit_type=UnitType.SPECIAL_TEAMS, team=team_abbr, side='def', params=self.config),
                qb=TeamQb(team=team_abbr, params=self.config)
            )
        return self.teams[team_abbr]
    
    def update_team(self, team: Team) -> None:
        '''Write team back to storage'''
        self.teams[team.team_abbr] = team
    
    def get_recal_obj(self, team: str, unit_type: str, season: int, week: int):
        '''
        Get recalibration object for a team/unit if available.
        
        Uses recalibration_set for lookups. If no recalibration_set is
        provided or week < min_recal_week, returns None (no recalibration).
        
        The ideal value is normalized to match UnitModel's scale before
        being used in the blend.
        
        Parameters:
        * team: Team abbreviation
        * unit_type: Unit type ('pass_off', 'rush_def', etc.)
        * season: Current season
        * week: Current week
        
        Returns:
        * RecalibrationObject if recalibration data available, None otherwise
        '''
        if self.recalibration_set is None:
            return None
        if week < self.min_recal_week:
            return None
        record = self.recalibration_set.get(season, week, team, unit_type)
        if record is None:
            return None
        ## normalize the ideal value to match UnitModel's scale ##
        normalized_value = self.recal_normalizer.normalize(
            recal_value=record.ideal_value,
            unit_type=unit_type,
            week=week
        )
        ## calculate blend weight using recal_config + s_curve ##
        recal_config = self.config.get('recal_config', {})
        activation_midpoint = recal_config.get('recal_activation_midpoint', 8.0)
        activation_steepness = recal_config.get('recal_activation_steepness', 0.5)
        activation_height = recal_config.get('recal_activation_height', 1.0)
        weight = s_curve(
            height=activation_height,
            mp=activation_midpoint,
            x=week,
            direction='up',
            steepness=activation_steepness
        )
        return RecalibrationObject(value=normalized_value, weight=weight)
    
    def process_game(self, row: pd.Series) -> Dict[str, Any]:
        '''
        Process a single game row
        
        Steps:
        1. Get team and opponent objects
        2. Access unit values (which handles regression)
        3. Update units for observed values
        4. Update all state 
        '''
        ## get team objects##
        home_team = self.get_team(row['home_team'])
        away_team = self.get_team(row['away_team'])
        ## get QB values and names ##
        home_qb_value = row['home_qb_value']
        away_qb_value = row['away_qb_value']
        home_qb_name = row['home_qb_name']
        away_qb_name = row['away_qb_name']
        league_qb_avg = self.league_qb.get_avg()
        ## calculate QB adjustments (handles season rollover and updates internally) ##
        home_qb_adj = home_team.qb.get_adjustment(home_qb_name, home_qb_value, row['season'])
        away_qb_adj = away_team.qb.get_adjustment(away_qb_name, away_qb_value, row['season'])
        ## create game context for weather and HFA adjustments ##
        game_context = GameContext(
            game_id=row['game_id'],
            config=self.config,
            hfa_base=row['hfa_base'],
            temp=row.get('temp'),
            wind=row.get('wind')
        )
        ## get recalibration objects ##
        ## if recalibrator is not used (ie none), then all will be none ##
        ## home ##
        home_pass_off_recal = self.get_recal_obj(row['home_team'], 'pass_off', row['season'], row['week'])
        home_rush_off_recal = self.get_recal_obj(row['home_team'], 'rush_off', row['season'], row['week'])
        home_st_off_recal = self.get_recal_obj(row['home_team'], 'st_off', row['season'], row['week'])
        home_pass_def_recal = self.get_recal_obj(row['home_team'], 'pass_def', row['season'], row['week'])
        home_rush_def_recal = self.get_recal_obj(row['home_team'], 'rush_def', row['season'], row['week'])
        home_st_def_recal = self.get_recal_obj(row['home_team'], 'st_def', row['season'], row['week'])
        ## away ##
        away_pass_off_recal = self.get_recal_obj(row['away_team'], 'pass_off', row['season'], row['week'])
        away_rush_off_recal = self.get_recal_obj(row['away_team'], 'rush_off', row['season'], row['week'])
        away_st_off_recal = self.get_recal_obj(row['away_team'], 'st_off', row['season'], row['week'])
        away_pass_def_recal = self.get_recal_obj(row['away_team'], 'pass_def', row['season'], row['week'])
        away_rush_def_recal = self.get_recal_obj(row['away_team'], 'rush_def', row['season'], row['week'])
        away_st_def_recal = self.get_recal_obj(row['away_team'], 'st_def', row['season'], row['week'])
        ## create records and access values ##
        ## accessing values will handle applicable regression and recalibration ##
        ## HOME ##
        home_game_record = {
            'game_id': row['game_id'],
            'season': row['season'],
            'week': row['week'],
            'team': row['home_team'],
            'opponent': row['away_team'],
            'is_home': True,
            'result': row['result'],
            'qb_value': home_qb_value,
            'qb_adj': home_qb_adj,
            'coach': row['home_coach'],
            ## get values and handle regression ##
            'pass_off_value_pre': home_team.pass_off.get_value(
                row['season'], row['home_coach'],
                home_team.qb.starter_value,
                league_qb_avg, home_pass_off_recal
            ),
            'rush_off_value_pre': home_team.rush_off.get_value(
                row['season'], row['home_coach'],
                team_qb_starter_value=home_team.qb.starter_value,
                league_qb_avg=league_qb_avg,
                recalibration=home_rush_off_recal
            ),
            'st_off_value_pre': home_team.st_off.get_value(
                row['season'], row['home_coach'],
                team_qb_starter_value=home_team.qb.starter_value,
                league_qb_avg=league_qb_avg,
                recalibration=home_st_off_recal
            ),
            'pass_def_value_pre': home_team.pass_def.get_value(
                row['season'], row['home_coach'],
                team_qb_starter_value=home_team.qb.starter_value,
                league_qb_avg=league_qb_avg,
                recalibration=home_pass_def_recal
            ),
            'rush_def_value_pre': home_team.rush_def.get_value(
                row['season'], row['home_coach'],
                team_qb_starter_value=home_team.qb.starter_value,
                league_qb_avg=league_qb_avg,
                recalibration=home_rush_def_recal
            ),
            'st_def_value_pre': home_team.st_def.get_value(
                row['season'], row['home_coach'],
                team_qb_starter_value=home_team.qb.starter_value,
                league_qb_avg=league_qb_avg,
                recalibration=home_st_def_recal
            ),
        }
        ## AWAY ##
        away_game_record = {
            'game_id': row['game_id'],
            'season': row['season'],
            'week': row['week'],
            'team': row['away_team'],
            'opponent': row['home_team'],
            'is_home': False,
            'result': -row['result'],
            'qb_value': away_qb_value,
            'qb_adj': away_qb_adj,
            'coach': row['away_coach'],
            ## get values and handle regression ##
            'pass_off_value_pre': away_team.pass_off.get_value(
                row['season'], row['away_coach'],
                team_qb_starter_value=away_team.qb.starter_value,
                league_qb_avg=league_qb_avg,
                recalibration=away_pass_off_recal
            ),
            'rush_off_value_pre': away_team.rush_off.get_value(
                row['season'], row['away_coach'],
                team_qb_starter_value=away_team.qb.starter_value,
                league_qb_avg=league_qb_avg,
                recalibration=away_rush_off_recal
            ),
            'st_off_value_pre': away_team.st_off.get_value(
                row['season'], row['away_coach'],
                team_qb_starter_value=away_team.qb.starter_value,
                league_qb_avg=league_qb_avg,
                recalibration=away_st_off_recal
            ),
            'pass_def_value_pre': away_team.pass_def.get_value(
                row['season'], row['away_coach'],
                team_qb_starter_value=away_team.qb.starter_value,
                league_qb_avg=league_qb_avg,
                recalibration=away_pass_def_recal
            ),
            'rush_def_value_pre': away_team.rush_def.get_value(
                row['season'], row['away_coach'],
                team_qb_starter_value=away_team.qb.starter_value,
                league_qb_avg=league_qb_avg,
                recalibration=away_rush_def_recal
            ),
            'st_def_value_pre': away_team.st_def.get_value(
                row['season'], row['away_coach'],
                team_qb_starter_value=away_team.qb.starter_value,
                league_qb_avg=league_qb_avg,
                recalibration=away_st_def_recal
            ),
        }
        ## Calculate elos ##
        home_elo = self.elo_translator.translate_to_elo(home_team)
        away_elo = self.elo_translator.translate_to_elo(away_team)
        ## Calculate context adjustments (weather only) ##
        home_context_adj = self.elo_translator.calculate_context_adj(home_team, game_context)
        away_context_adj = self.elo_translator.calculate_context_adj(away_team, game_context)
        ## Calculate elo diff with QB and HFA ##
        elo_diff = (
            home_elo + home_context_adj +
            home_qb_adj + row['hfa_base'] * 25
        ) - (
            away_elo + away_context_adj +
            away_qb_adj
        )
        ## Calculate win probability ##
        home_win_prob = calculate_win_probability(elo_diff)
        ## Store elo values in records ##
        home_game_record['elo'] = home_elo
        home_game_record['context_adj'] = home_context_adj
        home_game_record['win_prob'] = home_win_prob
        away_game_record['elo'] = away_elo
        away_game_record['context_adj'] = away_context_adj
        away_game_record['win_prob'] = 1-home_win_prob
        ## Update units ##
        for unit_type in ['pass', 'rush', 'st']:
            ## access units from team objects ##
            home_off_unit = getattr(home_team, f'{unit_type}_off')
            home_def_unit = getattr(home_team, f'{unit_type}_def')
            away_def_unit = getattr(away_team, f'{unit_type}_def')
            away_off_unit = getattr(away_team, f'{unit_type}_off')
            ## get league average for this unit type ##
            league_avg = self.league_baseline.get_avg(unit_type, row['season'])
            ## get adjustments for this unit type ##
            weather_adj = game_context.weather_adj(unit_type)
            home_hfa_adj = game_context.hfa_adj(unit_type, is_home=True)
            away_hfa_adj = game_context.hfa_adj(unit_type, is_home=False)
            ## calculate expected EPA before updating ##
            home_off_expected = home_off_unit.get_expected_epa(
                opponent_value=away_def_unit.value,
                hfa_adj=home_hfa_adj,
                home_qb_adj=home_qb_adj,
                away_qb_adj=away_qb_adj,
                weather_adj=weather_adj,
                is_home=True,
                league_avg=league_avg
            )
            home_def_expected = home_def_unit.get_expected_epa(
                opponent_value=away_off_unit.value,
                hfa_adj=home_hfa_adj,
                home_qb_adj=home_qb_adj,
                away_qb_adj=away_qb_adj,
                weather_adj=weather_adj,
                is_home=True,
                league_avg=league_avg
            )
            away_off_expected = away_off_unit.get_expected_epa(
                opponent_value=home_def_unit.value,
                hfa_adj=away_hfa_adj,
                home_qb_adj=home_qb_adj,
                away_qb_adj=away_qb_adj,
                weather_adj=weather_adj,
                is_home=False,
                league_avg=league_avg
            )
            away_def_expected = away_def_unit.get_expected_epa(
                opponent_value=home_off_unit.value,
                hfa_adj=away_hfa_adj,
                home_qb_adj=home_qb_adj,
                away_qb_adj=away_qb_adj,
                weather_adj=weather_adj,
                is_home=False,
                league_avg=league_avg
            )
            ## store expected and observed in records ##
            home_game_record[f'{unit_type}_off_expected'] = home_off_expected
            home_game_record[f'{unit_type}_off_observed'] = row[f'home_{unit_type}_epa']
            home_game_record[f'{unit_type}_def_expected'] = home_def_expected
            home_game_record[f'{unit_type}_def_observed'] = row[f'away_{unit_type}_epa']
            away_game_record[f'{unit_type}_off_expected'] = away_off_expected
            away_game_record[f'{unit_type}_off_observed'] = row[f'away_{unit_type}_epa']
            away_game_record[f'{unit_type}_def_expected'] = away_def_expected
            away_game_record[f'{unit_type}_def_observed'] = row[f'home_{unit_type}_epa']
            ## update units ##
            home_off_unit.update(
                observed_epa=row[f'home_{unit_type}_epa'],
                opponent_value=away_def_unit.value,
                hfa_adj=home_hfa_adj,
                home_qb_adj=home_qb_adj,
                away_qb_adj=away_qb_adj,
                weather_adj=weather_adj,
                season=row['season'],
                coach=row['home_coach'],
                is_home=True,
                league_avg=league_avg
            )
            home_def_unit.update(
                observed_epa=row[f'away_{unit_type}_epa'],
                opponent_value=away_off_unit.value,
                hfa_adj=home_hfa_adj,
                home_qb_adj=home_qb_adj,
                away_qb_adj=away_qb_adj,
                weather_adj=weather_adj,
                season=row['season'],
                coach=row['home_coach'],
                is_home=True,
                league_avg=league_avg
            )
            away_off_unit.update(
                observed_epa=row[f'away_{unit_type}_epa'],
                opponent_value=home_def_unit.value,
                hfa_adj=away_hfa_adj,
                home_qb_adj=away_qb_adj,
                away_qb_adj=home_qb_adj,
                weather_adj=weather_adj,
                season=row['season'],
                coach=row['away_coach'],
                is_home=False,
                league_avg=league_avg
            )
            away_def_unit.update(
                observed_epa=row[f'home_{unit_type}_epa'],
                opponent_value=home_off_unit.value,
                hfa_adj=away_hfa_adj,
                home_qb_adj=away_qb_adj,
                away_qb_adj=home_qb_adj,
                weather_adj=weather_adj,
                season=row['season'],
                coach=row['away_coach'],
                is_home=False,
                league_avg=league_avg
            )
            ## update league baseline (twice - once for each team) ##
            self.league_baseline.update(unit_type, row[f'home_{unit_type}_epa'], row['season'])
            self.league_baseline.update(unit_type, row[f'away_{unit_type}_epa'], row['season'])
        ## update league QB baseline ##
        self.league_qb.update(home_qb_value)
        self.league_qb.update(away_qb_value)
        ## update record for updated values ##
        home_game_record = home_game_record | {
            'pass_off_value_post': home_team.pass_off.value,
            'rush_off_value_post': home_team.rush_off.value,
            'st_off_value_post': home_team.st_off.value,
            'pass_def_value_post': home_team.pass_def.value,
            'rush_def_value_post': home_team.rush_def.value,
            'st_def_value_post': home_team.st_def.value,
        }
        away_game_record = away_game_record | {
            'pass_off_value_post': away_team.pass_off.value,
            'rush_off_value_post': away_team.rush_off.value,
            'st_off_value_post': away_team.st_off.value,
            'pass_def_value_post': away_team.pass_def.value,
            'rush_def_value_post': away_team.rush_def.value,
            'st_def_value_post': away_team.st_def.value,
        }
        ## update states ##
        self.update_team(home_team)
        self.update_team(away_team)
        ## add records to data ##
        self.team_game_records.append(home_game_record)
        self.team_game_records.append(away_game_record)
    
    def run(self) -> None:
        '''
        Main model execution - iterate through all games
        
        Handles week/season transitions and state collection.
        Recalibration is now handled via recalibration_set lookups.
        '''
        start_time = time.time()
        ## clear existing data only if not pre-initialized ##
        if not self.teams:
            self.teams = {}
        self.team_game_records = []
        ## track week/season for state collection ##
        current_week: Optional[int] = None
        current_season: Optional[int] = None
        ## process each game ##
        for idx, row in self.games.iterrows():
            game_season = row['season']
            game_week = row['week']
            ## handle week/season transitions ##
            if current_season is not None:
                if game_season != current_season:
                    ## new season - capture state for first week if collector exists ##
                    if self.state_collector is not None:
                        self.state_collector.on_week_boundary(
                            season=game_season,
                            for_week=game_week,
                            teams=self.teams,
                            league_baseline=self.league_baseline,
                            league_qb=self.league_qb
                        )
                    current_season = game_season
                    current_week = game_week
                elif game_week != current_week:
                    ## new week - capture state before processing ##
                    ## state is through current_week, FOR game_week ##
                    if self.state_collector is not None:
                        self.state_collector.on_week_boundary(
                            season=game_season,
                            for_week=game_week,
                            teams=self.teams,
                            league_baseline=self.league_baseline,
                            league_qb=self.league_qb
                        )
                    current_week = game_week
            else:
                current_season = game_season
                current_week = game_week
            self.process_game(row)
        ## track runtime ##
        end_time = time.time()
        self.model_runtime = end_time - start_time
    
    def get_results_df(self) -> pd.DataFrame:
        '''Return results as DataFrame'''
        return pd.DataFrame(self.team_game_records)
