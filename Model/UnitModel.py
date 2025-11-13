'''
UnitModel Class

Main model class that iterates through games and updates unit ratings.
'''

from typing import Dict, List, Any
import pandas as pd
import time
from .Types import UnitType, Side
from .Unit import Unit
from .Team import Team
from .LeagueBaseline import LeagueBaseline


class UnitModel:
    '''Main model for tracking unit ratings across games'''
    
    def __init__(self, games: pd.DataFrame, config: Dict[str, Any]):
        '''
        Initialize model
        
        Parameters:
        * games: Flattened team-game DataFrame from DataLoader.flatten_to_team_game()
        * config: Dictionary with smoothing factors and reversion rates for each unit
        '''
        self.games = games.sort_values(['season', 'week', 'game_id']).reset_index(drop=True)
        self.config = config
        ## storage ##
        self.teams: Dict[str, Team] = {} ## dict that holds units
        self.team_game_records: List[Dict[str, Any]] = []
        self.league_baseline: LeagueBaseline = LeagueBaseline(params=config)
        ## runtime tracking ##
        self.model_runtime: float = 0.0
    
    def get_team(self, team_abbr: str) -> Team:
        '''
        Get existing team or create new one with fresh units
        
        Creates team with 6 units:
        - 3 offensive (pass, rush, st)
        - 3 defensive (pass, rush, st)
        '''
        if team_abbr not in self.teams:
            self.teams[team_abbr] = Team(
                team_abbr=team_abbr,
                pass_off=Unit(unit_type=UnitType.PASS, team=team_abbr, side='off', params=self.config),
                rush_off=Unit(unit_type=UnitType.RUSH, team=team_abbr, side='off', params=self.config),
                st_off=Unit(unit_type=UnitType.SPECIAL_TEAMS, team=team_abbr, side='off', params=self.config),
                pass_def=Unit(unit_type=UnitType.PASS, team=team_abbr, side='def', params=self.config),
                rush_def=Unit(unit_type=UnitType.RUSH, team=team_abbr, side='def', params=self.config),
                st_def=Unit(unit_type=UnitType.SPECIAL_TEAMS, team=team_abbr, side='def', params=self.config)
            )
        return self.teams[team_abbr]
    
    def update_team(self, team: Team) -> None:
        '''
        Write team back to storage
        '''
        self.teams[team.team_abbr] = team
    
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
        ## get QB adjustments ##
        home_qb_adj = row['home_qb_adj']
        away_qb_adj = row['away_qb_adj']
        ## create records and access values ##
        ## HOME ##
        home_game_record = {
            'game_id': row['game_id'],
            'season': row['season'],
            'week': row['week'],
            'team': row['home_team'],
            'opponent': row['away_team'],
            'is_home': True,
            'qb_adj': home_qb_adj,
            'coach': row['home_coach'],
            ## get values and handle regression ##
            'pass_off_value_pre': home_team.pass_off.get_value(row['season'], row['home_coach']),
            'rush_off_value_pre': home_team.rush_off.get_value(row['season'], row['home_coach']),
            'st_off_value_pre': home_team.st_off.get_value(row['season'], row['home_coach']),
            'pass_def_value_pre': home_team.pass_def.get_value(row['season'], row['home_coach']),
            'rush_def_value_pre': home_team.rush_def.get_value(row['season'], row['home_coach']),
            'st_def_value_pre': home_team.st_def.get_value(row['season'], row['home_coach']),
        }
        ## AWAY ##
        away_game_record = {
            'game_id': row['game_id'],
            'season': row['season'],
            'week': row['week'],
            'team': row['away_team'],
            'opponent': row['home_team'],
            'is_home': False,
            'qb_adj': away_qb_adj,
            'coach': row['away_coach'],
            ## get values and handle regression ##
            'pass_off_value_pre': away_team.pass_off.get_value(row['season'], row['away_coach']),
            'rush_off_value_pre': away_team.rush_off.get_value(row['season'], row['away_coach']),
            'st_off_value_pre': away_team.st_off.get_value(row['season'], row['away_coach']),
            'pass_def_value_pre': away_team.pass_def.get_value(row['season'], row['away_coach']),
            'rush_def_value_pre': away_team.rush_def.get_value(row['season'], row['away_coach']),
            'st_def_value_pre': away_team.st_def.get_value(row['season'], row['away_coach']),
        }
        for unit_type in ['pass', 'rush', 'st']:
            ## access units from team objects ##
            home_off_unit = getattr(home_team, f'{unit_type}_off')
            home_def_unit = getattr(home_team, f'{unit_type}_def')
            away_def_unit = getattr(away_team, f'{unit_type}_def')
            away_off_unit = getattr(away_team, f'{unit_type}_off')
            ## get league average for this unit type ##
            league_avg = self.league_baseline.get_avg(unit_type, row['season'])
            ## calculate expected EPA before updating ##
            home_off_expected = home_off_unit.get_expected_epa(
                opponent_value=away_def_unit.value,
                hfa_base=row['hfa_base'],
                home_qb_adj=row['home_qb_adj'],
                away_qb_adj=row['away_qb_adj'],
                is_home=True,
                league_avg=league_avg
            )
            home_def_expected = home_def_unit.get_expected_epa(
                opponent_value=away_off_unit.value,
                hfa_base=row['hfa_base'],
                home_qb_adj=row['home_qb_adj'],
                away_qb_adj=row['away_qb_adj'],
                is_home=True,
                league_avg=league_avg
            )
            away_off_expected = away_off_unit.get_expected_epa(
                opponent_value=home_def_unit.value,
                hfa_base=row['hfa_base'],
                home_qb_adj=row['away_qb_adj'],
                away_qb_adj=row['home_qb_adj'],
                is_home=False,
                league_avg=league_avg
            )
            away_def_expected = away_def_unit.get_expected_epa(
                opponent_value=home_off_unit.value,
                hfa_base=row['hfa_base'],
                home_qb_adj=row['away_qb_adj'],
                away_qb_adj=row['home_qb_adj'],
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
                observed_epa=row[f'home_{unit_type}_epa'], ## observed EPA
                opponent_value=away_def_unit.value, ## expected value
                hfa_base=row['hfa_base'],
                home_qb_adj=row['home_qb_adj'],
                away_qb_adj=row['away_qb_adj'],
                season=row['season'],
                coach=row['home_coach'],
                is_home=True,
                league_avg=league_avg
            )
            home_def_unit.update(
                observed_epa=row[f'away_{unit_type}_epa'], ## observed EPA
                opponent_value=away_off_unit.value,
                hfa_base=row['hfa_base'],
                home_qb_adj=row['home_qb_adj'],
                away_qb_adj=row['away_qb_adj'],
                season=row['season'],
                coach=row['home_coach'],
                is_home=True,
                league_avg=league_avg
            )
            away_off_unit.update(
                observed_epa=row[f'away_{unit_type}_epa'],
                opponent_value=home_def_unit.value,
                hfa_base=row['hfa_base'],
                home_qb_adj=row['away_qb_adj'],
                away_qb_adj=row['home_qb_adj'],
                season=row['season'],
                coach=row['away_coach'],
                is_home=False,
                league_avg=league_avg
            )
            away_def_unit.update(
                observed_epa=row[f'home_{unit_type}_epa'],
                opponent_value=home_off_unit.value,
                hfa_base=row['hfa_base'],
                home_qb_adj=row['away_qb_adj'],
                away_qb_adj=row['home_qb_adj'],
                season=row['season'],
                coach=row['away_coach'],
                is_home=False,
                league_avg=league_avg
            )
            ## update league baseline (twice - once for each team) ##
            self.league_baseline.update(unit_type, row[f'home_{unit_type}_epa'], row['season'])
            self.league_baseline.update(unit_type, row[f'away_{unit_type}_epa'], row['season'])
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
        
        Mirrors qbelo run_model() pattern
        '''
        start_time = time.time()
        ## clear existing data ##
        self.teams = {}
        self.team_game_records = []
        self.league_baseline = LeagueBaseline(params=self.config)
        ## process each game ##
        for idx, row in self.games.iterrows():
            self.process_game(row)
        ## track runtime ##
        end_time = time.time()
        self.model_runtime = end_time - start_time
    
    def get_results_df(self) -> pd.DataFrame:
        '''Return results as DataFrame'''
        return pd.DataFrame(self.team_game_records)
    
