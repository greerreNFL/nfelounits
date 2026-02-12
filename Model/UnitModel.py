'''
UnitModel Class

Main model class that iterates through games and updates unit ratings.
'''

from typing import Dict, List, Any, Optional
import pandas as pd
import time

from .Entities import (
    UnitType, Side, Unit,
    Team, TeamQb,
    LeagueBaseline, LeagueQb,
    TeamGameRecord
)
from .Mechanics import GameContext, EloTranslator
from .State import UnitModelStateManager
from ..Utilities import calculate_win_probability


class UnitModel:
    '''Main model for tracking unit ratings across games'''
    
    def __init__(self, games: pd.DataFrame, config: Dict[str, Any]):
        '''
        Initialize model
        
        Parameters:
        * games: Flattened team-game DataFrame from DataLoader.flatten_to_team_game()
        * config: Dictionary with nested structure {unit_config: {...}, elo_config: {...}}
        '''
        self.games = games.sort_values(['season', 'week', 'game_id']).reset_index(drop=True)
        self.config = config
        ## storage ##
        self.teams: Dict[str, Team] = {} ## dict that holds units
        self.team_game_records: List[TeamGameRecord] = []
        self.league_baseline: LeagueBaseline = LeagueBaseline(params=config)
        self.league_qb: LeagueQb = LeagueQb(params=config)
        ## elo translator ##
        self.elo_translator: EloTranslator = EloTranslator(config.get('elo_config', {}))
        ## state manager ##
        self.state_manager: UnitModelStateManager = UnitModelStateManager()
        ## runtime tracking ##
        self.model_runtime: float = 0.0
    
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
        3. Update units for observed values (skipped for unplayed games)
        4. Update all state
        
        For unplayed games (result is NaN):
        - Pre-game values are still calculated (regression applied if needed)
        - Elo and win probability are calculated
        - Observations, post-game values, and updates are skipped
        - Units are left with pending_update=True (safety mechanism)
        '''
        ## check if game has result (determines if played) ##
        has_result = pd.notna(row['result'])
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
        ## create records and access values ##
        ## HOME ##
        home_game_record = {
            'game_id': row['game_id'],
            'season': row['season'],
            'week': row['week'],
            'team': row['home_team'],
            'opponent': row['away_team'],
            'is_home': True,
            'result': row['result'] if has_result else None,
            'qb_value': home_qb_value,  # actual starter value
            'qb_adj': home_qb_adj,  # calculated adjustment
            'coach': row['home_coach'],
            ## get values and handle regression ##
            'pass_off_value_pre': home_team.pass_off.get_value(row['season'], row['home_coach'], home_team.qb.starter_value, league_qb_avg),
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
            'result': -row['result'] if has_result else None,  ## flip sign for away team perspective
            'qb_value': away_qb_value,  # actual starter value
            'qb_adj': away_qb_adj,  # calculated adjustment
            'coach': row['away_coach'],
            ## get values and handle regression ##
            'pass_off_value_pre': away_team.pass_off.get_value(row['season'], row['away_coach'], away_team.qb.starter_value, league_qb_avg),
            'rush_off_value_pre': away_team.rush_off.get_value(row['season'], row['away_coach']),
            'st_off_value_pre': away_team.st_off.get_value(row['season'], row['away_coach']),
            'pass_def_value_pre': away_team.pass_def.get_value(row['season'], row['away_coach']),
            'rush_def_value_pre': away_team.rush_def.get_value(row['season'], row['away_coach']),
            'st_def_value_pre': away_team.st_def.get_value(row['season'], row['away_coach']),
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

        ## Process units - expected values always, observed/updates only for played games ##
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
            ## calculate expected EPA (always, even for unplayed) ##
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
            ## store expected in records (always) ##
            home_game_record[f'{unit_type}_off_expected'] = home_off_expected
            home_game_record[f'{unit_type}_def_expected'] = home_def_expected
            away_game_record[f'{unit_type}_off_expected'] = away_off_expected
            away_game_record[f'{unit_type}_def_expected'] = away_def_expected
            ## observed and updates only for played games ##
            if has_result:
                ## store observed in records ##
                home_game_record[f'{unit_type}_off_observed'] = row[f'home_{unit_type}_epa']
                home_game_record[f'{unit_type}_def_observed'] = row[f'away_{unit_type}_epa']
                away_game_record[f'{unit_type}_off_observed'] = row[f'away_{unit_type}_epa']
                away_game_record[f'{unit_type}_def_observed'] = row[f'home_{unit_type}_epa']
                ## calculate value created ##
                ## for pass units, need QB adjustments in EPA scale ##
                if unit_type == 'pass':
                    home_qb_adj_epa = home_qb_adj / 25
                    away_qb_adj_epa = away_qb_adj / 25
                else:
                    home_qb_adj_epa = 0
                    away_qb_adj_epa = 0
                ## offense value created: (observed - league_avg) - (context - opponent) ##
                home_game_record[f'{unit_type}_off_value_created'] = (
                    ## what happened ##
                    (row[f'home_{unit_type}_epa'] - league_avg) -
                    ## what we would have expected to happen independent of the offense ##
                    ((home_hfa_adj + weather_adj) - away_def_unit.value)
                )
                away_game_record[f'{unit_type}_off_value_created'] = (
                    ## what happened ##
                    (row[f'away_{unit_type}_epa'] - league_avg) -
                    ## what we would have expected to happen independent of the offense ##
                    ((away_hfa_adj + weather_adj) - home_def_unit.value)
                )
                ## defense value created: (opponent + context) - (observed - league_avg) ##
                home_game_record[f'{unit_type}_def_value_created'] = (
                    ## what we would have expected to happen ##
                    (away_off_unit.value + (away_qb_adj_epa + away_hfa_adj + weather_adj)) -
                    ## what happened ##
                    (row[f'away_{unit_type}_epa'] - league_avg)
                )
                away_game_record[f'{unit_type}_def_value_created'] = (
                    ## what we would have expected to happen ##
                    (home_off_unit.value + (home_qb_adj_epa + home_hfa_adj + weather_adj)) -
                    ## what happened ##
                    (row[f'home_{unit_type}_epa'] - league_avg)
                )
                ## store league avg ##
                home_game_record[f'{unit_type}_league_avg'] = league_avg
                away_game_record[f'{unit_type}_league_avg'] = league_avg
                ## opponent faced (positive = harder) ##
                home_game_record[f'{unit_type}_off_faced'] = away_def_unit.value - (home_hfa_adj + weather_adj)
                away_game_record[f'{unit_type}_off_faced'] = home_def_unit.value - (away_hfa_adj + weather_adj)
                home_game_record[f'{unit_type}_def_faced'] = away_off_unit.value + (away_qb_adj_epa + away_hfa_adj + weather_adj)
                away_game_record[f'{unit_type}_def_faced'] = home_off_unit.value + (home_qb_adj_epa + home_hfa_adj + weather_adj)
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
                    home_qb_adj=home_qb_adj,
                    away_qb_adj=away_qb_adj,
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
                    home_qb_adj=home_qb_adj,
                    away_qb_adj=away_qb_adj,
                    weather_adj=weather_adj,
                    season=row['season'],
                    coach=row['away_coach'],
                    is_home=False,
                    league_avg=league_avg
                )
                ## update league baseline ##
                self.league_baseline.update(unit_type, row[f'home_{unit_type}_epa'], row['season'])
                self.league_baseline.update(unit_type, row[f'away_{unit_type}_epa'], row['season'])
            else:
                ## unplayed: set observed, value_created, league_avg, and faced to NaN ##
                home_game_record[f'{unit_type}_off_observed'] = None
                home_game_record[f'{unit_type}_def_observed'] = None
                away_game_record[f'{unit_type}_off_observed'] = None
                away_game_record[f'{unit_type}_def_observed'] = None
                home_game_record[f'{unit_type}_off_value_created'] = None
                home_game_record[f'{unit_type}_def_value_created'] = None
                away_game_record[f'{unit_type}_off_value_created'] = None
                away_game_record[f'{unit_type}_def_value_created'] = None
                home_game_record[f'{unit_type}_league_avg'] = None
                away_game_record[f'{unit_type}_league_avg'] = None
                home_game_record[f'{unit_type}_off_faced'] = None
                home_game_record[f'{unit_type}_def_faced'] = None
                away_game_record[f'{unit_type}_off_faced'] = None
                away_game_record[f'{unit_type}_def_faced'] = None
        ## update league QB baseline (only for played games) ##
        if has_result:
            self.league_qb.update(home_qb_value)
            self.league_qb.update(away_qb_value)
        ## update record for post-game values ##
        if has_result:
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
        else:
            ## unplayed: set post-game values to NaN ##
            home_game_record = home_game_record | {
                'pass_off_value_post': None,
                'rush_off_value_post': None,
                'st_off_value_post': None,
                'pass_def_value_post': None,
                'rush_def_value_post': None,
                'st_def_value_post': None,
            }
            away_game_record = away_game_record | {
                'pass_off_value_post': None,
                'rush_off_value_post': None,
                'st_off_value_post': None,
                'pass_def_value_post': None,
                'rush_def_value_post': None,
                'st_def_value_post': None,
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
        self.league_qb = LeagueQb(params=self.config)
        ## process each game ##
        for idx, row in self.games.iterrows():
            self.process_game(row)
        ## track runtime ##
        end_time = time.time()
        self.model_runtime = end_time - start_time
    
    def get_results_df(self) -> pd.DataFrame:
        '''Return results as DataFrame'''
        return pd.DataFrame(self.team_game_records)
    
    def load_from_state(self, filepath: Optional[str] = None) -> None:
        '''
        Load model state from a saved JSON file.
        
        Restores teams, baselines, and history from state.
        Filters self.games to exclude any game_id already processed.
        
        Parameters:
            filepath: Path to state file. Defaults to Model/State/model_state.json
        '''
        state = self.state_manager.load(self.config, filepath)
        ## restore state ##
        self.teams = state.teams
        self.league_baseline = state.league_baseline
        self.league_qb = state.league_qb
        self.team_game_records = state.team_game_records
        ## filter games to exclude already-processed game_ids ##
        processed_game_ids = set(r['game_id'] for r in self.team_game_records)
        original_count = len(self.games)
        self.games = self.games[~self.games['game_id'].isin(processed_game_ids)].reset_index(drop=True)
        print(f"Loaded state: {len(self.teams)} teams, {len(self.team_game_records)} records")
        print(f"Filtered games: {original_count} -> {len(self.games)} remaining")
    
    def save_to_state(self, filepath: Optional[str] = None) -> str:
        '''
        Save current model state to a JSON file.
        
        Parameters:
            filepath: Path to save to. Defaults to Model/State/model_state.json
        
        Returns:
            The filepath where state was saved
        '''
        saved_path = self.state_manager.save(
            teams=self.teams,
            league_baseline=self.league_baseline,
            league_qb=self.league_qb,
            team_game_records=self.team_game_records,
            filepath=filepath
        )
        print(f"Saved state: {len(self.teams)} teams, {len(self.team_game_records)} records -> {saved_path}")
        return saved_path
    
