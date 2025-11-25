'''
DataLoader Class
Retrieves and preprocesses play-by-play data from nfelodcm.
'''
import pathlib
import json

import pandas as pd
import numpy

import nfelodcm as dcm

from ..Utilities import convert_gsis_ids

class DataLoader:
    '''Load and preprocess data'''
    
    def __init__(self):
        '''Initialize loader and load datasets from nfelodcm'''
        ## load datasets into db ##
        self.db = dcm.load([
            'pbp', 'games', ## core datasets for EPA and schedule
            'hfa', 'qbelo', ## for adding adjustments
            'qb_meta', ## needed to id QB plays
        ])
        ## access datasets ##
        self.pbp: pd.DataFrame = self.db['pbp']
        self.games: pd.DataFrame = self.db['games']
        self.hfa: pd.DataFrame = self.db['hfa']
        self.qbelo: pd.DataFrame = self.db['qbelo']
        self.qb_meta: pd.DataFrame = self.db['qb_meta']
        ## prepare data ##
        self.unit_games = self.prepare()
    
    def prepare(self) -> pd.DataFrame:
        '''
        Orchestrate the data preparation pipeline
        
        Returns:
        * Game-level DataFrame ready for model consumption
        '''
        pbp = self.pbp.copy()
        parsed_pbp = self.parse_pbp(pbp)
        game_level = self.aggregate_games(parsed_pbp)
        games_with_meta = self.add_meta(game_level)
        games_with_adjs = self.add_adjustments(games_with_meta)
        return games_with_adjs
    
    def parse_pbp(self, pbp: pd.DataFrame) -> pd.DataFrame:
        '''
        Parse PBP data to create unit_bin column for aggregation
        
        Unit Definitions:
        - pass: All pass plays + QB scrambles + QB designed runs
        - rush: All non-QB rushing plays
        - st: All kicking plays (punts, kickoffs, FGs, XPs)
        
        Parameters:
        * pbp: Raw PBP DataFrame
        
        Returns:
        * DataFrame with unit_bin column added
        '''
        pbp = pbp.copy()
        ## filter to regular season only ##
        pbp = pbp[pbp['season_type'] == 'REG'].copy()
        ## filter out plays with null EPA or zero EPA (pre-snap penalties, procedural) ##
        pbp = pbp[pbp['epa'].notna()].copy()
        pbp = pbp[pbp['epa'] != 0.0].copy()
        ## filter out kneeldowns ##
        pbp = pbp[~pbp['desc'].str.contains('kneel', case=False, na=False)].copy()
        ## filter out two point conversions ##
        pbp['two_point_conv_result'] = pbp['two_point_conv_result'].fillna('Normal Play')
        pbp = pbp[pbp['two_point_conv_result'] == 'Normal Play'].copy()
        ## === IDENTIFY QB PLAYS === ##
        ## combine passer and rusher ids to capture QB as rusher ##
        pbp['passer_id'] = pbp['passer_id'].combine_first(pbp['rusher_id'])
        ## convert ids to legacy gsis format ##
        pbp = convert_gsis_ids(pbp, id_fields=['passer_id'])
        ## defragment before adding many columns ##
        pbp = pbp.copy()
        ## capture "no play" or NaN play types that had epa and were a dropback ##
        pbp['desc_based_dropback'] = numpy.where(
            (
                ((pbp['play_type'] == 'no_play') | (pbp['play_type'].isna())) &
                (
                    (pbp['desc'].str.contains(' pass ', regex=False, na=False)) |
                    (pbp['desc'].str.contains(' sacked', regex=False, na=False)) |
                    (pbp['desc'].str.contains(' scramble', regex=False, na=False))
                )
            ),
            1,
            0
        )
        ## capture a run play that featured the qb ##
        pbp['designed_qb_run'] = numpy.where(
            (
                (pbp['play_type'] == 'run') &
                ~(pbp['desc'].str.contains('Aborted', regex=False, na=False)) &
                (pbp['passer_id'].isin(self.qb_meta['gsis_id'].astype('str').tolist()))
            ),
            1,
            0
        )
        ## flag all QB plays ##
        pbp['is_qb_play'] = numpy.where(
            (pbp['qb_dropback'] == 1) |
            ((pbp['play_type'] == 'no_play') & (pbp['desc_based_dropback'] == 1)) |
            (pbp['designed_qb_run'] == 1),
            1,
            0
        )
        ## === IDENTIFY RUSH PLAYS === ##
        ## identify runs from description for no_play or NaN play_type ##
        pbp['desc_based_run'] = numpy.where(
            (
                ((pbp['play_type'] == 'no_play') | (pbp['play_type'].isna())) &
                (
                    (pbp['desc'].str.contains(' left end', regex=False, na=False)) |
                    (pbp['desc'].str.contains(' left tackle', regex=False, na=False)) |
                    (pbp['desc'].str.contains(' left guard', regex=False, na=False)) |
                    (pbp['desc'].str.contains(' up the middle', regex=False, na=False)) |
                    (pbp['desc'].str.contains(' right guard', regex=False, na=False)) |
                    (pbp['desc'].str.contains(' right tackle', regex=False, na=False)) |
                    (pbp['desc'].str.contains(' right end', regex=False, na=False))
                )
            ),
            1,
            0
        )
        ## rush plays are run plays that are NOT qb plays, OR desc-based runs ##
        pbp['is_rush_play'] = numpy.where(
            ((pbp['play_type'] == 'run') & (pbp['is_qb_play'] == 0)) |
            ((pbp['desc_based_run'] == 1) & (pbp['is_qb_play'] == 0)),
            1,
            0
        )
        ## === IDENTIFY SPECIAL TEAMS PLAYS === ##
        ## identify ST from description for no_play or NaN play_type ##
        pbp['desc_based_st'] = numpy.where(
            (
                ((pbp['play_type'] == 'no_play') | (pbp['play_type'].isna())) &
                (
                    (pbp['desc'].str.contains(' punts ', regex=False, na=False)) |
                    (pbp['desc'].str.contains(' kicks ', regex=False, na=False)) |
                    (pbp['desc'].str.contains(' field goal', regex=False, na=False)) |
                    (pbp['desc'].str.contains(' extra point', regex=False, na=False))
                )
            ),
            1,
            0
        )
        pbp['is_st_play'] = numpy.where(
            (pbp['play_type'].isin(['punt', 'kickoff', 'field_goal', 'extra_point'])) |
            (pbp['desc_based_st'] == 1),
            1,
            0
        )
        ## === ASSIGN UNIT_BIN === ##
        pbp['unit_bin'] = None
        pbp['unit_bin'] = numpy.where(pbp['is_qb_play'] == 1, 'pass', pbp['unit_bin'])
        pbp['unit_bin'] = numpy.where(pbp['is_rush_play'] == 1, 'rush', pbp['unit_bin'])
        pbp['unit_bin'] = numpy.where(pbp['is_st_play'] == 1, 'st', pbp['unit_bin'])
        ## filter to only plays with a unit_bin assignment ##
        pbp = pbp[pbp['unit_bin'].notna()].copy()
        return pbp
    
    def aggregate_games(self, parsed_pbp: pd.DataFrame) -> pd.DataFrame:
        '''
        Aggregate play-by-play data to game level by unit
        
        Parameters:
        * parsed_pbp: PBP with unit_bin assignments
        
        Returns:
        * DataFrame with game_id, season, week, game_date, posteam, defteam, unit_bin, epa
        '''
        ## group by game, teams, and unit_bin to sum EPA ##
        agg = parsed_pbp.groupby([
            'game_id', 'season', 'week',
            'home_team', 'away_team',
            'posteam', 'defteam', 'unit_bin'
        ]).agg(
            epa=('epa', 'sum')
        ).reset_index()
        ## pivot to get EPA by unit for each home team ##
        home_units = agg[agg['posteam'] == agg['home_team']].pivot_table(
            index=['game_id', 'season', 'week', 'home_team'],
            columns='unit_bin',
            values='epa',
            aggfunc='sum'
        ).reset_index()
        ## pivot to get EPA by unit for each team ##
        away_units = agg[agg['posteam'] == agg['away_team']].pivot_table(
            index=['game_id', 'season', 'week', 'away_team'],
            columns='unit_bin',
            values='epa',
            aggfunc='sum'
        ).reset_index()
        ## remove column names ##
        home_units.columns.name = None
        away_units.columns.name = None
        ## merge home and away units ##
        games = pd.merge(
            home_units.rename(columns={
                'pass': 'home_pass_epa',
                'rush': 'home_rush_epa',
                'st': 'home_st_epa'
            }),
            away_units.rename(columns={
                'pass': 'away_pass_epa',
                'rush': 'away_rush_epa',
                'st': 'away_st_epa'
            }),
            on=['game_id', 'season', 'week'],
            how='outer'
        )
        return games

    def add_meta(self, game_level: pd.DataFrame) -> pd.DataFrame:
        '''
        Add game metadata (temp, wind, coaches, margin)
        
        Parameters:
        * games: Game-level EPA data
        
        Returns:
        * Game-level data with metadata added
        '''
        games = game_level.copy()
        meta = self.games[[
            'game_id', 'temp', 'wind',
            'home_coach', 'away_coach',
            'result', 'total', 'spread_line', 'total_line'
        ]].copy()
        result = pd.merge(
            games,
            meta,
            on=['game_id'],
            how='left'
        )
        return result
    
    def add_adjustments(self, games_with_meta: pd.DataFrame) -> pd.DataFrame:
        '''
        Add HFA, QB names, and QB pre-game values
        
        Parameters:
        * games_with_meta: Game-level data with metadata
        
        Returns:
        * Game-level data with adjustments added
        '''
        games = games_with_meta.copy()
        ## add HFA ##
        games = pd.merge(
            games,
            self.hfa[['game_id', 'hfa_base']],
            on=['game_id'],
            how='left'
        )
        ## add QB names and pre-game values ##
        games = pd.merge(
            games,
            self.qbelo[['game_id', 'qb1', 'qb2', 'qb1_value_pre', 'qb2_value_pre']].rename(columns={
                'qb1': 'home_qb_name',
                'qb2': 'away_qb_name',
                'qb1_value_pre': 'home_qb_value',
                'qb2_value_pre': 'away_qb_value'
            }),
            on=['game_id'],
            how='left'
        )
        return games
