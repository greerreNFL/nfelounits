'''
UnitTeamsProcessor Class

Processes team game records into the standard unit_teams.csv output format.
'''

import pandas as pd
from .BaseProcessor import BaseProcessor

from .Utils import forward_fill_weeks


class UnitTeamsProcessor(BaseProcessor):
    '''Processor for standard unit teams output.'''
    PROCESSOR_NAME = 'units'
    FILTER_COLS = [
        'season', 'week', 'team',
        # Pre-game values
        'pass_off_value_pre', 'rush_off_value_pre', 'st_off_value_pre',
        'pass_def_value_pre', 'rush_def_value_pre', 'st_def_value_pre',
        # Post-game values
        'pass_off_value_post', 'rush_off_value_post', 'st_off_value_post',
        'pass_def_value_post', 'rush_def_value_post', 'st_def_value_post'
    ]
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Process team game records into unit teams format.
        
        Selects and orders columns for standard output.
        '''
        ## reduce df to needed cols ##
        df = df[self.FILTER_COLS].copy()
        ## add total, offense, and defense cols ##
        df['off_value_pre'] = (
            df['pass_off_value_pre'] +
            df['rush_off_value_pre'] +
            df['st_off_value_pre']
        )
        df['def_value_pre'] = (
            df['pass_def_value_pre'] +
            df['rush_def_value_pre'] +
            df['st_def_value_pre']
        )
        df['total_value_pre'] = (
            df['off_value_pre'] +
            df['def_value_pre']
        )
        df['off_value_post'] = (
            df['pass_off_value_post'] +
            df['rush_off_value_post'] +
            df['st_off_value_post']
        )
        df['def_value_post'] = (
            df['pass_def_value_post'] +
            df['rush_def_value_post'] +
            df['st_def_value_post']
        )
        df['total_value_post'] = (
            df['off_value_post'] +
            df['def_value_post']
        )
        ## util function to ensure all teams for all weeks
        df = forward_fill_weeks(df)
        ## return ##
        return df
