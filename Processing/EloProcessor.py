'''
EloProcessor Class

Processes team game records into the standard elo.csv output format.
'''

import pandas as pd
from .BaseProcessor import BaseProcessor

from .Utils import forward_fill_weeks


class EloProcessor(BaseProcessor):
    '''Processor for team elo and qb adjustment output.'''
    PROCESSOR_NAME = 'elo'
    FILTER_COLS = [
        'season', 'week', 'team', 'game_id',
        'elo', 'qb_adj'
    ]
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Process team game records into elo output format.
        
        Outputs pre-game elo and qb_adj values for each team/week.
        '''
        ## reduce df to needed cols ##
        df = df[self.FILTER_COLS].copy()
        ## util function to ensure all teams for all weeks ##
        df = forward_fill_weeks(df)
        ## sort ##
        df = df.sort_values(
            by=['season', 'week', 'elo','team'],
            ascending=[True, True, False, True]
        ).reset_index(drop=True)
        ## return ##
        return df


