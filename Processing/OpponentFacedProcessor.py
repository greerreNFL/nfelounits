'''
OpponentFacedProcessor Class

Processes opponent quality faced into season-to-date averages with normalization.
'''

import pandas as pd
from .BaseProcessor import BaseProcessor

from .Utils import forward_fill_weeks, normalize


class OpponentFacedProcessor(BaseProcessor):
    '''
    Processor that calculates season-to-date average opponent quality faced
    for each unit, then normalizes for cross-era comparison.
    
    Positive = harder schedule.
    '''

    PROCESSOR_NAME = 'faced'
    FACED_COLS = [
        'pass_off_faced', 'pass_def_faced',
        'rush_off_faced', 'rush_def_faced',
        'st_off_faced', 'st_def_faced'
    ]
    FILTER_COLS = [
        'season', 'week', 'team',
    ] + FACED_COLS
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Process team game records into opponent_faced format.
        
        Steps:
        1. Filter to needed columns
        2. Calculate expanding mean (season-to-date avg) per season+team
        3. Forward fill weeks for bye coverage
        4. Normalize values
        5. Add percentiles
        '''
        ## reduce df to needed cols ##
        df = df[self.FILTER_COLS].copy()
        ## sort for expanding calculation ##
        df = df.sort_values(['season', 'team', 'week']).reset_index(drop=True)
        ## calculate expanding mean within season+team ##
        for col in self.FACED_COLS:
            df[col] = (
                df.groupby(['season', 'team'])[col]
                .expanding()
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
        ## forward fill weeks for bye coverage ##
        df = forward_fill_weeks(df)
        ## normalize values (group_by_week=True since sample size varies by week) ##
        df = normalize(df, self.FACED_COLS, group_by_week=True)
        ## build output columns with percentiles ##
        output_cols = ['season', 'week', 'team']
        norm_col_repl = {}
        for col in self.FACED_COLS:
            ## add raw expanding avg ##
            output_cols.append(col)
            ## rename normalized col ##
            norm_col_repl[f'{col}_norm'] = col.replace('_faced', '_norm_faced')
            output_cols.append(norm_col_repl[f'{col}_norm'])
            ## add ptile ##
            ptile_col = col.replace('_faced', '_ptile_faced')
            df[ptile_col] = df[f'{col}_norm'].rank(pct=True)
            output_cols.append(ptile_col)
        ## prep output ##
        df = df.rename(columns=norm_col_repl)
        df = df[output_cols].copy()
        ## return ##
        return df
