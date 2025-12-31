'''
ValueCreatedProcessor Class

Processes value_created into season-to-date averages with normalization.
'''

import pandas as pd
from .BaseProcessor import BaseProcessor

from .Utils import forward_fill_weeks, normalize


class ValueCreatedProcessor(BaseProcessor):
    '''
    Processor that calculates season-to-date average value_created
    for each unit, then normalizes for cross-era comparison.
    '''

    PROCESSOR_NAME = 'value_created'
    VALUE_CREATED_COLS = [
        'pass_off_value_created', 'pass_def_value_created',
        'rush_off_value_created', 'rush_def_value_created',
        'st_off_value_created', 'st_def_value_created'
    ]
    FILTER_COLS = [
        'season', 'week', 'team',
    ] + VALUE_CREATED_COLS
    
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Process team game records into value_created format.
        
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
        for col in self.VALUE_CREATED_COLS:
            df[col] = (
                df.groupby(['season', 'team'])[col]
                .expanding()
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
        ## forward fill weeks for bye coverage ##
        df = forward_fill_weeks(df)
        ## normalize values (group_by_week=True since sample size varies by week) ##
        df = normalize(df, self.VALUE_CREATED_COLS, group_by_week=True)
        ## build output columns with percentiles ##
        output_cols = ['season', 'week', 'team']
        norm_col_repl = {}
        for col in self.VALUE_CREATED_COLS:
            ## add raw expanding avg ##
            output_cols.append(col)
            ## rename normalized col ##
            norm_col_repl[f'{col}_norm'] = col.replace('_value_created', '_norm_value_created')
            output_cols.append(norm_col_repl[f'{col}_norm'])
            ## add ptile ##
            ptile_col = col.replace('_value_created', '_ptile_created')
            df[ptile_col] = df[f'{col}_norm'].rank(pct=True)
            output_cols.append(ptile_col)
        ## prep output ##
        df = df.rename(columns=norm_col_repl)
        df = df[output_cols].copy()
        ## return ##
        return df

