'''
UnitTeamsNormalizationProcessor Class

Normalizes team game records for cross-era comparison.
'''

import pandas as pd
from .BaseProcessor import BaseProcessor

from .Utils import forward_fill_weeks, normalize


class UnitTeamsNormalizationProcessor(BaseProcessor):
    '''
    Processor that normalizes unit values to a reference point
    for fairer comparison across eras.

    Reference point is week 1 of season 2015.
    '''

    PROCESSOR_NAME = 'units_normalized'
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
        ## util function to ensure all teams for all weeks
        df = forward_fill_weeks(df)
        ## add normalized values ##
        norm_cols = []
        for timing in ['pre', 'post']:
            for side in ['off', 'def']:
                for unit in ['pass', 'rush', 'st']:
                    norm_cols.append(f'{unit}_{side}_value_{timing}')
        df = normalize(df, norm_cols)
        ## loop to format and add ptiles ##
        output_cols = [
            'season', 'week', 'team'
        ]
        norm_col_repl = {}
        for col in norm_cols:
            ## repl the normalized cols ##
            norm_col_repl[
                f'{col}_norm'
            ] = col.replace('_norm','').replace('_value_', '_norm_value_')
            output_cols.append(norm_col_repl[f'{col}_norm'])
            ## add ptile ##
            ptile_col = col.replace('_value_', '_ptile_')
            df[ptile_col] = df[f'{col}_norm'].rank(pct=True)
            output_cols.append(ptile_col)
        ## prep output ##
        df = df.rename(columns=norm_col_repl)
        df = df[output_cols].copy()
        ## return ##
        return df
