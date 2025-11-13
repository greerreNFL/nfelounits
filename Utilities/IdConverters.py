'''
ID Converters

Functions for converting between new and legacy GSIS ID formats.
'''

import codecs
import pandas as pd
from typing import List


def convert_gsis_ids(df: pd.DataFrame, id_fields: List[str]) -> pd.DataFrame:
    '''
    Convert new style IDs into legacy gsis_id format for a list

    Parameters:
    * df: the dataframe to convert
    * id_fields: the list of id fields to convert

    Returns:
    * df: the converted dataframe
    '''
    ## helper for processing multiple rows ##
    def convert_row(row: pd.Series, checked_cols: List[str]) -> pd.Series:
        for col in checked_cols:
            try:
                if pd.isna(row[col]) or row[col] == '':
                    row[col] = None
                elif len(str(row[col])) > 10:
                    row[col] = codecs.decode(row[col][4:-8].replace('-',''),'hex').decode('utf-8')
                else:
                    row[col] = row[col]
            except:
                pass
        return row
    ## check that cols are in the df ##
    checked_fields = [f for f in id_fields if f in df.columns]
    if len(checked_fields) > 0:
        df = df.apply(convert_row, axis=1, checked_cols=checked_fields)
    return df

