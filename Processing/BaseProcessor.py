'''
BaseProcessor Class

Abstract base class for processing team game records into output files.
'''

from typing import List
from abc import ABC, abstractmethod
import os
import pathlib
import pandas as pd

from ..Model.Entities import TeamGameRecord


class BaseProcessor(ABC):
    '''
    Base class for processing TeamGameRecords (ie model output) into output files.

    Requires user defined process() method for transformations on the df composed
    from the TeamGameRecords
    
    '''
    
    ## Shared config ##
    ROUNDING_DECIMALS: int = 4
    DEFAULT_OUTPUT_PATH: str = f'{pathlib.Path(__file__).parent.parent.resolve()}/Output'
    PROCESSOR_NAME: str = None

    def __init__(self,
        team_game_records: List[TeamGameRecord]
    ):
        '''
        Initialize processor. Will translate passed recs into a processed
        frame on init.

        Successful df processing handled by _post_processor. If process does not return
        a df, error will be thrown. If no process name is passed, processor will fail on init.
        Since successful initialization ensures a save-able DF, user calls save() post init

        ie
        processor = SomeImplimentedProcessor(some_team_reords)
        processor.save(some_filepath)
        
        Parameters:
        * team_game_records: List of TeamGameRecord dicts
        '''
        self.df = pd.DataFrame(team_game_records)
        self.processed_df = self._post_process(self.process(self.df))
        ## ensure processer name passed ##
        if self.PROCESSOR_NAME is None:
            raise ValueError('Processor must have a name')
    
    @abstractmethod
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Process the DataFrame. Override in subclasses.
        
        Parameters:
        * df: Input DataFrame of team game records
        
        Returns:
        * Processed DataFrame
        '''
        pass

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Private class that runs standardized pos processing
        over the processed df, ie rounding, formatting, etc

        Round numeric columns to 4 decimals and save to CSV.
        '''
        numeric_cols = df.select_dtypes(include=['float64', 'float32']).columns
        for col in numeric_cols:
            df[col] = df[col].round(self.ROUNDING_DECIMALS)
        ## return ##
        return df
    
    def save(self, output_path: str = None) -> None:
        '''
        Saves the data frame to the default location unless one is provided

        Parameters:
        * output_path: Optional filepath (str) override

        Returns:
        * None
        '''
        ## use default if not provided ##
        if output_path is None:
            output_path = self.DEFAULT_OUTPUT_PATH
        ## if custom output_path passed, check it is valid ##
        elif not os.path.isdir(output_path):
            raise ValueError('Output path is not a valid directory')
        ## initializing checks, processing, and fp check gaurantee this df is saveable
        self.processed_df.to_csv(
            f'{output_path}/{self.PROCESSOR_NAME}.csv',
            index=False
        )




    

