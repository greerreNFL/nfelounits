"""
DataSplitter Class

Label data for train/test splits while maintaining full dataset for EWMA continuity.
"""

import pandas as pd


class DataSplitter:
    """Label data for chronological train/test splits"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize splitter
        
        Parameters:
        * df: DataFrame with 'season' column
        """
        self.df = df
    
    def label_train_test(self, n_test_seasons: int = 5, exclude_first_season: bool = True) -> pd.DataFrame:
        """
        Label data with 'data_set' column for train/test/exclude
        
        This maintains the full dataset for EWMA continuity while marking which
        records should be used for training objectives vs testing objectives.
        
        Parameters:
        * n_test_seasons: Number of seasons to hold out for testing (default 5)
        * exclude_first_season: Whether to exclude first season from scoring (default True)
        
        Returns:
        * DataFrame with 'data_set' column added ('exclude', 'train', or 'test')
        """
        df = self.df.copy()
        seasons = sorted(df['season'].unique())
        
        if len(seasons) <= n_test_seasons:
            raise ValueError(f'Not enough seasons ({len(seasons)}) to hold out {n_test_seasons} for testing')
        
        ## determine season ranges ##
        first_season = seasons[0]
        test_seasons = seasons[-n_test_seasons:]
        train_seasons = [s for s in seasons if s not in test_seasons]
        
        ## label data ##
        if exclude_first_season:
            ## first season is for EWMA warm-up only ##
            df.loc[df['season'] == first_season, 'data_set'] = 'exclude'
            df.loc[df['season'].isin(train_seasons) & (df['season'] != first_season), 'data_set'] = 'train'
        else:
            ## include first season in training ##
            df.loc[df['season'].isin(train_seasons), 'data_set'] = 'train'
        
        df.loc[df['season'].isin(test_seasons), 'data_set'] = 'test'
        
        return df
    
    def label_by_season(self, train_through_season: int, exclude_first_season: bool = True) -> pd.DataFrame:
        """
        Label data by specific season cutoff
        
        Parameters:
        * train_through_season: Last season to include in training set
        * exclude_first_season: Whether to exclude first season from scoring (default True)
        
        Returns:
        * DataFrame with 'data_set' column added ('exclude', 'train', or 'test')
        """
        df = self.df.copy()
        seasons = sorted(df['season'].unique())
        first_season = seasons[0]
        
        ## label data ##
        if exclude_first_season:
            df.loc[df['season'] == first_season, 'data_set'] = 'exclude'
            df.loc[(df['season'] > first_season) & (df['season'] <= train_through_season), 'data_set'] = 'train'
        else:
            df.loc[df['season'] <= train_through_season, 'data_set'] = 'train'
        
        df.loc[df['season'] > train_through_season, 'data_set'] = 'test'
        
        return df

