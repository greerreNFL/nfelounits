"""
DataSplitter Class

Season-based train/test splits for model validation.
Identical to dynamic_elo pattern.
"""

import pandas as pd
from typing import Tuple


class DataSplitter:
    """Create chronological train/test splits"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize splitter
        
        Parameters:
        * df: DataFrame with 'season' column
        """
        self.df = df
    
    def split_by_season(self, train_through_season: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split chronologically by season
        
        Parameters:
        * train_through_season: Last season to include in training set
        
        Returns:
        * Tuple of (train_df, test_df)
        """
        train_df = self.df[self.df['season'] <= train_through_season].copy()
        test_df = self.df[self.df['season'] > train_through_season].copy()
        
        return train_df, test_df
    
    def split_by_pct(self, train_pct: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split by percentage of seasons chronologically
        
        Parameters:
        * train_pct: Percentage of seasons to include in training (default 0.8)
        
        Returns:
        * Tuple of (train_df, test_df)
        """
        seasons = sorted(self.df['season'].unique())
        n_train_seasons = int(len(seasons) * train_pct)
        
        if n_train_seasons == 0:
            n_train_seasons = 1
        if n_train_seasons >= len(seasons):
            n_train_seasons = len(seasons) - 1
        
        train_seasons = seasons[:n_train_seasons]
        
        train_df = self.df[self.df['season'].isin(train_seasons)].copy()
        test_df = self.df[~self.df['season'].isin(train_seasons)].copy()
        
        return train_df, test_df
    
    def get_train_test_games(self, n_test_seasons: int = 5, exclude_first_season: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        Get train/test split with last N seasons held out for testing
        
        Parameters:
        * n_test_seasons: Number of seasons to hold out for testing (default 5)
        * exclude_first_season: Whether to exclude first season from objective (default True)
        
        Returns:
        * Tuple of (train_df, test_df, first_season_to_score)
        """
        seasons = sorted(self.df['season'].unique())
        
        if len(seasons) <= n_test_seasons:
            raise ValueError(f'Not enough seasons ({len(seasons)}) to hold out {n_test_seasons} for testing')
        
        ## determine split point ##
        test_seasons = seasons[-n_test_seasons:]
        train_seasons = seasons[:-n_test_seasons]
        
        ## create splits ##
        train_df = self.df[self.df['season'].isin(train_seasons)].copy()
        test_df = self.df[self.df['season'].isin(test_seasons)].copy()
        
        ## determine first season to include in objective ##
        first_season_to_score = seasons[1] if exclude_first_season else seasons[0]
        
        return train_df, test_df, first_season_to_score

