'''
Week Index Utilities

Converts (season, week) pairs to sequential week indices.
'''

import pandas as pd
from typing import Dict, Tuple


def create_week_index(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Add sequential week_idx column based on (season, week).
    
    Sorts unique (season, week) pairs chronologically and assigns
    sequential indices starting from 0.
    
    Parameters:
    * df: DataFrame with 'season' and 'week' columns
    
    Returns:
    * DataFrame with added 'week_idx' column
    '''
    ## get unique (season, week) pairs sorted ##
    week_keys = df[['season', 'week']].drop_duplicates().sort_values(['season', 'week'])
    week_keys = week_keys.reset_index(drop=True)
    week_keys['week_idx'] = week_keys.index
    ## merge back to original df ##
    return df.merge(week_keys, on=['season', 'week'], how='left')


def calculate_window(week_idx: int, total_weeks: int, w: int = 20) -> Tuple[int, int]:
    '''
    Calculate window range for a given week index.
    
    Returns (start, end) inclusive. Window size is 2w + 1.
    - If total_weeks <= 2w + 1: all weeks included
    - If week_idx <= w: first 2w + 1 weeks
    - If week_idx >= total_weeks - 1 - w: last 2w + 1 weeks
    - Otherwise: centered window [week_idx - w, week_idx + w]
    
    Parameters:
    * week_idx: The week index (0-based)
    * total_weeks: Total number of weeks in the dataset
    * w: Half-window size (default 20, giving 41-week window)
    
    Returns:
    * Tuple of (start_idx, end_idx) inclusive
    '''
    window_size = 2 * w + 1
    ## if fewer weeks than window size, return all ##
    if total_weeks <= window_size:
        return (0, total_weeks - 1)
    ## edge case: near start ##
    if week_idx <= w:
        return (0, window_size - 1)
    ## edge case: near end ##
    if week_idx >= total_weeks - 1 - w:
        return (total_weeks - window_size, total_weeks - 1)
    ## normal case: centered ##
    return (week_idx - w, week_idx + w)


def calculate_all_windows(total_weeks: int, w: int = 20) -> Dict[int, Tuple[int, int]]:
    '''
    Pre-compute windows for all week indices.
    
    Parameters:
    * total_weeks: Total number of weeks in the dataset
    * w: Half-window size (default 20)
    
    Returns:
    * Dict mapping week_idx to (start, end) tuple
    '''
    return {
        idx: calculate_window(idx, total_weeks, w)
        for idx in range(total_weeks)
    }


def forward_fill_weeks(df: pd.DataFrame, team_col: str = 'team') -> pd.DataFrame:
    '''
    Create complete (season, week, team) grid and forward fill missing values.
    
    Ensures every team has a record for every (season, week) in the data,
    with missing values forward-filled from the most recent game.
    Useful for byes and playoff weeks where teams don't play.
    
    Parameters:
    * df: DataFrame with 'season', 'week', and team columns
    * team_col: Name of team column (default 'team')
    
    Returns:
    * DataFrame with complete (season, week, team) coverage, forward-filled
    '''
    ## get all unique (season, week) pairs ##
    all_weeks = df[['season', 'week']].drop_duplicates().sort_values(['season', 'week'])
    ## get all unique teams ##
    all_teams = df[[team_col]].drop_duplicates()
    ## create complete grid ##
    grid = all_weeks.merge(all_teams, how='cross')
    ## merge original data ##
    result = grid.merge(df, on=['season', 'week', team_col], how='left')
    ## sort by team, then time ##
    result = result.sort_values([team_col, 'season', 'week'])
    ## forward fill within each team ##
    fill_cols = [c for c in result.columns if c not in ['season', 'week', team_col]]
    result[fill_cols] = result.groupby(team_col)[fill_cols].ffill()
    return result.reset_index(drop=True)

