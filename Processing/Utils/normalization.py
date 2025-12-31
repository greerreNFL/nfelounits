'''
Normalization Utilities

Era-adjusts values to a reference point using rolling windows.
'''

from typing import List

import pandas as pd

from .week_index import (
    create_week_index,
    calculate_all_windows,
    calculate_window,
)


def normalize(
    df: pd.DataFrame,
    columns: List[str],
    reference_season: int = 2015,
    reference_week: int = 1,
    w: int = 30,
    group_by_week: bool = False
) -> pd.DataFrame:
    '''
    Era-adjust columns to a reference point.
    
    For each column, calculates z-score relative to its era (rolling window),
    then applies that z-score to the reference era distribution. This puts
    all values on a common scale for cross-era comparison.
    
    Parameters:
    * df: DataFrame with 'season', 'week', and stat columns
    * columns: List of column names to normalize
    * reference_season: Season for reference distribution (default 2015)
    * reference_week: Week for reference distribution (default 1)
    * w: Half-window size (default 30, giving 61-week window)
    * group_by_week: If True, calculate mean/std within each week number
      (useful for metrics like value_created where sample size varies by week)
    
    Returns:
    * DataFrame with added '{col}_norm' columns
    '''
    result = df.copy()
    ## add week index ##
    result = create_week_index(result)
    ## get window mappings ##
    total_weeks = result['week_idx'].max() + 1
    windows = calculate_all_windows(total_weeks, w)
    ## find reference week_idx ##
    ref_match = result[(result['season'] == reference_season) & (result['week'] == reference_week)]
    if ref_match.empty:
        raise ValueError(f'Reference point ({reference_season}, {reference_week}) not found in data')
    ref_week_idx = ref_match['week_idx'].iloc[0]
    ref_start, ref_end = calculate_window(ref_week_idx, total_weeks, w)
    ## calculate reference distribution stats ##
    ref_window = result[(result['week_idx'] >= ref_start) & (result['week_idx'] <= ref_end)]
    ref_stats = {col: {'mean': ref_window[col].mean(), 'std': ref_window[col].std()} for col in columns}
    ## build era distribution stats per week_idx ##
    distribution_records = []
    for week_idx, (start, end) in windows.items():
        rec = {'week_idx': week_idx}
        window = result[(result['week_idx'] >= start) & (result['week_idx'] <= end)]
        if group_by_week:
            ## filter to same week number within window ##
            current_week = result[result['week_idx'] == week_idx]['week'].iloc[0]
            window = window[window['week'] == current_week]
        for col in columns:
            rec[f'{col}_mean'] = window[col].mean()
            rec[f'{col}_std'] = window[col].std()
        distribution_records.append(rec)
    ## merge era distribution stats to result ##
    dist_df = pd.DataFrame(distribution_records)
    result = result.merge(dist_df, on='week_idx', how='left')
    ## calculate normalized values per column ##
    for col in columns:
        ## z-score relative to era ##
        z_scores = (result[col] - result[f'{col}_mean']) / result[f'{col}_std']
        ## apply to reference distribution ##
        result[f'{col}_norm'] = ref_stats[col]['mean'] + z_scores * ref_stats[col]['std']
        ## handle edge case where era std is 0 ##
        result.loc[result[f'{col}_std'] == 0, f'{col}_norm'] = ref_stats[col]['mean']
        ## cleanup intermediate columns ##
        result = result.drop(columns=[f'{col}_mean', f'{col}_std'])
    ## cleanup week_idx ##
    result = result.drop(columns=['week_idx'])
    return result

