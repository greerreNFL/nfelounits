"""
Measure Priors - Prior Correlation Analysis

Calculates R² correlations between pre-season power ratings and end-of-season
power ratings to evaluate predictive value of different prior measures.

Output:
- Correlation matrix where rows are pre-season metrics and columns are end-of-season metrics
- Intermediate dataframe with (season, team) as composite key and all metrics as columns
"""
import pandas as pd
import numpy
import nfelodcm as dcm
from pathlib import Path


def main():
    print("=" * 80)
    print("PRIOR CORRELATION ANALYSIS")
    print("=" * 80)
    ## define paths ##
    output_dir = Path(__file__).parent.resolve()
    elo_path = output_dir.parent.parent / 'Output' / 'elo.csv'
    dvoa_path = output_dir / 'manual_data' / 'dvoa.csv'
    ## load external data ##
    print("\n1. Loading data...")
    db = dcm.load([
        'games', 'wt_ratings', 'srs_ratings', 'qbelo'
    ])
    games = db['games'].copy()
    wt_ratings = db['wt_ratings'].copy()
    srs_ratings = db['srs_ratings'].copy()
    qbelo = db['qbelo'].copy()
    units_elo = pd.read_csv(elo_path)
    dvoa = pd.read_csv(dvoa_path)
    print(f"   ✓ Loaded all datasets")
    ## filter to 2006-2024 ##
    seasons = range(2006, 2025)
    ## build base dataframe with all team-seasons ##
    print("\n2. Building base team-season dataframe...")
    team_seasons = wt_ratings[
        wt_ratings['season'].isin(seasons)
    ][['season', 'team']].drop_duplicates()
    df = team_seasons.copy()
    print(f"   ✓ {len(df)} team-season records")
    ## add wt_ratings pre-season metrics ##
    print("\n3. Adding wt_ratings metrics (pre-season only)...")
    wt_cols = wt_ratings[
        wt_ratings['season'].isin(seasons)
    ][[
        'season', 'team', 'wt_rating',
        'line', 'line_adj'
    ]].rename(columns={
        'wt_rating': 'wt_rating_pre',
        'line': 'win_total_line_pre',
        'line_adj': 'win_total_line_adj_pre'
    }).copy()
    df = pd.merge(
        df,
        wt_cols,
        on=['season', 'team'],
        how='left'
    ).copy()
    print(f"   ✓ Added wt_rating_pre, win_total_line_pre, win_total_line_adj_pre")
    ## add avg_mov from games (end-of-season) ##
    print("\n4. Adding avg_mov metrics from games...")
    ## filter to regular season ##
    games_reg = games[
        (games['season'].isin(seasons)) &
        (games['game_type'] == 'REG')
    ].copy()
    ## flatten to team perspective with MOV ##
    home_games = games_reg[[
        'season', 'home_team', 'result'
    ]].rename(columns={
        'home_team': 'team',
        'result': 'mov'
    })
    away_games = games_reg[[
        'season', 'away_team', 'result'
    ]].copy()
    away_games['result'] = -away_games['result']
    away_games = away_games.rename(columns={
        'away_team': 'team',
        'result': 'mov'
    })
    all_games = pd.concat([home_games, away_games])
    ## aggregate to avg_mov per team-season and add shifted prior ##
    avg_mov = all_games.groupby([
        'season', 'team'
    ])['mov'].mean().reset_index().rename(columns={
        'mov': 'avg_mov_end'
    }).sort_values([
        'team', 'season'
    ])
    avg_mov['avg_mov_last_season_pre'] = avg_mov.groupby('team')['avg_mov_end'].shift(1)
    df = pd.merge(
        df,
        avg_mov,
        on=['season', 'team'],
        how='left'
    ).copy()
    print(f"   ✓ Added avg_mov_end, avg_mov_last_season_pre")
    ## add SRS end-of-season metrics ##
    print("\n5. Adding SRS metrics (end-of-season only)...")
    srs_filtered = srs_ratings[
        srs_ratings['season'].isin(seasons)
    ].copy().sort_values([
        'season', 'team', 'week'
    ]).groupby([
        'season', 'team'
    ]).tail(1)
    df = pd.merge(
        df,
        srs_filtered[[
            'season', 'team', 'srs_rating_normalized'
        ]].rename(columns={
            'srs_rating_normalized': 'srs_end'
        }),
        on=['season', 'team'],
        how='left'
    ).copy()
    print(f"   ✓ Added srs_end")
    ## add units_elo pre and end ##
    print("\n6. Adding units_elo metrics (pre and end)...")
    units_filtered = units_elo[
        units_elo['season'].isin(seasons)
    ].copy().sort_values([
        'season', 'team', 'week'
    ])
    ## pre = week 1 ##
    units_pre = units_filtered.groupby([
        'season', 'team'
    ]).head(1)[[
        'season', 'team', 'elo'
    ]].rename(columns={
        'elo': 'units_elo_pre'
    })
    ## end = last week ##
    units_end = units_filtered.groupby([
        'season', 'team'
    ]).tail(1)[[
        'season', 'team', 'elo'
    ]].rename(columns={
        'elo': 'units_elo_end'
    })
    df = pd.merge(
        df,
        units_pre,
        on=['season', 'team'],
        how='left'
    ).copy()
    df = pd.merge(
        df,
        units_end,
        on=['season', 'team'],
        how='left'
    ).copy()
    print(f"   ✓ Added units_elo_pre, units_elo_end")
    ## add f38_elo pre and end from qbelo ##
    print("\n7. Adding f38_elo metrics (pre and end)...")
    ## filter to regular season ##
    qbelo_reg = qbelo[
        (qbelo['season'].isin(seasons)) &
        (qbelo['playoff'].isna() | (qbelo['playoff'] == ''))
    ].copy()
    ## flatten to team level first ##
    qbelo_t1 = qbelo_reg[[
        'season', 'week', 'team1', 'elo1_pre', 'elo1_post'
    ]].rename(columns={
        'team1': 'team',
        'elo1_pre': 'elo_pre',
        'elo1_post': 'elo_post'
    })
    qbelo_t2 = qbelo_reg[[
        'season', 'week', 'team2', 'elo2_pre', 'elo2_post'
    ]].rename(columns={
        'team2': 'team',
        'elo2_pre': 'elo_pre',
        'elo2_post': 'elo_post'
    })
    qbelo_flat = pd.concat([qbelo_t1, qbelo_t2]).sort_values([
        'season', 'team', 'week'
    ])
    ## pre = first week, end = last week ##
    f38_pre = qbelo_flat.groupby([
        'season', 'team'
    ]).head(1)[[
        'season', 'team', 'elo_pre'
    ]].rename(columns={
        'elo_pre': 'f38_elo_pre'
    })
    f38_end = qbelo_flat.groupby([
        'season', 'team'
    ]).tail(1)[[
        'season', 'team', 'elo_post'
    ]].rename(columns={
        'elo_post': 'f38_elo_end'
    })
    df = pd.merge(
        df,
        f38_pre,
        on=['season', 'team'],
        how='left'
    ).copy()
    df = pd.merge(
        df,
        f38_end,
        on=['season', 'team'],
        how='left'
    ).copy()
    print(f"   ✓ Added f38_elo_pre, f38_elo_end")
    ## add DVOA pre and end ##
    print("\n8. Adding DVOA metrics (pre and end)...")
    dvoa_cols = dvoa[
        dvoa['season'].isin(seasons)
    ][[
        'season', 'team', 'projected_total_dvoa', 'ending_total_dvoa'
    ]].rename(columns={
        'projected_total_dvoa': 'dvoa_pre',
        'ending_total_dvoa': 'dvoa_end'
    }).copy()
    df = pd.merge(
        df,
        dvoa_cols,
        on=['season', 'team'],
        how='left'
    ).copy()
    print(f"   ✓ Added dvoa_pre, dvoa_end")
    ## calculate correlation matrix ##
    print("\n9. Calculating R² correlation matrix...")
    pre_cols = [
        'wt_rating_pre',
        'win_total_line_pre',
        'win_total_line_adj_pre',
        'avg_mov_last_season_pre',
        'units_elo_pre',
        'f38_elo_pre',
        'dvoa_pre'
    ]
    end_cols = [
        'avg_mov_end',
        'srs_end',
        'units_elo_end',
        'f38_elo_end',
        'dvoa_end'
    ]
    ## calculate r² for each pre/end combination ##
    r2_matrix = pd.DataFrame(index=pre_cols, columns=end_cols, dtype=float)
    for pre_col in pre_cols:
        for end_col in end_cols:
            ## drop rows with NaN in either column ##
            valid = df[[pre_col, end_col]].dropna()
            if len(valid) > 0:
                corr = valid[pre_col].corr(valid[end_col])
                r2_matrix.loc[pre_col, end_col] = round(corr ** 2, 4)
            else:
                r2_matrix.loc[pre_col, end_col] = numpy.nan
    ## save correlation matrix ##
    r2_matrix.to_csv(output_dir / 'pre_end_rsq_matrix.csv')
    print(f"   ✓ Saved pre_end_rsq_matrix.csv")
    ## calculate pre-seaosn correlation matrix ##
    pre_r2_matrix = pd.DataFrame(index=pre_cols, columns=pre_cols, dtype=float)
    for pre_col in pre_cols:
        for pre_col2 in pre_cols:
            if pre_col != pre_col2:
                corr = df[[pre_col, pre_col2]].dropna()[pre_col].corr(df[[pre_col, pre_col2]].dropna()[pre_col2])
                pre_r2_matrix.loc[pre_col, pre_col2] = round(corr ** 2, 4)
    ## save pre-season correlation matrix ##
    pre_r2_matrix.to_csv(output_dir / 'pre_pre_rsq_matrix.csv')
    print(f"   ✓ Saved pre_pre_rsq_matrix.csv")
    ## calculate end-of-season correlation matrix ##
    end_r2_matrix = pd.DataFrame(index=end_cols, columns=end_cols, dtype=float)
    for end_col in end_cols:
        for end_col2 in end_cols:
            if end_col != end_col2:
                corr = df[[end_col, end_col2]].dropna()[end_col].corr(df[[end_col, end_col2]].dropna()[end_col2])
                end_r2_matrix.loc[end_col, end_col2] = round(corr ** 2, 4)
    ## save end-of-season correlation matrix ##
    end_r2_matrix.to_csv(output_dir / 'end_end_rsq_matrix.csv')
    print(f"   ✓ Saved end_end_rsq_matrix.csv")

if __name__ == '__main__':
    main()
