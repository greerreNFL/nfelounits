"""
Measure Prediction Accuracy

Calculates prediction accuracy metrics for game predictions using:
- Unit Elos (with QB adjustments and HFA)
- 538/nfelo QB Elos (with QB adjustments and HFA)
- Market implied probabilities (vig-free) as baseline

Metrics:
- Log loss (lower is better)
- Brier score (lower is better)
- 538 game points (higher is better)
- Accuracy (higher is better)

Output:
- By-season files for each metric
- Summary file with all metrics
"""
import pandas as pd
import numpy
import nfelodcm as dcm
from pathlib import Path

from nfelounits.Utilities import calculate_win_probability


def calculate_log_loss(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate log loss for binary predictions
    
    Parameters:
    * y_true: Actual outcomes (1 = home win, 0 = away win)
    * y_pred: Predicted home win probabilities
    
    Returns:
    * Log loss (lower is better)
    """
    ## clip predictions to avoid log(0) ##
    y_pred = y_pred.clip(0.001, 0.999)
    return -1 * (y_true * numpy.log(y_pred) + (1 - y_true) * numpy.log(1 - y_pred)).mean()


def calculate_brier(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate Brier score (mean squared error of probabilities)
    
    Parameters:
    * y_true: Actual outcomes (1 = home win, 0 = away win)
    * y_pred: Predicted home win probabilities
    
    Returns:
    * Brier score (lower is better)
    """
    return ((y_pred - y_true) ** 2).mean()


def calculate_f38_game_points(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate 538 prediction game points
    
    Formula: points = 50 + 25 * log2(p_winner)
    - Where p_winner is the probability assigned to the winning team
    - Maximum 50 points per game (when p_winner = 1.0)
    - 25 points for a coin flip (p_winner = 0.5)
    - 0 points when p_winner = 0.25
    
    Parameters:
    * y_true: Actual outcomes (1 = home win, 0 = away win)
    * y_pred: Predicted home win probabilities
    
    Returns:
    * Average points per game (higher is better)
    """
    ## clip predictions to avoid log(0) ##
    y_pred = y_pred.clip(0.001, 0.999)
    ## get probability assigned to winning team ##
    p_winner = numpy.where(y_true == 1, y_pred, 1 - y_pred)
    ## calculate points: 50 + 25 * log2(p_winner) ##
    points = 50 + 25 * numpy.log2(p_winner)
    return points.mean()


def calculate_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Calculate prediction accuracy (% games correctly predicted)
    
    Parameters:
    * y_true: Actual outcomes (1 = home win, 0 = away win)
    * y_pred: Predicted home win probabilities
    
    Returns:
    * Accuracy (0 to 1, higher is better)
    """
    return ((y_pred > 0.5) == (y_true == 1)).mean()


def implied_prob_from_moneyline(ml: float) -> float:
    """
    Convert American moneyline to implied probability
    
    Parameters:
    * ml: American moneyline (e.g., -150 or +130)
    
    Returns:
    * Implied probability (0 to 1)
    """
    if pd.isna(ml):
        return numpy.nan
    if ml < 0:
        return -ml / (-ml + 100)
    else:
        return 100 / (ml + 100)


def vig_free_probability(home_ml: float, away_ml: float) -> float:
    """
    Calculate vig-free home win probability from moneylines
    
    Parameters:
    * home_ml: Home team moneyline
    * away_ml: Away team moneyline
    
    Returns:
    * Vig-free home win probability
    """
    home_implied = implied_prob_from_moneyline(home_ml)
    away_implied = implied_prob_from_moneyline(away_ml)
    if pd.isna(home_implied) or pd.isna(away_implied):
        return numpy.nan
    ## remove vig by normalizing ##
    total = home_implied + away_implied
    return home_implied / total


def main():
    print("=" * 80)
    print("PREDICTION ACCURACY ANALYSIS")
    print("=" * 80)
    ## define paths ##
    output_dir = Path(__file__).parent.resolve()
    elo_path = output_dir.parent.parent / 'Output' / 'elo.csv'
    ## load data ##
    print("\n1. Loading data...")
    db = dcm.load([
        'games', 'qbelo', 'hfa'
    ])
    games = db['games'].copy()
    qbelo = db['qbelo'].copy()
    hfa = db['hfa'].copy()
    units_elo = pd.read_csv(elo_path)
    print(f"   ✓ Loaded all datasets")
    ## filter to completed regular season games ##
    print("\n2. Filtering to completed regular season games...")
    games = games[
        (games['game_type'] == 'REG') &
        (games['result'].notna())
    ].copy()
    print(f"   ✓ {len(games)} completed regular season games")
    ## create home win indicator ##
    games['home_win'] = (games['result'] > 0).astype(int)
    ## add HFA ##
    print("\n3. Adding HFA adjustments...")
    games = pd.merge(
        games,
        hfa[['game_id', 'hfa_adj']],
        on=['game_id'],
        how='left'
    ).copy()
    print(f"   ✓ Added HFA adjustments")
    ## === UNITS ELO === ##
    print("\n4. Processing Units Elo predictions...")
    ## deduplicate elo data (bye weeks forward-fill game_id) ##
    units_elo_dedup = units_elo.drop_duplicates(
        subset=['game_id', 'team'],
        keep='first'
    )
    print(f"   ✓ Deduplicated elo data: {len(units_elo)} -> {len(units_elo_dedup)} rows")
    ## get home team pre-game elo ##
    units_home = units_elo_dedup[[
        'game_id', 'team', 'elo', 'qb_adj'
    ]].rename(columns={
        'team': 'home_team',
        'elo': 'units_home_elo',
        'qb_adj': 'units_home_qb_adj'
    })
    ## get away team pre-game elo ##
    units_away = units_elo_dedup[[
        'game_id', 'team', 'elo', 'qb_adj'
    ]].rename(columns={
        'team': 'away_team',
        'elo': 'units_away_elo',
        'qb_adj': 'units_away_qb_adj'
    })
    ## join to games ##
    games = pd.merge(
        games,
        units_home,
        on=['game_id', 'home_team'],
        how='left'
    ).copy()
    games = pd.merge(
        games,
        units_away,
        on=['game_id', 'away_team'],
        how='left'
    ).copy()
    ## calculate units elo diff (home perspective) ##
    ## elo diff = (home_elo + home_qb_adj + hfa) - (away_elo + away_qb_adj) ##
    games['units_elo_diff'] = (
        (games['units_home_elo'] + games['units_home_qb_adj'] + games['hfa_adj'] * 25) -
        (games['units_away_elo'] + games['units_away_qb_adj'])
    )
    games['units_home_prob'] = games['units_elo_diff'].apply(calculate_win_probability)
    print(f"   ✓ Calculated Units Elo predictions")
    ## === QBELO (538) === ##
    print("\n5. Processing 538 QB Elo predictions...")
    ## qbelo already has team1/team2 structure where team1 is home ##
    qbelo_cols = qbelo[
        qbelo['playoff'].isna() | (qbelo['playoff'] == '')
    ][[
        'game_id', 'elo1_pre', 'elo2_pre', 'qb1_adj', 'qb2_adj'
    ]].rename(columns={
        'elo1_pre': 'f38_home_elo',
        'elo2_pre': 'f38_away_elo',
        'qb1_adj': 'f38_home_qb_adj',
        'qb2_adj': 'f38_away_qb_adj'
    })
    games = pd.merge(
        games,
        qbelo_cols,
        on=['game_id'],
        how='left'
    ).copy()
    ## calculate 538 QB elo diff (with QB adjustments) ##
    games['f38_qb_elo_diff'] = (
        (games['f38_home_elo'] + games['f38_home_qb_adj'] + games['hfa_adj'] * 25) -
        (games['f38_away_elo'] + games['f38_away_qb_adj'])
    )
    games['f38_qb_home_prob'] = games['f38_qb_elo_diff'].apply(calculate_win_probability)
    ## calculate 538 base elo diff (no QB adjustments) ##
    games['f38_base_elo_diff'] = (
        (games['f38_home_elo'] + games['hfa_adj'] * 25) -
        (games['f38_away_elo'])
    )
    games['f38_base_home_prob'] = games['f38_base_elo_diff'].apply(calculate_win_probability)
    print(f"   ✓ Calculated 538 Elo predictions (with and without QB adj)")
    ## === MARKET BASELINE === ##
    print("\n6. Processing market implied probabilities...")
    games['market_home_prob'] = games.apply(
        lambda r: vig_free_probability(r['home_moneyline'], r['away_moneyline']),
        axis=1
    )
    print(f"   ✓ Calculated market implied probabilities")
    ## === FILTER TO VALID GAMES === ##
    print("\n7. Filtering to games with all predictions...")
    valid_games = games[
        games['units_home_prob'].notna() &
        games['f38_qb_home_prob'].notna() &
        games['f38_base_home_prob'].notna() &
        games['market_home_prob'].notna() &
        games['home_win'].notna()
    ].copy()
    print(f"   ✓ {len(valid_games)} games with all predictions")
    ## define models for iteration ##
    models = [
        ('units_elo', 'units_home_prob'),
        ('f38_base_elo', 'f38_base_home_prob'),
        ('f38_qb_elo', 'f38_qb_home_prob'),
        ('market', 'market_home_prob')
    ]
    ## split into first/second half ##
    first_half = valid_games[valid_games['week'] <= 8].copy()
    second_half = valid_games[valid_games['week'] >= 9].copy()
    print(f"   ✓ First half (weeks 1-8): {len(first_half):,} games")
    print(f"   ✓ Second half (weeks 9-18): {len(second_half):,} games")
    ## === CALCULATE ALL METRICS === ##
    print("\n8. Calculating metrics...")
    ## === BY SEASON FILES === ##
    print("\n" + "=" * 80)
    print("BY SEASON RESULTS")
    print("=" * 80)
    ## log loss by season ##
    ll_results = []
    for season in sorted(valid_games['season'].unique()):
        sg = valid_games[valid_games['season'] == season]
        row = {'season': season, 'n_games': len(sg)}
        for name, col in models:
            row[name] = round(calculate_log_loss(sg['home_win'], sg[col]), 4)
        ll_results.append(row)
    ll_df = pd.DataFrame(ll_results)
    ll_df.to_csv(output_dir / 'log_loss_by_season.csv', index=False)
    print(f"\n   ✓ Saved log_loss_by_season.csv")
    print(ll_df.to_string(index=False))
    ## brier by season ##
    brier_results = []
    for season in sorted(valid_games['season'].unique()):
        sg = valid_games[valid_games['season'] == season]
        row = {'season': season, 'n_games': len(sg)}
        for name, col in models:
            row[name] = round(calculate_brier(sg['home_win'], sg[col]), 4)
        brier_results.append(row)
    brier_df = pd.DataFrame(brier_results)
    brier_df.to_csv(output_dir / 'brier_by_season.csv', index=False)
    print(f"\n   ✓ Saved brier_by_season.csv")
    ## f38 game points by season ##
    f38_points_results = []
    for season in sorted(valid_games['season'].unique()):
        sg = valid_games[valid_games['season'] == season]
        row = {'season': season, 'n_games': len(sg)}
        for name, col in models:
            row[name] = round(calculate_f38_game_points(sg['home_win'], sg[col]), 2)
        f38_points_results.append(row)
    f38_points_df = pd.DataFrame(f38_points_results)
    f38_points_df.to_csv(output_dir / 'f38_points_by_season.csv', index=False)
    print(f"   ✓ Saved f38_points_by_season.csv")
    ## accuracy by season ##
    acc_results = []
    for season in sorted(valid_games['season'].unique()):
        sg = valid_games[valid_games['season'] == season]
        row = {'season': season, 'n_games': len(sg)}
        for name, col in models:
            row[name] = round(calculate_accuracy(sg['home_win'], sg[col]), 4)
        acc_results.append(row)
    acc_df = pd.DataFrame(acc_results)
    acc_df.to_csv(output_dir / 'accuracy_by_season.csv', index=False)
    print(f"   ✓ Saved accuracy_by_season.csv")
    ## === SUMMARY TABLE === ##
    print("\n" + "=" * 80)
    print("SUMMARY (all seasons)")
    print("=" * 80)
    summary_rows = []
    for name, col in models:
        row = {
            'model': name,
            'n_games': len(valid_games),
            'log_loss': round(calculate_log_loss(valid_games['home_win'], valid_games[col]), 4),
            'log_loss_wk1_8': round(calculate_log_loss(first_half['home_win'], first_half[col]), 4),
            'log_loss_wk9_18': round(calculate_log_loss(second_half['home_win'], second_half[col]), 4),
            'brier': round(calculate_brier(valid_games['home_win'], valid_games[col]), 4),
            'f38_points': round(calculate_f38_game_points(valid_games['home_win'], valid_games[col]), 2),
            'accuracy': round(calculate_accuracy(valid_games['home_win'], valid_games[col]), 4)
        }
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / 'model_summary.csv', index=False)
    print(f"\n   ✓ Saved model_summary.csv")
    print(summary_df.to_string(index=False))
    ## === MODEL CORRELATION MATRIX === ##
    print("\n" + "=" * 80)
    print("MODEL CORRELATION MATRIX")
    print("=" * 80)
    prob_cols = [col for _, col in models]
    corr_df = valid_games[prob_cols].corr().round(4)
    corr_df.index = [name for name, _ in models]
    corr_df.columns = [name for name, _ in models]
    corr_df.to_csv(output_dir / 'model_correlation.csv')
    print(f"\n   ✓ Saved model_correlation.csv")
    print(corr_df.to_string())
    ## === PRINT DETAILED RESULTS === ##
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    print(f"\nGames: {len(valid_games):,} total | {len(first_half):,} first half | {len(second_half):,} second half")
    print("\nLOG LOSS (lower is better):")
    for name, col in models:
        ll_all = calculate_log_loss(valid_games['home_win'], valid_games[col])
        ll_first = calculate_log_loss(first_half['home_win'], first_half[col])
        ll_second = calculate_log_loss(second_half['home_win'], second_half[col])
        print(f"  {name:15s}  all: {ll_all:.4f}  wk1-8: {ll_first:.4f}  wk9-18: {ll_second:.4f}")
    print("\nBRIER SCORE (lower is better):")
    for name, col in models:
        print(f"  {name:15s}  {calculate_brier(valid_games['home_win'], valid_games[col]):.4f}")
    print("\n538 GAME POINTS (higher is better):")
    for name, col in models:
        print(f"  {name:15s}  {calculate_f38_game_points(valid_games['home_win'], valid_games[col]):.2f}")
    print("\nACCURACY (higher is better):")
    for name, col in models:
        print(f"  {name:15s}  {calculate_accuracy(valid_games['home_win'], valid_games[col]):.1%}")


if __name__ == '__main__':
    main()
