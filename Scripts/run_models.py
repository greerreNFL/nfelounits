'''
Run Models Script

Convenience function to run the model and save results to output file.
'''
import pandas as pd
from pathlib import Path

from ..Data import DataLoader
from ..Model import UnitModel
from ..Optimizer import ModelConfig


def run(output_path: str = None):
    '''
    Run the unit model and save results to CSV
    
    Parameters:
    * output_path: Path to save results (default 'output/unit_teams.csv')
    
    Output columns:
    * season, week, team, opponent
    * {unit}_off_value_pre, {unit}_def_value_pre (for pass, rush, st)
    * elo_pre
    * {unit}_off_value_post, {unit}_def_value_post (for pass, rush, st)
    * elo_post
    '''
    print("=" * 80)
    print("RUNNING UNIT MODEL")
    print("=" * 80)
    ## set output path if not provided ##
    if output_path is None:
        output_path = f'{Path(__file__).parent.parent.resolve()}/Output'
    ## Load data
    print("\n1. Loading data...")
    loader = DataLoader()
    print(f"   ✓ Loaded {len(loader.unit_games):,} games")
    ## Load config
    print("\n2. Loading config...")
    config = ModelConfig.from_file()
    print(f"   ✓ Loaded {len(config.params)} parameters")
    ## Run model
    print("\n3. Running model...")
    model = UnitModel(loader.unit_games, config.values)
    model.run()
    print("   ✓ Model run complete")
    ## Get results
    print("\n4. Preparing results...")
    results = model.get_results_df()
    ## round output ##
    results['elo'] = results['elo'].round(4)
    results['qb_adj'] = results['qb_adj'].round(4)
    for unit in ['pass', 'rush', 'st']:
        for side in ['off', 'def']:
            results[f'{unit}_{side}_value_pre'] = results[f'{unit}_{side}_value_pre'].round(4)
            results[f'{unit}_{side}_value_post'] = results[f'{unit}_{side}_value_post'].round(4)
    ## Select and order columns for output
    output_cols = [
        'season', 'week', 'team', 'opponent',
        'elo','qb_adj',
        # Pre-game values
        'pass_off_value_pre', 'pass_def_value_pre',
        'rush_off_value_pre', 'rush_def_value_pre',
        'st_off_value_pre', 'st_def_value_pre',
        # Post-game values
        'pass_off_value_post', 'pass_def_value_post',
        'rush_off_value_post', 'rush_def_value_post',
        'st_off_value_post', 'st_def_value_post',
        'elo_post'
    ]
    ## Filter to available columns
    available_cols = [col for col in output_cols if col in results.columns]
    output_df = results[available_cols].copy()
    ## Save to file
    output_df.to_csv(f'{output_path}/unit_teams.csv', index=False)
    print(f"   ✓ Saved")

