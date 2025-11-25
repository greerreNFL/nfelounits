'''
Standardized Model Optimization Script

Runs both UnitOptimizer and EloOptimizer with multiple random starts,
selects the best results, and updates the model configuration.
'''

import pandas as pd
import datetime

from ..Data import DataLoader, DataSplitter
from ..Model import UnitModel
from ..Optimizer import ModelConfig, UnitOptimizer, EloOptimizer

def optimize_unit_params_by_unit(labeled_data: pd.DataFrame, config: ModelConfig, n_rounds: int = 10) -> dict:
    '''
    Run UnitOptimizer separately for each unit (pass, rush, st) and combine results
    
    Parameters:
    * labeled_data: DataFrame with train/test labels
    * config: ModelConfig object
    * n_rounds: Number of optimization rounds per unit (default 10)
    
    Returns:
    * Dictionary with best parameters from all units (rounded to 4 decimals)
    '''
    print("\n" + "=" * 80)
    print("OPTIMIZING UNIT PARAMETERS (BY UNIT)")
    print("=" * 80)
    
    # Define parameter subsets for each unit (includes weather params)
    unit_subsets = {
        'pass': [
            'unit_config.pass_hfa_share',
            'unit_config.pass_off_sf',
            'unit_config.pass_def_sf',
            'unit_config.pass_off_reversion',
            'unit_config.pass_def_reversion',
            'unit_config.league_pass_sf',
            'unit_config.league_pass_reversion',
            'unit_config.pass_wind_disc_height',
            'unit_config.pass_temp_disc_height',
            'unit_config.pass_off_qb_reversion',
            'unit_config.league_qb_sf',
        ],
        'rush': [
            'unit_config.rush_hfa_share',
            'unit_config.rush_off_sf',
            'unit_config.rush_def_sf',
            'unit_config.rush_off_reversion',
            'unit_config.rush_def_reversion',
            'unit_config.league_rush_sf',
            'unit_config.league_rush_reversion',
            'unit_config.rush_wind_disc_height',
            'unit_config.rush_temp_disc_height'
        ],
        'st': [
            'unit_config.st_hfa_share',
            'unit_config.st_off_sf',
            'unit_config.st_def_sf',
            'unit_config.st_off_reversion',
            'unit_config.st_def_reversion',
            'unit_config.league_st_sf',
            'unit_config.league_st_reversion',
            'unit_config.st_wind_disc_height',
            'unit_config.st_temp_disc_height'
        ]
    }
    
    all_unit_params = {}
    
    # Optimize each unit separately
    for unit_name, subset in unit_subsets.items():
        print(f"\n{'=' * 80}")
        print(f"OPTIMIZING {unit_name.upper()} UNIT ({len(subset)} parameters)")
        print("=" * 80)
        
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        best_records = []
        
        # Run multiple rounds with random starts
        for round_num in range(1, n_rounds + 1):
            print(f"\nRound {round_num}/{n_rounds}")
            print("-" * 40)
            
            optimizer = UnitOptimizer(
                data=labeled_data,
                config=config,
                subset=subset,
                subset_name=f'{unit_name}_params',
                randomize_bgs=True,
                run_id=run_id
            )
            
            optimizer.optimize(save_result=False, update_config=False)
            best_record = optimizer.get_best_record()
            best_records.append({**best_record, 'round_num': round_num})
            
            print(f"  Best avg MAE this round: {best_record['avg_mae']:.4f}")
            # Show unit-specific MAE if available
            unit_mae_keys = [k for k in best_record.keys() if k.startswith(f'mae_{unit_name}_')]
            if unit_mae_keys:
                for key in unit_mae_keys:
                    print(f"    {key}: {best_record[key]:.4f}")
        
        # Find best result for this unit
        df = pd.DataFrame(best_records)
        df = df.sort_values('avg_mae')
        best_result = df.iloc[0].to_dict()
        
        print(f"\n{unit_name.upper()} UNIT OPTIMIZATION COMPLETE")
        print(f"Best avg MAE: {best_result['avg_mae']:.4f}")
        print(f"From round: {int(best_result['round_num'])}")
        
        # Extract and round parameters for this unit
        unit_params = {k: round(v, 4) for k, v in best_result.items() 
                      if k.startswith('unit_config.') and isinstance(v, (int, float))}
        
        print(f"\nOptimal {unit_name} parameters:")
        for param, value in unit_params.items():
            print(f"  {param}: {value:.4f}")
        
        # Add to combined results
        all_unit_params.update(unit_params)
        
        # Update config with this unit's params before optimizing next unit
        config.update_config(unit_params)
    
    print("\n" + "=" * 80)
    print("ALL UNIT OPTIMIZATIONS COMPLETE")
    print("=" * 80)
    print(f"\nTotal parameters optimized: {len(all_unit_params)}")
    
    return all_unit_params


def optimize_elo_params(labeled_data: pd.DataFrame, config: ModelConfig, n_rounds: int = 10) -> dict:
    '''
    Run EloOptimizer multiple times and return the best result
    
    Parameters:
    * labeled_data: DataFrame with train/test labels
    * config: ModelConfig object
    * n_rounds: Number of optimization rounds to run (default 10)
    
    Returns:
    * Dictionary with best elo parameters (rounded to 4 decimals)
    '''
    print("\n" + "=" * 80)
    print("OPTIMIZING ELO PARAMETERS")
    print("=" * 80)
    ## run set up ##
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_records = []
    ## run multiple times with random starting points to find the best result ##
    for round_num in range(1, n_rounds + 1):
        print(f"\nRound {round_num}/{n_rounds}")
        print("-" * 40)
        ## set up the optimizer ##
        optimizer = EloOptimizer(
            data=labeled_data,
            config=config,
            subset=[],  # empty = all elo_config params
            subset_name='elo_params',
            randomize_bgs=True,
            calculate_test=True,
            run_id=run_id
        )
        ## run the optimizer ##
        optimizer.optimize(save_result=False, update_config=False)
        best_record = optimizer.get_best_record()
        best_records.append({**best_record, 'round_num': round_num})
        print(f"  Train log loss: {best_record['train_log_loss']:.6f}")
        if best_record.get('test_log_loss') is not None:
            print(f"  Test log loss:  {best_record['test_log_loss']:.6f}")
    # Find the best overall result
    df = pd.DataFrame(best_records)
    df = df.sort_values('train_log_loss')
    best_result = df.iloc[0].to_dict()
    print("\n" + "=" * 80)
    print("ELO OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nBest train log loss: {best_result['train_log_loss']:.6f}")
    if best_result.get('test_log_loss') is not None:
        print(f"Best test log loss:  {best_result['test_log_loss']:.6f}")
    print(f"From round: {int(best_result['round_num'])}")
    # Round all elo parameter values to 4 decimals
    elo_result = {k: round(v, 4) for k, v in best_result.items() 
                  if k.startswith('elo_config.') and isinstance(v, (int, float))}
    return elo_result


def optimize_models(n_rounds: int = 10, n_test_seasons: int = 5):
    '''
    Main optimization workflow - optimizes all model parameters
    
    Parameters:
    * n_rounds: Number of optimization rounds per optimizer (default 10)
    * n_test_seasons: Number of seasons to hold out for testing (default 5)
    '''
    print("=" * 80)
    print("STANDARDIZED MODEL OPTIMIZATION")
    print("=" * 80)
    # Load data
    print("\n1. Loading data...")
    loader = DataLoader()
    print(f"   ✓ Loaded {len(loader.unit_games):,} games")
    # Label data for train/test
    print("\n2. Labeling data for train/test split...")
    splitter = DataSplitter(loader.unit_games)
    labeled_data = splitter.label_train_test(
        n_test_seasons=n_test_seasons,
        exclude_first_season=True
    )
    print(f"   ✓ Total games: {len(labeled_data):,}")
    print(f"   ✓ Exclude (warm-up): {len(labeled_data[labeled_data['data_set']=='exclude']):,} games")
    print(f"   ✓ Train: {len(labeled_data[labeled_data['data_set']=='train']):,} games")
    print(f"   ✓ Test: {len(labeled_data[labeled_data['data_set']=='test']):,} games")
    # Load config
    print("\n3. Loading current config...")
    config = ModelConfig.from_file()
    print(f"   ✓ {len(config.params)} total parameters")
    # Optimize unit parameters (by unit for faster convergence)
    unit_params = optimize_unit_params_by_unit(labeled_data, config, n_rounds=n_rounds)
    print("\n4. Updating config with optimal unit parameters...")
    config.update_config(unit_params)
    config.to_file()
    print("   ✓ Unit parameters updated and saved")
    print("\nOptimal unit parameters:")
    for param, value in unit_params.items():
        print(f"  {param}: {value:.4f}")
    # Reload config to get the updated values for elo optimization
    config = ModelConfig.from_file()
    # Optimize elo parameters
    elo_params = optimize_elo_params(labeled_data, config, n_rounds=n_rounds)
    print("\n5. Updating config with optimal elo parameters...")
    config.update_config(elo_params)
    config.to_file()
    print("   ✓ Elo parameters updated and saved")
    print("\nOptimal elo parameters:")
    for param, value in elo_params.items():
        print(f"  {param}: {value:.4f}")
    print("\n" + "=" * 80)
    print("ALL OPTIMIZATIONS COMPLETE")
    print("=" * 80)
    print("\n✓ Model configuration has been updated with optimal parameters")
    print("✓ Config saved to: config.json")

