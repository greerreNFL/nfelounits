'''
Calibrate Recalibration Normalizer
Computes normalization coefficients to scale recalibration values
to match UnitModel's scale before blending.
'''
import json
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from ..Data import DataLoader
from ..Model import UnitModel, RecalibrationSet, UnitModelStateCollector
from ..Optimizer import ModelConfig

UNIT_TYPES = ['pass_off', 'pass_def', 'rush_off', 'rush_def', 'st_off', 'st_def']

def run(output_config: bool = True, verbose: bool = True):
    '''
    Calibrate the recalibration normalizer coefficients.
    Parameters:
    * output_config: Whether to write results to config.json (default True)
    * verbose: Whether to print progress (default True)
    Returns:
    * Dict with coefficients for each unit type
    '''
    if verbose:
        print("=" * 80)
        print("CALIBRATING RECALIBRATION NORMALIZER")
        print("=" * 80)
    ## paths ##
    package_path = Path(__file__).parent.parent.resolve()
    recal_path = package_path / 'Output' / 'recalibration_values.csv'
    config_path = package_path / 'config.json'
    ## load recalibration set ##
    if verbose:
        print("\n1. Loading recalibration values...")
    recal_set = RecalibrationSet.from_csv(str(recal_path))
    if verbose:
        print(f"   ✓ Loaded {len(recal_set.records):,} recalibration records")
    ## load data and config ##
    if verbose:
        print("\n2. Loading data and config...")
    loader = DataLoader()
    config = ModelConfig.from_file()
    if verbose:
        print(f"   ✓ Loaded {len(loader.unit_games):,} games")
    ## run model with state collection (no recalibration) ##
    if verbose:
        print("\n3. Running UnitModel with state collection...")
    collector = UnitModelStateCollector()
    model = UnitModel(
        games=loader.unit_games,
        config=config.values,
        state_collector=collector
    )
    model.run()
    if verbose:
        print(f"   ✓ Collected {len(collector.get_all_states())} week states")
    ## build dataframe of unit model values at week boundaries ##
    if verbose:
        print("\n4. Building comparison dataset...")
    model_records = []
    for state in collector.get_all_states():
        for team_abbr, team in state.teams.items():
            for unit_type in UNIT_TYPES:
                unit = getattr(team, unit_type)
                model_records.append({
                    'season': state.season,
                    'for_week': state.for_week,
                    'team': team_abbr,
                    'unit_type': unit_type,
                    'model_value': unit.value
                })
    model_df = pd.DataFrame(model_records)
    ## build dataframe of recal values ##
    recal_records = [{
        'season': r.season,
        'for_week': r.for_week,
        'team': r.team,
        'unit_type': r.unit_type,
        'recal_value': r.ideal_value
    } for r in recal_set.records]
    recal_df = pd.DataFrame(recal_records)
    ## join ##
    joined = pd.merge(
        model_df, recal_df,
        on=['season', 'for_week', 'team', 'unit_type'],
        how='inner'
    )
    if verbose:
        print(f"   ✓ Joined {len(joined):,} records")
    ## compute normalization coefficients ##
    if verbose:
        print("\n5. Computing normalization coefficients...")
    results = {}
    for unit_type in UNIT_TYPES:
        unit_data = joined[joined['unit_type'] == unit_type].copy()
        ## compute m for each week (regression: model_value = m * recal_value, through origin) ##
        week_coefficients = []
        weeks = sorted(unit_data['for_week'].unique())
        for week in weeks:
            week_data = unit_data[unit_data['for_week'] == week]
            ## regression through origin using statsmodels ##
            ols = sm.OLS(week_data['model_value'], week_data['recal_value']).fit()
            week_coefficients.append({'week': week, 'm': ols.params.iloc[0]})
        ## fit regression: m = slope * week + intercept ##
        coef_df = pd.DataFrame(week_coefficients)
        coef_df['constant'] = 1
        ols = sm.OLS(coef_df['m'], coef_df[['week', 'constant']]).fit()
        results[unit_type] = {
            'm': round(ols.params['week'], 5),
            'b': round(ols.params['constant'], 5),
            'r_squared': ols.rsquared
        }
        if verbose:
            print(f"   {unit_type}: m={ols.params['week']:.6f}, b={ols.params['constant']:.6f}, R²={ols.rsquared:.4f}")
    ## write to config ##
    if output_config:
        if verbose:
            print("\n6. Writing to config.json...")
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        for unit_type, coefs in results.items():
            config_data['recal_normalizer'][f'{unit_type}_m']['value'] = round(coefs['m'], 6)
            config_data['recal_normalizer'][f'{unit_type}_b']['value'] = round(coefs['b'], 6)
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        if verbose:
            print(f"   ✓ Updated {config_path}")
    if verbose:
        print("\n" + "=" * 80)
        print("CALIBRATION COMPLETE")
        print("=" * 80)
    return results
