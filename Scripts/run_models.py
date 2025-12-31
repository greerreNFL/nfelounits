'''
Run Models Script

Convenience function to run the model and save results to output file.
'''
from pathlib import Path

from ..Data import DataLoader
from ..Model import UnitModel
from ..Optimizer import ModelConfig
from ..Performance import UnitGrader
from ..Processing import UnitTeamsProcessor, UnitTeamsNormalizationProcessor, ValueCreatedProcessor, OpponentFacedProcessor


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
    
    ## Calculate and print grades
    grader = UnitGrader(results)
    grader.print_grades()
    
    ## Process and save output
    processor = UnitTeamsProcessor(model.team_game_records)
    processor.save(output_path)
    print(f"   ✓ Saved Output")
    processor = UnitTeamsNormalizationProcessor(model.team_game_records)
    processor.save(output_path)
    print(f"   ✓ Saved Normalized Output")
    processor = ValueCreatedProcessor(model.team_game_records)
    processor.save(output_path)
    print(f"   ✓ Saved Value Created Output")
    processor = OpponentFacedProcessor(model.team_game_records)
    processor.save(output_path)
    print(f"   ✓ Saved Opponent Faced Output")
