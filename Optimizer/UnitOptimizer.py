'''
UnitOptimizer Class

Optimizer for tuning unit model configuration parameters to minimize MAE prediction error.
'''

from typing import List, Optional
import pathlib
import pandas as pd

from .BaseOptimizer import BaseOptimizer
from .ModelConfig import ModelConfig
from ..Model import UnitModel


class UnitOptimizer(BaseOptimizer):
    '''
    Optimizer that returns the optimal value for each parameter in the model config
    '''
    config_section = 'unit_config'
    
    def __init__(self,
        data: pd.DataFrame,
        config: ModelConfig,
        tol: float = 0.000001,
        step: float = 0.00001,
        method: str = 'SLSQP',
        subset: List[str] = [],
        subset_name: str = 'subset',
        randomize_bgs: bool = False,
        run_id: Optional[str] = None
    ):
        '''
        Initialize optimizer
        
        Parameters:
        * data: Full game-level DataFrame with 'data_set' column from DataSplitter
        * config: ModelConfig object with parameters to optimize
        * tol: Tolerance for optimization convergence
        * step: Step size for numerical gradient
        * method: Optimization method (default 'SLSQP')
        * subset: List of parameter names to optimize (empty = all unit_config params)
        * subset_name: Name for this subset (for saving results)
        * randomize_bgs: Whether to randomize initial guesses
        * run_id: Unique identifier for this optimization run (defaults to timestamp if not provided)
        '''
        ## initialize base class ##
        super().__init__(
            data=data,
            config=config,
            tol=tol,
            step=step,
            method=method,
            subset=subset,
            subset_name=subset_name,
            randomize_bgs=randomize_bgs,
            run_id=run_id
        )
    
    def get_metric_name(self) -> str:
        '''Return the name of the metric being optimized'''
        return 'avg_mae'
    
    def objective(self, x: List[float]) -> float:
        '''MAE objective function for the optimizer'''
        ## increment the round number ##
        self.round_number += 1
        ## create denormalized config ##
        denormalized_config = self.denormalize_optimizer_values(x)
        ## create the model ##
        model = UnitModel(self.data, denormalized_config)
        ## run the model ##
        model.run()
        ## get results ##
        results = model.get_results_df()
        ## join data_set labels back to results ##
        if 'data_set' in self.data.columns:
            data_set_labels = self.data[['game_id', 'data_set']].drop_duplicates()
            results = pd.merge(results, data_set_labels, on='game_id', how='left')
        ## filter to train data set ##
        if 'data_set' in results.columns:
            results = results[results['data_set'] == 'train'].copy()
        ## calculate MAE for each unit ##
        mae_values = {}
        for unit in ['pass', 'rush', 'st']:
            for side in ['off', 'def']:
                unit_name = f'{unit}_{side}'
                expected_col = f'{unit}_{side}_expected'
                observed_col = f'{unit}_{side}_observed'
                if expected_col in results.columns and observed_col in results.columns:
                    mae = (results[expected_col] - results[observed_col]).abs().mean()
                    mae_values[f'mae_{unit_name}'] = mae
        ## calculate average MAE across all units ##
        avg_mae = sum(mae_values.values()) / len(mae_values)
        ## create scored record ##
        scored_record = {
            'round': self.round_number,
            'avg_mae': avg_mae,
            **mae_values,
        }
        ## add denormalized values ##
        for i, feature in enumerate(self.features):
            denormalized_value = self.denormalize_param(x[i], self.config.params[feature])
            scored_record[feature] = denormalized_value
        
        ## add the record to the optimization records ##
        self.optimization_records.append(scored_record)
        ## save the record if it is a new best, or if it an interval of 100 rounds ##
        save_record = False
        if self.best_obj is None:
            self.best_obj = avg_mae
            save_record = True
        elif avg_mae < self.best_obj:
            self.best_obj = avg_mae
            save_record = True
        if self.round_number % 100 == 0:
            save_record = True
        if save_record:
            df = pd.DataFrame(self.optimization_records)
            output_dir = pathlib.Path(__file__).parent.resolve() / 'runs'
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(f'{output_dir}/{self.run_id}_{self.subset_name}_inflight_round.csv', index=False)
        ## return average MAE (what we're optimizing) ##
        return avg_mae

