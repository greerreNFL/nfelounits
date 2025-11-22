'''
EloOptimizer Class

Optimizer for tuning elo translation coefficients to minimize log loss on win probability predictions.
'''

from typing import List, Optional
import pathlib
import pandas as pd
import numpy as np

from .BaseOptimizer import BaseOptimizer
from .ModelConfig import ModelConfig
from ..Model import UnitModel


class EloOptimizer(BaseOptimizer):
    '''
    Optimizer that returns the optimal elo coefficients to minimize log loss on win predictions
    '''
    config_section = 'elo_config'
    
    def __init__(self,
        data: pd.DataFrame,
        config: ModelConfig = None,
        tol: float = 0.000001,
        step: float = 0.00001,
        method: str = 'SLSQP',
        subset: List[str] = [],
        subset_name: str = 'elo_params',
        randomize_bgs: bool = False,
        calculate_test: bool = True,
        run_id: Optional[str] = None
    ):
        '''
        Initialize optimizer
        
        Parameters:
        * data: Full game-level DataFrame with 'data_set' column from DataSplitter
        * config: ModelConfig object with parameters
        * tol: Tolerance for optimization convergence
        * step: Step size for numerical gradient
        * method: Optimization method (default 'SLSQP')
        * subset: List of parameter names to optimize (empty = all elo_config params)
        * subset_name: Name for this subset (for saving results)
        * randomize_bgs: Whether to randomize initial guesses
        * calculate_test: Whether to calculate test metrics (default True)
        * run_id: Unique identifier for this optimization run (defaults to timestamp if not provided)
        '''
        self.calculate_test: bool = calculate_test
        
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
        return 'train_log_loss'
    
    def calculate_log_loss(self, results: pd.DataFrame, data_set: str) -> float:
        '''
        Calculate log loss on specific data set
        
        Parameters:
        * results: Full results DataFrame
        * data_set: Which data_set to score ('train' or 'test')
        
        Returns:
        * Log loss for the specified data set
        '''
        ## filter to specified data set ##
        if 'data_set' in results.columns:
            filtered_results = results[results['data_set'] == data_set].copy()
        else:
            filtered_results = results.copy()
        
        ## filter to home team records only (avoid double counting) ##
        home_results = filtered_results[filtered_results['is_home'] == True].copy()
        
        if len(home_results) == 0:
            return float('nan')
        
        ## determine actual outcome from result field ##
        ## result is home score - away score, so positive = home won ##
        home_results['home_won'] = (home_results['result'] > 0).astype(int)
        
        ## Calculate log loss ##
        ## Log loss = -1/N * sum(y * log(p) + (1-y) * log(1-p)) ##
        epsilon = 1e-15  ## avoid log(0) ##
        probs = np.clip(home_results['win_prob'].values, epsilon, 1 - epsilon)
        actual = home_results['home_won'].values
        
        log_loss = -np.mean(actual * np.log(probs) + (1 - actual) * np.log(1 - probs))
        return log_loss
    
    def objective(self, x: List[float]) -> float:
        '''Log loss objective function for the optimizer'''
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
        
        ## calculate train log loss ##
        train_log_loss = self.calculate_log_loss(results, 'train')
        
        ## calculate test log loss if requested ##
        test_log_loss = None
        if self.calculate_test:
            test_log_loss = self.calculate_log_loss(results, 'test')
        
        ## create scored record ##
        scored_record = {
            'round': self.round_number,
            'train_log_loss': train_log_loss,
            'test_log_loss': test_log_loss,
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
            self.best_obj = train_log_loss
            save_record = True
        elif train_log_loss < self.best_obj:
            self.best_obj = train_log_loss
            save_record = True
        if self.round_number % 100 == 0:
            save_record = True
        if save_record:
            df = pd.DataFrame(self.optimization_records)
            output_dir = pathlib.Path(__file__).parent.resolve() / 'runs'
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(f'{output_dir}/{self.run_id}_{self.subset_name}_inflight_round.csv', index=False)
        ## return train log loss (what we're optimizing) ##
        return train_log_loss
    
    def optimize(
            self,
            save_result: bool = True,
            update_config: bool = False
        ):
        '''
        Core optimization function
        
        Overrides base class to add test_log_loss to optimization results after running.
        '''
        ## call base class optimize ##
        super().optimize(save_result=save_result, update_config=update_config)
        
        ## add test log loss from best record if available ##
        if self.calculate_test and len(self.optimization_records) > 0:
            best_record = self.get_best_record()
            self.optimization_results['test_log_loss'] = best_record.get('test_log_loss')

