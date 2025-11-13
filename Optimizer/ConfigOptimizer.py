'''
ConfigOptimizer Class

Optimizer for tuning model configuration parameters to minimize prediction error.
'''

from typing import List, Tuple, Any, Optional, Dict
import pathlib
import time
import datetime
import pandas as pd
import numpy
from scipy.optimize import minimize

from .ModelConfig import ModelConfig, ModelParam
from ..Model import UnitModel


class ConfigOptimizer:
    '''
    Optimizer that returns the optimal value for each parameter in the model config
    '''
    def __init__(self,
        data: pd.DataFrame,
        config: ModelConfig,
        objective_name: str = 'avg_mae',
        tol: float = 0.000001,
        step: float = 0.00001,
        method: str = 'SLSQP',
        subset: List[str] = [],
        subset_name: str = 'subset',
        obj_normalization: float = 5.0,
        randomize_bgs: bool = False,
        first_season_to_score: Optional[int] = None,
        run_id: Optional[str] = None
    ):
        '''
        Initialize optimizer
        
        Parameters:
        * data: Game-level DataFrame from DataLoader.unit_games
        * config: ModelConfig object with parameters to optimize
        * objective_name: Name of objective function (default 'avg_mae')
        * tol: Tolerance for optimization convergence
        * step: Step size for numerical gradient
        * method: Optimization method (default 'SLSQP')
        * subset: List of parameter names to optimize (empty = all)
        * subset_name: Name for this subset (for saving results)
        * obj_normalization: Normalization factor for objective function
        * randomize_bgs: Whether to randomize initial guesses
        * first_season_to_score: First season to include in objective (excludes earlier seasons)
        * run_id: Unique identifier for this optimization run (defaults to timestamp if not provided)
        '''
        self.data: pd.DataFrame = data
        self.config: ModelConfig = config
        self.objective_name: str = objective_name
        self.subset: List[str] = subset
        self.subset_name: str = subset_name
        self.run_id: str = run_id if run_id else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ## optimizer setup ##
        self.features: List[str] = []
        self.bgs: List[float] = []
        self.bounds: List[Tuple[float, float]] = []
        self.tol: float = tol
        self.step: float = step
        self.method: str = method
        self.obj_normalization: float = obj_normalization
        self.randomize_bgs: bool = randomize_bgs
        self.first_season_to_score: Optional[int] = first_season_to_score
        self.init_features()
        ## in-optimization data ##
        self.round_number: int = 0
        self.optimization_records: List[dict] = []
        self.best_obj: Optional[float] = None
        ## post optimization data ##
        self.solution: Any = None
        self.optimization_results: dict = {}
    
    def normalize_param(self, value: float, param: ModelParam) -> float:
        '''Normalize a parameter to a value between 0 and 1'''
        return (value - param.opti_min) / (param.opti_max - param.opti_min)
    
    def denormalize_param(self, value: float, param: ModelParam) -> float:
        '''Denormalize a parameter from a value between 0 and 1'''
        return value * (param.opti_max - param.opti_min) + param.opti_min
    
    def denormalize_optimizer_values(self, x: List[float]) -> Dict[str, float]:
        '''Denormalizes an optimizer values list into a config dictionary'''
        ## start with a copy of the config ##
        local_config = self.config.values.copy()
        ## update the config with the new values that are being optimized ##
        for i, k in enumerate(x):
            local_config[self.features[i]] = self.denormalize_param(k, self.config.params[self.features[i]])
        return local_config
    
    def init_features(self):
        '''Initialize the features, bgs, and bounds for the optimizer'''
        for k, v in self.config.params.items():
            if len(self.subset) > 0 and k not in self.subset:
                continue
            self.features.append(k)
            self.bgs.append(
                self.normalize_param(v.value, v) if not self.randomize_bgs
                else numpy.random.uniform(0, 1)
            )
            self.bounds.append((0,1)) ## all features are normalized ##
    
    def objective(self, x: List[float]) -> float:
        '''Objective function for the optimizer'''
        ## increment the round number ##
        self.round_number += 1
        ## create denormalized config ##
        denormalized_dict = self.denormalize_optimizer_values(x)
        ## create the model ##
        model = UnitModel(
            self.data,
            denormalized_dict
        )
        ## run the model ##
        model.run()
        ## get results ##
        results = model.get_results_df()
        ## filter to seasons to score ##
        if self.first_season_to_score is not None:
            results = results[results['season'] >= self.first_season_to_score].copy()
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
            **denormalized_dict
        }
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
        ## return normalized objective ##
        return avg_mae / self.obj_normalization
    
    def update_config(self, x: List[float]):
        '''Update the config with the new values and save the result'''
        ## create the updated config ##
        updated_config = self.denormalize_optimizer_values(x)
        ## apply additional rounding ##
        for k, v in updated_config.items():
            updated_config[k] = round(v, 6)
        ## update the config ##
        self.config.update_config(updated_config)
        ## save to package config file ##
        self.config.to_file()
    
    def optimize(
            self,
            save_result: bool = True,
            update_config: bool = False
        ):
        '''Core optimization function'''
        ## run the optimizer ##
        ## start timer ##
        start_time = float(time.time())
        solution = minimize(
            self.objective,
            self.bgs,
            bounds=self.bounds,
            method=self.method,
            options={
                'ftol' : self.tol,
                'eps' : self.step
            }
        )
        ## end timer ##
        end_time = float(time.time())
        ## save the solution ##
        self.solution = solution
        ## create an optimization result object ##
        ## values ##
        optimal_config = self.denormalize_optimizer_values(solution.x)
        ## add objective function reached ##
        self.optimization_results['avg_mae'] = solution.fun * self.obj_normalization
        self.optimization_results['runtime'] = end_time - start_time
        ## extend the optimization results with the optimal config ##
        self.optimization_results = self.optimization_results | optimal_config
        ## save as needed ##
        if save_result:
            df = pd.DataFrame([self.optimization_results])
            output_dir = pathlib.Path(__file__).parent.resolve() / 'runs'
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(f'{output_dir}/{self.run_id}_{self.subset_name}_final.csv', index=False)
        ## update the config if needed ##
        if update_config:
            self.update_config(solution.x)
    
    def get_best_record(self) -> dict:
        '''Gets the best record from the stored optimization records'''
        df = pd.DataFrame(self.optimization_records)
        return df.sort_values(
            by=['avg_mae'],
            ascending=[True]
        ).reset_index(drop=True).to_dict(orient='records')[0]

