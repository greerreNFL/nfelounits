'''
BaseOptimizer Abstract Class

Base class for all optimizers that provides common functionality for parameter optimization.
Subclasses only need to implement the objective function and metric name.
'''

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Optional, Dict
import pathlib
import time
import datetime
import pandas as pd
import numpy as np
from scipy.optimize import minimize

from .ModelConfig import ModelConfig, ModelParam
from ..Model import UnitModel


class BaseOptimizer(ABC):
    '''
    Abstract base class for optimizers that tune model configuration parameters.
    
    Subclasses must:
    - Define config_section as a class attribute (e.g., 'unit_config', 'elo_config')
    - Implement objective(x: List[float]) -> float: Calculate the optimization objective
    - Implement get_metric_name() -> str: Return the name of the metric being optimized
    '''
    config_section: str = None  # Must be set by subclasses
    
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
        * subset: List of parameter names to optimize (empty = all params in config_section)
        * subset_name: Name for this subset (for saving results)
        * randomize_bgs: Whether to randomize initial guesses
        * run_id: Unique identifier for this optimization run (defaults to timestamp if not provided)
        '''
        self.data: pd.DataFrame = data
        self.config: ModelConfig = config
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
        self.randomize_bgs: bool = randomize_bgs
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
    
    def denormalize_optimizer_values(self, x: List[float]) -> Dict[str, Any]:
        '''Denormalizes an optimizer values list into a nested config dictionary'''
        ## start with a copy of the config ##
        local_config = self.config.values.copy()
        ## update the config with the new values that are being optimized ##
        for i, feature in enumerate(self.features):
            denormalized_value = self.denormalize_param(x[i], self.config.params[feature])
            ## handle both nested (section.param) and flat (param) naming ##
            if '.' in feature:
                section, param_name = feature.split('.', 1)
                if section not in local_config:
                    local_config[section] = {}
                local_config[section][param_name] = denormalized_value
            else:
                ## flat parameter name ##
                local_config[feature] = denormalized_value
        return local_config
    
    def init_features(self):
        '''Initialize the features, bgs, and bounds for the optimizer'''
        ## if subset is empty, use all parameters from this optimizer's config section ##
        if len(self.subset) == 0:
            if self.config_section is None:
                raise ValueError(f"{self.__class__.__name__} must define config_section class attribute")
            params_to_optimize = [k for k in self.config.params.keys() 
                                 if k.startswith(f'{self.config_section}.')]
        else:
            params_to_optimize = self.subset
        
        for k in params_to_optimize:
            if k in self.config.params:
                param = self.config.params[k]
                self.features.append(k)
                self.bgs.append(
                    self.normalize_param(param.value, param) if not self.randomize_bgs
                    else np.random.uniform(0, 1)
                )
                self.bounds.append((0, 1))  ## all features are normalized ##
    
    @abstractmethod
    def objective(self, x: List[float]) -> float:
        '''
        Objective function for the optimizer - must be implemented by subclasses
        
        This method should:
        1. Increment self.round_number
        2. Create and run the model with denormalized parameters
        3. Calculate the optimization metric
        4. Store the record in self.optimization_records
        5. Save records periodically and when a new best is found
        6. Return the metric value to be minimized
        
        Parameters:
        * x: Normalized parameter values
        
        Returns:
        * Objective value to minimize
        '''
        pass
    
    @abstractmethod
    def get_metric_name(self) -> str:
        '''
        Return the name of the metric being optimized
        
        This is used for sorting records to find the best result.
        Examples: 'train_log_loss', 'avg_mae'
        '''
        pass
    
    def update_config(self, x: List[float]):
        '''Update the config with the new values and save the result'''
        ## create the updated config ##
        updated_values = {}
        for i, feature in enumerate(self.features):
            denormalized_value = self.denormalize_param(x[i], self.config.params[feature])
            updated_values[feature] = round(denormalized_value, 6)
        ## update the config ##
        self.config.update_config(updated_values)
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
                'ftol': self.tol,
                'eps': self.step
            }
        )
        ## end timer ##
        end_time = float(time.time())
        ## save the solution ##
        self.solution = solution
        ## create an optimization result object ##
        optimal_config = {}
        for i, feature in enumerate(self.features):
            denormalized_value = self.denormalize_param(solution.x[i], self.config.params[feature])
            optimal_config[feature] = denormalized_value
        ## add objective function reached and runtime ##
        metric_name = self.get_metric_name()
        self.optimization_results[metric_name] = solution.fun
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
        metric_name = self.get_metric_name()
        return df.sort_values(
            by=[metric_name],
            ascending=[True]
        ).reset_index(drop=True).to_dict(orient='records')[0]

