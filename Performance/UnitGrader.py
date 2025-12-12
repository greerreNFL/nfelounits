"""
UnitGrader Class

Calculate performance metrics for unit predictions.
"""

import numpy
import pandas as pd
from typing import Dict, Any, List, Optional


class UnitGrader:
    """Calculate performance metrics for unit model"""
    
    UNIT_KEYS = ['pass_off', 'pass_def', 'rush_off', 'rush_def', 'st_off', 'st_def']
    
    def __init__(self, results: pd.DataFrame):
        """
        Initialize grader
        
        Parameters:
        * results: DataFrame from model.get_results_df()
        """
        self.results = results
        self.grades: Dict[str, float] = {}
    
    def calculate_unit_metrics(self, unit_key: str) -> Dict[str, float]:
        """
        Calculate metrics for a single unit
        
        Parameters:
        * unit_key: 'pass_off', 'pass_def', 'rush_off', 'rush_def', 'st_off', or 'st_def'
        
        Returns:
        * dict with rmse, mae, r_squared for this unit
        """
        expected_col = f"{unit_key}_expected"
        observed_col = f"{unit_key}_observed"
        expected = self.results[expected_col]
        observed = self.results[observed_col]
        ## calculate metrics ##
        squared_error = (expected - observed) ** 2
        abs_error = numpy.abs(expected - observed)
        rmse = numpy.sqrt(squared_error.mean())
        mae = abs_error.mean()
        ## R² calculation ##
        ss_res = squared_error.sum()
        ss_tot = ((observed - observed.mean()) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        return {
            f"{unit_key}_rmse": rmse,
            f"{unit_key}_mae": mae,
            f"{unit_key}_r_squared": r_squared
        }
    
    def grade(self, subset: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate all performance metrics
        
        Parameters:
        * subset: Optional list of unit keys to include in overall metrics
                  Default None = all units ['pass_off', 'pass_def', 'rush_off', 'rush_def', 'st_off', 'st_def']
        
        Returns:
        * dict with metrics for each unit plus overall (based on subset)
        """
        if subset is None:
            subset = self.UNIT_KEYS
        ## calculate metrics for all units ##
        all_metrics = {}
        for unit_key in self.UNIT_KEYS:
            metrics = self.calculate_unit_metrics(unit_key)
            all_metrics.update(metrics)
        self.grades.update(all_metrics)
        ## calculate overall metrics based on subset ##
        subset_maes = [all_metrics[f'{k}_mae'] for k in subset]
        subset_rmses = [all_metrics[f'{k}_rmse'] for k in subset]
        subset_r2s = [all_metrics[f'{k}_r_squared'] for k in subset]
        self.grades['overall_mae'] = numpy.mean(subset_maes)
        self.grades['overall_rmse'] = numpy.mean(subset_rmses)
        self.grades['overall_r_squared'] = numpy.mean(subset_r2s)
        return self.grades
    
    def print_grades(self) -> None:
        """Print formatted performance metrics"""
        print('\nUnit Model Performance:')
        print('\nPass Offense:')
        print(f"  RMSE: {self.grades['pass_off_rmse']:.3f}")
        print(f"  MAE: {self.grades['pass_off_mae']:.3f}")
        print(f"  R²: {self.grades['pass_off_r_squared']:.3f}")
        print('\nPass Defense:')
        print(f"  RMSE: {self.grades['pass_def_rmse']:.3f}")
        print(f"  MAE: {self.grades['pass_def_mae']:.3f}")
        print(f"  R²: {self.grades['pass_def_r_squared']:.3f}")
        print('\nRush Offense:')
        print(f"  RMSE: {self.grades['rush_off_rmse']:.3f}")
        print(f"  MAE: {self.grades['rush_off_mae']:.3f}")
        print(f"  R²: {self.grades['rush_off_r_squared']:.3f}")
        print('\nRush Defense:')
        print(f"  RMSE: {self.grades['rush_def_rmse']:.3f}")
        print(f"  MAE: {self.grades['rush_def_mae']:.3f}")
        print(f"  R²: {self.grades['rush_def_r_squared']:.3f}")
        print('\nSpecial Teams Offense:')
        print(f"  RMSE: {self.grades['st_off_rmse']:.3f}")
        print(f"  MAE: {self.grades['st_off_mae']:.3f}")
        print(f"  R²: {self.grades['st_off_r_squared']:.3f}")
        print('\nSpecial Teams Defense:')
        print(f"  RMSE: {self.grades['st_def_rmse']:.3f}")
        print(f"  MAE: {self.grades['st_def_mae']:.3f}")
        print(f"  R²: {self.grades['st_def_r_squared']:.3f}")
        print('\nOverall:')
        print(f"  RMSE: {self.grades['overall_rmse']:.3f}")
        print(f"  MAE: {self.grades['overall_mae']:.3f}")
        print(f"  R²: {self.grades['overall_r_squared']:.3f}")
