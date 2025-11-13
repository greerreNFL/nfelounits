"""
UnitGrader Class

Calculate performance metrics for unit predictions.
Mirrors qbelo scoring patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


class UnitGrader:
    """Calculate performance metrics for unit model"""
    
    def __init__(self, results: pd.DataFrame):
        """
        Initialize grader
        
        Parameters:
        * results: DataFrame from model.get_results_df()
        """
        self.results = results
        self.grades: Dict[str, float] = {}
    
    def calculate_unit_metrics(
        self,
        unit_prefix: str
    ) -> Dict[str, float]:
        """
        Calculate metrics for a single unit
        
        Parameters:
        * unit_prefix: 'pass', 'rush', or 'st'
        
        Returns:
        * dict with rmse, mae, r_squared
        """
        expected_col = f"{unit_prefix}_expected"
        observed_col = f"{unit_prefix}_observed"
        
        expected = self.results[expected_col]
        observed = self.results[observed_col]
        
        # Calculate metrics
        squared_error = (expected - observed) ** 2
        abs_error = np.abs(expected - observed)
        
        rmse = np.sqrt(squared_error.mean())
        mae = abs_error.mean()
        
        # R² calculation
        ss_res = squared_error.sum()
        ss_tot = ((observed - observed.mean()) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            f"{unit_prefix}_rmse": rmse,
            f"{unit_prefix}_mae": mae,
            f"{unit_prefix}_r_squared": r_squared
        }
    
    def grade(self) -> Dict[str, float]:
        """
        Calculate all performance metrics
        
        Returns:
        * dict with metrics for each unit plus overall
        """
        # Grade each unit
        pass_metrics = self.calculate_unit_metrics('pass')
        rush_metrics = self.calculate_unit_metrics('rush')
        st_metrics = self.calculate_unit_metrics('st')
        
        self.grades.update(pass_metrics)
        self.grades.update(rush_metrics)
        self.grades.update(st_metrics)
        
        # Calculate overall metrics (average across units)
        self.grades['overall_rmse'] = np.mean([
            pass_metrics['pass_rmse'],
            rush_metrics['rush_rmse'],
            st_metrics['st_rmse']
        ])
        
        self.grades['overall_mae'] = np.mean([
            pass_metrics['pass_mae'],
            rush_metrics['rush_mae'],
            st_metrics['st_mae']
        ])
        
        self.grades['overall_r_squared'] = np.mean([
            pass_metrics['pass_r_squared'],
            rush_metrics['rush_r_squared'],
            st_metrics['st_r_squared']
        ])
        
        return self.grades
    
    def print_grades(self) -> None:
        """Print formatted performance metrics"""
        print('\nUnit Model Performance:')
        print('\nPass Unit:')
        print(f"  RMSE: {self.grades['pass_rmse']:.3f}")
        print(f"  MAE: {self.grades['pass_mae']:.3f}")
        print(f"  R²: {self.grades['pass_r_squared']:.3f}")
        
        print('\nRush Unit:')
        print(f"  RMSE: {self.grades['rush_rmse']:.3f}")
        print(f"  MAE: {self.grades['rush_mae']:.3f}")
        print(f"  R²: {self.grades['rush_r_squared']:.3f}")
        
        print('\nSpecial Teams Unit:')
        print(f"  RMSE: {self.grades['st_rmse']:.3f}")
        print(f"  MAE: {self.grades['st_mae']:.3f}")
        print(f"  R²: {self.grades['st_r_squared']:.3f}")
        
        print('\nOverall:')
        print(f"  RMSE: {self.grades['overall_rmse']:.3f}")
        print(f"  MAE: {self.grades['overall_mae']:.3f}")
        print(f"  R²: {self.grades['overall_r_squared']:.3f}")

