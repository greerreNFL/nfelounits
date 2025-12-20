"""
UnitGrader Class

Calculate performance metrics for unit predictions.
Mirrors qbelo scoring patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List


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
    
    def calculate_series_metrics(
        self,
        expected: pd.Series,
        observed: pd.Series,
        prefix: str
    ) -> Dict[str, float]:
        """
        Calculate metrics for a data series pair
        
        Parameters:
        * expected: Series of expected values
        * observed: Series of observed values
        * prefix: Prefix for metric keys (e.g. 'pass_off')
        
        Returns:
        * dict with rmse, mae, r_squared
        """
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
            f"{prefix}_rmse": rmse,
            f"{prefix}_mae": mae,
            f"{prefix}_r_squared": r_squared
        }
    
    def grade(self) -> Dict[str, float]:
        """
        Calculate all performance metrics
        
        Returns:
        * dict with metrics for each unit component plus overall
        """
        units = ['pass', 'rush', 'st']
        sides = ['off', 'def']
        
        all_maes = []
        all_rmses = []
        all_r_squareds = []
        
        for unit in units:
            unit_maes = []
            unit_rmses = []
            unit_r_squareds = []
            
            for side in sides:
                col_prefix = f"{unit}_{side}"
                expected_col = f"{col_prefix}_expected"
                observed_col = f"{col_prefix}_observed"
                
                if expected_col in self.results.columns and observed_col in self.results.columns:
                    metrics = self.calculate_series_metrics(
                        self.results[expected_col],
                        self.results[observed_col],
                        col_prefix
                    )
                    self.grades.update(metrics)
                    
                    # Track for aggregation
                    all_maes.append(metrics[f"{col_prefix}_mae"])
                    all_rmses.append(metrics[f"{col_prefix}_rmse"])
                    all_r_squareds.append(metrics[f"{col_prefix}_r_squared"])
                    
                    unit_maes.append(metrics[f"{col_prefix}_mae"])
                    unit_rmses.append(metrics[f"{col_prefix}_rmse"])
                    unit_r_squareds.append(metrics[f"{col_prefix}_r_squared"])
            
            # Aggregate for unit type (average of off/def)
            if unit_maes:
                self.grades[f"{unit}_mae"] = np.mean(unit_maes)
                self.grades[f"{unit}_rmse"] = np.mean(unit_rmses)
                self.grades[f"{unit}_r_squared"] = np.mean(unit_r_squareds)
        
        # Calculate overall metrics (average across all 6 components)
        if all_maes:
            self.grades['avg_mae'] = np.mean(all_maes)
            self.grades['overall_mae'] = np.mean(all_maes) # Alias for backward compatibility/clarity
            self.grades['overall_rmse'] = np.mean(all_rmses)
            self.grades['overall_r_squared'] = np.mean(all_r_squareds)
        
        return self.grades
    
    def print_grades(self) -> None:
        """Print formatted performance metrics"""
        if not self.grades:
            self.grade()
            
        print('\nUnit Model Performance:')
        
        for unit in ['pass', 'rush', 'st']:
            print(f'\n{unit.capitalize()} Unit:')
            if f"{unit}_mae" in self.grades:
                print(f"  RMSE: {self.grades.get(f'{unit}_rmse', 0):.4f}")
                print(f"  MAE: {self.grades.get(f'{unit}_mae', 0):.4f}")
                print(f"  R²: {self.grades.get(f'{unit}_r_squared', 0):.4f}")
                # Optional: Print off/def splits if desired, keeping it clean for now
            else:
                print("  No data")
        
        print('\nOverall (Avg across 6 components):')
        print(f"  RMSE: {self.grades.get('overall_rmse', 0):.4f}")
        print(f"  MAE: {self.grades.get('overall_mae', 0):.4f}")
        print(f"  R²: {self.grades.get('overall_r_squared', 0):.4f}")
