import unittest
import pandas as pd
from nfelounits.Data import DataLoader, DataSplitter
from nfelounits.Optimizer import ModelConfig
from nfelounits.Model import UnitModel
from nfelounits.Performance import UnitGrader

class TestModel(unittest.TestCase):
    def test_model_execution_and_quality(self):
        """
        Test that the model runs successfully and produces reasonable quality metrics.
        Includes checking performance on standard Train/Test split.
        """
        # 1. Load Data
        print("\nLoading data...")
        loader = DataLoader()
        self.assertTrue(len(loader.unit_games) > 0, "No games loaded")
        
        # 2. Load Config
        print("Loading config...")
        config = ModelConfig.from_file()
        self.assertIsNotNone(config.values, "Config not loaded")
        
        # 3. Run Model
        print("Running model...")
        model = UnitModel(loader.unit_games, config.values)
        model.run()
        
        # 4. Get Results
        results = model.get_results_df()
        self.assertFalse(results.empty, "Model produced no results")
        
        # 5. Apply Data Split Logic to Results
        # We need to apply the split logic to the *results* dataframe to filter for grading
        # Note: DataSplitter expects a DataFrame with 'season' column, which results has.
        print("Applying Train/Test split...")
        splitter = DataSplitter(results)
        labeled_results = splitter.label_train_test(n_test_seasons=5, exclude_first_season=True)
        
        # 6. Check Quality on Splits
        # Test Set Performance
        test_data = labeled_results[labeled_results['data_set'] == 'test']
        print(f"\nTest Set Performance (Last 5 Seasons, n={len(test_data)} games):")
        test_grader = UnitGrader(test_data)
        test_grades = test_grader.grade()
        
        # Print only overall metrics for Test
        print(f"  RMSE: {test_grades.get('overall_rmse', 0):.4f}")
        print(f"  MAE: {test_grades.get('overall_mae', 0):.4f}")
        print(f"  R²: {test_grades.get('overall_r_squared', 0):.4f}")

        # Training Set Performance
        train_data = labeled_results[labeled_results['data_set'] == 'train']
        print(f"\nTraining Set Performance (Earlier Seasons, Excl First, n={len(train_data)} games):")
        train_grader = UnitGrader(train_data)
        train_grades = train_grader.grade()
        
        # Print only overall metrics for Train
        print(f"  RMSE: {train_grades.get('overall_rmse', 0):.4f}")
        print(f"  MAE: {train_grades.get('overall_mae', 0):.4f}")
        print(f"  R²: {train_grades.get('overall_r_squared', 0):.4f}")
        
        # Assertions on TEST set (most important for generalization)
        self.assertLess(test_grades['overall_rmse'], 20.0, f"Test RMSE too high: {test_grades['overall_rmse']}")
        self.assertLess(test_grades['overall_mae'], 20.0, f"Test MAE too high: {test_grades['overall_mae']}")

if __name__ == '__main__':
    unittest.main()
