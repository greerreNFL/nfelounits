import unittest
import os
import tempfile
from nfelounits.Data import DataLoader
from nfelounits.Optimizer import ModelConfig
from nfelounits.Model import UnitModel


class TestState(unittest.TestCase):
    """Tests for UnitModel state save/load functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Load data and config once for all tests."""
        cls.loader = DataLoader()
        cls.config = ModelConfig.from_file()
        # Use 2023 season for quick tests
        cls.games_2023 = cls.loader.unit_games[
            cls.loader.unit_games['season'] == 2023
        ].copy()
    
    def test_save_and_load_state(self):
        """Test that state can be saved and loaded correctly."""
        # Run model on first 10 weeks
        historical = self.games_2023[self.games_2023['week'] <= 10].copy()
        model = UnitModel(historical, self.config.values)
        model.run()
        
        # Capture values before save
        kc_pass_off = model.teams['KC'].pass_off.value
        record_count = len(model.team_game_records)
        team_count = len(model.teams)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            model.save_to_state(temp_path)
            
            # Create fresh model and load state
            model2 = UnitModel(self.games_2023, self.config.values)
            model2.load_from_state(temp_path)
            
            # Verify state restored
            self.assertEqual(len(model2.teams), team_count)
            self.assertEqual(len(model2.team_game_records), record_count)
            self.assertAlmostEqual(
                model2.teams['KC'].pass_off.value, 
                kc_pass_off, 
                places=6
            )
        finally:
            os.unlink(temp_path)
    
    def test_game_filtering(self):
        """Test that loading state filters out already-processed games."""
        # Run on weeks 1-10
        historical = self.games_2023[self.games_2023['week'] <= 10].copy()
        model = UnitModel(historical, self.config.values)
        model.run()
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            model.save_to_state(temp_path)
            
            # Load into model with ALL games
            model2 = UnitModel(self.games_2023, self.config.values)
            original_count = len(model2.games)
            model2.load_from_state(temp_path)
            
            # Should have filtered out processed games
            self.assertLess(len(model2.games), original_count)
            
            # Remaining games should all be week > 10
            self.assertTrue(all(model2.games['week'] > 10))
        finally:
            os.unlink(temp_path)
    
    def test_state_manager_exists(self):
        """Test that state_manager.exists() works correctly."""
        model = UnitModel(self.games_2023, self.config.values)
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        os.unlink(temp_path)  # Delete so it doesn't exist
        
        # Should not exist
        self.assertFalse(model.state_manager.exists(temp_path))
        
        # Run and save
        model.run()
        model.save_to_state(temp_path)
        
        try:
            # Now should exist
            self.assertTrue(model.state_manager.exists(temp_path))
        finally:
            os.unlink(temp_path)


if __name__ == '__main__':
    unittest.main()

