# nfelounits

A Python package for decomposing NFL team performance into measurable unit ratings using EPA (Expected Points Added) from play-by-play data, with translation to Elo values

## Overview

`nfelounits` breaks down team performance into three distinct units:
- **Passing** (QB/Passing offense and pass defense)
- **Rushing** (Rush offense and rush defense)  
- **Special Teams** (ST offense and ST defense)

The model uses EWMA (Exponentially Weighted Moving Average) to track unit performance over time, with offseason regression and contextual game adjustments. The unit's rating represents the predicted point differential the unit will create against an average team Summing all units together represents the team's estimated margin against an average team.

The Unit Model is trained to minimize the absolute error between the predicted points and the actual points at the unit level. In addition to providing unit ratings, there is a second model, EloTranslator, that combines the unit ratings into a single Elo rating for each team. EloTranslator is trained to minimize the log loss when using these composed Elo ratings to predict the win probability of a game.


## Quick Start

```python
from nfelounits import DataLoader, UnitModel, ModelConfig

# Load play-by-play data
loader = DataLoader()
unit_games = loader.unit_games

# Load default configuration
config = ModelConfig.from_file()

# Run the model
model = UnitModel(unit_games, config.values)
model.run()

# Get results with unit ratings and win probabilities
results = model.get_results_df()
print(results[['team', 'win_prob', 'pass_off_expected', 'rush_off_expected']])
```

## Core Components

### Data Loading
```python
from nfelounits import DataLoader

# Load and prepare data
loader = DataLoader()
unit_games = loader.unit_games  # Game-level unit EPA data
pbp = loader.pbp                # Play-by-play data
```

### Model Execution
```python
from nfelounits import UnitModel, ModelConfig

# Load configuration
config = ModelConfig.from_file()

# Create and run model
model = UnitModel(data=unit_games, config=config.values)
model.run()

# Access results
results_df = model.get_results_df()
team_ratings = model.teams  # Dict of Team objects
```

### Configuration Management
```python
from nfelounits import ModelConfig

# Load from file
config = ModelConfig.from_file()

# Access parameter values
smoothing_factor = config.params['pass_off_sf'].value

# Update parameters
config.update_config({
    'pass_off_sf': 0.08,
    'rush_def_reversion': 0.40
})

# Save updated config
config.to_file()
```

### Parameter Optimization

The package provides two specialized optimizers:

#### UnitOptimizer - Optimize unit model parameters
```python
from nfelounits import DataLoader, DataSplitter, UnitOptimizer, ModelConfig

# Load and split data
loader = DataLoader()
splitter = DataSplitter(loader.unit_games)
labeled_data = splitter.label_train_test(n_test_seasons=5)

# Load config
config = ModelConfig.from_file()

# Optimize unit parameters to minimize MAE
optimizer = UnitOptimizer(
    data=labeled_data,
    config=config,
    subset=['pass_off_sf', 'pass_def_sf']  # Specific params to optimize
)

optimizer.optimize(save_result=True, update_config=True)
print(f"Best MAE: {optimizer.optimization_results['avg_mae']:.4f}")
```

#### EloOptimizer - Optimize Elo translation coefficients
```python
from nfelounits import EloOptimizer, ModelConfig

# Optimize Elo coefficients to minimize log loss
optimizer = EloOptimizer(
    data=labeled_data,
    config=config,
    calculate_test=True  # Track test set performance
)

optimizer.optimize(save_result=True, update_config=True)
print(f"Train log loss: {optimizer.optimization_results['train_log_loss']:.4f}")
print(f"Test log loss: {optimizer.optimization_results['test_log_loss']:.4f}")
```

### Performance Grading
```python
from nfelounits import UnitGrader

# Grade team units
grader = UnitGrader(model.teams)
grades = grader.get_grades(season=2023, week=10)
```

## Configuration Parameters

The model uses parameters organized into two main sections:

### Unit Config - Unit performance tracking
**Smoothing Factors** (EWMA update rate)
- `unit_config.pass_off_sf`, `unit_config.pass_def_sf`
- `unit_config.rush_off_sf`, `unit_config.rush_def_sf`
- `unit_config.st_off_sf`, `unit_config.st_def_sf`
- `unit_config.league_pass_sf`, `unit_config.league_rush_sf`, `unit_config.league_st_sf`

**Reversion Rates** (Offseason regression)
- `unit_config.pass_off_reversion`, `unit_config.pass_def_reversion`
- `unit_config.rush_off_reversion`, `unit_config.rush_def_reversion`
- `unit_config.st_off_reversion`, `unit_config.st_def_reversion`
- `unit_config.league_pass_reversion`, `unit_config.league_rush_reversion`, `unit_config.league_st_reversion`

**Home Field Advantage Shares**
- `unit_config.pass_hfa_share`, `unit_config.rush_hfa_share`, `unit_config.st_hfa_share`

### Elo Config - Win probability translation
**Elo Translation Coefficients** (Unit EPA → Elo conversion)
- `elo_config.pass_off_coef`, `elo_config.pass_def_coef`
- `elo_config.rush_off_coef`, `elo_config.rush_def_coef`
- `elo_config.st_off_coef`, `elo_config.st_def_coef`

All parameters include optimization bounds (`opti_min`, `opti_max`) for automated tuning.

## Data Requirements

The package expects play-by-play data with the following key fields:
- `game_id`, `season`, `week`
- `posteam`, `defteam`, `home_team`, `away_team`
- `play_type` (pass, run, punt, kickoff, field_goal)
- `epa` (Expected Points Added)
- `passer_id` (for QB adjustments)

Compatible with data from [nflfastR](https://www.nflfastr.com/) and similar sources.

## API Reference

### Classes

#### `DataLoader`
Loads and prepares play-by-play data for modeling.

**Methods:**
- `__init__()` - Initialize and load data
- Properties: `pbp`, `unit_games`

#### `DataSplitter`
Labels data for train/test splits while maintaining full dataset for EWMA continuity.

**Methods:**
- `label_train_test(n_test_seasons, exclude_first_season)` - Label data with 'data_set' column
- `label_by_season(train_through_season, exclude_first_season)` - Label by specific season cutoff

#### `UnitModel`
Core model for tracking unit performance over time.

**Methods:**
- `__init__(data, config)` - Initialize model
- `run()` - Execute model through all games
- `get_results_df()` - Export results as DataFrame
- `get_team(team_code)` - Access specific team's units

#### `ModelConfig`
Configuration container with parameter management and nested structure support.

**Methods:**
- `from_file(filepath)` - Load from JSON
- `to_file(filepath)` - Save to JSON
- `update_config(updates)` - Update parameter values
- Property: `values` - Nested dict of parameter values
- Property: `params` - Dict of ModelParam objects with metadata

#### `EloTranslator`
Translates unit EPA ratings to Elo scores and win probabilities.

**Methods:**
- `__init__(elo_config)` - Initialize with Elo coefficients
- `get_team_elo(team_units, league_baseline)` - Calculate team Elo from units
- `get_win_probability(home_elo, away_elo)` - Calculate win probability

#### `BaseOptimizer` (Abstract)
Base class for all parameter optimizers using scipy.minimize.

**Abstract Methods:**
- `objective(x)` - Calculate optimization metric (implemented by subclasses)
- `get_metric_name()` - Return name of metric being optimized

**Methods:**
- `optimize(save_result, update_config)` - Run optimization
- `get_best_record()` - Get best parameter set found

#### `UnitOptimizer`
Optimizes unit model parameters to minimize MAE on unit performance predictions.

**Methods:**
- `__init__(data, config, subset, ...)` - Initialize optimizer
- `objective(x)` - Calculate MAE across all units
- `get_metric_name()` - Returns 'avg_mae'

#### `EloOptimizer`
Optimizes Elo translation coefficients to minimize log loss on win predictions.

**Methods:**
- `__init__(data, config, subset, calculate_test, ...)` - Initialize optimizer
- `objective(x)` - Calculate log loss on win probabilities
- `get_metric_name()` - Returns 'train_log_loss'
- `calculate_log_loss(results, data_set)` - Helper to compute log loss

#### `UnitGrader`
Performance evaluation and grading.

**Methods:**
- `__init__(teams)` - Initialize with team data
- `get_grades(season, week)` - Calculate unit grades

## Examples

### Example 1: Simple Model Run
```python
from nfelounits import DataLoader, UnitModel, ModelConfig

loader = DataLoader()
config = ModelConfig.from_file()

model = UnitModel(loader.unit_games, config.values)
model.run()

results = model.get_results_df()
print(results.head())
```

### Example 2: Win Probability Predictions
```python
from nfelounits import DataLoader, UnitModel, ModelConfig

loader = DataLoader()
config = ModelConfig.from_file()

model = UnitModel(loader.unit_games, config.values)
model.run()

results = model.get_results_df()

# View win probabilities
home_games = results[results['is_home'] == True]
print(home_games[['team', 'opponent', 'win_prob', 'result']])
```

### Example 3: Custom Configuration
```python
from nfelounits import ModelConfig

config = ModelConfig.from_file()

# Customize parameters (using nested structure)
config.update_config({
    'unit_config.pass_off_sf': 0.10,
    'unit_config.rush_off_sf': 0.05,
    'elo_config.pass_off_coef': 15.0
})

# Use in model
model = UnitModel(data, config.values)
```

### Example 4: Unit Parameter Optimization
```python
from nfelounits import DataLoader, DataSplitter, UnitOptimizer, ModelConfig

loader = DataLoader()
splitter = DataSplitter(loader.unit_games)
labeled_data = splitter.label_train_test(n_test_seasons=5, exclude_first_season=True)

config = ModelConfig.from_file()

# Optimize passing unit parameters
optimizer = UnitOptimizer(
    data=labeled_data,
    config=config,
    subset=['unit_config.pass_off_sf', 'unit_config.pass_def_sf', 
            'unit_config.pass_off_reversion', 'unit_config.pass_def_reversion']
)

optimizer.optimize(save_result=True, update_config=True)
print(f"Optimized MAE: {optimizer.optimization_results['avg_mae']:.4f}")
```

### Example 5: Elo Coefficient Optimization
```python
from nfelounits import EloOptimizer, ModelConfig

# Optimize Elo translation to improve win predictions
optimizer = EloOptimizer(
    data=labeled_data,
    config=config,
    calculate_test=True  # Calculate test set performance
)

optimizer.optimize(save_result=True, update_config=False)

# View results
print(f"Train log loss: {optimizer.optimization_results['train_log_loss']:.6f}")
print(f"Test log loss:  {optimizer.optimization_results['test_log_loss']:.6f}")

# View optimal coefficients
for coef in ['pass_off_coef', 'rush_off_coef', 'st_off_coef']:
    key = f'elo_config.{coef}'
    print(f"{coef}: {optimizer.optimization_results[key]:.4f}")
```

## Project Structure

```
nfelounits/
├── Data/
│   ├── DataLoader.py      # Data loading and preparation
│   └── DataSplitter.py    # Train/test labeling utilities
├── Model/
│   ├── UnitModel.py       # Main model implementation
│   ├── Team.py            # Team container with units
│   ├── Unit.py            # Individual unit tracking
│   ├── EloTranslator.py   # Unit EPA → Elo → Win probability
│   ├── LeagueBaseline.py  # League-wide EPA tracking
│   ├── GameContext.py     # Game context adjustments
│   └── Types.py           # Enum definitions
├── Optimizer/
│   ├── BaseOptimizer.py   # Abstract base optimizer class
│   ├── UnitOptimizer.py   # Optimize unit parameters (MAE)
│   ├── EloOptimizer.py    # Optimize Elo coefficients (log loss)
│   └── ModelConfig.py     # Configuration management
├── Performance/
│   └── UnitGrader.py      # Performance evaluation
├── Utilities/
│   ├── IdConverters.py    # ID format conversion helpers
│   ├── CurveUtils.py      # Activation funcs for adjustments
│   └── EloUtils.py        # Elo calculation utilities
└── config.json            # Default model parameters
```

## Extending the Package

### Creating Custom Optimizers

The `BaseOptimizer` class makes it easy to create new optimizers for different objectives:

```python
from nfelounits.Optimizer import BaseOptimizer
from typing import List

class MyCustomOptimizer(BaseOptimizer):
    def get_metric_name(self) -> str:
        return 'my_metric'
    
    def objective(self, x: List[float]) -> float:
        # Increment round counter
        self.round_number += 1
        
        # Denormalize parameters and run model
        config = self.denormalize_optimizer_values(x)
        model = UnitModel(self.data, config)
        model.run()
        results = model.get_results_df()
        
        # Calculate your custom metric
        my_metric = calculate_my_metric(results)
        
        # Store record
        self.optimization_records.append({
            'round': self.round_number,
            'my_metric': my_metric,
            **{f: self.denormalize_param(x[i], self.config.params[f]) 
               for i, f in enumerate(self.features)}
        })
        
        # Save periodically
        if self.round_number % 100 == 0:
            self.save_progress()
        
        return my_metric  # Value to minimize
```
