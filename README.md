# nfelounits

A Python package for decomposing NFL team performance into measurable unit ratings using EPA (Expected Points Added) from play-by-play data.

## Overview

`nfelounits` breaks down team performance into three distinct units:
- **Passing** (QB/Passing offense and pass defense)
- **Rushing** (Rush offense and rush defense)  
- **Special Teams** (ST offense and ST defense)

The model uses EWMA (Exponentially Weighted Moving Average) to track unit performance over time, with offseason regression and contextual game adjustments.


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

# Get results
results = model.get_results_df()
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
```python
from nfelounits import ConfigOptimizer, ModelConfig

# Load data and config
loader = DataLoader()
config = ModelConfig.from_file()

# Create optimizer
optimizer = ConfigOptimizer(
    data=loader.unit_games,
    config=config,
    objective_name='avg_mae',
    subset=['pass_off_sf', 'pass_def_sf']  # Optimize specific params
)

# Run optimization
optimizer.optimize(save_result=True, update_config=True)

# Get results
best_params = optimizer.optimization_results
```

### Performance Grading
```python
from nfelounits import UnitGrader

# Grade team units
grader = UnitGrader(model.teams)
grades = grader.get_grades(season=2023, week=10)
```

## Configuration Parameters

The model uses 18 core parameters organized by unit type:

### Smoothing Factors (EWMA update rate)
- `pass_off_sf`, `pass_def_sf`
- `rush_off_sf`, `rush_def_sf`
- `st_off_sf`, `st_def_sf`
- `league_pass_sf`, `league_rush_sf`, `league_st_sf`

### Reversion Rates (Offseason regression)
- `pass_off_reversion`, `pass_def_reversion`
- `rush_off_reversion`, `rush_def_reversion`
- `st_off_reversion`, `st_def_reversion`
- `league_pass_reversion`, `league_rush_reversion`, `league_st_reversion`

### Home Field Advantage Shares
- `pass_hfa_share`, `rush_hfa_share`, `st_hfa_share`

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
Splits data for cross-validation and testing.

**Methods:**
- `split_by_season(n_splits)` - Chronological splits by season
- `get_train_test(test_season)` - Train/test split by season

#### `UnitModel`
Core model for tracking unit performance over time.

**Methods:**
- `__init__(data, config)` - Initialize model
- `run()` - Execute model through all games
- `get_results_df()` - Export results as DataFrame
- `get_team(team_code)` - Access specific team's units

#### `ModelConfig`
Configuration container with parameter management.

**Methods:**
- `from_file(filepath)` - Load from JSON
- `to_file(filepath)` - Save to JSON
- `update_config(updates)` - Update parameter values
- Property: `values` - Dict of parameter values

#### `ConfigOptimizer`
Automated parameter optimization using scipy.minimize.

**Methods:**
- `__init__(data, config, ...)` - Initialize optimizer
- `optimize(save_result, update_config)` - Run optimization
- `get_best_record()` - Get best parameter set found

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

### Example 2: Custom Configuration
```python
from nfelounits import ModelConfig

config = ModelConfig.from_file()

# Customize parameters
config.update_config({
    'pass_off_sf': 0.10,
    'rush_off_sf': 0.05,
    'pass_off_reversion': 0.35
})

# Use in model
model = UnitModel(data, config.values)
```

### Example 3: Parameter Optimization
```python
from nfelounits import DataLoader, ConfigOptimizer, ModelConfig

loader = DataLoader()
config = ModelConfig.from_file()

# Optimize passing unit parameters
optimizer = ConfigOptimizer(
    data=loader.unit_games,
    config=config,
    subset=['pass_off_sf', 'pass_def_sf', 
            'pass_off_reversion', 'pass_def_reversion']
)

optimizer.optimize(save_result=True, update_config=True)
```

## Project Structure

```
nfelounits/
├── Data/
│   ├── DataLoader.py      # Data loading and preparation
│   └── DataSplitter.py    # Train/test splitting utilities
├── Model/
│   ├── UnitModel.py       # Main model implementation
│   ├── Team.py            # Team container with units
│   ├── Unit.py            # Individual unit tracking
│   ├── LeagueBaseline.py  # League-wide EPA tracking
│   └── Types.py           # Enum definitions
├── Optimizer/
│   ├── ConfigOptimizer.py # Parameter optimization
│   └── ModelConfig.py     # Configuration management
├── Performance/
│   └── UnitGrader.py      # Performance evaluation
├── Utilities/
│   └── IdConverters.py    # ID format conversion helpers
└── config.json            # Default model parameters
```
