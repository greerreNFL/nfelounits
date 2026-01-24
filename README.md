# nfelounits

A Python package for decomposing NFL team performance into measurable unit ratings using EPA (Expected Points Added) from play-by-play data, with translation to Elo values for win probability predictions.

## Overview

`nfelounits` breaks down team performance into six distinct units:

| Offense | Defense |
|---------|---------|
| Pass Offense | Pass Defense |
| Rush Offense | Rush Defense |
| Special Teams Offense | Special Teams Defense |

The model uses EWMA (Exponentially Weighted Moving Average) to track unit performance over time, with:
- **Opponent adjustment** - Performance measured relative to opponent quality
- **Context adjustment** - Weather and home field advantage effects
- **Offseason regression** - Units regress toward league average between seasons
- **QB adjustment** - Pass offense accounts for quarterback quality

Unit ratings represent predicted EPA above/below average. Summing all units gives the team's estimated margin against an average opponent.

### Win Probability

The `EloTranslator` converts unit EPA ratings into Elo scores, which are then used to calculate game win probabilities.

## Installation

```bash
# Clone the repository
git clone https://github.com/greerreNFL/nfelounits.git

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
- pandas
- numpy
- scipy
- [nfelodcm](https://github.com/greerreNFL/nfelodcm)

## Quick Start

```python
from nfelounits import DataLoader, UnitModel, ModelConfig

# Load play-by-play data
loader = DataLoader()

# Load default configuration
config = ModelConfig.from_file()

# Run the model
model = UnitModel(loader.unit_games, config.values)
model.run()

# Get results
results = model.get_results_df()
print(results[['team', 'opponent', 'win_prob', 'pass_off_value_pre']])
```

### Running All Processors

```python
from nfelounits import run

# Run model and save all output files
run()
```

This produces:
- `Output/units.csv` - Raw unit values
- `Output/units_normalized.csv` - Era-adjusted unit values
- `Output/value_created.csv` - Performance vs expectation
- `Output/faced.csv` - Schedule difficulty
- `Output/elo.csv` - Elo ratings and win probabilities

## Model Performance

Validated against established baselines over 5,000+ games:

| Model | Log Loss | Accuracy |
|-------|----------|----------|
| Unit Model Elo | 0.623 | 64.3% |
| 538 Base Elo | 0.629 | 64.3% |
| 538 QB Elo | 0.622 | 64.9% |
| Market (Vegas) | 0.609 | 66.5% |

See [Analytics/README.md](Analytics/README.md) for detailed accuracy analysis.

## Project Structure

```
nfelounits/
├── Analytics/          # Model validation scripts
│   └── README.md
├── Data/               # Data loading and preparation
│   └── README.md
├── Model/              # Core model implementation
│   ├── Entities/       # Team, Unit, Types
│   ├── Mechanics/      # EloTranslator, GameContext
│   ├── State/          # State persistence
│   └── README.md
├── Optimizer/          # Configuration and parameter tuning
│   └── README.md
├── Output/             # Generated CSV files
├── Processing/         # Output transformations
│   └── README.md
├── Performance/        # Grading utilities
│   └── README.md
├── Scripts/            # Convenience functions
│   └── README.md
├── Tests/              # Unit tests
├── Utilities/          # Helper functions
├── config.json         # Model parameters
└── requirements.txt    # Dependencies
```

## Documentation

Each module has its own README with detailed API documentation:

- [**Analytics**](Analytics/README.md) - Model validation and accuracy metrics
- [**Data**](Data/README.md) - DataLoader and DataSplitter usage
- [**Model**](Model/README.md) - UnitModel, entities, and mechanics
- [**Optimizer**](Optimizer/README.md) - Configuration and parameter optimization
- [**Performance**](Performance/README.md) - Unit-level grading and metrics
- [**Processing**](Processing/README.md) - Output processors and utilities
- [**Scripts**](Scripts/README.md) - Convenience functions for common workflows

## Key Concepts

### Unit Value

A unit's value represents its expected EPA contribution relative to league average:
- **Positive** = better than average
- **Negative** = worse than average
- Range typically -3 to +3 EPA per game

### Value Created

Performance above/below what opponent + context would predict:
- **Offense**: `(observed - league_avg) - (context - opponent_def)`
- **Defense**: `(opponent_off + context) - (observed - league_avg)`

### Opponent Faced

Schedule difficulty adjusted for context:
- **Positive** = harder than average schedule
- **Negative** = easier than average schedule

## Configuration

Model parameters are stored in `config.json` with two sections:

- **unit_config** - EWMA smoothing factors, reversion rates, weather adjustments
- **elo_config** - Coefficients for translating unit EPA to Elo

See [Optimizer/README.md](Optimizer/README.md) for parameter details and tuning.

## Examples

### Get Team Ratings

```python
from nfelounits import DataLoader, UnitModel, ModelConfig

loader = DataLoader()
config = ModelConfig.from_file()
model = UnitModel(loader.unit_games, config.values)
model.run()

# Access a specific team
kc = model.teams['KC']
print(f"KC Pass Offense: {kc.pass_off.value:.2f}")
print(f"KC Rush Defense: {kc.rush_def.value:.2f}")
```

### Optimize Parameters

```python
from nfelounits import DataLoader, DataSplitter, UnitOptimizer, ModelConfig

loader = DataLoader()
splitter = DataSplitter(loader.unit_games)
labeled = splitter.label_train_test(n_test_seasons=5)

config = ModelConfig.from_file()
optimizer = UnitOptimizer(data=labeled, config=config)
optimizer.optimize(save_result=True, update_config=True)
```

### Process Custom Output

```python
from nfelounits.Processing import BaseProcessor

class MyProcessor(BaseProcessor):
    def get_filename(self) -> str:
        return 'my_output.csv'
    
    def get_columns(self):
        return ['season', 'week', 'team', 'win_prob']
    
    def process(self):
        df = pd.DataFrame(self.records)
        return df[self.get_columns()]
```

## Testing

```bash
python -m pytest Tests/
```

## License

This project is provided for research and educational purposes.
