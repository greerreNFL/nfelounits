# Processing

Output processors that transform model results into analysis-ready formats.

## Why Processors?

The model's raw output (`TeamGameRecord`) contains everything needed for predictions, but analysis often requires derived metrics and transformations:

- **Accessibility** - The model's raw output is a typed dictionary, which is not easily ingested by other datasets. The processed files make joining much more convenient.
- **Normalization** - Compare a 2024 rating to a 2010 rating fairly
- **Aggregation** - Season-to-date averages to make "through X weeks" comparisons
- **Derived metrics** - Units Values are meant to be predictive, not descriptive. However using concepts like averaging Observed - Expected, allow us to create descriptive metrics as well. Think of these as more in line with something like DVOA

Processors encapsulate these transformations so they're consistent and reusable.

## Overview

All processors inherit from `BaseProcessor` and produce CSV files in the `Output/` directory.

```python
from nfelounits.Processing import (
    UnitTeamsProcessor,
    UnitTeamsNormalizationProcessor,
    ValueCreatedProcessor,
    OpponentFacedProcessor,
    EloProcessor
)

# After running model
processor = ValueCreatedProcessor(model.team_game_records)
processor.save()  # Saves to Output/value_created.csv
```

## Available Processors

### UnitTeamsProcessor

Raw unit values (pre and post game).

**Output**: `units.csv`

| Column | Description |
|--------|-------------|
| `season`, `week`, `team`, `opponent` | Game identifiers |
| `{unit}_{side}_value_pre` | Pre-game unit rating |
| `{unit}_{side}_value_post` | Post-game unit rating |

### UnitTeamsNormalizationProcessor

Era-adjusted unit values with percentiles.

**Output**: `units_normalized.csv`

| Column | Description |
|--------|-------------|
| `{unit}_{side}_value_pre_normalized` | Z-score within era |
| `{unit}_{side}_value_pre_percentile` | Percentile within era |

**Why Normalization?**

Raw unit values aren't comparable across eras because:
- League-wide passing EPA has increased over time (rule changes favor passing)
- Team strategies evolve (more passing attempts, fewer rushes)
- A +2 EPA pass offense in 2010 might be equivalent to +1 EPA today

Z-scores and percentiles put all values on a common scale.

**Why Group by Week?**

Normalization uses rolling mean/std calculated per week number because:
- Week 1 ratings are based on ~1 game of data (high uncertainty)
- Week 17 ratings are based on ~17 games (lower uncertainty)
- Comparing a Week 1 z-score to a Week 17 z-score is misleading without adjustment

### ValueCreatedProcessor

Season-to-date average value created per unit.

**Output**: `value_created.csv`

**Value Created** measures performance above/below what opponent + context would predict for an average team:
- **Offense**: `(observed - league_avg) - (context - opponent_def)`
- **Defense**: `(opponent_off + context) - (observed - league_avg)`
- **Positive = good performance**

| Column | Description |
|--------|-------------|
| `{unit}_{side}_value_created` | Season-to-date expanding mean |
| `{unit}_{side}_value_created_normalized` | Era-adjusted z-score |

**Why Value Created vs Unit Values?**

Unit Values are a rolling average of value created, tuned for predictive accuracy. As a result, they are not descriptive of how a team has achieved their performance. Value Created, on the other hand, is a simple seasonal mean, which is far more descriptive of how a team has performed. Unlike other models, which are geared towards either prediction or description, by including both values, the Unit Model framework can do both.

### OpponentFacedProcessor

Season-to-date average opponent difficulty faced.

**Output**: `faced.csv`

**Opponent Faced** measures schedule difficulty adjusted for context:
- **Offense**: `opponent_def - (hfa_adj + weather_adj)`
- **Defense**: `opponent_off + (qb_adj + hfa_adj + weather_adj)`
- **Positive = harder schedule**

| Column | Description |
|--------|-------------|
| `{unit}_{side}_faced` | Season-to-date expanding mean |
| `{unit}_{side}_faced_normalized` | Era-adjusted z-score |

### EloProcessor

Pre-game Elo ratings and win probabilities.

**Output**: `elo.csv`

| Column | Description |
|--------|-------------|
| `elo` | Team's composite Elo rating |
| `qb_adj` | QB adjustment in Elo points |
| `context_adj` | Weather adjustment in Elo points |
| `win_prob` | Pre-game win probability |

## BaseProcessor

Abstract base class for creating custom processors.

```python
from nfelounits.Processing import BaseProcessor

class MyProcessor(BaseProcessor):
    def get_filename(self) -> str:
        return 'my_output.csv'
    
    def get_columns(self) -> List[str]:
        return ['season', 'week', 'team', 'my_metric']
    
    def process(self) -> pd.DataFrame:
        # Transform self.records into output format
        df = pd.DataFrame(self.records)
        df['my_metric'] = calculate_metric(df)
        return df[self.get_columns()]
```

## Utilities

### Forward Fill Weeks

Handles bye weeks by forward-filling team data to maintain weekly continuity.

```python
from nfelounits.Processing.Utils import forward_fill_weeks
df = forward_fill_weeks(df)
```

### Normalization

Era-adjusted z-scores and percentiles.

```python
from nfelounits.Processing.Utils import normalize

df = normalize(
    df,
    columns=['value_created'],
    group_by_week=True  # Calculate stats per week number
)
```

When `group_by_week=True`, normalization accounts for sample size differences (week 1 has ~1 game per team vs week 17 with ~17 games).
