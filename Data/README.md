# Data

Data loading and preparation utilities for the Unit Model.

## DataLoader

Loads play-by-play data from [nfelodcm](https://github.com/greerreNFL/nfelodcm) and prepares it for model consumption.

### Usage

```python
from nfelounits import DataLoader

loader = DataLoader()

# Access prepared game-level data
unit_games = loader.unit_games  # Ready for UnitModel

# Access raw datasets
pbp = loader.pbp        # Play-by-play data
games = loader.games    # Schedule data
hfa = loader.hfa        # Home field advantage adjustments
qbelo = loader.qbelo    # QB Elo data
qb_meta = loader.qb_meta  # QB metadata
```

### Data Pipeline

1. **Load datasets** from nfelodcm (pbp, games, hfa, qbelo, qb_meta)
2. **Build games** - Filter to regular season, include played games + next unplayed week
3. **Parse PBP** - Categorize plays into units (pass, rush, special teams)
4. **Aggregate** - Sum EPA by game and unit
5. **Add adjustments** - Join location effect (HFA) and QB data

### Unit Definitions

| Unit | Included Plays |
|------|----------------|
| **Pass** | Pass plays, QB scrambles, designed QB runs |
| **Rush** | Non-QB rushing plays |
| **Special Teams** | Punts, kickoffs, field goals, extra points |

### Output Schema

The `unit_games` DataFrame contains:

| Column | Description |
|--------|-------------|
| `game_id` | Unique game identifier |
| `season`, `week` | Game timing |
| `home_team`, `away_team` | Team abbreviations |
| `home_pass_epa`, `away_pass_epa` | Pass unit EPA totals |
| `home_rush_epa`, `away_rush_epa` | Rush unit EPA totals |
| `home_st_epa`, `away_st_epa` | Special teams EPA totals |
| `home_pass_int_epa`, `away_pass_int_epa` | EPA from interceptions (for volatile play discounting) |
| `home_pass_qb_fumble_epa`, `away_pass_qb_fumble_epa` | EPA from QB fumbles (sack, scramble, designed run) |
| `home_pass_nonqb_fumble_epa`, `away_pass_nonqb_fumble_epa` | EPA from non-QB fumbles on pass plays |
| `home_rush_nonqb_fumble_epa`, `away_rush_nonqb_fumble_epa` | EPA from fumbles on rush plays |
| `home_pass_plays`, `away_pass_plays` | Pass play counts (for pace tracking) |
| `home_rush_plays`, `away_rush_plays` | Rush play counts (for pace tracking) |
| `home_st_plays`, `away_st_plays` | Special teams play counts (for pace tracking) |
| `temp`, `wind` | Weather conditions |
| `home_coach`, `away_coach` | Head coaches |
| `hfa_base` | Location effect base factor |
| `home_qb_value`, `away_qb_value` | Pre-game QB values |
| `result` | Point differential (home perspective), NaN for unplayed |

## DataSplitter

Labels data for train/test splits while maintaining full dataset for EWMA continuity.

### Usage

```python
from nfelounits import DataSplitter

splitter = DataSplitter(loader.unit_games)

# Split by number of test seasons
labeled = splitter.label_train_test(n_test_seasons=5, exclude_first_season=True)

# Or split by specific season cutoff
labeled = splitter.label_by_season(train_through_season=2019)
```

### Why Labels Instead of Splitting?

The EWMA model requires processing games chronologically - it can't skip games. The splitter adds a `data_set` column ('train', 'test', or 'excluded') so you can:

1. Run the model on the full dataset
2. Filter results for evaluation on specific subsets

### Parameters

- `n_test_seasons` - Number of most recent seasons for test set
- `exclude_first_season` - Exclude the first season (model warm-up period)
- `train_through_season` - Train on all seasons up to and including this year
