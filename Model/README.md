# Model

Core model implementation for tracking NFL team unit performance.

## Overview

The Unit Model decomposes team performance into six units:
- **Pass Offense** / **Pass Defense**
- **Rush Offense** / **Rush Defense**
- **Special Teams Offense** / **Special Teams Defense**

Each unit tracks EPA (Expected Points Added) using EWMA (Exponentially Weighted Moving Average) with opponent adjustment and offseason regression.

## Why Unit Decomposition?

Traditional team ratings (like Elo) treat teams as monolithic entities. Decomposing into units provides:

1. **Signal extraction** - A team's passing game might be elite while their run defense is poor; averaging hides this
2. **Matchup analysis** - Unit ratings enable predicting how specific matchups will play out
3. **Model Granualarity** - With decomposed units, they become sub models that can be controlled and tuned independently, which lends itself to different regression and update treatment for teams depending on what drove their results.

## Why EWMA?

The 'true value' of a team is both recursive and changing. Our understanding of a team's value is predicated on the value of the teams they played, which is predicated on the value of the teams those teams played, and so forth. Team values also evolve and change over time. A team's true value in Week 18 is often very different than it is in Week 1.

This creates a challenge with respect to modeling true value in a given week. If a team beats expectation vs an opponent, a model's estiamte of their true value should increase. If in later weeks, it is revealed that the opponent they exceeded expectation against is worse than we originally thought, we might discount the team we adjusted up. This is the recursiveness of the problem, and it is handled well by Maximum Likelihood Estimators like a Bradley-Terry Model, where ratings are those those that are most likely to explain all observations.

While this can correct adjustments made from bad priors as more data is collected, it ignores that team ratings do _legitimately_ change over time. Sequential rating systems like Elo or an EWMA are better suited to problems where true value changes as they only adjust off of the previous week's rating based on the immediately new information they received from the current week. They are more of a snapshot than a landscape.

For the NFL, Elo and EWMA do seem to work better out of the box for more applications as the non-stationary dynamic is stronger than the system dynamic. So to start, Units are derived from an EWMA. That will evolve over time.

## UnitModel

Main model class that processes games chronologically.

### Usage

```python
from nfelounits import DataLoader, UnitModel, ModelConfig

loader = DataLoader()
config = ModelConfig.from_file()

model = UnitModel(loader.unit_games, config.values)
model.run()

# Get results
results = model.get_results_df()
team_ratings = model.teams  # Dict of Team objects
```

### Processing Flow

For each game:
1. Get/create team objects (with regression if new season)
2. Calculate QB adjustments
3. Build game context (weather, HFA)
4. Calculate pre-game unit values and expected EPA
5. Calculate Elo and win probability
6. For played games: update units with observed EPA
7. Store TeamGameRecord

### State Persistence

```python
# Save state after processing
model.save_to_state('path/to/state.json')

# Load state and continue from checkpoint
model2 = UnitModel(new_games, config.values)
model2.load_from_state('path/to/state.json')
model2.run()  # Only processes games not in state
```

## Entities

### Team

Container for a team's six units plus QB tracker.

```python
team = model.get_team('KC')
team.pass_off.value  # Pass offense rating
team.rush_def.value  # Rush defense rating
team.qb.starter_value  # Current starter's value
```

### Unit

Individual unit with EWMA tracking.

**Key Methods:**
- `get_value(season, coach, ...)` - Get current value (handles regression)
- `get_expected_epa(...)` - Calculate expected EPA given opponent/context
- `update(observed_epa, ...)` - Update rating with observed performance

**The Adjustment Ethos**

Even though models use a simple EWMA framework, they draw from a bayesian ethos of expect, observe, and update. A unit's EWMA is not a rolling raw EPA, it is a rolling prediction error, which can be interpretted as an expectation above average.

```
adjustment = observed_epa - expected_epa
new_value = old_value + smoothing_factor * adjustment
```

If a team has +5 EPA against a bad defense in a dome (expected: 4), it would be graded as a +1 performance. In this way, Unit values are comprised of rolling differences between what we would expect a league average unit to do in similar situations.

**Regression:**
- Triggered on first access in a new season
- Reverts toward league average based on `reversion` parameter
- Pass offense also regresses toward Week 1 QB starter value

**Why Offseason Regression?**

NFL rosters change significantly between seasons. Regression accounts for:
- Player departures/arrivals
- Coaching changes
- Scheme adjustments
- Natural performance mean reversion

Different units regress at different rates - pass offense (QB-dependent) tends to be more stable than special teams.

### TeamGameRecord

TypedDict containing all per-team, per-game outputs:

| Category | Fields |
|----------|--------|
| Identifiers | `game_id`, `season`, `week`, `team`, `opponent`, `is_home` |
| QB | `qb_value`, `qb_adj`, `coach` |
| Pre-game values | `{unit}_{side}_value_pre` |
| Elo | `elo`, `context_adj`, `win_prob` |
| Expected/Observed | `{unit}_{side}_expected`, `{unit}_{side}_observed` |
| Value Created | `{unit}_{side}_value_created` |
| Opponent Faced | `{unit}_{side}_faced` |
| League Averages | `{unit}_league_avg` |
| Post-game values | `{unit}_{side}_value_post` |

## Mechanics

### EloTranslator

Converts unit EPA ratings to Elo scores and win probabilities.

```python
translator = EloTranslator(elo_config)
team_elo = translator.translate_to_elo(team)
win_prob = calculate_win_probability(home_elo - away_elo)
```

Each unit has a coefficient controlling its contribution to overall Elo.

**Why Translate to Elo?**

EPA is on a "points per game" scale, but the nfelo model, which is the ecosystem in which Units sit, is Elo denominated. Thus, a team's unit values need to be translated into a single Elo rating. The translation:
1. Weights each unit by its empirical importance (coefficients learned via EloOptimizer)
2. Sums to a single Elo rating

### GameContext

Handles game-specific adjustments:
- **Weather** - Wind and temperature effects on passing/kicking
- **Home Field Advantage** - Distributed across units

```python
context = GameContext(game_id, config, hfa_base, temp, wind)
weather_adj = context.weather_adj('pass')  # Negative for bad weather
hfa_adj = context.hfa_adj('pass', is_home=True)
```

## State

### UnitModelStateManager

Handles serialization/deserialization of model state to JSON.

Persists:
- All Team objects (with unit values)
- League baselines
- League QB average
- TeamGameRecords processed so far

Enables incremental updates when new games are played.
