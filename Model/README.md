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
3. **Model Granularity** - With decomposed units, they become sub models that can be controlled and tuned independently, which lends itself to different regression and update treatment for teams depending on what drove their results.

## Why EWMA?

The 'true value' of a team is both recursive and changing. Our understanding of a team's value is predicated on the value of the teams they played, which is predicated on the value of the teams those teams played, and so forth. Team values also evolve and change over time. A team's true value in Week 18 is often very different than it is in Week 1.

This creates a challenge with respect to modeling true value in a given week. If a team beats expectation vs an opponent, a model's estimate of their true value should increase. If in later weeks, it is revealed that the opponent they exceeded expectation against is worse than we originally thought, we might discount the team we adjusted up. This is the recursiveness of the problem, and it is handled well by Maximum Likelihood Estimators like a Bradley-Terry Model, where ratings are those that are most likely to explain all observations.

While this can correct adjustments made from bad priors as more data is collected, it ignores that team ratings do _legitimately_ change over time. Sequential rating systems like Elo or an EWMA are better suited to problems where true value changes as they only adjust off of the previous week's rating based on the immediately new information they received from the current week. They are more of a snapshot than a landscape.

For the NFL, Elo and EWMA do seem to work better out of the box for more applications as the non-stationary dynamic is stronger than the system dynamic. So to start, Units are derived from an EWMA. That will evolve over time.

## Key Dynamics

### The Adjustment Ethos

The model draws from a Bayesian ethos of expect, observe, and update. A unit's EWMA is not a rolling raw EPA -- it is a rolling prediction error, which can be interpreted as an expectation above or below average.

```
adjustment = observed_epa - expected_epa
new_value = old_value + smoothing_factor * adjustment
```

If a team generates +5 EPA against a bad defense in a dome (expected: +4), it would be graded as a +1 performance. Unit values are comprised of rolling differences between what we would expect a league average unit to do in similar situations and what was actually observed.

### League Baselines

NFL EPA is not stationary -- league passing and rushing efficiency drifts over years as rules and schemes evolve. Failing to account for this contaminates unit ratings with era effects. The model tracks league-wide EPA baselines via separate EWMA per unit type (`league_pass_sf`, `league_rush_sf`, `league_st_sf`), with offseason reversion rates. Subtracting these baselines from observed EPA centers unit updates around the current league environment.

### Location Effect

The model distributes a base location effect (derived from historical home-field advantage data) across units using learned shares (`pass_location_effect_share`, `rush_location_effect_share`, `st_location_effect_share`).

Counterintuitively, the pass and rush shares are both **negative**. The model operates on total EPA per game, not per-play efficiency. While home teams are more efficient per play, they also are more likely to have the lead, which imapacts play mix -- passing less and running more. Since rushing carries a negative expected EPA on average, more rushing attempts produce more total negative EPA. The net effect is that being home _reduces_ total pass EPA (fewer pass attempts) and total rush EPA (more carries at negative expected value). In both cases, a negative location effect share discounts the home team's expected EPA, preventing the model from interpreting play-mix-driven EPA as signal about the unit itself.

Special teams is the only unit with a positive share (~0.24), reflecting genuine home-field advantage on a per play basis without meaningful discounting to the number of special team plays.

### Weather

Wind and temperature affect unit performance through sigmoid discount curves. For each unit type, a `_wind_disc_height` and `_temp_disc_height` parameter controls the maximum EPA reduction at extreme weather. The model learns these heights via optimization; midpoints are hardcoded (wind: 18 mph, temp: 32°F).

The consistent signal across every optimization run: wind hurts passing the most (~4.1 EPA), cold moderately hurts passing (~2.2), and everything else is negligible. Rush and ST temperature effects converge to zero every time. Weather adjustments are subtracted from both expected and observed EPA, so units aren't penalized (or credited) for conditions outside their control.

### QB Adjustment

Pass offense and defense units are adjusted for quarterback quality. Each game's starting QBs have a pre-game value (from an external QB Elo model via `nfelodcm`). The QB adjustment scales the difference between a QB's value and the league average into EPA space (`qb_adj / 25`). This prevents the model from attributing a QB downgrade or upgrade to the pass unit itself -- if a team starts a backup, the expected pass EPA already reflects that.

Importantly, if the starter is playing, there is no adjustment, even if that starter's value has changed. Conceptually, this means a team's pass unit value is a reflection of their value with the starter.

### Holt Trend Smoothing

Standard EWMA reacts to performance but doesn't model momentum. Holt-style trend smoothing adds a second component that tracks the _direction_ of change. Each unit has a `_trend_sf` parameter; when non-zero, the unit's forecast is `value + trend`, allowing the model to extrapolate trajectories.

Only 3 of 6 units exhibit meaningful trend: **pass offense** (strongest, ~0.05), **rush defense** (~0.03), and **ST defense** (~0.03). This suggests passing offense momentum is real (scheme changes and adaptation, QB development, weapons, etc), and defensive improvement/decline follows trajectories for rush and ST. Pass defense doesn't trend (inherently reactive), and rush/ST offense are too stochastic for meaningful trend detection.

### Volatile Play Discounts

Turnovers carry extreme EPA values that can distort unit ratings. The model partially discounts their influence on the EWMA update (not the observed record). Three parameters control discounting: `pass_int_disc` (interceptions), `pass_qb_fumble_disc` (sack fumbles, scramble fumbles, designed QB run fumbles), and `nonqb_fumble_disc` (receiver fumbles, RB fumbles). Non-QB fumbles are the most heavily discounted -- the model treats them as mostly noise.

Instead of discounting the smoothing factor when EPA is muddled with potentially unreliable plays, we try to discount observed performance directly so as not to discount the rest of plays which
posses signal.

### Pace Tracking

Each unit tracks its team's plays-per-game via a separate EWMA (`Pace`), building a running estimate of that team's typical pace (mean and variance). When a game has an unusual play count relative to the team's norm (measured in z-scores), the smoothing factor for that game's unit update is discounted via a Lorentzian decay: `discount = 1 / (1 + (z / threshold)²)`. The `_pace_disc_threshold` parameter controls where the half-power point falls.

This prevents games with abnormal pace (ie unusual game script) from carrying full weight in the update. Pass pace tracking is the strongest signal (higher SF, wider threshold), reflecting how much pass volume varies game-to-game.

At the league level, `LeaguePace` tracks average play counts per unit type, providing regression targets and initialization values for new teams as the league play style evolves.

### Offseason Regression

Triggered on first access in a new season. All units regress toward zero (league average) at rates set by `_reversion` parameters. Different units regress at different rates -- pass defense reverts hardest (~0.43), while pass offense doesn't revert at all (0.0) once trend smoothing is in place.

Pass offense has special QB-starter regression: in addition to general reversion, it blends the current value toward the Week 1 QB starter's expected contribution (`pass_off_qb_reversion`). This accounts for the outsized impact of QB changes between seasons.

Pace also regresses toward league averages at the offseason, and trend is reset to zero (no momentum carries over between seasons).

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
3. Build game context (weather, location effect)
4. Calculate pre-game unit values and expected EPA
5. Apply volatile play discounts to EPA for unit updates
6. Calculate Elo and win probability
7. For played games: update units with observed EPA and pace
8. Update league baselines and league pace
9. Store TeamGameRecord

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

Individual unit with EWMA tracking and Holt trend.

**Key Methods:**
- `get_value(season, coach, ...)` - Get current value (handles regression and pace initialization)
- `get_expected_epa(...)` - Calculate expected EPA given opponent/context
- `update(observed_epa, ...)` - Update rating with observed performance and pace

### Pace

Per-unit EWMA tracker for plays-per-game. Provides `get_sf_discount()` for dampening updates on abnormal-pace games. Initialized from league averages, regresses toward league at offseason.

### LeagueBaseline

Tracks league-wide EPA averages per unit type via EWMA. Provides `get_avg()` for centering unit updates.

### LeaguePace

Tracks league-wide average play counts per unit type. Provides initialization values for new `Pace` instances and regression targets at offseason.

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
- **Weather** - Wind and temperature sigmoid curves reducing expected EPA
- **Location Effect** - Distributed across units with learned shares (negative for pass/rush, positive for ST)

```python
context = GameContext(game_id, config, location_effect_base, temp, wind)
weather_adj = context.weather_adj('pass')  # Negative for bad weather
location_adj = context.location_effect_adj('pass', is_home=True)
```

## State

### UnitModelStateManager

Handles serialization/deserialization of model state to JSON.

Persists:
- All Team objects (with unit values, trends, and pace)
- League baselines
- League QB average
- League pace
- TeamGameRecords processed so far

Enables incremental updates when new games are played.
