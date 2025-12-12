# Recalibration Caching Feature

This document describes an experimental feature for "recalibrating" unit ratings by computing Maximum Likelihood Estimate (MLE) priors and blending them back into the main UnitModel. The feature is implemented but did not provide meaningful predictive lift, so it is being shelved for future exploration.

**Purpose of this document**: Provide complete context for a future developer to understand the codebase changes, the problem we were solving, why the current approach didn't work, and what directions might be worth exploring.

---

## Table of Contents

1. [The Problem](#the-problem)
2. [The Recalibration Concept](#the-recalibration-concept)
3. [Why This Approach vs Bradley-Terry](#why-this-approach-vs-bradley-terry)
4. [The Runtime Challenge](#the-runtime-challenge)
5. [The Pre-Computation Insight](#the-pre-computation-insight)
6. [Architecture Overview](#architecture-overview)
7. [Key Files and Their Purposes](#key-files-and-their-purposes)
8. [Config Additions](#config-additions)
9. [Why It Didn't Work](#why-it-didnt-work)
10. [Hypotheses for Future Exploration](#hypotheses-for-future-exploration)
11. [What NOT to Try Again](#what-not-to-try-again)
12. [Usage Examples](#usage-examples)
13. [Key Code Patterns](#key-code-patterns)

---

## The Problem

### UnitModel is Directional

The UnitModel operates like an Elo rating system - it processes games sequentially and updates unit ratings based on observed performance. This creates a fundamental limitation: **bad priors propagate forward**.

Consider this scenario:
- A team enters Week 1 with a pass offense rating of 0.0 (neutral)
- They play a collection of tough opponents, but do well
- By Week 8, their pass offense rating has increased to +3.0
- As it turns out, the teams they played were actually not as tough as originally thought
- If we were to use the current ratings as the opponent adjustment, we'd see the true skill was indeed 0.0.

The problem: The model started with priors that were "behind" on the opponents faced. Since it overrepresented their true skill, the model ended up overestimating the skill of their opponents. Though we can observe the error in hindsight, it is propegated forward. Evaluations are made based on available information at the time, priors are updated, and the decision is never revisited.

### The Elo Limitation

This is a fundamental limitation of Elo-style models. They are computationally efficient and intuitive, but they:
1. Cannot revise past estimates based on new information
2. Are sensitive to initial conditions (priors)
3. May never fully converge to the "true" rating if priors were far off

The degree to which this is a problem depends on the accuracy of the initial priors.

---

## The Recalibration Concept

### What is Recalibration?

Recalibration is the idea of continuously calculating what the OPTIMAL priors would have been, given all observed data up to a point, and using those optimal priors to recalculate a rating that the main model can be nudged towards.

### MLE vs Elo

| Aspect | Elo (UnitModel) | MLE (Recalibration) |
|--------|-----------------|---------------------|
| Approach | Sequential updates | Global optimization |
| Computation | O(n) - one pass | O(n × iterations) - many passes |
| Prior sensitivity | High | None - finds best priors |
| Revision | Cannot revise | Implicitly revises by finding optimal starting point |

### The Blending Approach

Rather than replacing UnitModel ratings with MLE ratings, we BLEND them using a weighted average:

```
final_value = (1 - weight) × unit_model_value + weight × mle_value
```

The weight is calculated using an S-curve (sigmoid) that increases with the week number:
- Early season (weeks 1-4): Low weight → trust UnitModel's priors
- Mid season (weeks 8-12): Medium weight → blend both
- Late season (weeks 14+): High weight → trust MLE more

The intuition: Early in the season, there's not enough data for MLE to be reliable. Late in the season, MLE has seen enough games to find truly optimal priors.

### S-Curve Parameters

Three parameters control the blend:
- `recal_activation_height`: Maximum blend weight (0-1, typically 1.0)
- `recal_activation_midpoint`: Week where blend weight = 0.5 (typically 8)
- `recal_activation_steepness`: How quickly the curve transitions (typically 0.5)

---

## Why This Approach vs other MLE approaches (Bradley-Terry, etc.)

### Why We Chose Prior Optimization

We chose to optimize UnitModel's PRIORS rather than build a separate MLE model because:

1. **Diagnosing the Problem**: The hypothesis was that UnitModel is fundamentally sound, but hampered by suboptimal priors. If true, fixing priors, and keeping all other aspects of the model the same, should yield apples-to-apples values that are simply more accurate. A new model type could introduce fundamental differences in model behavior for specific teams or circumstances, which might make it harder to blend.

2. **Architectural Simplicity**: UnitModel is already optimized and tested. Adding a blend is simpler than maintaining two parallel models.

3. **Configuration Consistency**: All other parameters (learning rates, regression rates, etc.) remain the same. We're only adjusting the starting point.

4. **Potential for Joint Optimization**: If this worked, we could potentially optimize priors and model parameters together.

### Was This the Right Choice?

**Unclear.** The approach didn't provide lift, but we can't conclusively say something like a Bradley-Terry model would have done better. It remains a hypothesis to explore.

---

## The Runtime Challenge

### The Original Implementation Problem

The first implementation created a new UnitRecalibrator at the start of each week during `UnitModel.run()`:

```python
# Old approach (inside UnitModel.run())
for game in games:
    if new_week:
        recalibrator = UnitRecalibrator(games, config, season, week, ...)
        # This runs scipy.optimize with ~2000 iterations
        # Each iteration runs a mini UnitModel on all games up to this week
```

**The math is brutal:**
- ~460 weeks of data (2010-2024)
- Each week runs optimization with ~2000 iterations
- Each iteration runs UnitModel on games up to that week
- Result: ~2.5 hours just for recalibration computation

To OPTIMIZE the s-curve parameters, you need to run the full model many times:
- 1000+ optimization rounds × 2.5 hours = **impossible**

### Optimization Tricks We Tried

1. **Smart maxiter**: Determine how many iterations are needed to converge on 90% of the available MAE reduction in the optimizer. Determine how many iters the model can run for a given week within a time budget of X. Allow the model to optimize for the shorter of the 90% convergence iter, or the time budget iter value. Early weeks are an easier optimization problem and are able to converge to 90% well under the budget (but critically, we dont allow them to continue converging for little lift). Later weeks take longer, can achieve less lift overall, and are less important because the optimial priors they find are updated away. They are capped on run time.

2. **Warm starting**: Use current UnitModel values as initial guesses for the optimizer (faster convergence)

3. **Bounds normalization**: Normalize all parameters to 0-1 range for more stable optimization

These got per-week recalibration down to ~30 seconds, but the full model still took ~2.5 hours, making s-curve optimization impractical.

---

## The Pre-Computation Insight

### The Key Realization

The "ideal" MLE priors for a given week depend on:
- Games played (fixed data)
- Unit config parameters (learning rates, regression rates, etc.)

They do NOT depend on:
- S-curve parameters (those control the blend, not the ideal values themselves)
- The main UnitModel's current ratings

**This means ideal values can be pre-computed ONCE and cached.**

### The New Flow

**One-time computation (~2.5 hours):**
```
Run UnitModel with StateCollector
    → Capture state at each week boundary
    → For each state, run UnitRecalibrator
    → Store ideal values in CSV
```

**During s-curve optimization (instant):**
```
Load cached ideal values
Run UnitModel with RecalibrationSet
    → Look up ideal values instantly
    → Apply current s-curve parameters for blending
    → Iterate quickly to find optimal s-curve params
```

This decoupling made s-curve optimization feasible.

---

## Architecture Overview

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ONE-TIME CACHE GENERATION                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  DataLoader                                                             │
│      │                                                                  │
│      ▼                                                                  │
│  UnitModel + UnitModelStateCollector                                    │
│      │                                                                  │
│      │ (captures state at each week boundary)                           │
│      ▼                                                                  │
│  List[UnitModelState]                                                   │
│      │ (season, for_week, teams snapshot, league_baseline, league_qb)   │
│      │                                                                  │
│      ▼                                                                  │
│  UnitRecalibrator (for each state)                                      │
│      │ (scipy.optimize.minimize to find MLE priors)                     │
│      │                                                                  │
│      ▼                                                                  │
│  RecalibrationSet                                                       │
│      │                                                                  │
│      ▼                                                                  │
│  Output/recalibration_values.csv                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     RUNTIME (with cached values)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  RecalibrationSet.from_csv()                                            │
│      │                                                                  │
│      ▼                                                                  │
│  UnitModel(recalibration_set=recal_set)                                 │
│      │                                                                  │
│      │ For each game:                                                   │
│      │   1. get_recal_obj(team, unit_type, season, week)                │
│      │   2. Lookup ideal value from RecalibrationSet                    │
│      │   3. Normalize value via RecalibrationNormalizer                 │
│      │   4. Calculate blend weight via s_curve()                        │
│      │   5. Return RecalibrationObject(value, weight)                   │
│      │   6. Unit.get_value() applies blend                              │
│      │                                                                  │
│      ▼                                                                  │
│  Predictions with recalibration applied                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Week Semantics (Critical Detail)

When capturing state for recalibration, the timing matters:

```python
# In UnitModel.run()
if game_week != current_week:
    # We've encountered the FIRST game of a new week
    # BEFORE processing this game, capture state
    # State contains all games THROUGH current_week
    # This state is FOR PREDICTING game_week
    state_collector.on_week_boundary(
        season=game_season,
        for_week=game_week,  # NOT current_week
        teams=self.teams,
        ...
    )
```

Example: Processing games, `current_week = 4`, encounter week 5 game → capture state with `for_week = 5`. This state will be used to recalibrate predictions FOR week 5.

---

## Key Files and Their Purposes

### Model/RecalibrationObject.py (25 lines)

Simple dataclass holding recalibration data for blending:

```python
@dataclass
class RecalibrationObject:
    value: float   # Ideal value from MLE optimization
    weight: float  # Blend weight from s-curve (0-1)
```

Exists in its own file to avoid circular imports (Unit.py needs it for type hints).

### Model/UnitModelState.py (98 lines)

**UnitModelState** (dataclass): Captures model state at a week boundary:
- `season`: Year
- `for_week`: The week this state is FOR (about to predict)
- `teams`: Deep copy of all Team objects
- `league_baseline`: Deep copy of LeagueBaseline
- `league_qb`: Deep copy of LeagueQb

**UnitModelStateCollector**: Observer pattern class that:
- Gets called by UnitModel at week transitions
- Deep copies state to avoid mutation during later optimization
- Provides lookup by (season, for_week)

### Model/UnitRecalibrator.py (305 lines)

The optimization engine. Key responsibilities:

1. **Initialization**: Filter games to current season up to current week
2. **Feature mapping**: Create mapping from optimizer array index to (team, unit_type)
3. **Normalization**: All parameters normalized to 0-1 for optimizer stability
4. **Optimization**: Use scipy.optimize.minimize (SLSQP method) to find MLE priors
5. **Smart maxiter**: Calculate iteration budget based on time constraints

Key constants:
```python
UNIT_BOUNDS = {
    'pass_off': (-20.0, 20.0),
    'pass_def': (-15.0, 15.0),
    'rush_off': (-12.0, 12.0),
    'rush_def': (-9.0, 9.0),
    'st_off': (-4.5, 4.5),
    'st_def': (-3.5, 3.5)
}
TARGET_ROUNDS = 2000  # Rounds needed for ~95% of convergence
TIME_LIMIT_SECONDS = 100  # Max time per week
```

### Model/RecalibrationCache.py (350 lines)

**RecalibrationRecord** (dataclass): Single cache entry:
- `season`, `for_week`, `team`, `unit_type`, `ideal_value`

**RecalibrationSet**: Collection with efficient lookup:
- `from_csv()` / `to_csv()`: Persistence
- `get(season, for_week, team, unit_type)`: O(1) lookup via index
- `get_missing_weeks()`: Find gaps for incremental updates
- `upsert()`: Add/update records

**RecalibrationManager**: Orchestration:
- `generate_all()`: Full backfill (~2.5 hours)
- `update()`: Incremental update (only missing weeks)
- Coordinates UnitModel, StateCollector, and UnitRecalibrator

### Model/RecalibrationNormalizer.py (46 lines)

Scales recalibration values to match UnitModel's scale.

**The problem**: MLE values optimize on a small sample and can be extreme. UnitModel values are more conservative. Direct blending causes scale mismatch.

**The solution**: Linear normalization per unit type, varying by week:
```python
normalized_value = m(week) × recal_value
where m(week) = slope × week + intercept
```

Coefficients are computed by regressing UnitModel values against RecalibrationSet values across all weeks, then fitting m = f(week).

### Scripts/calibrate_normalizer.py (134 lines)

Computes normalization coefficients by:
1. Running UnitModel with StateCollector (no recalibration)
2. Loading RecalibrationSet
3. Joining on (season, for_week, team, unit_type)
4. For each unit type, for each week: regress model_value ~ recal_value (through origin)
5. Fit m = slope × week + intercept
6. Write coefficients to config.json

Output example:
```
pass_off: m=0.036466, b=0.103311, R²=0.9819
pass_def: m=0.026648, b=0.025358, R²=0.9971
rush_off: m=0.023419, b=-0.001415, R²=0.9952
rush_def: m=0.017425, b=0.034170, R²=0.9560
st_off: m=0.023305, b=0.016911, R²=0.9959
st_def: m=0.021346, b=0.064145, R²=0.9566
```

High R² values confirm strong linear relationship.

### local/scripts/generate_recalibration_cache.py (88 lines)

One-time script for full cache generation. Run from command line:
```bash
cd nfelounits/local/scripts
python generate_recalibration_cache.py
```

Expected runtime: ~2.5 hours for full historical backfill.

---

## Config Additions

### recal_config (s-curve parameters)

```json
"recal_config": {
    "recal_activation_height": {
        "value": 1.0,
        "description": "Height of sigmoid activation for recalibration blend",
        "opti_min": 0.0,
        "opti_max": 1.0
    },
    "recal_activation_midpoint": {
        "value": 8.0,
        "description": "Week where recalibration blend weight = 0.5",
        "opti_min": 4.0,
        "opti_max": 14.0
    },
    "recal_activation_steepness": {
        "value": 0.5,
        "description": "Steepness of sigmoid activation for recalibration blend",
        "opti_min": 0.1,
        "opti_max": 1.0
    }
}
```

### recal_normalizer (per-unit scaling)

```json
"recal_normalizer": {
    "pass_off_m": {"value": 0.036466, "description": "Slope for pass offense"},
    "pass_off_b": {"value": 0.103311, "description": "Intercept for pass offense"},
    "pass_def_m": {"value": 0.026648, "description": "..."},
    "pass_def_b": {"value": 0.025358, "description": "..."},
    // ... etc for all 6 unit types
}
```

---

## Why It Didn't Work

### What We Observed

After implementing the full system:
1. Recalibration values correlate highly with UnitModel values (validation passed)
2. Normalization coefficients have excellent R² (0.95-0.99)
3. The system runs correctly and produces sensible outputs
4. **But prediction accuracy did not improve**

### The Core Mystery

We expected: MLE-optimized priors → better predictions, especially early-mid season.

We observed: No meaningful lift at any blend level.

### Potential Explanations

1. **UnitModel is already well-optimized for the no-recalibration case**
   
   The UnitModel has been extensively tuned. Its learning rates and regression parameters may already account for prior uncertainty. By the time recalibration would kick in (week 4+), the model has already updated away from bad priors. Additionally, the recalibrator uses the same config as the UnitModel since it embeds a localized UnitModel to run the optimization. Perhaps the recalibrator could have used different config values to produce a better result not for minimizing observation error, but for predicting future unobserved games.

2. **MLE overfitting**
   
   MLE priors minimize error on PAST games. This doesn't guarantee better FUTURE predictions. As evidenced by the extreme values that needed to be normalized, MLE models suffer from a lack of sample. The MLE model could have been severly overfit, creating error when blended to the stabler UnitModel. Said another way, instead of improving the priors based on new information, we tossed them out completely and started from scratch.

4. **We took a bad MLE approach**
   
   Our UnitModel was designed for sequential updates on informed priors. So while we were trying to keep things "apples to apples", by using it to find "ideal" priors, we ignored the core problem we were trying to solve, which is finding updated priors. It was the wrong tool for the job. Instead, we _should_ have used a Bradley-Terry, or even more simply, taking the models current, sensible value and recycling it as the prior to generate an updated value.

---

## Hypotheses for Future Exploration

### 1. Joint Optimization (Most Promising, Most Difficult, likely not possible)

**Hypothesis**: UnitModel is optimized to update quickly away from bad priors because it "knows" it won't get recalibration help. If we optimize UnitModel AND recalibration together, UnitModel could learn to update more slowly, trusting that recalibration will guide it to the right place.

**Implementation approach**:
- Add recal_config parameters to the main optimization
- For each optimization iteration: run UnitModel with that iteration's recal params
- This requires running recalibration live (can't pre-compute)

**Challenge**: Removes the pre-computation benefit. Optimization might take months.

**Mitigation**: 
- Use warm-starting aggressively
- Consider Bayesian optimization instead of grid/random search
- Run on cloud compute

### 2. MLE Discounting

**Hypothesis**: MLE priors are too extreme because they overfit to small samples. We should "discount" them back towards the original prior before calculating values to blend back in.

**Implementation approach**:
```python
discounted_prior = (1 - discount) × mle_prior + discount × original_prior
```

Pre-compute caches at multiple discount levels (0%, 10%, 20%, ..., 90%).

Add `recal_discount` parameter to config. Optimizer searches over discount levels.

**Benefit**: Preserves pre-computation. Simple to implement.

**Challenge**: It's not clear that the issue is simply in the extremeness of the values. Overfitting could yield extreme, but directionally useful values, in which case, MLE discounting would be sensible. However, it could also be the case that the values overfit to the point of being useless, in which case, the optimal discount level would be 100%, and we'd be exactly where we are now.

### 3. Different Blending Strategy

**Hypothesis**: Additive blending isn't the right approach.

**Alternatives to explore**:
- Multiplicative: `final = unit_model × (1 + weight × (mle/unit_model - 1))`
- Constrained: Use MLE as a "soft constraint" that penalizes deviation
- Ensemble: Make separate predictions and combine at prediction time
- Variance-weighted: Weight by inverse variance of each estimate

### 4. Full Bradley-Terry Model

**Hypothesis**: Optimizing UnitModel priors is the wrong abstraction. Build a proper BT model.

**Implementation**:
- Each week, fit Bradley-Terry on observed unit matchups
- Use BT ratings directly (not as priors to UnitModel)
- Compare BT predictions to UnitModel predictions

**Benefit**: Clean separation. If BT works, it validates the recalibration concept. Even though the original concern was the "apples-to-oranges" nature of this approach, it could actually be a benefit. As with any diversification, problem, usefullness that doesn't correlated is actually _the best_ thing to blend with as the trade-off of muting variance and signal is favorable.

### 6. Recycling

**Hypothesis**: The UnitModel gets close to the right place by itself, but can be a little off due to bad priors. By recycling the current value, which we believe to be pretty close to true value, as the prior, we generate slightly better ratings that blend well and are not extreme.

**Implementation**:
- Each week, calculate the MAE of the system on a season-to-date basis.
- Recycle the current values as priors and run the model again. Calculate MAE again.
- Continue to do this until the MAE lift elboughs / plateaus
- Find the optimal numbers of recursions, likely decreasing by week as 1) models converge, and 2) bad priors matter less
- Optimize the blend

**Challenge:**
This is the most milquetoast approach as it would move the model the least. While computationally it would be cheap since it requires no optimization, it adds complexity, which runs the risk of greater overfitting. It's also possible that the approach would yield negative MAE left, as true value is not stationary. The value in Week 10 would be a poor prior for a team that had substantially changed in value (ie injuries, adaptation to scheme, etc.)

---

## What NOT to Try Again

### 1. On-the-Fly Recalibration Per Week

Running UnitRecalibrator during UnitModel.run() makes optimization impossible. The pre-computation approach is essential if recalibration is to be optimized.

### 2. Direct MLE Values Without Normalization

MLE values are on a different scale than UnitModel values. Direct blending without normalization produces poor results.

---

## Usage Examples

### Generate Cache (One-Time, ~2.5 hours)

```python
from nfelounits.Data import DataLoader
from nfelounits.Model import RecalibrationManager
from nfelounits.Optimizer import ModelConfig

# Load data
loader = DataLoader()
config = ModelConfig.from_file()

# Generate all recalibration values
manager = RecalibrationManager(
    games=loader.unit_games,
    config=config.values,
    filepath='Output/recalibration_values.csv',
    min_week=1
)
manager.generate_all(verbose=True)
manager.save()
```

### Use Cached Values in UnitModel

```python
from nfelounits.Model import UnitModel, RecalibrationSet
from nfelounits.Data import DataLoader
from nfelounits.Optimizer import ModelConfig

# Load everything
loader = DataLoader()
config = ModelConfig.from_file()
recal_set = RecalibrationSet.from_csv('Output/recalibration_values.csv')

# Run model with recalibration
model = UnitModel(
    games=loader.unit_games,
    config=config.values,
    recalibration_set=recal_set
)
model.run()
results = model.get_results_df()
```

### Calibrate Normalizer Coefficients

```python
from nfelounits.Scripts import calibrate_normalizer
calibrate_normalizer()  # Writes to config.json
```

### Optimize S-Curve Parameters

```python
from nfelounits.Optimizer import UnitOptimizer, ModelConfig
from nfelounits.Data import DataLoader

loader = DataLoader()
config = ModelConfig.from_file()

optimizer = UnitOptimizer(
    data=loader.unit_games,
    config=config,
    subset=['recal_config.recal_activation_height',
            'recal_config.recal_activation_midpoint',
            'recal_config.recal_activation_steepness'],
    n_iterations=500
)
optimizer.run()
```

Note: UnitOptimizer automatically loads RecalibrationSet when optimizing recal_config params.

---

## Key Code Patterns

### State Injection for Localized Models

UnitModel now supports state injection for creating "localized" models:

```python
model = UnitModel(
    games=filtered_games,
    config=config,
    teams=existing_teams,           # Pre-initialized teams
    league_baseline=existing_lb,    # Pre-initialized baseline
    league_qb=existing_qb           # Pre-initialized QB tracker
)
```

This is used by UnitRecalibrator to avoid running from scratch.

### Deep Copying State

When capturing state for later use, ALWAYS deep copy:

```python
state = UnitModelState(
    teams=copy.deepcopy(teams),
    league_baseline=copy.deepcopy(league_baseline),
    league_qb=copy.deepcopy(league_qb)
)
```

The model mutates teams during run(). Without deep copy, captured state would be corrupted.

### Normalization Pattern in Optimizer

UnitRecalibrator normalizes all parameters to 0-1 range:

```python
def normalize(self, value: float, unit_type: str) -> float:
    min_val, max_val = self.UNIT_BOUNDS[unit_type]
    return (value - min_val) / (max_val - min_val)

def denormalize(self, value: float, unit_type: str) -> float:
    min_val, max_val = self.UNIT_BOUNDS[unit_type]
    return value * (max_val - min_val) + min_val
```

This makes scipy optimization more stable across different unit scales.

### Recalibration Set Lookup Pattern

```python
record = recal_set.get(season, week, team, unit_type)
if record is not None:
    normalized = normalizer.normalize(record.ideal_value, unit_type, week)
    weight = s_curve(height, midpoint, week, 'up', steepness)
    return RecalibrationObject(value=normalized, weight=weight)
return None
```

---

## Files Changed Summary

| File | Status | Purpose |
|------|--------|---------|
| `Model/RecalibrationObject.py` | NEW | Dataclass for blend data |
| `Model/UnitModelState.py` | NEW | State capture classes |
| `Model/RecalibrationCache.py` | NEW | Persistence and orchestration |
| `Model/UnitRecalibrator.py` | NEW | MLE optimization engine |
| `Model/RecalibrationNormalizer.py` | NEW | Scale normalization |
| `Model/UnitModel.py` | MODIFIED | State injection, recal lookups |
| `Model/Unit.py` | MODIFIED | Recalibration blend in get_value() |
| `Model/__init__.py` | MODIFIED | Export new classes |
| `Utilities/CurveUtils.py` | MODIFIED | Added steepness param to s_curve |
| `Performance/UnitGrader.py` | MODIFIED | Per-unit metrics |
| `Optimizer/UnitOptimizer.py` | MODIFIED | RecalSet loading, UnitGrader usage |
| `Scripts/calibrate_normalizer.py` | NEW | Normalizer coefficient calculation |
| `Scripts/__init__.py` | MODIFIED | Export calibrate_normalizer |
| `__init__.py` | MODIFIED | Export calibrate_normalizer |
| `config.json` | MODIFIED | Added recal_config, recal_normalizer |
| `Output/recalibration_values.csv` | NEW | Cached ideal values (~83K records) |
| `local/scripts/generate_recalibration_cache.py` | NEW | One-time generation script |

---

## Conclusion

This feature represents significant engineering effort to enable fast optimization of recalibration parameters. The architecture is sound and the implementation is correct. The lack of predictive lift suggests the problem may require a different approach entirely (joint optimization, alternative blending, or Bradley-Terry), not just parameter tuning.

The pre-computation insight and state injection patterns are valuable contributions that could be reused in future experiments.

