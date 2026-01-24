# Scripts

Convenience functions for common workflows.

## run()

Runs the full model pipeline and saves all output files.

```python
from nfelounits import run

run()  # Uses default output path (Output/)
run(output_path='/custom/path')
```

### What It Does

1. Loads play-by-play data via `DataLoader`
2. Loads configuration from `config.json`
3. Runs `UnitModel` through all games
4. Prints performance grades
5. Runs all processors and saves output CSVs


## optimize_models()

Runs the full optimization pipeline to tune all model parameters.

```python
from nfelounits import optimize_models

optimize_models(n_rounds=10, n_test_seasons=5)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_rounds` | 10 | Random restarts per optimizer |
| `n_test_seasons` | 5 | Seasons held out for testing |

### Why Multiple Random Starts?

The optimization landscape is non-convex with local minima. Running multiple times with random starting points:
- Explores more of the parameter space
- Reduces chance of getting stuck in a bad local minimum
- Gives confidence that the result is robust

10 rounds is a reasonable default; more rounds = better coverage but longer runtime.

### Why Optimize By Unit?

The script optimizes unit parameters (pass, rush, st) separately rather than all at once because:

1. **Overfit Mitigation** - By tuning to average accuracy across all units, we reduce the risk of overfitting as this objective function is distinct from the one we actually want to optimize for, which is overall team rating. The ethos is to make the individual unit predictions as accurate as possible, and then let their collective accuracy "bubble up" into a predictive team rating, without explicitly optimizing for it.
3. **Interpretability** - You can see which unit improved and by how much and see which units are contributing most to overall accuracy (or inaccuracy)

### Optimization Order

```
1. Pass unit parameters (smoothing, reversion, weather, QB)
2. Rush unit parameters (smoothing, reversion, weather)
3. ST unit parameters (smoothing, reversion, weather)
4. Elo coefficients (unit → Elo translation)
```

### Output

- Updates `config.json` with optimal parameters
- Prints progress and final results
- Saves optimization run history to `Optimizer/runs/`

### Typical Runtime

- ~5-10 minutes per unit × 3 units = 15-30 minutes for unit params
- ~5-10 minutes for Elo params
- **Total**: 20-40 minutes depending on hardware and n_rounds

### When to Re-optimize

- After significant code changes to the model
- When adding a new adjustment type
- Annually when a new season's data is available
- Generally not needed for routine use
