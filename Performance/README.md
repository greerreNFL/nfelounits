# Performance

Utilities for evaluating model quality.

## UnitGrader

Calculates prediction accuracy metrics comparing expected vs observed EPA.

### Usage

```python
from nfelounits import UnitGrader

# After running model
results = model.get_results_df()
grader = UnitGrader(results)

# Get all grades
grades = grader.grade()
print(grades['overall_mae'])
print(grades['pass_off_mae'])

# Or print formatted summary
grader.print_grades()
```

### Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **MAE** | Mean Absolute Error | Average EPA prediction error. Lower is better. |
| **RMSE** | Root Mean Squared Error | Penalizes large errors more heavily. Lower is better. |
| **R²** | Coefficient of Determination | Variance explained by the model. Higher is better (max 1.0). |

### Why These Metrics?

- **MAE** is the primary optimization target because it's intuitive (average error in EPA) and robust to outliers
- **RMSE** is included because it penalizes large misses, which matter more for game outcomes
- **R²** shows how much of the variance in unit performance the model captures vs attributing to noise

### Output Structure

```python
grades = grader.grade()

# Per-unit metrics (6 units × 3 metrics = 18)
grades['pass_off_mae']      # Pass offense MAE
grades['rush_def_rmse']     # Rush defense RMSE
grades['st_off_r_squared']  # ST offense R²

# Unit type averages (3 types × 3 metrics = 9)
grades['pass_mae']   # Average of pass_off and pass_def
grades['rush_rmse']
grades['st_r_squared']

# Overall averages (3 metrics)
grades['overall_mae']        # Average MAE across all 6 units
grades['overall_rmse']
grades['overall_r_squared']
grades['avg_mae']            # Alias for overall_mae (used by optimizer)
```

### Typical Values

For a well-tuned model on held-out test data:
- **MAE**: ~7-8 EPA (unit games have high variance)
- **RMSE**: ~10-12 EPA
- **R²**: ~0.03-0.05 (low because single-game EPA is noisy)

## Integration with Optimization

The `UnitOptimizer` uses `UnitGrader` internally:

```python
# Inside UnitOptimizer.objective()
grader = UnitGrader(results[results['data_set'] == 'train'])
grades = grader.grade()
return grades['avg_mae']  # This is what gets minimized
```

