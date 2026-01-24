# Optimizer

Configuration management and parameter optimization for the Unit Model.

## Why Optimization?

The model has ~30 parameters controlling smoothing rates, regression amounts, weather effects, and Elo translation. These could be set by intuition, but optimization:

1. **Finds non-obvious values** - e.g., pass defense regresses more than pass offense
2. **Balances tradeoffs** - Faster updates capture changes but amplify noise
3. **Validates assumptions** - If an optimal value hits a bound, the assumption may be wrong
4. **Enables reproducibility** - Anyone can re-run optimization on new data

## ModelConfig

Loads, manages, and persists model parameters from `config.json`.

### Usage

```python
from nfelounits import ModelConfig

# Load from default location
config = ModelConfig.from_file()

# Access nested parameter values
values = config.values  # {'unit_config': {...}, 'elo_config': {...}}

# Access parameter metadata
param = config.params['unit_config.pass_off_sf']
print(param.value, param.opti_min, param.opti_max)

# Update parameters
config.update_config({
    'unit_config.pass_off_sf': 0.08,
    'elo_config.pass_off_coef': 20.0
})

# Save changes
config.to_file()
```

### Configuration Structure

```json
{
    "unit_config": {
        "pass_off_sf": {"value": 0.0625, "opti_min": 0.0, "opti_max": 0.15, ...},
        "pass_def_sf": {...},
        ...
    },
    "elo_config": {
        "pass_off_coef": {"value": 22.97, "opti_min": 0.0, "opti_max": 50.0, ...},
        ...
    }
}
```

### Parameter Categories

| Category | Parameters | Description |
|----------|------------|-------------|
| **Smoothing Factors** | `*_sf` | EWMA update rate (0 = no update, 1 = full replacement) |
| **Reversion Rates** | `*_reversion` | Offseason regression toward mean (0 = no reversion, 1 = full) |
| **Weather Adjustments** | `*_wind_disc_height`, `*_temp_disc_height` | Max EPA reduction for weather |
| **HFA Shares** | `*_hfa_share` | Portion of HFA attributed to each unit |
| **Elo Coefficients** | `*_coef` | Unit EPA → Elo conversion factor |

### Why Bounds?

Each parameter has `opti_min` and `opti_max` bounds because:
- **Physical constraints** - Smoothing factors must be 0-1
- **Stability** - Extreme values cause erratic behavior
- **Interpretability** - Values outside reasonable ranges indicate model issues

If optimization consistently hits a bound, consider widening it.

## UnitOptimizer

Optimizes unit model parameters to minimize Mean Absolute Error (MAE) on unit predictions.

### Why MAE?

MAE (Mean Absolute Error) was chosen over alternatives:
- **vs MSE/RMSE** - MAE is more robust to outliers, which are common in NFL (blowouts, garbage time)
- **vs Log Loss** - Log loss is for probabilities; unit predictions are EPA values
- **Interpretability** - "Average error of 7 EPA" is intuitive

### Usage

```python
from nfelounits import DataLoader, DataSplitter, UnitOptimizer, ModelConfig

# Prepare data with train/test labels
loader = DataLoader()
splitter = DataSplitter(loader.unit_games)
labeled_data = splitter.label_train_test(n_test_seasons=5)

config = ModelConfig.from_file()

# Optimize specific parameters
optimizer = UnitOptimizer(
    data=labeled_data,
    config=config,
    subset=['unit_config.pass_off_sf', 'unit_config.pass_def_sf']
)

optimizer.optimize(save_result=True, update_config=True)
print(f"Best MAE: {optimizer.optimization_results['avg_mae']:.4f}")
```

### Optimization Details

- **Objective**: Minimize average MAE across all unit predictions (expected vs observed EPA)
- **Method**: scipy.optimize.minimize with L-BFGS-B
- **Train Set**: Uses only games labeled 'train' for optimization
- **Normalization**: Parameters are normalized to [0,1] using opti_min/opti_max bounds

## EloOptimizer

Optimizes Elo translation coefficients to minimize log loss on win predictions.

### Why Separate Optimizers?

Unit parameters and Elo coefficients serve different purposes:
- **Unit parameters** affect EPA predictions (what the model "sees")
- **Elo coefficients** affect how EPA translates to win probability (how predictions are combined)

Optimizing them separately:
1. Allows using different objective functions (MAE vs log loss)
2. Reduces parameter space per optimization (faster, more stable)
3. Ensures unit ratings are meaningful before combining them

### Usage

```python
from nfelounits import EloOptimizer, ModelConfig

optimizer = EloOptimizer(
    data=labeled_data,
    config=config,
    calculate_test=True  # Track test set performance
)

optimizer.optimize(save_result=True, update_config=False)

print(f"Train log loss: {optimizer.optimization_results['train_log_loss']:.4f}")
print(f"Test log loss: {optimizer.optimization_results['test_log_loss']:.4f}")
```

### Optimization Details

- **Objective**: Minimize log loss on train set win probabilities
- **Test Tracking**: Optionally calculates test set log loss each round
- **Parameters**: Optimizes `elo_config.*_coef` values

## BaseOptimizer

Abstract base class for creating custom optimizers.

### Creating a Custom Optimizer

```python
from nfelounits.Optimizer import BaseOptimizer

class MyOptimizer(BaseOptimizer):
    def get_metric_name(self) -> str:
        return 'my_metric'
    
    def objective(self, x: List[float]) -> float:
        self.round_number += 1
        config = self.denormalize_optimizer_values(x)
        
        model = UnitModel(self.data, config)
        model.run()
        results = model.get_results_df()
        
        # Calculate your metric
        metric = calculate_metric(results)
        
        # Store record
        self.optimization_records.append({
            'round': self.round_number,
            'my_metric': metric,
            **self.get_param_dict(x)
        })
        
        return metric  # Value to minimize
```

## Output

Optimization runs save progress to `Optimizer/runs/`:
- `{optimizer}_{timestamp}.csv` - All rounds with parameters and metrics
- Enables analysis of optimization landscape and parameter sensitivity
