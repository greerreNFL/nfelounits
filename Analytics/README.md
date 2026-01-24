# Analytics

Scripts and outputs for validating model performance against baselines.

## Why External Validation?

The model's internal metrics (unit MAE, R²) measure how well it predicts EPA at the unit level. But the ultimate question is: **does this translate to better game predictions?**

External validation answers this by comparing win probability predictions against:
- Other public models (FiveThirtyEight/nfelo Elo)
- The market (Vegas moneylines)

This reveals whether the unit decomposition approach adds value over simpler team-level ratings.

## Prediction Accuracy

Compares the Unit Model's game predictions against established baselines:

| Model | Description |
|-------|-------------|
| `units_elo` | Unit Model Elo ratings with QB adjustments and HFA |
| `f38_base_elo` | FiveThirtyEight/nfelo Elo ratings (no QB adjustment) |
| `f38_qb_elo` | FiveThirtyEight/nfelo Elo ratings with QB adjustments |
| `market` | Vegas moneyline implied probabilities (vig-free) |

### Metrics

- **Log Loss** - Measures probability calibration (lower is better)
- **Brier Score** - Mean squared error of probabilities (lower is better)
- **538 Game Points** - FiveThirtyEight's scoring system (higher is better)
- **Accuracy** - Percentage of games correctly predicted (higher is better)

### Why These Metrics?

**Log Loss** is the primary metric because it:
- Penalizes confident wrong predictions heavily
- Rewards well-calibrated probabilities (a 70% prediction should win ~70% of the time)
- Is the standard for probabilistic forecasts

**Accuracy** is included for interpretability but is less informative - a model predicting 51% on every game would have ~50% accuracy but terrible log loss.

**538 Points** provides a different perspective - it rewards confidence on correct picks without penalizing incorrect picks as harshly as log loss.

### Running the Analysis

```python
from nfelounits.Analytics.prediction_accuracy.measure_accuracy import main
main()
```

Or run directly:
```bash
python -m nfelounits.Analytics.prediction_accuracy.measure_accuracy
```

### Output Files

| File | Description |
|------|-------------|
| `model_summary.csv` | Overall metrics for all models |
| `log_loss_by_season.csv` | Log loss broken down by season |
| `brier_by_season.csv` | Brier score broken down by season |
| `f38_points_by_season.csv` | 538 points broken down by season |
| `accuracy_by_season.csv` | Accuracy broken down by season |
| `model_correlation.csv` | Correlation matrix between model predictions |

### Sample Results

```
model          n_games  log_loss  brier   f38_points  accuracy
units_elo      5065     0.623     0.2174  27.53       0.6432
f38_base_elo   5065     0.6287    0.2198  27.32       0.6434
f38_qb_elo     5065     0.6224    0.2171  27.55       0.6492
market         5065     0.6093    0.2111  28.02       0.6648
```

### Interpreting Results

The Unit Model (`units_elo`) performs comparably to FiveThirtyEight's QB-adjusted Elo:
- Slightly worse log loss (0.623 vs 0.622)
- Similar accuracy (~64%)

The market remains the benchmark - it incorporates information (injuries, motivation, weather details) that pure statistical models don't capture.

**Key insight**: The model splits results by first half (weeks 1-8) vs second half (weeks 9-18). Models typically perform better in the second half once they've accumulated more data on each team. A likely explanation is that 538's QB Elo incorporates win totals to set preseason priors, making it more accurate at the start of the season.

## Pre-Season Priors

Analysis of pre-season predictions using external sources like DVOA.

### Files

- `measure_priors.py` - Script to analyze pre-season prior accuracy
- `manual_data/dvoa.csv` - Historical DVOA data for comparison
