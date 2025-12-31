# Changelog

## 2025-12-30

### Added

**Value Created Metric**
- New `value_created` fields in `TeamGameRecord` for each unit (pass/rush/st × off/def)
- Measures performance above/below what opponent + context would predict for an average team
- Offense formula: `(observed - league_avg) - (context - opponent_def)`
- Defense formula: `(opponent_off + opp_context) - (observed - league_avg)`
- Positive = good performance

**Opponent Faced Metric**
- New `faced` fields in `TeamGameRecord` for each unit
- Measures schedule difficulty adjusted for context
- Offense formula: `opponent_def - (hfa_adj + weather_adj)`
- Defense formula: `opponent_off + (qb_adj + hfa_adj + weather_adj)`
- Positive = harder schedule

**New Processors**
- `ValueCreatedProcessor` - outputs `value_created.csv` with season-to-date averages
- `OpponentFacedProcessor` - outputs `faced.csv` with season-to-date averages
- Both use expanding mean within season+team and era-adjusted normalization

**Normalization Enhancement**
- Added `group_by_week` parameter to `normalize()` function
- When True, calculates mean/std within each week number for fairer comparison across sample sizes
- Used by ValueCreatedProcessor and OpponentFacedProcessor since early-season samples are smaller

**League Average Storage**
- New `{unit}_league_avg` fields in `TeamGameRecord` (pass/rush/st)
- Enables derived calculations in processors

### Changed

**Weather Adjustment Sign Convention**
- `GameContext.weather_adj()` now returns negative values for bad weather
- Consistent convention: all adjustments are added directly, sign determines effect
- Updated `Unit.update()` and `Unit.get_expected_epa()` to add weather_adj instead of subtract

### Fixed

**Offense Value Created**
- Removed QB adjustment from offense value_created (was incorrectly included)
- Value created for offense should only reflect performance vs opponent defense + context

