# CP2 protocol v2 — changes fixed BEFORE the v2 run

Written 2026-10-19, after the v1 (proposal-locked) results were seen. Every
change is listed with its justification and whether it was motivated by v1
results. v1 results are kept unchanged in `results_v1_locked/` and reported
alongside v2. No change below may be revised after the v2 run is seen.

| # | Change | Justification | Informed by v1 results? |
|---|---|---|---|
| 1 | Warm-up 260 → 208 weeks; use every fold that fits (8 folds, test origins Jan 2015 – Dec 2022, was 6 folds, Dec 2015 – Dec 2021) | v1's main limitation was low power and only 11 outbreak episodes; the 2014–15 epidemic and all of 2022 were never tested although the data existed. 208 weeks is the shortest warm-up whose first training window still spans a full post-2014 year | Yes — by the diagnosed lack of power, not by which model won |
| 2 | Equal-weight ensemble: mean of Random Forest, XGBoost and LSTM (mean of 5 seeds) | v1 found the three learners statistically indistinguishable, so no single one can be preferred; averaging indistinguishable forecasts reduces variance (forecast-combination literature). Weights are fixed at 1/3, not tuned | Yes — by the indistinguishability result |
| 3 | Significance: the pre-specified comparisons are the primary inference (as in the CP1 proposal); Holm adjustment reported as a robustness check. Ensemble vs naive added as one extra pre-specified comparison per horizon | v1 report applied Holm as primary, which the proposal did not specify | Yes — corrects a v1 reporting choice |
| 4 | Alarm comparison: paired moving-block bootstrap (8-week blocks, 2,000 resamples) for the difference in sensitivity and specificity, best model vs naive | Wilson intervals assume independent weeks; outbreak weeks are strongly autocorrelated, so v1 intervals were too narrow | Yes — corrects an inference weakness, direction of effect unknown in advance |
| 5 | COVID-19 movement-control indicator (1 from 18 Mar 2020 to 31 Mar 2022) as an ADDITIONAL feature-set variant in the ablation only; the main comparison does not use it | Dengue notification fell under movement control; v1 learners failed in 2021 | Yes — post hoc; reported only as a disclosed sensitivity analysis |

Unchanged: data, features (Table 3.4), targets (change for trees/LSTM),
hyperparameter grids, nested tuning, LSTM seeds, outbreak threshold
definition (fold training-window 75th percentile), metrics, horizons.
