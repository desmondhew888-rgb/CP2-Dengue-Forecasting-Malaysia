# CP2 — Comparative Evaluation of ML Techniques for Dengue Outbreak Forecasting in Malaysia

Implements the controlled comparison specified in the CP1 proposal (Chapter 3):
naive, seasonal naive, SARIMA, Random Forest, XGBoost and LSTM, evaluated at
1/4/8-week horizons under one leakage-free rolling-origin protocol.

## Reproduce

```
pip install -r requirements.txt
python fetch_climate_data.py            # builds dengue_model_dataset.csv (CP1)
python -m pytest -q tests               # verification suite (must pass first)
python -m src.run --data dengue_model_dataset.csv
python -m src.report                    # tables -> results/tables, figures -> figures/
```

Smoke test without data: `python -m src.run --synthetic --quick`.

## Layout

| Path | Role |
|---|---|
| `src/config.py` | Locked settings: folds, horizons, Table 3.5 grids, seeds |
| `src/data.py` | Weekly grid, imputation of the two missing weeks, Table 3.4 features |
| `src/folds.py` | Rolling-origin folds, inner tuning split, per-fold outbreak threshold |
| `src/models.py` | The six models; trees and LSTM predict the change y[t+h]−y[t] |
| `src/evaluate.py` | Relative MAE, pooled alarm metrics, Diebold–Mariano (HLN), break test |
| `src/run.py` | Runs the full experiment, ablation and target-spec comparison |
| `src/report.py` | Chapter 4 tables and figures |
| `tests/` | Leakage and implementation checks (V&V evidence) |

## Changes from the CP1 harness

* **Training-row gap.** CP1's `evaluate()` trained on every row before the first
  test origin *o*, which includes rows whose target y[t+h] falls after *o*
  (h−1 rows of future information). Training rows now satisfy t + h ≤ o.
  `tests/test_leakage.py::test_training_targets_observed_at_origin` enforces it.
* **Fixed fold calendar.** Test origins are the same six 52-week blocks for
  every horizon, so horizons are directly comparable.
* **Missing weeks imputed** by linear interpolation on a complete weekly grid
  (CP1 left the two gaps in place, which shifts every lag feature).
