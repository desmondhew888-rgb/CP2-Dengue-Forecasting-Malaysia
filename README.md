# CP2 — Comparative Evaluation of Machine Learning Techniques for Dengue Outbreak Forecasting in Malaysia

Capstone Project 2 (PRJ3213), Hew Zhen Wei (24026973), Sunway University.
Supervisor: Prof. Ts. Dr.-Ing. Lau Sian Lun.
Continues the CP1 repository: https://github.com/desmondhew888-rgb/CP1-Dengue-Forecasting-Malaysia

A controlled comparison in which only the technique varies. Naive, seasonal naive,
SARIMA, Random Forest, XGBoost, LSTM (5 seeds) and an equal-weight ensemble forecast
weekly national dengue cases (2011–2022) 1, 4 and 8 weeks ahead, evaluated by
leakage-free rolling-origin validation (8 folds, test years 2015–2022).

## Headline results (protocol v2)

| Model | Rel. MAE 1 wk | 4 wk | 8 wk |
|---|---|---|---|
| Naive | 1.000 | 1.000 | 1.000 |
| SARIMA | 1.053 | 1.008 | 1.002 |
| Random Forest | 0.980 | 0.797 | **0.742** |
| XGBoost | 1.000 | 0.801 | 0.767 |
| LSTM (5-seed mean) | **0.946** | 0.843 | 0.823 |
| Ensemble (RF+XGB+LSTM) | 0.947 | **0.784** | 0.744 |

Relative MAE = model MAE ÷ naive MAE at the same horizon (below 1 beats persistence).
RF, XGBoost and the ensemble beat persistence significantly at 4 and 8 weeks
(Diebold–Mariano); the learners cannot be separated from each other. At 8 weeks RF
raises outbreak-week sensitivity from 0.711 to 0.868 (block-bootstrap 95% CI of the
gain 0.045–0.304).

## Reproduce

Python 3.13. Place the three raw files in `data/` (not redistributed here; see
Data below) and check them against `raw_checksums.sha256`.

```
pip install -r requirements.lock.txt
cd data && python ../fetch_climate_data.py && cd ..   # builds data/dengue_model_dataset.csv
python -m pytest -q tests                             # 16 verification tests, must pass
python -m src.run --data data/dengue_model_dataset.csv   # ~75 min on a 2-core CPU
python -m src.report                                  # Chapter 4 tables and figures
python -m src.extra                                   # supplementary post hoc analyses
```

Smoke test without data: `python -m src.run --synthetic --quick`.

An independent re-run on Google Colab (PyTorch 2.11 instead of 2.14.1) reproduced all
12,400 forecasts; maximum relative-MAE difference 0.000025.

## Data

| File | Source |
|---|---|
| `Target_Dataset.csv` | OpenDengue v1.3, weekly national cases, Malaysia |
| `mys-rainfall-subnat-full.csv` | CHIRPS subnational dekadal rainfall (HDX / WFP) |
| `nasa_power_daily_states.csv` | NASA POWER (MERRA-2) daily climate, 15 state points; downloaded automatically if absent |

## Layout

| Path | Role |
|---|---|
| `src/config.py` | Locked v2 settings: horizons, warm-up, folds, threshold, seeds, grids |
| `src/data.py` | Weekly grid, interpolation of two missing weeks, 51 features |
| `src/folds.py` | Rolling-origin folds (t + h ≤ origin), inner tuning split, per-fold threshold |
| `src/models.py` | Baselines, SARIMA, Random Forest, XGBoost, LSTM with nested tuning |
| `src/evaluate.py` | Metrics, alarm metrics, episode detection, Diebold–Mariano (HLN), Holm, break test |
| `src/run.py` | Full experiment, permutation importance, ablation, target-spec test |
| `src/report.py` | Ensemble, Chapter 4 tables, block bootstrap, figures |
| `src/extra.py` | Post hoc descriptive analyses (bias, epidemic phase, direction, threshold sensitivity) |
| `tests/test_leakage.py` | Verification suite (data integrity, leakage, statistics) |
| `results/`, `figures/` | v2 outputs; `results_v1_locked/`, `figures_v1_locked/` keep the proposal-locked v1 run |
| `CHANGES_v2.md` | Protocol revision, committed before the v2 run |

## Key correction to the CP1 harness

CP1 trained each fold on every row before the first test origin, including the last
h − 1 rows whose targets lie after the origin. Training rows now satisfy t + h ≤ origin,
enforced by `tests/test_leakage.py::test_training_targets_observed_at_origin`.
