"""Locked CP2 configuration. Every value here is fixed by the CP1 proposal
(Chapter 3) and must not be changed after results are seen."""

HORIZONS = [1, 4, 8]          # weeks ahead (Section 3.6.3)
TRAIN_MIN = 260               # ~5-year warm-up before first origin (3.5.7)
TEST_LEN = 52                 # weeks per outer fold
STEP = 52                     # origin advance -> 6 folds
INNER_VAL = 52                # inner expanding-window validation length
OUTBREAK_Q = 0.75             # per-fold training-window 75th percentile (3.6.3)
BREAK_DATE = "2014-01-01"     # documented lab-confirmation policy boundary
SEASON = 52
SEED = 42
LSTM_SEEDS = [0, 1, 2, 3, 4]  # instability made visible (3.5.5)
LSTM_WINDOW = 12              # sequence length fed to the LSTM

# Table 3.5 -- grids fixed before modelling
RF_GRID = {"n_estimators": [300, 600], "max_depth": [None, 12],
           "min_samples_leaf": [2, 5]}                          # 8 configs
XGB_GRID = {"max_depth": [3, 5, 7], "learning_rate": [0.03, 0.10],
            "n_estimators": [300, 600]}                         # 12 configs
XGB_FIXED = {"subsample": 0.8, "colsample_bytree": 0.8}
LSTM_GRID = {"units": [32, 64], "layers": [1, 2], "dropout": [0.1, 0.3]}  # 8
LSTM_FIXED = {"batch": 32, "max_epochs": 200, "patience": 15, "lr": 1e-3}

# SARIMA candidate orders (identified on first training window by AIC)
SARIMA_P = [0, 1, 2]
SARIMA_Q = [0, 1, 2]
SARIMA_SP = [0, 1]
SARIMA_SQ = [0, 1]

MODELS = ["naive", "snaive", "sarima", "rf", "xgb", "lstm"]
CANDIDATES = ["sarima", "rf", "xgb", "lstm"]
