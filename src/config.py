"""CP2 configuration, protocol v2. Values follow the CP1 proposal (Chapter 3)
except the changes listed and justified in CHANGES_v2.md, fixed before the v2
run. Nothing here may change after v2 results are seen."""

HORIZONS = [1, 4, 8]          # weeks ahead (Section 3.6.3)
TRAIN_MIN = 208               # v2: 4-year warm-up (CHANGES_v2.md #1); v1 used 260
TEST_LEN = 52                 # weeks per outer fold
STEP = 52                     # origin advance -> 8 folds under v2
INNER_VAL = 52                # inner expanding-window validation length
OUTBREAK_Q = 0.75             # per-fold training-window 75th percentile (3.6.3)
BREAK_DATE = "2014-01-01"     # documented lab-confirmation policy boundary
COVID_START, COVID_END = "2020-03-18", "2022-03-31"   # MCO -> endemic phase (v2 #5)
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
