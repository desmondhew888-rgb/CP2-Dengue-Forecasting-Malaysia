"""Load the assembled weekly series, fill the two missing weeks, build
features. Every feature uses only information available at the origin t."""
import numpy as np
import pandas as pd

from .config import BREAK_DATE

BASE = ["dengue_total", "rain_chirps", "rain_chirps_pop", "temp_mean",
        "humidity"]

FEATURE_GROUPS = {
    "cases": lambda c: c.startswith("cases_"),
    "rain": lambda c: c.startswith(("rain_lag", "rain_roll", "rainpop_")),
    "temp": lambda c: c.startswith("temp_"),
    "humidity": lambda c: c.startswith("hum_"),
    "calendar": lambda c: c in ("week_of_year", "month", "year"),
    "regime": lambda c: c == "regime",
}


def load_series(path):
    """Accepts either dengue_model_dataset.csv (CP1 output) or any CSV with
    'start' + BASE columns. Reindexes to a complete weekly grid and imputes
    missing case weeks by linear interpolation (Section 3.4.1)."""
    df = pd.read_csv(path, parse_dates=["start"])
    df = df[["start"] + [c for c in BASE if c in df.columns]]
    df = df.sort_values("start").drop_duplicates("start")
    full = pd.date_range(df["start"].min(), df["start"].max(), freq="7D")
    df = df.set_index("start").reindex(full)
    df.index.name = "start"
    df["imputed"] = df["dengue_total"].isna().astype(int)
    df = df.interpolate("linear", limit_direction="both")
    return df.reset_index()


def build_features(df, add_regime=True):
    df = df.copy()
    y = df["dengue_total"]
    for lag in [1, 2, 3, 4, 6, 8, 12]:
        df[f"cases_lag{lag}"] = y.shift(lag)
    for w in [4, 8, 12, 16]:
        df[f"cases_roll{w}"] = y.shift(1).rolling(w).mean()
        df[f"temp_roll{w}"] = df["temp_mean"].shift(1).rolling(w).mean()
        df[f"hum_roll{w}"] = df["humidity"].shift(1).rolling(w).mean()
        df[f"rain_roll{w}"] = df["rain_chirps"].shift(1).rolling(w).mean()
        df[f"rainpop_roll{w}"] = df["rain_chirps_pop"].shift(1).rolling(w).mean()
    for w in [8, 12]:
        for lag in [4, 8]:
            df[f"temp_roll{w}_lag{lag}"] = df["temp_mean"].shift(lag).rolling(w).mean()
            df[f"hum_roll{w}_lag{lag}"] = df["humidity"].shift(lag).rolling(w).mean()
    for lag in [2, 4, 8, 12]:
        df[f"rain_lag{lag}"] = df["rain_chirps"].shift(lag)
        df[f"temp_lag{lag}"] = df["temp_mean"].shift(lag)
        df[f"hum_lag{lag}"] = df["humidity"].shift(lag)
    df["week_of_year"] = df["start"].dt.isocalendar().week.astype(int)
    df["month"] = df["start"].dt.month
    df["year"] = df["start"].dt.year
    if add_regime:
        df["regime"] = (df["start"] >= pd.Timestamp(BREAK_DATE)).astype(int)
    return df


def feature_cols(df, groups=None):
    excl = set(BASE) | {"start", "imputed", "temp_max", "temp_min",
                        "precip_power"}
    cols = [c for c in df.columns if c not in excl]
    if groups is None:
        return cols
    return [c for c in cols if any(FEATURE_GROUPS[g](c) for g in groups)]


def group_of(col):
    for g, f in FEATURE_GROUPS.items():
        if f(col):
            return g
    return "other"


def synthetic(n=626, seed=0):
    """Synthetic series with trend, seasonality, epidemics, a 2014 level shift
    and lagged climate dependence. Used only to test the pipeline."""
    rng = np.random.default_rng(seed)
    start = pd.date_range("2011-01-02", periods=n, freq="7D")
    t = np.arange(n)
    temp = 27 + 0.8 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 0.3, n)
    hum = 80 + 4 * np.cos(2 * np.pi * t / 52) + rng.normal(0, 1.5, n)
    rain = np.clip(60 + 30 * np.sin(2 * np.pi * (t + 10) / 52)
                   + rng.normal(0, 15, n), 0, None)
    log_y = 5.8 + 0.002 * t + 0.35 * np.sin(2 * np.pi * (t - 5) / 52)
    log_y += 0.7 * (start >= pd.Timestamp("2014-01-01"))
    temp_eff = pd.Series(temp).rolling(8, min_periods=1).mean().shift(6).bfill()
    log_y += 0.4 * (temp_eff.values - 27)
    for c in [170, 230, 430]:                       # epidemics
        log_y += 0.8 * np.exp(-0.5 * ((t - c) / 8) ** 2)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.9 * e[i - 1] + rng.normal(0, 0.06)
    y = np.round(np.exp(log_y + e))
    df = pd.DataFrame({"start": start, "dengue_total": y, "rain_chirps": rain,
                       "rain_chirps_pop": rain * 1.1 + rng.normal(0, 5, n),
                       "temp_mean": temp, "humidity": hum})
    return df.drop(index=[200, 410]).reset_index(drop=True)  # 2 missing weeks
