"""Verification tests (V&V evidence for Chapter 4 / poster demo).

    python -m pytest -q tests
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import HORIZONS, TRAIN_MIN
from src.data import build_features, feature_cols, load_series, synthetic
from src.evaluate import diebold_mariano, holm
from src.folds import inner_split, outbreak_threshold, outer_folds

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def df(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "s.csv"
    synthetic().to_csv(p, index=False)
    return build_features(load_series(p))


# R1 -- data integrity --------------------------------------------------------
def test_complete_weekly_grid(df):
    gaps = df["start"].diff().dropna().dt.days
    assert (gaps == 7).all()
    assert df["imputed"].sum() == 2          # the two absent weeks, filled


def test_pcode_mapping_is_cod_not_iso():
    spec = importlib.util.spec_from_file_location("f", ROOT / "fetch_climate_data.py")
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    m = f.PCODE_TO_STATE
    assert m["MY10"] == "Perlis" and m["MY17"] == "Selangor"
    assert m["MY04"] == "WP Kuala Lumpur"


# R2 -- no temporal leakage ---------------------------------------------------
@pytest.mark.parametrize("h", HORIZONS)
def test_training_targets_observed_at_origin(df, h):
    for _, train, test in outer_folds(len(df), h):
        assert train.max() + h <= test.min()


def test_cp1_harness_would_have_leaked():
    """Documents the CP1 defect fixed here: training on all rows < o admits
    rows whose target y[t+h] lies after the origin."""
    o, h = TRAIN_MIN, 8
    cp1_train = np.arange(0, o)
    assert (cp1_train + h > o).sum() == h - 1


@pytest.mark.parametrize("h", HORIZONS)
def test_inner_split_never_touches_test(df, h):
    for _, train, test in outer_folds(len(df), h):
        itr, val = inner_split(train, h)
        assert val.max() < test.min()
        assert itr.max() + h <= val.min()


def test_features_have_no_lookahead(df, tmp_path):
    """Corrupt every value AFTER week t; features at rows <= t must not move."""
    raw = load_series(_write(synthetic(), tmp_path))
    t = 300
    corrupt = raw.copy()
    for c in ["dengue_total", "rain_chirps", "rain_chirps_pop", "temp_mean",
              "humidity"]:
        corrupt.loc[t + 1:, c] = corrupt.loc[t + 1:, c] * 10 + 999
    a, b = build_features(raw), build_features(corrupt)
    cols = feature_cols(a)
    pd.testing.assert_frame_equal(a.loc[:t, cols], b.loc[:t, cols])


def test_threshold_uses_training_window_only():
    y = np.r_[np.ones(300), np.full(52, 1e6)]
    assert outbreak_threshold(y, 299) == 1.0


def test_folds_share_calendar_origins(df):
    origins = {h: [t[0] for _, _, t in outer_folds(len(df), h)] for h in HORIZONS}
    assert origins[1] == origins[4] == origins[8]
    assert len(origins[1]) == 8


# R5 -- statistical test implementation ---------------------------------------
def test_dm_identical_forecasts_is_zero():
    e = np.random.default_rng(0).normal(size=300)
    r = diebold_mariano(e, e.copy(), 1)
    assert r["mean_loss_diff"] == 0


def test_dm_detects_clearly_better_model():
    rng = np.random.default_rng(1)
    e_good, e_bad = rng.normal(0, 1, 300), rng.normal(0, 3, 300)
    r = diebold_mariano(e_good, e_bad, 4)
    assert r["dm_stat"] < 0 and r["p_value"] < 0.001


def test_dm_size_under_null():
    """Equal-accuracy errors: rejection rate at 5% should be near 5%."""
    rng = np.random.default_rng(2)
    rej = [diebold_mariano(rng.normal(size=300), rng.normal(size=300), 4)
           ["p_value"] < 0.05 for _ in range(400)]
    assert 0.02 < np.mean(rej) < 0.09


def test_holm_monotone():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5])


def _write(d, tmp):
    p = tmp / "x.csv"
    d.to_csv(p, index=False)
    return p
