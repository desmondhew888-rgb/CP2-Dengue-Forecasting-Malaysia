"""Supplementary descriptive analyses for Chapter 4 (added 26 Oct 2026).

Computed from the frozen v2 predictions only; no model is refitted and no
setting is changed. All are post hoc descriptions of existing forecasts and
are reported as such.

    python -m src.extra
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_series
from .evaluate import confusion, episode_detection, pooled_alarm
from .folds import outbreak_threshold
from .report import load

RES = Path("results")
OUT = RES / "tables"
MODELS = ["naive", "sarima", "rf", "xgb", "lstm", "ens"]


def frame():
    p = load()
    lstm = (p[p.base == "lstm"].groupby(["h", "origin"], as_index=False)["pred"].mean())
    base = p[p.model == "rf"].drop(columns=["pred"])
    lstm = base.merge(lstm, on=["h", "origin"]).assign(model="lstm", base="lstm")
    p = pd.concat([p[~p.model.str.startswith("lstm_s")], lstm[p.columns]],
                  ignore_index=True)
    return p[p.model.isin(MODELS)]


def series():
    raw = load_series("data/dengue_model_dataset.csv")
    return raw["dengue_total"].to_numpy(float), raw["start"]


def bias_and_spread(p):
    rows = []
    for (m, h), q in p.groupby(["model", "h"]):
        e = q.pred - q.actual
        ae = e.abs()
        ob = q.actual > q.thr
        n = p[(p.model == "naive") & (p.h == h)]
        nae = (n.pred - n.actual).abs()
        rows.append({"model": m, "h": h,
                     "mean_error": e.mean(),
                     "mean_error_pct_of_mean_actual": 100 * e.mean() / q.actual.mean(),
                     "mean_error_outbreak": e[ob].mean(),
                     "mean_error_outbreak_pct": 100 * e[ob].mean() / q.actual[ob].mean(),
                     "mean_error_quiet": e[~ob].mean(),
                     "median_ae": ae.median(), "p90_ae": ae.quantile(0.9),
                     "median_ae_rel_naive": ae.median() / nae.median(),
                     "p90_ae_rel_naive": ae.quantile(0.9) / nae.quantile(0.9),
                     "max_ae": ae.max()})
    return pd.DataFrame(rows)


def phase_table(p, y):
    """Classify each TARGET week by the actual 4-week trajectory ending there."""
    rows = []
    for (m, h), q in p.groupby(["model", "h"]):
        t = (q.origin + h).to_numpy()
        prev = y[t - 4]
        chg = (y[t] - prev) / prev
        phase = np.where(chg > 0.10, "rising", np.where(chg < -0.10, "falling", "stable"))
        for ph in ["rising", "stable", "falling"]:
            s = phase == ph
            n = p[(p.model == "naive") & (p.h == h)]
            nt = (n.origin + h).to_numpy()
            nchg = (y[nt] - y[nt - 4]) / y[nt - 4]
            nph = np.where(nchg > 0.10, "rising", np.where(nchg < -0.10, "falling", "stable"))
            rel = (np.abs(q.pred - q.actual)[s].mean()
                   / np.abs(n.pred - n.actual)[nph == ph].mean())
            rows.append({"model": m, "h": h, "phase": ph, "n": int(s.sum()),
                         "rel_mae": rel})
    return pd.DataFrame(rows)


def direction(p):
    """Share of origins where the forecast change has the sign of the actual
    change, among origins whose actual change exceeded 10% of y(t)."""
    rows = []
    for (m, h), q in p.groupby(["model", "h"]):
        if m == "naive":
            continue
        act = q.actual - q.naive
        prd = q.pred - q.naive
        big = act.abs() > 0.10 * q.naive
        hit = np.sign(act[big]) == np.sign(prd[big])
        up, dn = act[big] > 0, act[big] < 0
        rows.append({"model": m, "h": h, "n_moves": int(big.sum()),
                     "direction_hit": hit.mean(),
                     "hit_rises": hit[up].mean(), "hit_falls": hit[dn].mean()})
    return pd.DataFrame(rows)


def threshold_sensitivity(p, y):
    rows = []
    for qv in [0.70, 0.75, 0.80, 0.90]:
        for (m, h), q in p.groupby(["model", "h"]):
            if h == 1:
                continue
            thr = q.groupby("fold").origin.transform("min").map(
                lambda o: outbreak_threshold(y, int(o), qv))
            cm = confusion(q.actual.values, q.pred.values, thr.values)
            a = pooled_alarm(cm)
            f = q.assign(thr=thr.values)
            eps, det = episode_detection(f)
            rows.append({"q": qv, "model": m, "h": h,
                         "outbreak_weeks": cm["tp"] + cm["fn"],
                         "sensitivity": a["sensitivity"],
                         "specificity": a["specificity"], "ppv": a["ppv"],
                         "episodes": eps, "detected": det})
    return pd.DataFrame(rows)


def peaks(p):
    rows = []
    for (m, h), q in p.groupby(["model", "h"]):
        q = q.assign(year=q.target_week.dt.year)
        for yr in [2015, 2019]:
            s = q[q.year == yr]
            if s.empty:
                continue
            ia, ip = s.actual.idxmax(), s.pred.idxmax()
            rows.append({"model": m, "h": h, "year": yr,
                         "actual_peak": s.actual.max(),
                         "actual_peak_week": s.target_week[ia].date(),
                         "forecast_peak": s.pred.max(),
                         "forecast_peak_week": s.target_week[ip].date(),
                         "peak_ratio": s.pred.max() / s.actual.max(),
                         "timing_weeks": (s.target_week[ip] - s.target_week[ia]).days // 7})
    return pd.DataFrame(rows)


def main():
    p = frame()
    y, _ = series()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in [("X1_bias_spread", bias_and_spread(p)),
                     ("X2_phase", phase_table(p, y)),
                     ("X3_direction", direction(p)),
                     ("X4_threshold_sensitivity", threshold_sensitivity(p, y)),
                     ("X5_peaks", peaks(p))]:
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"\n== {name}\n", df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
