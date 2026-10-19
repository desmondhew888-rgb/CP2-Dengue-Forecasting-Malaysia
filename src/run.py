"""CP2 experiment runner.

    python -m src.run --data data/dengue_model_dataset.csv
    python -m src.run --synthetic --quick        # pipeline smoke test

Writes long-format predictions and every result table to results/.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import models as M
from .config import (BREAK_DATE, CANDIDATES, HORIZONS, LSTM_SEEDS, OUTBREAK_Q,
                     SEED, TRAIN_MIN)
from .data import (build_features, feature_cols, group_of, load_series,
                   synthetic)
from .evaluate import break_test
from .folds import outbreak_threshold, outer_folds

RES = Path("results")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def grouped_permutation(model, X, y, idx, h, cols, rng, reps=5):
    base = M.mae(y[idx + h], model.predict(X[idx]) + y[idx])
    out = {}
    groups = sorted({group_of(c) for c in cols})
    for g in groups:
        j = [i for i, c in enumerate(cols) if group_of(c) == g]
        incs = []
        for _ in range(reps):
            Xp = X[idx].copy()
            Xp[:, j] = Xp[rng.permutation(len(idx))][:, j]
            incs.append(M.mae(y[idx + h], model.predict(Xp) + y[idx]) - base)
        out[g] = float(np.mean(incs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="1 LSTM seed, smaller grids: smoke test only")
    ap.add_argument("--skip-lstm", action="store_true")
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    meta = {"run_started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "synthetic": args.synthetic, "quick": args.quick}

    raw = synthetic() if args.synthetic else load_series(args.data)
    if args.synthetic:
        tmp = RES / "_synthetic_input.csv"
        raw.to_csv(tmp, index=False)
        raw = load_series(tmp)
    seeds = LSTM_SEEDS[:1] if args.quick else LSTM_SEEDS
    if args.quick:
        M.RF_GRID.update(n_estimators=[100])
        M.XGB_GRID.update(n_estimators=[100], max_depth=[3])
        M.LSTM_GRID.update(units=[16], layers=[1])
        M.LSTM_FIXED.update(max_epochs=30)

    # ---- 1. structural break test (Section 3.3.1.1) -----------------------
    bt = break_test(raw["start"], raw["dengue_total"], BREAK_DATE)
    use_regime = bt["chow_p"] < 0.05 or bt["shift_p_hac"] < 0.05
    bt["regime_indicator_added"] = bool(use_regime)
    pd.DataFrame([bt]).to_csv(RES / "break_test.csv", index=False)
    log(f"break test: shift {bt['shift_pct']:.1f}% p={bt['shift_p_hac']:.2g}; "
        f"Chow p={bt['chow_p']:.2g} -> regime indicator {use_regime}")

    df = build_features(raw, add_regime=use_regime)
    cols = feature_cols(df)
    X = df[cols].to_numpy(float)
    y = df["dengue_total"].to_numpy(float)
    weeks = df["start"].to_numpy()
    first_valid = int(np.where(~np.isnan(X).any(1))[0][0])
    n = len(df)
    meta.update(n_weeks=n, n_imputed=int(df["imputed"].sum()),
                n_features=len(cols), first_valid_row=first_valid,
                features=cols, first_week=str(df["start"].iloc[0].date()),
                last_week=str(df["start"].iloc[-1].date()))
    log(f"{n} weeks, {len(cols)} features, first complete row {first_valid}")

    rows, tuned, perm = [], [], []

    def add(model, h, k, idx, pred, thr):
        for o, p in zip(idx, pred):
            rows.append({"model": model, "h": h, "fold": k, "origin": int(o),
                         "origin_week": weeks[o], "target_week": weeks[o + h],
                         "actual": y[o + h], "pred": float(p),
                         "naive": y[o], "thr": thr})

    # ---- 2. SARIMA: orders identified once on the first training window ---
    order, sorder = M.select_sarima(y[: TRAIN_MIN + 1])
    meta["sarima_order"], meta["sarima_seasonal_order"] = order, sorder
    log(f"SARIMA{order}x{sorder}")
    sar = {}
    for k, _, test in outer_folds(n, 1, first_valid=first_valid):
        o = test[0]
        tst = np.arange(o, min(o + 52, n - 1))
        sar[k] = M.sarima_fold(y, o, tst, HORIZONS, order, sorder)
        log(f"  SARIMA fold {k} done")

    # ---- 3. outer loop -----------------------------------------------------
    rng = np.random.default_rng(SEED)
    for h in HORIZONS:
        for k, train, test in outer_folds(n, h, first_valid=first_valid):
            o = test[0]
            thr = outbreak_threshold(y, o)            # train window only
            add("naive", h, k, test, M.naive(y, test, h), thr)
            add("snaive", h, k, test, M.snaive(y, test, h), thr)
            add("sarima", h, k, test, [sar[k][h][t] for t in test], thr)
            for kind in ["rf", "xgb"]:
                pred, best, mdl = M.tree_fold(kind, X, y, train, test, h)
                add(kind, h, k, test, pred, thr)
                tuned.append({"model": kind, "h": h, "fold": k, **best})
                perm.append({"model": kind, "h": h, "fold": k,
                             **grouped_permutation(mdl, X, y, test, h, cols, rng)})
            if not args.skip_lstm:
                preds, best = M.lstm_fold(X, y, train, test, h, seeds)
                for s, p in preds.items():
                    add(f"lstm_s{s}", h, k, test, p, thr)
                tuned.append({"model": "lstm", "h": h, "fold": k, **best})
            log(f"h={h} fold {k} ({str(weeks[o])[:10]}) done")

    pred = pd.DataFrame(rows)
    pred.to_csv(RES / "predictions.csv", index=False)
    pd.DataFrame(tuned).to_csv(RES / "tuned_configs.csv", index=False)
    pd.DataFrame(perm).to_csv(RES / "permutation_importance.csv", index=False)

    # ---- 4. feature-group ablation (Objective 2) --------------------------
    abl = []
    df_c = build_features(raw, add_regime=use_regime, add_covid=True)
    sets = {"cases_only": (df, ["cases"]),
            "cases+calendar+regime": (df, ["cases", "calendar", "regime"]),
            "full": (df, None),
            "full+covid (post hoc)": (df_c, None)}      # CHANGES_v2.md #5
    for name, (frame, groups) in sets.items():
        c = feature_cols(frame, groups)
        Xa = frame[c].to_numpy(float)
        for kind in ["rf", "xgb"]:
            for h in HORIZONS:
                for k, train, test in outer_folds(n, h, first_valid=first_valid):
                    p, _, _ = M.tree_fold(kind, Xa, y, train, test, h)
                    abl.append({"set": name, "model": kind, "h": h, "fold": k,
                                "abs_err_sum": float(np.abs(y[test + h] - p).sum()),
                                "naive_abs_err_sum": float(np.abs(y[test + h] - y[test]).sum()),
                                "n": len(test)})
    pd.DataFrame(abl).to_csv(RES / "ablation.csv", index=False)
    log("ablation done")

    # ---- 5. target specification, run once at h=4 (Section 3.5.6) --------
    ts = []
    for k, train, test in outer_folds(n, 4, first_valid=first_valid):
        rf_lvl = M.RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                         random_state=SEED, n_jobs=-1)
        rf_lvl.fit(X[train], y[train + 4])
        p_lvl = rf_lvl.predict(X[test])
        p_dlt, _, _ = M.tree_fold("rf", X, y, train, test, 4)
        rf_d = M.RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                       random_state=SEED, n_jobs=-1)
        rf_d.fit(X[train], y[train + 4] - y[train])
        p_d = rf_d.predict(X[test]) + y[test]
        for spec, p in [("level", p_lvl), ("delta", p_d)]:
            ts.append({"spec": spec, "fold": k,
                       "abs_err_sum": float(np.abs(y[test + 4] - p).sum()),
                       "naive_abs_err_sum": float(np.abs(y[test + 4] - y[test]).sum()),
                       "max_pred": float(p.max()), "max_actual": float(y[test + 4].max()),
                       "train_max": float(y[train + 4].max())})
    pd.DataFrame(ts).to_csv(RES / "target_spec.csv", index=False)

    meta["run_finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (RES / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log("all done")


if __name__ == "__main__":
    main()
