"""Turn results/*.csv into Chapter 4 tables (results/tables/) and figures
(figures/). Every number in the report should trace to a file written here.

    python -m src.report
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import CANDIDATES, HORIZONS
from .evaluate import (accuracy, confusion, diebold_mariano, episode_detection,
                       holm, pooled_alarm)

RES, FIG = Path("results"), Path("figures")
TAB = RES / "tables"

# fixed model -> colour mapping (reference palette, light mode, fixed order)
# same model -> colour mapping as the report's interactive charts
COL = {"rf": "#2a78d6", "xgb": "#eb6834", "lstm": "#1baf7a",
       "ens": "#6250d6", "sarima": "#8f8d87", "naive": "#52514e",
       "snaive": "#b5b3ad"}
NAME = {"naive": "Naive", "snaive": "Seasonal naive", "sarima": "SARIMA",
        "rf": "Random Forest", "xgb": "XGBoost", "lstm": "LSTM",
        "ens": "Ensemble (RF+XGB+LSTM)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 200, "savefig.bbox": "tight",
    "legend.frameon": False})


def load():
    p = pd.read_csv(RES / "predictions.csv",
                    parse_dates=["origin_week", "target_week"])
    p["base"] = p["model"].str.replace(r"_s\d+$", "", regex=True)
    # CHANGES_v2.md #2: equal-weight ensemble, weights fixed at 1/3 (not tuned);
    # the LSTM member is the mean of its five seeds.
    keys = ["h", "origin"]
    lstm = p[p.base == "lstm"].groupby(keys, as_index=False)["pred"].mean()
    members = [p[p.model == "rf"][keys + ["pred"]], p[p.model == "xgb"][keys + ["pred"]], lstm]
    if all(len(m) for m in members):
        ens = pd.concat(members).groupby(keys, as_index=False)["pred"].mean()
        base = p[p.model == "rf"].drop(columns="pred")
        ens = base.merge(ens, on=keys).assign(model="ens", base="ens")
        p = pd.concat([p, ens[p.columns]], ignore_index=True)
    return p


def fold_year(q):
    """Label a fold by the calendar year most of its target weeks fall in."""
    return str(q.target_week.dt.year.mode().iloc[0])


def seed_models(p):
    return sorted(m for m in p["model"].unique() if m.startswith("lstm_s"))


def representative_lstm(p, h):
    """The LSTM seed with the median pooled relative MAE at horizon h."""
    s = seed_models(p)
    if not s:
        return None
    r = {m: accuracy(*[p[(p.model == m) & (p.h == h)][c].values
                       for c in ("actual", "pred", "naive")])["rel_mae"]
         for m in s}
    return sorted(r, key=r.get)[len(r) // 2]


# ------------------------------------------------------------------ tables --
def t_accuracy(p):
    rows = []
    for h in HORIZONS:
        for m in p["model"].unique():
            q = p[(p.model == m) & (p.h == h)]
            rows.append({"model": m, "h": h,
                         **accuracy(q.actual.values, q.pred.values, q.naive.values)})
    a = pd.DataFrame(rows)
    a["base"] = a["model"].str.replace(r"_s\d+$", "", regex=True)
    agg = a.groupby(["base", "h"]).agg(
        rel_mae=("rel_mae", "mean"), rel_mae_sd=("rel_mae", "std"),
        mae=("mae", "mean"), rmse=("rmse", "mean"),
        skill_rmse_pct=("skill_rmse_pct", "mean"), n_seeds=("model", "size")
    ).reset_index()
    agg.loc[agg.n_seeds == 1, "rel_mae_sd"] = np.nan
    a.to_csv(TAB / "accuracy_per_seed.csv", index=False)
    agg.to_csv(TAB / "T4_1_accuracy.csv", index=False)
    return agg


def t_per_fold(p):
    rows = []
    for (m, h, k), q in p.groupby(["base", "h", "fold"]):
        acc = accuracy(q.actual.values, q.pred.values, q.naive.values) \
            if m != "lstm" else None
        if m == "lstm":   # mean over seeds within fold
            accs = [accuracy(g.actual.values, g.pred.values, g.naive.values)
                    for _, g in q.groupby("model")]
            acc = {k2: np.mean([x[k2] for x in accs]) for k2 in accs[0]}
        one = q[q.model == q.model.iloc[0]]
        rows.append({"model": m, "h": h, "fold": k, "year": fold_year(q),
                     "outbreak_wks": int((one.actual > one.thr).sum()),
                     "thr": one.thr.iloc[0], **acc})
    f = pd.DataFrame(rows)
    f["period"] = np.where(f.outbreak_wks > 0, "epidemic", "quiet")
    f.to_csv(TAB / "T4_2_per_fold.csv", index=False)
    # epidemic vs quiet pooled
    rows = []
    folds = f[["h", "fold", "period"]].drop_duplicates()
    q2 = p.merge(folds, on=["h", "fold"])
    for (m, h, per), q in q2.groupby(["model", "h", "period"]):
        rows.append({"model": m, "h": h, "period": per,
                     **accuracy(q.actual.values, q.pred.values, q.naive.values)})
    e = pd.DataFrame(rows)
    e["base"] = e.model.str.replace(r"_s\d+$", "", regex=True)
    e = e.groupby(["base", "h", "period"]).mean(numeric_only=True).reset_index()
    e.to_csv(TAB / "T4_3_epidemic_vs_quiet.csv", index=False)
    return f, e


def t_alarm(p):
    rows = []
    for (m, h), q in p.groupby(["model", "h"]):
        cm = {"tp": 0, "fn": 0, "fp": 0, "tn": 0}
        for _, g in q.groupby("fold"):
            c = confusion(g.actual.values, g.pred.values, g.thr.iloc[0])
            cm = {k: cm[k] + c[k] for k in cm}
        eps, det = episode_detection(q[["target_week", "actual", "pred", "thr"]])
        rows.append({"model": m, "h": h, **pooled_alarm(cm),
                     "episodes": eps, "episodes_detected": det})
    a = pd.DataFrame(rows)
    a["base"] = a.model.str.replace(r"_s\d+$", "", regex=True)
    a.to_csv(TAB / "alarm_per_seed.csv", index=False)
    agg = a.groupby(["base", "h"]).mean(numeric_only=True).reset_index()
    agg.to_csv(TAB / "T4_4_alarm.csv", index=False)
    return agg


def t_dm(p, acc):
    """Pre-specified comparisons per horizon (CHANGES_v2.md #3): each candidate
    vs naive (4), the ensemble vs naive (1) and the best vs runner-up among the
    four individual techniques (1). The raw p within this restricted set is the
    primary inference, as in the CP1 proposal; Holm across all tests is a
    robustness check. Absolute-error loss, matching the headline metric."""
    rows = []
    for h in HORIZONS:
        rep = {c: c for c in CANDIDATES}
        rep["lstm"] = representative_lstm(p, h)
        series = {}
        for c, m in rep.items():
            if m is None:
                continue
            q = p[(p.model == m) & (p.h == h)].sort_values("origin")
            series[c] = (q.actual - q.pred).values
        q = p[(p.model == "naive") & (p.h == h)].sort_values("origin")
        e_naive = (q.actual - q.pred).values
        for c, e in series.items():
            rows.append({"h": h, "comparison": f"{c} vs naive",
                         **diebold_mariano(e, e_naive, h)})
        q = p[(p.model == "ens") & (p.h == h)].sort_values("origin")
        if len(q):
            rows.append({"h": h, "comparison": "ens vs naive",
                         **diebold_mariano((q.actual - q.pred).values, e_naive, h)})
        r = acc[(acc.h == h) & acc.base.isin(series)].sort_values("rel_mae")
        b, ru = r.base.iloc[0], r.base.iloc[1]
        rows.append({"h": h, "comparison": f"{b} vs {ru} (best vs runner-up)",
                     **diebold_mariano(series[b], series[ru], h)})
    d = pd.DataFrame(rows)
    d["p_holm"] = holm(d.p_value)
    d.to_csv(TAB / "T4_5_diebold_mariano.csv", index=False)
    return d


def t_alarm_bootstrap(p, block=8, reps=2000, seed=0):
    """CHANGES_v2.md #4: paired moving-block bootstrap of the difference in
    pooled sensitivity and specificity, model minus naive, per horizon.
    Origins are resampled in blocks of `block` consecutive weeks so the
    autocorrelation of outbreak weeks is preserved."""
    rng = np.random.default_rng(seed)
    out = []
    p2 = p.copy()
    lstm = p2[p2.base == "lstm"].groupby(["h", "origin"], as_index=False)["pred"].mean()
    for h in HORIZONS:
        nv = p2[(p2.model == "naive") & (p2.h == h)].sort_values("origin")
        a = (nv.actual > nv.thr).values
        pn = (nv.pred > nv.thr).values
        n = len(nv)
        starts = rng.integers(0, n - block + 1, size=(reps, int(np.ceil(n / block))))
        idx = (starts[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :n]
        for m in ["rf", "xgb", "lstm", "ens"]:
            if m == "lstm":
                q = nv[["origin"]].merge(lstm[lstm.h == h], on="origin")
            else:
                q = nv[["origin"]].merge(p2[(p2.model == m) & (p2.h == h)][["origin", "pred"]], on="origin")
            if len(q) != n:
                continue
            pm = (q.pred.values > nv.thr.values)
            def stats_(ii):
                aa, mm, nn = a[ii], pm[ii], pn[ii]
                pos, neg = aa.sum(), (~aa).sum()
                ds = ((mm & aa).sum() - (nn & aa).sum()) / pos if pos else np.nan
                dp = ((~mm & ~aa).sum() - (~nn & ~aa).sum()) / neg if neg else np.nan
                return ds, dp
            est = stats_(np.arange(n))
            bs = np.array([stats_(ii) for ii in idx])
            out.append({"model": m, "h": h, "d_sens": est[0],
                        "d_sens_lo": np.nanpercentile(bs[:, 0], 2.5),
                        "d_sens_hi": np.nanpercentile(bs[:, 0], 97.5),
                        "p_sens_le0": float(np.mean(bs[:, 0] <= 0)),
                        "d_spec": est[1],
                        "d_spec_lo": np.nanpercentile(bs[:, 1], 2.5),
                        "d_spec_hi": np.nanpercentile(bs[:, 1], 97.5)})
    b = pd.DataFrame(out)
    b.to_csv(TAB / "T4_9_alarm_bootstrap.csv", index=False)
    return b


def t_features():
    pi = pd.read_csv(RES / "permutation_importance.csv")
    g = pi.drop(columns="fold").groupby(["model", "h"]).mean().reset_index()
    g.to_csv(TAB / "T4_6_permutation_importance.csv", index=False)
    ab = pd.read_csv(RES / "ablation.csv")
    a = ab.groupby(["set", "model", "h"])[["abs_err_sum", "naive_abs_err_sum"]].sum()
    a["rel_mae"] = a.abs_err_sum / a.naive_abs_err_sum
    a = a.reset_index()[["set", "model", "h", "rel_mae"]]
    a.to_csv(TAB / "T4_7_ablation.csv", index=False)
    ts = pd.read_csv(RES / "target_spec.csv")
    t = ts.groupby("spec")[["abs_err_sum", "naive_abs_err_sum"]].sum()
    t["rel_mae"] = t.abs_err_sum / t.naive_abs_err_sum
    t.to_csv(TAB / "T4_8_target_spec.csv")
    return g, a, t


# ----------------------------------------------------------------- figures --
def f_leadtime(acc):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ax.text(6, 0.993, "Naive = 1", va="top", ha="center", color=MUTED, fontsize=8)
    ends = []
    for m in ["sarima", "rf", "xgb", "lstm", "ens"]:
        r = acc[acc.base == m].sort_values("h")
        if r.empty:
            continue
        ax.plot(r.h, r.rel_mae, color=COL[m], lw=2, marker="o", ms=5,
                label=NAME[m], mec="white", mew=1)
        if m == "lstm" and r.rel_mae_sd.notna().any():
            ax.fill_between(r.h, r.rel_mae - r.rel_mae_sd,
                            r.rel_mae + r.rel_mae_sd, color=COL[m], alpha=.15,
                            lw=0)
        ends.append([r.rel_mae.iloc[-1], NAME[m]])
    # de-collide end labels: enforce a minimum vertical gap in data units
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for yv, lab in ends:
        ax.text(8.25, yv, lab, va="center", color=INK, fontsize=8)
    ax.set_xticks(HORIZONS, [f"{h} wk" for h in HORIZONS])
    ax.set_xlim(0.6, 9.6)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Relative MAE (model / naive)")
    ax.set_ylim(top=max(1.06, ax.get_ylim()[1]))
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, ncol=4)
    fig.savefig(FIG / "fig4_1_leadtime_relmae.png")
    plt.close(fig)


def f_per_fold(f, h=4):
    q = f[(f.h == h) & f.model.isin(["sarima", "rf", "xgb", "lstm"])]
    years = sorted(q.year.unique())
    ms = ["sarima", "rf", "xgb", "lstm"]
    w = 0.19
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    for i, m in enumerate(ms):
        r = q[q.model == m].set_index("year").reindex(years)
        ax.bar(np.arange(len(years)) + (i - 1.5) * w, r.rel_mae, w - 0.02,
               color=COL[m], label=NAME[m])
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ob = q[q.model == "rf"].set_index("year").reindex(years).outbreak_wks
    ax.set_xticks(range(len(years)),
                  [f"{y}\n({int(o)})" for y, o in zip(years, ob)])
    ax.set_xlabel("Test fold (outbreak weeks in fold)")
    ax.set_ylabel(f"Relative MAE, h = {h} wk")
    ax.legend(fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0, 1.12))
    fig.savefig(FIG / f"fig4_2_per_fold_h{h}.png")
    plt.close(fig)


def f_forecast(p, model, h=4):
    q = p[(p.model == model) & (p.h == h)].sort_values("target_week")
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(q.target_week, q.actual, color=INK, lw=1.4, label="Observed")
    ax.plot(q.target_week, q.pred, color=COL[model.split("_")[0]], lw=1.6,
            label=f"{NAME[model.split('_')[0]]} forecast ({h} wk ahead)")
    for k, g in q.groupby("fold"):
        ax.hlines(g.thr.iloc[0], g.target_week.min(), g.target_week.max(),
                  color="#e34948", lw=1, ls=":")
    ax.plot([], [], color="#e34948", lw=1, ls=":",
            label="Outbreak threshold (fold q75)")
    ax.set_ylabel("Weekly cases")
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(FIG / f"fig4_3_forecast_{model}_h{h}.png")
    plt.close(fig)


def f_alarm(al):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    for ax, met in zip(axes, ["sensitivity", "specificity"]):
        for m in ["naive", "sarima", "rf", "xgb", "lstm"]:
            r = al[al.base == m].sort_values("h")
            if r.empty:
                continue
            ax.plot(r.h, r[met], color=COL[m], lw=2, marker="o", ms=5,
                    mec="white", mew=1, label=NAME[m],
                    ls="--" if m == "naive" else "-")
        ax.set_title(met.capitalize(), color=INK, fontsize=9, loc="left")
        ax.set_xticks(HORIZONS, [f"{h} wk" for h in HORIZONS])
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Pooled across folds")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.savefig(FIG / "fig4_4_alarm_leadtime.png")
    plt.close(fig)


def f_importance(g):
    groups = ["cases", "rain", "temp", "humidity", "calendar", "regime"]
    q = g[g.h == 4]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    w = 0.38
    for i, m in enumerate(["rf", "xgb"]):
        r = q[q.model == m]
        vals = [float(r[c].iloc[0]) if c in r else 0 for c in groups]
        ax.barh(np.arange(len(groups)) + (i - .5) * w, vals, w - .03,
                color=COL[m], label=NAME[m])
    ax.set_yticks(range(len(groups)), [c.capitalize() for c in groups])
    ax.invert_yaxis()
    ax.set_xlabel("MAE increase when group is permuted (cases/week), h = 4 wk")
    ax.legend(fontsize=8)
    fig.savefig(FIG / "fig4_5_feature_importance_h4.png")
    plt.close(fig)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    p = load()
    acc = t_accuracy(p)
    f, e = t_per_fold(p)
    al = t_alarm(p)
    dm = t_dm(p, acc)
    bb = t_alarm_bootstrap(p)
    g, ab, ts = t_features()
    f_leadtime(acc)
    f_per_fold(f)
    best = acc[(acc.h == 4) & acc.base.isin(CANDIDATES)].sort_values("rel_mae").base.iloc[0]
    f_forecast(p, best if best != "lstm" else representative_lstm(p, 4))
    f_alarm(al)
    f_importance(g)
    summ = {"best_model_h4": best}
    (RES / "report_summary.json").write_text(json.dumps(summ, indent=2))
    pd.set_option("display.width", 160)
    print(acc.round(3).to_string(index=False))
    print(dm.round(4).to_string(index=False))
    print(bb.round(3).to_string(index=False))
    print(al[["base", "h", "sensitivity", "specificity", "episodes",
              "episodes_detected"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
