"""Metrics, alarm evaluation, Diebold-Mariano test and the structural break
test. The DM test is written out here (not in statsmodels) -- Appendix code."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


# --------------------------------------------------------------- accuracy --
def accuracy(actual, pred, naive):
    e, en = actual - pred, actual - naive
    mae_m, mae_n = np.mean(np.abs(e)), np.mean(np.abs(en))
    rmse_m, rmse_n = np.sqrt(np.mean(e ** 2)), np.sqrt(np.mean(en ** 2))
    return {"rel_mae": mae_m / mae_n if mae_n else np.nan,
            "mae": mae_m, "rmse": rmse_m,
            "skill_rmse_pct": 100 * (1 - rmse_m / rmse_n) if rmse_n else np.nan}


# --------------------------------------------------------- Diebold-Mariano --
def diebold_mariano(e1, e2, h, loss="abs"):
    """H0: equal predictive accuracy. Loss differential d = L(e1) - L(e2);
    negative mean => model 1 more accurate. Long-run variance uses
    autocovariances up to lag h-1 (h-step forecasts are MA(h-1) under
    optimality); Harvey-Leybourne-Newbold small-sample correction; t(n-1)."""
    L = (lambda e: np.abs(e)) if loss == "abs" else (lambda e: e ** 2)
    d = L(np.asarray(e1)) - L(np.asarray(e2))
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    gam = [np.dot(dc[k:], dc[: n - k]) / n for k in range(h)]
    var = gam[0] + 2 * sum(gam[1:])
    if var <= 0:                                   # fall back to Bartlett weights
        var = gam[0] + 2 * sum((1 - k / h) * gam[k] for k in range(1, h))
    if var <= 0 or not np.isfinite(var):           # identical forecasts
        return {"dm_stat": 0.0, "p_value": 1.0, "mean_loss_diff": dbar, "n": n}
    dm = dbar / np.sqrt(var / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    stat = hln * dm
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return {"dm_stat": stat, "p_value": p, "mean_loss_diff": dbar, "n": n}


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m, running = len(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ------------------------------------------------------------------ alarms --
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    m = (p + z * z / (2 * n)) / (1 + z * z / n)
    return (m - c, m + c)


def confusion(actual, pred, thr):
    a, p = actual > thr, pred > thr
    return {"tp": int((a & p).sum()), "fn": int((a & ~p).sum()),
            "fp": int((~a & p).sum()), "tn": int((~a & ~p).sum())}


def pooled_alarm(cm):
    tp, fn, fp, tn = cm["tp"], cm["fn"], cm["fp"], cm["tn"]
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    lo_s, hi_s = wilson(tp, tp + fn)
    lo_p, hi_p = wilson(tn, tn + fp)
    return {**cm, "sensitivity": sens, "sens_lo": lo_s, "sens_hi": hi_s,
            "specificity": spec, "spec_lo": lo_p, "spec_hi": hi_p, "ppv": ppv}


def episode_detection(frame):
    """frame: one model, one horizon; columns target_week, actual, pred, thr.
    An episode = a run of consecutive outbreak weeks (actual > thr) by target
    week. Detected if the forecast issued h weeks before the FIRST outbreak
    week of the run already exceeded the threshold."""
    f = frame.sort_values("target_week").reset_index(drop=True)
    out = (f["actual"] > f["thr"]).values
    eps, det = 0, 0
    for i in range(len(f)):
        new = out[i] and (i == 0 or not out[i - 1]
                          or (f["target_week"][i] - f["target_week"][i - 1]).days > 7)
        if new:
            eps += 1
            det += int(f["pred"][i] > f["thr"][i])
    return eps, det


# --------------------------------------------------------- structural break --
def break_test(start, y, break_date, K=3, hac_lags=8):
    """log(y) ~ trend + K Fourier pairs (+ regime dummy). Reports (a) the HAC
    t-test on the regime level shift and (b) a Chow F-test that all
    coefficients differ either side of the break."""
    t = np.arange(len(y))
    X = {"trend": t / 52}
    for k in range(1, K + 1):
        X[f"s{k}"] = np.sin(2 * np.pi * k * t / 52.18)
        X[f"c{k}"] = np.cos(2 * np.pi * k * t / 52.18)
    X = sm.add_constant(pd.DataFrame(X))
    z = np.log1p(np.asarray(y, float))
    D = (pd.to_datetime(start) >= pd.Timestamp(break_date)).astype(int).values
    Xd = X.assign(regime=D)
    r = sm.OLS(z, Xd).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    # Chow
    rss_p = sm.OLS(z, X).fit().ssr
    rss_1 = sm.OLS(z[D == 0], X[D == 0]).fit().ssr
    rss_2 = sm.OLS(z[D == 1], X[D == 1]).fit().ssr
    k, n = X.shape[1], len(z)
    F = ((rss_p - rss_1 - rss_2) / k) / ((rss_1 + rss_2) / (n - 2 * k))
    pF = stats.f.sf(F, k, n - 2 * k)
    return {"shift_coef_log": r.params["regime"],
            "shift_pct": 100 * (np.exp(r.params["regime"]) - 1),
            "shift_t_hac": r.tvalues["regime"], "shift_p_hac": r.pvalues["regime"],
            "chow_F": F, "chow_p": pF, "n_pre": int((D == 0).sum()),
            "n_post": int((D == 1).sum())}
