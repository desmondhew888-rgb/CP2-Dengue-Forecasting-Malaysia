"""The six models. Feature-based learners predict the CHANGE y[t+h]-y[t]
(Section 3.5.6); the level forecast is recovered by adding y[t]."""
import itertools
import warnings

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBRegressor

from .config import (LSTM_FIXED, LSTM_GRID, LSTM_WINDOW, RF_GRID, SARIMA_P,
                     SARIMA_Q, SARIMA_SP, SARIMA_SQ, SEASON, SEED, XGB_FIXED,
                     XGB_GRID)
from .folds import inner_split

warnings.filterwarnings("ignore")
torch.set_num_threads(4)


def grid(g):
    keys = list(g)
    return [dict(zip(keys, v)) for v in itertools.product(*g.values())]


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


# ---------------------------------------------------------------- baselines --
def naive(y, idx, h):
    return y[idx]


def snaive(y, idx, h):
    return y[idx + h - SEASON]


# ------------------------------------------------------------------- SARIMA --
def choose_d(z):
    return 0 if adfuller(z, autolag="AIC")[1] < 0.05 else 1


def select_sarima(y_train):
    """Two-stage AIC identification on the FIRST training window only; orders
    are then held fixed across folds and horizons (parameters re-estimated)."""
    z = np.log1p(y_train)
    d = choose_d(z)
    best = (np.inf, None)
    for p, q in itertools.product(SARIMA_P, SARIMA_Q):
        try:
            r = SARIMAX(z, order=(p, d, q), trend="c" if d == 0 else "n"
                        ).fit(disp=False)
            best = min(best, (r.aic, (p, d, q)), key=lambda x: x[0])
        except Exception:
            pass
    order = best[1]
    sbest = (best[0], (0, 0, 0, SEASON))
    for P, Q in itertools.product(SARIMA_SP, SARIMA_SQ):
        if P == Q == 0:
            continue
        try:
            r = SARIMAX(z, order=order, seasonal_order=(P, 0, Q, SEASON),
                        trend="c" if d == 0 else "n").fit(disp=False,
                                                          maxiter=200)
            if r.aic < sbest[0]:
                sbest = (r.aic, (P, 0, Q, SEASON))
        except Exception:
            pass
    return order, sbest[1]


def sarima_fold(y, train_end, test_idx, horizons, order, sorder):
    """Fit on y[:train_end+1]; then filter forward one week at a time with
    fixed parameters, forecasting every horizon from every test origin."""
    z = np.log1p(y)
    d = order[1]
    res = SARIMAX(z[: train_end + 1], order=order, seasonal_order=sorder,
                  trend="c" if d == 0 else "n").fit(disp=False, maxiter=200)
    out = {h: {} for h in horizons}
    t = train_end
    for o in test_idx:
        if o > t:
            res = res.extend(z[t + 1: o + 1])
            t = o
        f = res.forecast(max(horizons))
        for h in horizons:
            out[h][o] = float(np.expm1(f[h - 1]))
    return out


# ------------------------------------------------------------- tree models --
def _fit_tree(kind, params, X, yd, seed=SEED):
    if kind == "rf":
        m = RandomForestRegressor(random_state=seed, n_jobs=-1, **params)
    else:
        m = XGBRegressor(random_state=seed, n_jobs=4, verbosity=0,
                         **XGB_FIXED, **params)
    return m.fit(X, yd)


def tree_fold(kind, X, y, train_idx, test_idx, h):
    yd = y[np.minimum(np.arange(len(y)) + h, len(y) - 1)] - y   # delta target
    itr, val = inner_split(train_idx, h)
    cands = grid(RF_GRID if kind == "rf" else XGB_GRID)
    scores = []
    for p in cands:
        m = _fit_tree(kind, p, X[itr], yd[itr])
        scores.append(mae(y[val + h], m.predict(X[val]) + y[val]))
    best = cands[int(np.argmin(scores))]
    m = _fit_tree(kind, best, X[train_idx], yd[train_idx])
    pred = m.predict(X[test_idx]) + y[test_idx]
    return pred, best, m


# -------------------------------------------------------------------- LSTM --
class Net(torch.nn.Module):
    def __init__(self, n_in, units, layers, dropout):
        super().__init__()
        self.rnn = torch.nn.LSTM(n_in, units, num_layers=layers,
                                 batch_first=True,
                                 dropout=dropout if layers > 1 else 0.0)
        self.drop = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(units, 1)

    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(self.drop(o[:, -1])).squeeze(-1)


def _seq(Xs, idx):
    L = LSTM_WINDOW
    return np.stack([Xs[i - L + 1: i + 1] for i in idx]).astype(np.float32)


def _train_lstm(Xs, yd_s, idx, p, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cut = int(len(idx) * 0.85)                      # chronological tail
    tr, es = idx[:cut], idx[cut:]
    Xt, yt = torch.tensor(_seq(Xs, tr)), torch.tensor(yd_s[tr], dtype=torch.float32)
    Xe, ye = torch.tensor(_seq(Xs, es)), torch.tensor(yd_s[es], dtype=torch.float32)
    net = Net(Xs.shape[1], p["units"], p["layers"], p["dropout"])
    opt = torch.optim.Adam(net.parameters(), lr=LSTM_FIXED["lr"])
    lossf = torch.nn.MSELoss()
    best, best_state, wait = np.inf, None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(LSTM_FIXED["max_epochs"]):
        net.train()
        perm = torch.randperm(len(Xt), generator=g)
        for b in range(0, len(Xt), LSTM_FIXED["batch"]):
            j = perm[b: b + LSTM_FIXED["batch"]]
            opt.zero_grad()
            lossf(net(Xt[j]), yt[j]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = lossf(net(Xe), ye).item()
        if v < best - 1e-6:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
        else:
            wait += 1
            if wait >= LSTM_FIXED["patience"]:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net


def _predict_lstm(net, Xs, idx):
    with torch.no_grad():
        return net(torch.tensor(_seq(Xs, idx))).numpy()


def lstm_fold(X, y, train_idx, test_idx, h, seeds):
    """Scaling statistics come from the training rows only."""
    train_idx = train_idx[train_idx >= train_idx[0] + LSTM_WINDOW - 1]
    mu, sd = X[train_idx].mean(0), X[train_idx].std(0) + 1e-8
    Xs = (X - mu) / sd
    yd = y[np.minimum(np.arange(len(y)) + h, len(y) - 1)] - y
    dmu, dsd = yd[train_idx].mean(), yd[train_idx].std() + 1e-8
    yd_s = (yd - dmu) / dsd
    itr, val = inner_split(train_idx, h)
    cands = grid(LSTM_GRID)
    scores = []
    for p in cands:
        net = _train_lstm(Xs, yd_s, itr, p, seeds[0])
        pv = _predict_lstm(net, Xs, val) * dsd + dmu + y[val]
        scores.append(mae(y[val + h], pv))
    best = cands[int(np.argmin(scores))]
    preds = {}
    for s in seeds:
        net = _train_lstm(Xs, yd_s, train_idx, best, s)
        preds[s] = _predict_lstm(net, Xs, test_idx) * dsd + dmu + y[test_idx]
    return preds, best
