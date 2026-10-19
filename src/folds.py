"""Rolling-origin fold construction.

Row t is a forecast ORIGIN: everything up to and including week t is known;
the target is y[t+h].

CP1 correction: the CP1 harness trained on every row before the first test
origin o, including rows t in (o-h, o) whose targets y[t+h] lie AFTER o -- i.e.
future data relative to the forecast origin. Here a training row is admitted
only if its target is already observed at the origin: t + h <= o.
"""
import numpy as np

from .config import HORIZONS, INNER_VAL, STEP, TEST_LEN, TRAIN_MIN


def n_folds_for(n):
    """Every fold whose first origin can still be scored at the longest
    horizon. Same count for every horizon, so fold calendars match."""
    k = 0
    while TRAIN_MIN + k * STEP < n - max(HORIZONS):
        k += 1
    return k


def outer_folds(n, h, n_folds=None, first_valid=0):
    """Yield (k, train_idx, test_idx). Fold test windows start at the same
    calendar origins for every horizon, so horizons are comparable."""
    for k in range(n_folds_for(n) if n_folds is None else n_folds):
        o = TRAIN_MIN + k * STEP
        test = np.arange(o, min(o + TEST_LEN, n - h))
        train = np.arange(first_valid, o - h + 1)     # t + h <= o
        if len(test):
            yield k, train, test


def outbreak_threshold(y, origin, q=None):
    """Fold outbreak threshold: q-quantile of cases observed up to the fold's
    first origin only -- never test-period data (Section 3.6.3)."""
    from .config import OUTBREAK_Q
    return float(np.quantile(np.asarray(y)[: origin + 1],
                             OUTBREAK_Q if q is None else q))


def inner_split(train_idx, h):
    """Expanding-window inner split for tuning: last INNER_VAL origins of the
    training window validate; inner-train rows satisfy t + h <= first val
    origin. The outer test fold is never touched."""
    v0 = train_idx[-INNER_VAL]
    val = train_idx[train_idx >= v0]
    itr = train_idx[train_idx + h <= v0]
    return itr, val
