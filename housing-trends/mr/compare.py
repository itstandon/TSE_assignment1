"""Tolerance-aware comparison of two contract tables. All helpers return a list of human-readable problems
(empty list = the relation holds), capped so a badly broken run does not flood the logs."""
from __future__ import annotations

import numpy as np
import pandas as pd

CAP = 5


def _fmt(v):
    return "NaN" if v != v else f"{v:.6g}"


def mismatches(expected: pd.DataFrame, actual: pd.DataFrame, cols, *, rtol: float = 1e-9, atol: float = 1e-9, months=None):
    """Months and columns where `actual` differs from `expected` beyond atol + rtol*max(|a|,|b|). NaN must match NaN."""
    months = list(expected.index if months is None else months)
    out = []
    with np.errstate(invalid="ignore"):
        for col in cols:
            x = expected[col].reindex(months).to_numpy(float)
            y = actual[col].reindex(months).to_numpy(float)
            same = (np.isnan(x) & np.isnan(y)) | (np.abs(x - y) <= atol + rtol * np.maximum(np.abs(x), np.abs(y)))
            out += [f"{col} {m}: expected {_fmt(a)}, got {_fmt(b)}" for m, a, b, ok in zip(months, x, y, same) if not ok]
    return out[:CAP]


def not_below(lower: pd.DataFrame, upper: pd.DataFrame, col: str, *, tol: float = 0.0, what: str = ""):
    """Months where upper[col] < lower[col] - tol (i.e. `upper` should be >= `lower`). NaN on both sides is fine;
    NaN on exactly one side is reported."""
    months = list(lower.index)
    x, y = lower[col].reindex(months).to_numpy(float), upper[col].reindex(months).to_numpy(float)
    out = []
    with np.errstate(invalid="ignore"):
        for m, a, b in zip(months, x, y):
            if np.isnan(a) and np.isnan(b):
                continue
            if np.isnan(a) != np.isnan(b) or b < a - tol:
                out.append(f"{col} {m}: {what or 'must not decrease'} ({_fmt(a)} -> {_fmt(b)})")
    return out[:CAP]
