"""Point-in-time (no look-ahead) checks shared by all phases (plan Section 6)."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def perturb_after(df: pd.DataFrame | pd.Series, t: pd.Timestamp, seed: int = 0):
    """Copy of ``df`` whose rows strictly after ``t`` are scrambled.

    Parameters
    ----------
    df : DataFrame or Series indexed by date.
    t : cut-off date; rows dated <= t are left untouched.
    seed : random seed for the scrambling.

    Returns
    -------
    Same type as ``df``; rows after ``t`` are multiplied by U(0.5, 1.5) noise, so
    positive values stay positive and NaNs stay NaN.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    mask = out.index > t
    out.loc[mask] = out.loc[mask] * rng.uniform(0.5, 1.5, size=out.loc[mask].shape)
    return out


def assert_point_in_time(func: Callable, data, t: pd.Timestamp, seed: int = 0) -> None:
    """Assert that every output row dated <= t is unchanged when data after t changes.

    Parameters
    ----------
    func : maps ``data`` to a DataFrame/Series (or a tuple of them) indexed by date.
    data : DataFrame or Series indexed by date.
    t : cut-off date.
    seed : random seed passed to ``perturb_after``.

    Raises
    ------
    AssertionError if any output row dated <= t changes, or if no output row is
    dated <= t (the check would be vacuous).
    """
    base = func(data)
    moved = func(perturb_after(data, t, seed))
    base = base if isinstance(base, tuple) else (base,)
    moved = moved if isinstance(moved, tuple) else (moved,)
    for b, m in zip(base, moved):
        assert (b.index <= t).any(), "no output rows at or before t: test is vacuous"
        if isinstance(b, pd.Series):
            pd.testing.assert_series_equal(b[b.index <= t], m[m.index <= t])
        else:
            pd.testing.assert_frame_equal(b[b.index <= t], m[m.index <= t])
