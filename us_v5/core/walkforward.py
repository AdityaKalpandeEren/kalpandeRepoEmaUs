"""
Purged, embargoed, expanding-window walk-forward (Lopez de Prado style).

For a test block [test_start, test_end):
  train = rows whose label window ENDS before the embargo cut-off
          (label_end < cut, cut = test_start minus `embargo_days` sessions),
  so no training label overlaps the test period and serial correlation at
  the boundary can't leak. Test = rows dated inside the block.
Blocks are `refit_months` long, from validation.first_test up to (not
including) validation.holdout_start. The holdout is never touched here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_cut: pd.Timestamp       # train rows need label_end < train_cut


def folds(dates: pd.DatetimeIndex, cfg: dict) -> list[Fold]:
    v = cfg["validation"]
    dates = pd.DatetimeIndex(sorted(set(dates)))
    starts = pd.date_range(v["first_test"], v["holdout_start"], freq=f"{v['refit_months']}MS")
    out = []
    for s, e in zip(starts[:-1], starts[1:]):
        i = dates.searchsorted(s)
        cut = dates[max(i - v["embargo_days"], 0)]
        out.append(Fold(pd.Timestamp(s), pd.Timestamp(e), cut))
    return out


def split(df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[(df["label_end"] < fold.train_cut) & df["fwd_ret"].notna()]
    test = df[(df["date"] >= fold.test_start) & (df["date"] < fold.test_end)]
    return train, test


def holdout_split(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Final model: train on everything whose label ends before the holdout
    (minus embargo); test on the holdout dates."""
    v = cfg["validation"]
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    hs = pd.Timestamp(v["holdout_start"])
    cut = dates[max(dates.searchsorted(hs) - v["embargo_days"], 0)]
    train = df[(df["label_end"] < cut) & df["fwd_ret"].notna()]
    test = df[df["date"] >= hs]
    return train, test


def check_no_overlap(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise if any training label window reaches into the test period."""
    if len(train) and len(test):
        assert train["label_end"].max() < test["date"].min(), "label overlap between train and test"
        assert np.all(train["date"] < test["date"].min()), "train rows dated inside test"
