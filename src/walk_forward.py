"""
walk_forward.py — Strict time-ordered walk-forward cross-validation.

Folds:
  Train 2010–2018 → Test 2019
  Train 2010–2019 → Test 2020
  Train 2010–2020 → Test 2021

No data leakage: test set is NEVER seen during training or scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

from src.utils import WALK_FORWARD_FOLDS, get_logger

log = get_logger(__name__)


@dataclass
class WalkForwardFold:
    fold_id: int
    train_end: int              # last training year (inclusive)
    test_year: int
    train_idx: np.ndarray       # integer positions in df
    test_idx: np.ndarray


def generate_folds(df: pd.DataFrame) -> list[WalkForwardFold]:
    """
    Generate WalkForwardFold objects for each entry in WALK_FORWARD_FOLDS.
    Rows outside all configured test years are silently skipped
    (we don't fail if the data doesn't reach 2021).

    Parameters
    ----------
    df : pd.DataFrame with DatetimeIndex

    Returns
    -------
    List of WalkForwardFold (may be shorter than WALK_FORWARD_FOLDS if data
    doesn't cover the test year).
    """
    folds = []

    for fold_id, spec in enumerate(WALK_FORWARD_FOLDS):
        train_end  = spec["train_end"]
        test_year  = spec["test_year"]

        train_mask = df.index.year <= train_end
        test_mask  = df.index.year == test_year

        if not train_mask.any():
            log.warning("Fold %d: no training data up to %d — skipping.", fold_id, train_end)
            continue
        if not test_mask.any():
            log.warning("Fold %d: no data for test year %d — skipping.", fold_id, test_year)
            continue

        folds.append(WalkForwardFold(
            fold_id   = fold_id,
            train_end = train_end,
            test_year = test_year,
            train_idx = np.where(train_mask)[0],
            test_idx  = np.where(test_mask)[0],
        ))
        log.info(
            "Fold %d | train: %d rows (up to %d) | test: %d rows (%d)",
            fold_id, train_mask.sum(), train_end, test_mask.sum(), test_year,
        )

    if not folds:
        raise ValueError(
            "No valid walk-forward folds could be constructed. "
            "Check that your data covers the years in WALK_FORWARD_FOLDS."
        )

    return folds
