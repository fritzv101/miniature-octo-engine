"""Splitting strategies and regression metrics.

Random splits over-state performance in medicinal chemistry because analogues of
the same series land on both sides; scaffold and group splits are the honest
defaults here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd

from oncomap.chem import murcko_scaffold


def scaffold_split(
    smiles: Sequence[str], test_fraction: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/test row indices with no Murcko scaffold shared between them."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[murcko_scaffold(s) or s].append(i)
    return _assign_groups(list(groups.values()), len(smiles), test_fraction, seed)


def group_split(
    groups: Sequence[str], test_fraction: float = 0.2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole groups (e.g. unseen compounds or unseen cell lines)."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        buckets[str(g)].append(i)
    return _assign_groups(list(buckets.values()), len(groups), test_fraction, seed)


def _assign_groups(
    buckets: list[list[int]], n: int, test_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(buckets))
    target = test_fraction * n
    test: list[int] = []
    for idx in order:
        if len(test) >= target:
            break
        test.extend(buckets[idx])
    test_set = set(test)
    train = np.array([i for i in range(n) if i not in test_set], dtype=int)
    return train, np.array(sorted(test_set), dtype=int)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return {"n": 0}
    resid = y_true - y_pred
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    metrics = {
        "n": int(y_true.size),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "r2": float(1 - (resid**2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
    }
    if y_true.size > 2 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        metrics["pearson"] = float(np.corrcoef(y_true, y_pred)[0, 1])
        metrics["spearman"] = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
    return metrics


def per_group_metrics(
    frame: pd.DataFrame, group_col: str, true_col: str, pred_col: str, min_n: int = 5
) -> pd.DataFrame:
    """Metrics per target / per compound, e.g. to find targets the model can't learn."""
    rows = []
    for key, sub in frame.groupby(group_col):
        if len(sub) < min_n:
            continue
        m = regression_metrics(sub[true_col].to_numpy(), sub[pred_col].to_numpy())
        rows.append({group_col: key, **m})
    return pd.DataFrame(rows)
