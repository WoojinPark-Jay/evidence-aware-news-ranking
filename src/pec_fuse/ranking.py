"""Validation-configured ExtraTrees fusion and within-query ranking."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from .schema import FEATURES_25


class PECFuseRanker:
    """Pointwise fusion scorer whose outputs are sorted within each query."""

    def __init__(
        self,
        *,
        features: Sequence[str] = FEATURES_25,
        n_estimators: int,
        max_depth: int | None,
        min_samples_leaf: int,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.features = tuple(features)
        self.model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, frame: pd.DataFrame, label_col: str = "label") -> "PECFuseRanker":
        self.model.fit(frame.loc[:, self.features], frame[label_col].astype(int))
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(frame.loc[:, self.features])[:, 1]


def rank_within_query(
    frame: pd.DataFrame,
    scores: Sequence[float],
    *,
    query_col: str = "query_id",
    score_col: str = "score",
) -> pd.DataFrame:
    """Attach scores and return descending rank within each query."""

    out = frame.copy()
    out[score_col] = np.asarray(scores, dtype=float)
    out["rank"] = out.groupby(query_col)[score_col].rank(
        method="first", ascending=False
    ).astype(int)
    return out.sort_values([query_col, "rank"]).reset_index(drop=True)
