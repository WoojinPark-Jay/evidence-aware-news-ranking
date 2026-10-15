"""Deterministic construction of the PEC-Fuse candidate record."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import BASE_INPUT_FEATURES, FEATURES_25


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required candidate fields: {missing}")


def build_feature_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return candidate rows with the exact ordered PEC-Fuse record.

    The caller supplies source-specific summaries computed without target-label
    access. This function derives exposure, count summaries, and the six fixed
    cross-source interactions.
    """

    _require_columns(candidates, BASE_INPUT_FEATURES)
    out = candidates.copy()

    numeric = [column for column in BASE_INPUT_FEATURES if column != "retained_rank"]
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="raise")
    out["retained_rank"] = pd.to_numeric(out["retained_rank"], errors="coerce")

    membership = out["same_day_recommended"].astype(float)
    if not membership.isin([0.0, 1.0]).all():
        raise ValueError("same_day_recommended must be binary")

    valid_rank = out["retained_rank"].notna() & out["retained_rank"].gt(0)
    if (membership.eq(1.0) & ~valid_rank).any():
        raise ValueError("retained-trace members require a positive retained_rank")
    if (membership.eq(0.0) & valid_rank).any():
        raise ValueError("candidates outside the retained trace cannot have retained_rank")

    out["reciprocal_rank"] = np.where(
        membership.eq(1.0),
        1.0 / out["retained_rank"].astype(float),
        0.0,
    )
    out["P_count"] = (
        out["pref_category_weight"] + out["pref_publisher_weight"]
    ).clip(upper=1.0)
    out["C_count"] = (
        out["hist_category_click_share"] + out["hist_publisher_click_share"]
    ).clip(upper=1.0)
    out["E_exposure"] = membership + out["reciprocal_rank"]

    out["P_semantic_x_E"] = out["P_semantic"] * out["E_exposure"]
    out["C_semantic_x_E"] = out["C_semantic"] * out["E_exposure"]
    out["P_semantic_x_C_semantic"] = out["P_semantic"] * out["C_semantic"]
    out["C_attention_x_E"] = out["C_attention"] * out["E_exposure"]
    out["C_max_x_E"] = out["C_max"] * out["E_exposure"]
    out["P_semantic_x_C_attention"] = out["P_semantic"] * out["C_attention"]

    if not np.isfinite(out[list(FEATURES_25)].to_numpy(dtype=float)).all():
        raise ValueError("PEC-Fuse features must be finite and non-missing")
    return out
