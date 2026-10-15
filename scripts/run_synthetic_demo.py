"""Run the public PEC-Fuse contract on non-private synthetic rows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pec_fuse import (
    FEATURES_25,
    STRONG_CONTROL_FEATURES,
    PECFuseRanker,
    aggregate_metrics,
    build_feature_frame,
    query_metrics,
    rank_within_query,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    inputs = pd.read_csv(ROOT / "examples" / "synthetic" / "candidate_inputs.csv")
    rows = build_feature_frame(inputs)
    train = rows[rows["split"].eq("train")].copy()
    test = rows[rows["split"].eq("test")].copy()

    configs = json.loads((ROOT / "configs" / "extra_trees.json").read_text())
    regime = configs["recent_window"]

    control = PECFuseRanker(features=STRONG_CONTROL_FEATURES, **regime["strong_control"])
    full = PECFuseRanker(features=FEATURES_25, **regime["pec_fuse"])
    control.fit(train)
    full.fit(train)

    control_ranked = rank_within_query(test, control.score(test), score_col="control_score")
    full_ranked = rank_within_query(test, full.score(test), score_col="pec_fuse_score")
    control_metrics = aggregate_metrics(query_metrics(control_ranked, score_col="control_score"))
    full_metrics = aggregate_metrics(query_metrics(full_ranked, score_col="pec_fuse_score"))

    display = full_ranked[["query_id", "candidate_id", "label", "pec_fuse_score", "rank"]]
    print(display.to_string(index=False))
    print("strong_control", control_metrics)
    print("pec_fuse", full_metrics)
    print("Synthetic scores are an interface demonstration, not a paper result.")


if __name__ == "__main__":
    main()
