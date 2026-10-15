"""Query-averaged ranking metrics and paired user-clustered intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _dcg(labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, labels.size + 2))
    return float(np.sum(labels * discounts))


def query_metrics(
    frame: pd.DataFrame,
    *,
    score_col: str = "score",
    query_col: str = "query_id",
    user_col: str = "user_id",
    label_col: str = "label",
    k: int = 10,
) -> pd.DataFrame:
    """Compute NDCG@k, Recall@k, and MRR for every query."""

    rows: list[dict[str, object]] = []
    for query_id, group in frame.groupby(query_col, sort=False):
        ranked = group.sort_values(score_col, ascending=False, kind="stable")
        labels = ranked[label_col].to_numpy(dtype=float)
        top = labels[:k]
        ideal = np.sort(labels)[::-1][:k]
        idcg = _dcg(ideal)
        positives = float(labels.sum())
        positive_ranks = np.flatnonzero(labels > 0)
        rows.append(
            {
                "query_id": query_id,
                "user_id": ranked[user_col].iloc[0],
                f"ndcg@{k}": _dcg(top) / idcg if idcg > 0 else 0.0,
                f"recall@{k}": float(top.sum() / positives) if positives > 0 else 0.0,
                "mrr": float(1.0 / (positive_ranks[0] + 1)) if positive_ranks.size else 0.0,
            }
        )
    return pd.DataFrame(rows)


def aggregate_metrics(per_query: pd.DataFrame) -> dict[str, float]:
    """Return unrounded query-averaged ranking metrics."""

    metric_columns = [column for column in per_query if column not in {"query_id", "user_id"}]
    return {column: float(per_query[column].mean()) for column in metric_columns}


def paired_user_cluster_bootstrap(
    challenger: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    metric: str = "ndcg@10",
    iterations: int = 2000,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Bootstrap a paired metric delta by resampling users with replacement."""

    paired = challenger[["query_id", "user_id", metric]].merge(
        reference[["query_id", metric]],
        on="query_id",
        suffixes=("_challenger", "_reference"),
        validate="one_to_one",
    )
    paired["delta"] = paired[f"{metric}_challenger"] - paired[f"{metric}_reference"]
    clusters = {user: group["delta"].to_numpy(dtype=float) for user, group in paired.groupby("user_id")}
    users = np.asarray(list(clusters), dtype=object)
    if users.size == 0:
        raise ValueError("No paired users available for bootstrap")

    rng = np.random.default_rng(random_state)
    bootstrap_means = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled_users = rng.choice(users, size=len(users), replace=True)
        sampled_deltas = np.concatenate([clusters[user] for user in sampled_users])
        bootstrap_means[index] = sampled_deltas.mean()

    return {
        "queries": int(len(paired)),
        "users": int(len(users)),
        "mean_delta": float(paired["delta"].mean()),
        "ci_low": float(np.percentile(bootstrap_means, 2.5)),
        "ci_high": float(np.percentile(bootstrap_means, 97.5)),
        "p_delta_le_zero": float(np.mean(bootstrap_means <= 0.0)),
    }
