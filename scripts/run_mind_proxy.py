#!/usr/bin/env python3
"""Run the public MIND-small PEC proxy with one text encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from pec_fuse.mind import (
    MIND_FEATURE_SETS,
    build_mind_rows,
    load_mind_split,
    load_news,
    training_statistics,
)


def encode_news(news: pd.DataFrame, model_name: str, batch_size: int) -> dict:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Install public-proxy dependencies with: pip install -e '.[mind]'"
        ) from exc
    model = SentenceTransformer(model_name)
    matrix = model.encode(
        news["text"].tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return {
        item: np.asarray(vector)
        for item, vector in zip(news["news_id"], matrix)
    }


def query_metrics(frame: pd.DataFrame) -> dict[str, float]:
    values = []
    for _, group in frame.groupby("query_id", sort=False):
        ranked = group.sort_values("score", ascending=False, kind="stable")
        labels = ranked["label"].to_numpy(dtype=float)
        top = labels[:10]
        discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
        dcg = float(np.dot(top, discounts))
        ideal = np.sort(labels)[::-1][:10]
        idcg = float(np.dot(ideal, discounts[: len(ideal)]))
        positives = max(float(labels.sum()), 1.0)
        positive_ranks = np.flatnonzero(labels > 0)
        values.append(
            {
                "ndcg@10": dcg / idcg if idcg else 0.0,
                "recall@10": float(top.sum()) / positives,
                "mrr": 1.0 / (int(positive_ranks[0]) + 1)
                if len(positive_ranks)
                else 0.0,
            }
        )
    return pd.DataFrame(values).mean().to_dict()


def select_config(
    train: pd.DataFrame, valid: pd.DataFrame, features: list[str], seed: int
):
    candidates = [
        {"n_estimators": 240, "max_depth": 12, "min_samples_leaf": 3},
        {"n_estimators": 420, "max_depth": None, "min_samples_leaf": 4},
        {"n_estimators": 640, "max_depth": 16, "min_samples_leaf": 6},
    ]
    best = None
    for config in candidates:
        model = ExtraTreesClassifier(
            **config,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(train[features], train["label"])
        scored = valid[["query_id", "label"]].copy()
        scored["score"] = model.predict_proba(valid[features])[:, 1]
        metric = query_metrics(scored)["ndcg@10"]
        if best is None or metric > best[0]:
            best = (metric, config)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mind-root", type=Path, required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--train-limit", type=int, default=30000)
    parser.add_argument("--dev-limit", type=int, default=30000)
    parser.add_argument("--history-window", type=int, default=50)
    parser.add_argument("--unexposed-per-query", type=int, default=10)
    parser.add_argument("--popular-pool-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/mind_proxy")
    )
    args = parser.parse_args()

    news = load_news(args.mind_root)
    train_behaviors = load_mind_split(
        args.mind_root, "train", args.train_limit
    )
    dev_behaviors = load_mind_split(args.mind_root, "dev", args.dev_limit)
    vectors = encode_news(news, args.encoder, args.batch_size)
    options = {
        "history_window": args.history_window,
        "unexposed_per_query": args.unexposed_per_query,
        "popular_pool_size": args.popular_pool_size,
        "seed": args.seed,
    }
    cutoff = int(len(train_behaviors) * 0.85)
    stats = training_statistics(train_behaviors.iloc[:cutoff], news)
    fit_rows = build_mind_rows(
        train_behaviors.iloc[:cutoff], news, vectors, stats, **options
    )
    valid_rows = build_mind_rows(
        train_behaviors.iloc[cutoff:], news, vectors, stats, **options
    )
    final_stats = training_statistics(train_behaviors, news)
    dev_rows = build_mind_rows(dev_behaviors, news, vectors, final_stats, **options)
    all_train = build_mind_rows(train_behaviors, news, vectors, final_stats, **options)

    output = {
        "encoder": args.encoder,
        "candidate_protocol": "logged_plus_sampled_weak_negatives",
        "models": {},
    }
    for name, features in MIND_FEATURE_SETS.items():
        validation_ndcg, config = select_config(
            fit_rows, valid_rows, features, args.seed
        )
        model = ExtraTreesClassifier(
            **config,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=args.seed,
        )
        model.fit(all_train[features], all_train["label"])
        scored = dev_rows[["query_id", "user_id", "label"]].copy()
        scored["score"] = model.predict_proba(dev_rows[features])[:, 1]
        output["models"][name] = {
            "features": features,
            "selected_config": config,
            "validation_ndcg@10": validation_ndcg,
            "dev_metrics": query_metrics(scored),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / f"{args.encoder.replace('/', '__')}.json"
    target.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
