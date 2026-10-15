"""Public MIND-small proxy construction for PEC-Fuse boundary checks."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


E_FEATURES = ["E_seen", "E_rr", "E_rank_pct"]
P_FEATURES = ["P_proxy", "P_category_share", "P_subcategory_share"]
C_FEATURES = ["C_mean", "C_max", "C_top3", "C_attention"]
CONTROL_FEATURES = [
    "news_ctr",
    "log_news_impressions",
    "category_ctr",
    "subcategory_ctr",
    "log_history_len",
    "log_slate_size",
    "log_title_len",
    "log_abstract_len",
]
INTERACTIONS = [
    "P_proxy_x_E_seen",
    "C_mean_x_E_seen",
    "C_attention_x_E_seen",
    "C_max_x_E_seen",
    "P_proxy_x_C_mean",
    "P_proxy_x_C_attention",
]
MIND_FEATURE_SETS = {
    "exposure_only": CONTROL_FEATURES + E_FEATURES,
    "p_c_without_e": CONTROL_FEATURES + P_FEATURES + C_FEATURES,
    "full_pec_proxy": (
        CONTROL_FEATURES + P_FEATURES + E_FEATURES + C_FEATURES + INTERACTIONS
    ),
}

NEWS_COLUMNS = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]
BEHAVIOR_COLUMNS = ["impression_id", "user_id", "time", "history", "impressions"]


def load_news(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    frames = [
        pd.read_csv(
            root / split / "news.tsv",
            sep="\t",
            names=NEWS_COLUMNS,
            dtype=str,
            keep_default_na=False,
        )
        for split in ("train", "dev")
    ]
    news = pd.concat(frames, ignore_index=True).drop_duplicates("news_id")
    news["text"] = (
        news["title"].str.strip() + " [SEP] " + news["abstract"].str.strip()
    )
    news["log_title_len"] = np.log1p(news["title"].str.split().str.len())
    news["log_abstract_len"] = np.log1p(news["abstract"].str.split().str.len())
    return news.set_index("news_id", drop=False)


def load_mind_split(
    root: str | Path, split: str, limit: int | None = None
) -> pd.DataFrame:
    return pd.read_csv(
        Path(root) / split / "behaviors.tsv",
        sep="\t",
        names=BEHAVIOR_COLUMNS,
        dtype=str,
        keep_default_na=False,
        nrows=limit,
    )


def parse_impressions(value: str) -> list[tuple[str, int]]:
    parsed = []
    for token in value.split():
        news_id, label = token.rsplit("-", 1)
        parsed.append((news_id, int(label)))
    return parsed


def training_statistics(behaviors: pd.DataFrame, news: pd.DataFrame) -> dict:
    imps, clicks = Counter(), Counter()
    cat_imps, cat_clicks = Counter(), Counter()
    sub_imps, sub_clicks = Counter(), Counter()
    for value in behaviors["impressions"]:
        for news_id, label in parse_impressions(value):
            if news_id not in news.index:
                continue
            category = news.at[news_id, "category"]
            subcategory = news.at[news_id, "subcategory"]
            imps[news_id] += 1
            clicks[news_id] += label
            cat_imps[category] += 1
            cat_clicks[category] += label
            sub_imps[subcategory] += 1
            sub_clicks[subcategory] += label
    return {
        "impressions": imps,
        "clicks": clicks,
        "category_impressions": cat_imps,
        "category_clicks": cat_clicks,
        "subcategory_impressions": sub_imps,
        "subcategory_clicks": sub_clicks,
        "popular": [item for item, _ in imps.most_common()],
    }


def _ctr(clicks: Counter, impressions: Counter, key: str) -> float:
    return (clicks[key] + 1.0) / (impressions[key] + 2.0)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def _sample_added(
    popular: list[str], blocked: set[str], count: int, seed: int, query_id: str
) -> list[str]:
    pool = [item for item in popular if item not in blocked]
    if len(pool) <= count:
        return pool
    digest = hashlib.sha256(f"{seed}:{query_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    return [
        pool[int(index)]
        for index in rng.choice(len(pool), size=count, replace=False)
    ]


def build_mind_rows(
    behaviors: pd.DataFrame,
    news: pd.DataFrame,
    vectors: dict[str, np.ndarray],
    stats: dict,
    *,
    history_window: int = 50,
    unexposed_per_query: int = 10,
    popular_pool_size: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    popular = [
        item for item in stats["popular"][:popular_pool_size] if item in vectors
    ]
    for behavior in behaviors.itertuples(index=False):
        query_id = str(behavior.impression_id)
        history = [
            item for item in behavior.history.split() if item in vectors
        ][-history_window:]
        logged = [
            (item, label)
            for item, label in parse_impressions(behavior.impressions)
            if item in vectors
        ]
        logged_ids = [item for item, _ in logged]
        added = _sample_added(
            popular,
            set(history) | set(logged_ids),
            unexposed_per_query,
            seed,
            query_id,
        )
        candidates = logged + [(item, 0) for item in added]
        rank = {item: index + 1 for index, item in enumerate(logged_ids)}
        history_matrix = (
            np.vstack([vectors[item] for item in history]) if history else None
        )
        profile = (
            _unit(history_matrix.mean(axis=0))
            if history_matrix is not None
            else None
        )
        history_categories = Counter(news.at[item, "category"] for item in history)
        history_subcategories = Counter(
            news.at[item, "subcategory"] for item in history
        )

        for news_id, label in candidates:
            vector = vectors[news_id]
            category = news.at[news_id, "category"]
            subcategory = news.at[news_id, "subcategory"]
            seen = float(news_id in rank)
            rr = 1.0 / rank[news_id] if seen else 0.0
            rank_pct = (
                1.0 - (rank[news_id] - 1) / max(len(logged_ids) - 1, 1)
                if seen
                else 0.0
            )
            if history_matrix is None:
                p_proxy = c_mean = c_max = c_top3 = c_attention = 0.0
            else:
                similarities = history_matrix @ vector
                p_proxy = float(profile @ vector)
                c_mean = float(similarities.mean())
                c_max = float(similarities.max())
                c_top3 = float(np.sort(similarities)[-3:].mean())
                weights = np.exp((similarities - similarities.max()) / 0.18)
                c_attention = float(
                    np.dot(weights / weights.sum(), similarities)
                )
            row = {
                "query_id": query_id,
                "user_id": str(behavior.user_id),
                "news_id": news_id,
                "label": int(label),
                "news_ctr": _ctr(
                    stats["clicks"], stats["impressions"], news_id
                ),
                "log_news_impressions": math.log1p(
                    stats["impressions"][news_id]
                ),
                "category_ctr": _ctr(
                    stats["category_clicks"],
                    stats["category_impressions"],
                    category,
                ),
                "subcategory_ctr": _ctr(
                    stats["subcategory_clicks"],
                    stats["subcategory_impressions"],
                    subcategory,
                ),
                "log_history_len": math.log1p(len(history)),
                "log_slate_size": math.log1p(len(logged_ids)),
                "log_title_len": float(news.at[news_id, "log_title_len"]),
                "log_abstract_len": float(
                    news.at[news_id, "log_abstract_len"]
                ),
                "P_proxy": p_proxy,
                "P_category_share": history_categories[category]
                / max(len(history), 1),
                "P_subcategory_share": history_subcategories[subcategory]
                / max(len(history), 1),
                "E_seen": seen,
                "E_rr": rr,
                "E_rank_pct": rank_pct,
                "C_mean": c_mean,
                "C_max": c_max,
                "C_top3": c_top3,
                "C_attention": c_attention,
                "P_proxy_x_E_seen": p_proxy * seen,
                "C_mean_x_E_seen": c_mean * seen,
                "C_attention_x_E_seen": c_attention * seen,
                "C_max_x_E_seen": c_max * seen,
                "P_proxy_x_C_mean": p_proxy * c_mean,
                "P_proxy_x_C_attention": p_proxy * c_attention,
            }
            rows.append(row)
    return pd.DataFrame(rows)
