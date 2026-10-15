"""Public PEC-Fuse feature, ranking, and evaluation interface."""

from .evaluation import aggregate_metrics, paired_user_cluster_bootstrap, query_metrics
from .features import build_feature_frame
from .ranking import PECFuseRanker, rank_within_query
from .schema import FEATURES_25, STRONG_CONTROL_FEATURES

__all__ = [
    "FEATURES_25",
    "STRONG_CONTROL_FEATURES",
    "PECFuseRanker",
    "aggregate_metrics",
    "build_feature_frame",
    "paired_user_cluster_bootstrap",
    "query_metrics",
    "rank_within_query",
]
