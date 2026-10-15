"""Canonical feature groups and the ordered 25-dimensional record."""

P_FEATURES = ("P_semantic",)

E_FEATURES = (
    "same_day_recommended",
    "reciprocal_rank",
    "E_exposure",
)

C_FEATURES = (
    "C_semantic",
    "C_max",
    "C_top3",
    "C_attention",
)

CONTROL_FEATURES = (
    "pref_category_weight",
    "pref_publisher_weight",
    "hist_category_click_share",
    "hist_publisher_click_share",
    "log_user_prior_clicks",
    "P_count",
    "C_count",
    "popularity_score",
    "recency_score",
    "log_article_len",
    "is_breaking",
)

INTERACTION_FEATURES = (
    "P_semantic_x_E",
    "C_semantic_x_E",
    "P_semantic_x_C_semantic",
    "C_attention_x_E",
    "C_max_x_E",
    "P_semantic_x_C_attention",
)

FEATURES_25 = (
    "pref_category_weight",
    "pref_publisher_weight",
    "same_day_recommended",
    "reciprocal_rank",
    "hist_category_click_share",
    "hist_publisher_click_share",
    "log_user_prior_clicks",
    "P_semantic",
    "C_semantic",
    "P_count",
    "C_count",
    "E_exposure",
    "P_semantic_x_E",
    "C_semantic_x_E",
    "P_semantic_x_C_semantic",
    "popularity_score",
    "recency_score",
    "log_article_len",
    "is_breaking",
    "C_max",
    "C_top3",
    "C_attention",
    "C_attention_x_E",
    "C_max_x_E",
    "P_semantic_x_C_attention",
)

# B: the 11-field operational control used by the corrected headline fits.
# H retains the three derived summaries for the shared-context fusion audit.
STRONG_CONTROL_FEATURES = (
    "pref_category_weight",
    "pref_publisher_weight",
    "same_day_recommended",
    "reciprocal_rank",
    "hist_category_click_share",
    "hist_publisher_click_share",
    "log_user_prior_clicks",
    "popularity_score",
    "recency_score",
    "log_article_len",
    "is_breaking",
)
SHARED_CONTEXT_FEATURES = tuple(
    name for name in FEATURES_25
    if name in STRONG_CONTROL_FEATURES or name in ("P_count", "C_count", "E_exposure")
)

BASE_INPUT_FEATURES = (
    "pref_category_weight",
    "pref_publisher_weight",
    "same_day_recommended",
    "retained_rank",
    "hist_category_click_share",
    "hist_publisher_click_share",
    "log_user_prior_clicks",
    "P_semantic",
    "C_semantic",
    "popularity_score",
    "recency_score",
    "log_article_len",
    "is_breaking",
    "C_max",
    "C_top3",
    "C_attention",
)
