"""End-to-end pipeline used by both the CLI (run_analysis.py) and the Streamlit app."""
from __future__ import annotations

import pandas as pd

from .anomalies import detect_anomalies, evaluate_anomalies
from .comment_clusters import cluster_comments, post_audience_profiles
from .content_performance import add_engagement_metrics
from .data_io import validate
from .recommendations import build_recommendations

META = ["post_id", "brand", "platform", "campaign", "content_type"]
COMPLAINT_NEEDS = ["Delivery complaints", "Price objections"]


def fit(posts: pd.DataFrame, comments: pd.DataFrame, k: int | None = 28, z_threshold: float = 3.5, seed: int = 42) -> dict:
    """Heavy step (clustering + anomaly detection). Returns enriched tables and diagnostics."""
    posts, comments, warnings = validate(posts, comments)
    pm = add_engagement_metrics(posts)
    cl = cluster_comments(comments, k=k, seed=seed)
    cm = cl["comments"].merge(pm[META], on="post_id", how="left")
    complaint_share = cm.groupby("post_id")["need"].apply(lambda s: s.isin(COMPLAINT_NEEDS).mean())
    pm = detect_anomalies(pm, complaint_share, z_threshold)
    pm = pm.merge(post_audience_profiles(cl["comments"]), on="post_id", how="left")
    return dict(posts=pm, comments=cm, groups=cl["groups"], needs=cl["needs"], k=cl["k"], k_scores=cl["k_scores"],
                warnings=warnings, evaluation=dict(clustering=cl["evaluation"], anomalies=evaluate_anomalies(pm)))


def recommend(posts: pd.DataFrame, cm: pd.DataFrame) -> pd.DataFrame:
    """Light step - can be re-run on filtered subsets."""
    return build_recommendations(posts, cm)
