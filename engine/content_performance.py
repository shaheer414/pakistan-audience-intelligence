"""Q1 - Which content attracts attention?

Raw likes are misleading: a 40-day-old post on a big account always beats a 2-day-old
post on a small one. We therefore compute an **engagement index** = actual engagement /
expected engagement for that brand, platform and post age (1.0 = typical post).
Content type / campaign are NOT in the baseline, so their effect shows up in the index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

REQUIRED_POST_COLS = ["post_id", "brand", "platform", "campaign", "content_type", "likes", "comments_count", "days_since_post"]


def add_engagement_metrics(posts: pd.DataFrame) -> pd.DataFrame:
    df = posts.copy()
    shares = df["shares"].fillna(0) if "shares" in df else 0
    df["engagement"] = df["likes"] + df["comments_count"] + shares
    if "views" in df:
        df["engagement_rate"] = np.where(df["views"] > 0, df["engagement"] / df["views"], np.nan)
    df["log_eng"] = np.log1p(df["engagement"])
    resid = baseline_residual(df, ["brand", "platform"])
    df["engagement_index"] = np.exp(resid)
    return df


def design_matrix(df: pd.DataFrame, group_cols: list[str]) -> np.ndarray:
    d = df["days_since_post"].clip(lower=1).astype(float)
    parts = [np.ones(len(df)), np.log(d), 1 / d]
    if group_cols:
        parts += list(pd.get_dummies(df[group_cols].astype(str), drop_first=True).astype(float).T.values)
    return np.column_stack(parts)


def baseline_residual(df: pd.DataFrame, group_cols: list[str], y: str = "log_eng") -> np.ndarray:
    """Residual of log engagement after removing group effects and post-age effect (OLS)."""
    X, yv = design_matrix(df, group_cols), df[y].values
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    return yv - X @ beta


def _boot_median_ci(v: np.ndarray, rng: np.random.Generator, b: int = 400) -> tuple[float, float]:
    if len(v) < 5:
        return (np.nan, np.nan)
    m = np.median(v[rng.integers(0, len(v), (b, len(v)))], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def performance_table(posts: pd.DataFrame, by: str, seed: int = 0) -> pd.DataFrame:
    """Median engagement index per group with a 95% bootstrap CI, sorted best-first."""
    rng = np.random.default_rng(seed)
    rows = []
    for key, g in posts.groupby(by):
        lo, hi = _boot_median_ci(g["engagement_index"].values, rng)
        rows.append({by: key, "posts": len(g), "median_index": g["engagement_index"].median(), "ci_low": lo, "ci_high": hi,
                     "median_likes": g["likes"].median(), "median_comments": g["comments_count"].median(),
                     "median_shares": g["shares"].median() if "shares" in g else np.nan,
                     "median_views": g["views"].median() if "views" in g else np.nan,
                     "median_eng_rate": g["engagement_rate"].median() if "engagement_rate" in g else np.nan})
    return pd.DataFrame(rows).sort_values("median_index", ascending=False).reset_index(drop=True)


def group_difference_pvalue(posts: pd.DataFrame, by: str) -> float:
    """Kruskal-Wallis test: do the groups' engagement indices differ more than chance?"""
    groups = [g["engagement_index"].values for _, g in posts.groupby(by) if len(g) >= 5]
    return float(stats.kruskal(*groups).pvalue) if len(groups) >= 2 else np.nan


def type_platform_matrix(posts: pd.DataFrame) -> pd.DataFrame:
    return posts.pivot_table(index="content_type", columns="platform", values="engagement_index", aggfunc="median")
