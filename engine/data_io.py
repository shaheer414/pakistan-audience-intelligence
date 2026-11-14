"""Loading and validating posts (Table A) and comments (Table B)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .content_performance import REQUIRED_POST_COLS

NUMERIC = ["likes", "comments_count", "shares", "views", "days_since_post"]
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _num(s: pd.Series) -> pd.Series:
    """'1,250' -> 1250 ; '' / 'n/a' -> NaN."""
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def validate(posts: pd.DataFrame, comments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return cleaned (posts, comments, warnings). Raises ValueError for unusable input."""
    warn: list[str] = []
    posts, comments = posts.copy(), comments.copy()
    posts.columns, comments.columns = [c.strip().lower() for c in posts.columns], [c.strip().lower() for c in comments.columns]
    missing = [c for c in REQUIRED_POST_COLS if c not in posts.columns]
    if missing:
        raise ValueError(f"posts file is missing required column(s): {', '.join(missing)}")
    miss_c = [c for c in ("post_id", "comment_text") if c not in comments.columns]
    if miss_c:
        raise ValueError(f"comments file is missing required column(s): {', '.join(miss_c)}")

    for c in NUMERIC:
        if c in posts:
            posts[c] = _num(posts[c])
    n0 = len(posts)
    posts = posts.drop_duplicates("post_id")
    if len(posts) < n0:
        warn.append(f"{n0 - len(posts)} duplicate post_id row(s) removed.")
    n0 = len(posts)
    posts = posts.dropna(subset=["likes", "comments_count", "days_since_post"])
    posts = posts[(posts["likes"] >= 0) & (posts["comments_count"] >= 0) & (posts["days_since_post"] >= 0)]
    if len(posts) < n0:
        warn.append(f"{n0 - len(posts)} post(s) with missing/negative likes, comments or age removed.")
    for c in ("brand", "platform", "campaign", "content_type"):
        posts[c] = posts[c].astype(str).str.strip()

    comments["comment_text"] = comments["comment_text"].astype(str).str.strip()
    n0 = len(comments)
    comments = comments[(comments["comment_text"] != "") & (comments["comment_text"].str.lower() != "nan")]
    comments = comments[comments["post_id"].isin(posts["post_id"])]
    if len(comments) < n0:
        warn.append(f"{n0 - len(comments)} comment(s) removed (empty text or post_id not found in posts).")
    if len(posts) < 20:
        raise ValueError(f"Only {len(posts)} usable posts - at least 20 are needed for comparisons.")
    if len(comments) < 50:
        raise ValueError(f"Only {len(comments)} usable comments - at least 50 are needed for clustering.")
    if "data_provenance" in posts and posts["data_provenance"].astype(str).str.lower().eq("simulated").any():
        warn.append("Dataset contains SIMULATED records: findings describe the simulation, not real audiences.")
    return posts.reset_index(drop=True), comments.reset_index(drop=True), warn


def load_demo() -> tuple[pd.DataFrame, pd.DataFrame]:
    p, c = DATA_DIR / "posts.csv", DATA_DIR / "comments.csv"
    if p.exists() and c.exists():
        return pd.read_csv(p), pd.read_csv(c)
    from .data_generator import generate_dataset
    return generate_dataset()
