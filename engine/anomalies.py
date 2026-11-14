"""Q3 - Which marketing patterns are unusual?

A post is flagged when it differs strongly from COMPARABLE posts. We build four
features and score each with a robust z-score (median / MAD, so outliers do not distort the
yardstick):

* engagement vs expectation - residual of log engagement after removing brand, platform,
  content-type and post-age effects (age matters: a 1-day-old post has fewer likes)
* comments-per-like ratio   - vs posts of the same platform and content type
* shares-per-like ratio     - same peers (only where shares are public)
* engagement-per-view       - same peers (only where views are public)

``anomaly_score`` = largest |z|. Posts at or above ``z_threshold`` (default 3.5, the
conventional robust-z cut-off) are flagged and given a plain-language type and reason.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .content_performance import baseline_residual


def robust_z(values: pd.Series, groups: pd.Series | None = None, min_group: int = 8, floor: float = 0.05) -> pd.Series:
    """(x - median) / (1.4826 * MAD), computed inside each peer group (global fallback for small groups)."""
    g = groups if groups is not None else pd.Series(0, index=values.index)
    med = values.groupby(g).transform("median")
    mad = (values - med).abs().groupby(g).transform("median") * 1.4826
    size = values.groupby(g).transform("count")      # only rows where the value exists
    gmed = values.median()
    gmad = (values - gmed).abs().median() * 1.4826
    med, mad = med.where(size >= min_group, gmed), mad.where(size >= min_group, gmad)
    return ((values - med) / mad.clip(lower=floor)).fillna(0.0)


def detect_anomalies(posts: pd.DataFrame, complaint_share: pd.Series | None = None, z_threshold: float = 3.5) -> pd.DataFrame:
    df = posts.copy()
    peers = df["platform"].astype(str) + "|" + df["content_type"].astype(str)
    df["z_engagement"] = robust_z(pd.Series(baseline_residual(df, ["brand", "platform", "content_type"]), index=df.index))
    df["z_comments"] = robust_z(np.log((df["comments_count"] + 1) / (df["likes"] + 1)), peers)
    shares = df["shares"] if "shares" in df else pd.Series(np.nan, index=df.index)
    sr = np.log((shares + 1) / (df["likes"] + 1))
    df["z_shares"] = robust_z(sr, peers)
    views = df["views"] if "views" in df else pd.Series(np.nan, index=df.index)
    vc = np.log((df["engagement"] + 1) / (views + 1))
    df["z_views"] = robust_z(vc, peers)

    z = df[["z_engagement", "z_comments", "z_shares", "z_views"]]
    df["anomaly_score"] = z.abs().max(axis=1)
    df["is_anomaly"] = df["anomaly_score"] >= z_threshold
    cs = df["post_id"].map(complaint_share) if complaint_share is not None else pd.Series(np.nan, index=df.index)
    t = z_threshold
    types, reasons = [], []
    for r, c, flagged in zip(df.itertuples(), cs, df["is_anomaly"]):
        if not flagged:
            types.append(""); reasons.append(""); continue
        ze, zc, zs, zv = r.z_engagement, r.z_comments, r.z_shares, r.z_views
        per100 = 100 * (r.comments_count + 1) / (r.likes + 1)
        if zc <= -t and ze >= 1.5:
            ty, why = "Likes without conversation", f"Likes are far above expectation (z={ze:+.1f}) but comments per like are unusually low (z={zc:+.1f}): possible paid or bot-driven likes."
        elif zc >= t:
            if pd.notna(c) and c >= 0.4:
                ty, why = "Complaint storm", f"{c:.0%} of sampled comments are complaints/objections and comments per like are far above peers (z={zc:+.1f})."
            else:
                ty, why = "Comment spike", f"{r.comments_count} comments on {r.likes} likes ({per100:.0f} per 100 likes) is far above comparable posts (z={zc:+.1f})."
        elif ze >= t:
            ty, why = "Viral / over-performing", f"Engagement is far above what a {r.content_type} on {r.platform} of this age normally gets (z={ze:+.1f})."
        elif ze <= -t:
            ty, why = "Under-performing", f"Engagement is far below what comparable posts of this age get (z={ze:+.1f})."
        elif zc <= -t:
            ty, why = "Unusually few comments", f"Comments per like are far below comparable posts (z={zc:+.1f})."
        else:
            ty, why = "Unusual share/view pattern", f"Shares or views relative to likes differ strongly from peers (z shares={zs:+.1f}, views={zv:+.1f})."
        types.append(ty); reasons.append(why)
    df["anomaly_type"], df["anomaly_reason"] = types, reasons
    return df


def evaluate_anomalies(df: pd.DataFrame) -> dict:
    """Precision / recall against simulated ground truth (column sim_anomaly_type)."""
    if "sim_anomaly_type" not in df:
        return {}
    truth = df["sim_anomaly_type"].fillna("").astype(str).str.len() > 0
    hit = df["is_anomaly"]
    tp = int((truth & hit).sum())
    per = {k: float(hit[df["sim_anomaly_type"] == k].mean()) for k in sorted(set(df["sim_anomaly_type"].dropna()) - {""})}
    return dict(true_anomalies=int(truth.sum()), flagged=int(hit.sum()), true_positives=tp,
                precision=tp / max(int(hit.sum()), 1), recall=tp / max(int(truth.sum()), 1), recall_by_type=per)
