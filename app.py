"""Pakistan Social Audience & Campaign Intelligence Engine - Streamlit app.

Run:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from engine.comment_clusters import need_by_group
from engine.content_performance import group_difference_pvalue, performance_table, type_platform_matrix
from engine.data_io import load_demo
from engine.pipeline import fit, recommend

st.set_page_config(page_title="Audience & Campaign Intelligence", page_icon="📊", layout="wide")


@st.cache_data(show_spinner="Cleaning data, clustering comments, scanning for unusual posts...")
def run(posts: pd.DataFrame, comments: pd.DataFrame, k: int, z: float) -> dict:
    return fit(posts, comments, k=k, z_threshold=z)


def csv_button(label: str, df: pd.DataFrame, name: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8-sig"), file_name=name, mime="text/csv")


# ------------------------------------------------------------------ sidebar
st.sidebar.title("📊 Engine settings")
source = st.sidebar.radio("Data source", ["Demo (simulated)", "Upload my CSV files"])
if source.startswith("Demo"):
    posts_raw, comments_raw = load_demo()
else:
    up_p = st.sidebar.file_uploader("Posts CSV (Table A)", type="csv")
    up_c = st.sidebar.file_uploader("Comments CSV (Table B)", type="csv")
    if not (up_p and up_c):
        st.title("Pakistan Social Audience & Campaign Intelligence Engine")
        st.info("Upload both files. Required posts columns: post_id, brand, platform, campaign, content_type, likes, "
                "comments_count, days_since_post (optional: shares, views, product_category). "
                "Required comments columns: post_id, comment_text (optional: language_type, topic, sentiment, purchase_intent).")
        st.stop()
    posts_raw, comments_raw = pd.read_csv(up_p), pd.read_csv(up_c)

k = st.sidebar.slider("Comment groups (k)", 12, 40, 28, help="Fine-grained phrase groups; they are merged into named audience needs.")
z = st.sidebar.slider("Anomaly sensitivity (robust z)", 2.5, 5.0, 3.5, 0.5, help="Lower = flag more posts.")
try:
    R = run(posts_raw, comments_raw, k, z)
except ValueError as e:
    st.error(str(e))
    st.stop()

posts, cm = R["posts"], R["comments"]
st.sidebar.markdown("---\n**Filters** (applied to every tab)")
sel = {}
for col, label in [("platform", "Platform"), ("brand", "Brand"), ("campaign", "Campaign")]:
    opts = sorted(posts[col].unique())
    sel[col] = st.sidebar.multiselect(label, opts, default=opts)
mask = np.ones(len(posts), bool)
for col, chosen in sel.items():
    mask &= posts[col].isin(chosen).values
fp = posts[mask].reset_index(drop=True)
fcm = cm[cm["post_id"].isin(fp["post_id"])].reset_index(drop=True)

# ------------------------------------------------------------------ header
st.title("Pakistan Social Audience & Campaign Intelligence Engine")
for w in R["warnings"]:
    (st.warning if "SIMULATED" in w else st.info)(w)
if fp.empty or len(fcm) < 50:
    st.warning("Not enough posts/comments under the current filters.")
    st.stop()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Posts", f"{len(fp):,}")
c2.metric("Comments analysed", f"{len(fcm):,}")
c3.metric("Unusual posts", f"{int(fp['is_anomaly'].sum())}")
c4.metric("Audience needs found", f"{fcm['need'].nunique()}")

tabs = st.tabs(["1 · Content", "2 · What the audience wants", "3 · Unusual posts", "4 · Audience needs & segments",
                "5 · Recommended actions", "Data quality & accuracy"])

# ------------------------------------------------------------------ 1 content
with tabs[0]:
    st.subheader("Which content attracts attention?")
    st.caption("Engagement index = (likes + comments + shares) relative to what a typical post of the same brand, platform and age "
               "receives. 1.0 = typical; 2.0 = twice as engaging. Bars show 95% bootstrap confidence intervals.")
    by = st.selectbox("Compare by", ["content_type", "campaign", "platform", "brand"], key="by1")
    t = performance_table(fp, by)
    fig = px.bar(t, x=by, y="median_index", error_y=t["ci_high"] - t["median_index"], error_y_minus=t["median_index"] - t["ci_low"],
                 labels={"median_index": "Median engagement index"}, text=t["median_index"].round(2))
    fig.add_hline(y=1, line_dash="dot", annotation_text="typical post")
    st.plotly_chart(fig)
    p = group_difference_pvalue(fp, by)
    if not np.isnan(p):
        st.caption(f"Kruskal-Wallis test across {by.replace('_', ' ')} groups: p = {p:.3g} "
                   f"({'differences are statistically clear' if p < 0.05 else 'differences could be chance'}).")
    st.dataframe(t.round(3), hide_index=True)
    if fp["content_type"].nunique() > 1 and fp["platform"].nunique() > 1:
        st.markdown("**Content type × platform (median engagement index)**")
        st.plotly_chart(px.imshow(type_platform_matrix(fp), text_auto=".2f", color_continuous_scale="RdBu_r", color_continuous_midpoint=1, aspect="auto"))

# ------------------------------------------------------------------ 2 audience wants
with tabs[1]:
    st.subheader("What does the audience want?")
    st.caption("Comments in English, Roman Urdu and Urdu script are grouped by meaning (unsupervised clustering), then named by their dominant theme.")
    needs = (fcm.groupby("need").agg(comments=("need", "size"), negative_share=("negative", "mean")).reset_index()
             .assign(share=lambda d: d["comments"] / d["comments"].sum()).sort_values("share"))
    st.plotly_chart(px.bar(needs, x="share", y="need", orientation="h", text=needs["share"].map("{:.1%}".format),
                           color="negative_share", color_continuous_scale="Reds", range_color=(0, 1),
                           labels={"share": "Share of comments", "need": "", "negative_share": "Negative"}))
    info = R["needs"].copy()
    info["share"] = info["share"].map("{:.1%}".format)
    info["negative_share"] = info["negative_share"].map("{:.0%}".format)
    st.dataframe(info[["need", "comments", "share", "negative_share", "top_terms", "examples", "languages"]], hide_index=True)
    csv_button("Download needs table", R["needs"], "audience_needs.csv")
    cA, cB = st.columns(2)
    if "language_type" in fcm:
        cA.markdown("**Language mix**")
        cA.plotly_chart(px.pie(fcm, names="language_type", hole=.4))
    if fcm["city"].notna().any():
        cb = fcm[fcm["city"].notna()]["city"].value_counts().head(8).reset_index()
        cB.markdown("**Cities mentioned in comments**")
        cB.plotly_chart(px.bar(cb, x="count", y="city", orientation="h").update_yaxes(autorange="reversed"))
    with st.expander("Fine-grained phrase groups (before merging into needs)"):
        st.dataframe(R["groups"][["cluster", "need", "comments", "theme_purity", "top_terms", "examples"]].round(2), hide_index=True)

# ------------------------------------------------------------------ 3 anomalies
with tabs[2]:
    st.subheader("Which marketing patterns are unusual?")
    st.caption("Posts that differ strongly from comparable posts (same platform and content type, adjusted for brand and post age). "
               "These are leads to investigate, not verdicts.")
    plot = fp.assign(kind=fp["anomaly_type"].replace("", "Normal"), comments_per_100_likes=100 * fp["comments_count"] / (fp["likes"] + 1))
    st.plotly_chart(px.scatter(plot, x="engagement", y="comments_per_100_likes", color="kind", log_x=True, log_y=True, opacity=.65,
                               hover_data=["post_id", "brand", "platform", "content_type", "campaign"],
                               labels={"engagement": "Engagement (likes + comments + shares)"}))
    flagged = fp[fp["is_anomaly"]].sort_values("anomaly_score", ascending=False)
    kinds = ["All"] + sorted(flagged["anomaly_type"].unique())
    pick = st.selectbox("Show type", kinds)
    show = flagged if pick == "All" else flagged[flagged["anomaly_type"] == pick]
    cols = ["post_id", "brand", "platform", "campaign", "content_type", "likes", "comments_count", "shares", "views",
            "days_since_post", "anomaly_type", "anomaly_score", "anomaly_reason"]
    st.dataframe(show[cols].round(2), hide_index=True)
    csv_button("Download flagged posts", flagged[cols], "unusual_posts.csv")

# ------------------------------------------------------------------ 4 segments
with tabs[3]:
    st.subheader("Which audience needs should the brand address?")
    g = st.selectbox("Where do the needs concentrate? Group by", ["content_type", "campaign", "brand", "platform"], key="by4")
    share, lift = need_by_group(fcm, fp, g)
    st.caption("Lift = how much more (red) or less (blue) common a need is in that group than overall. 1.0 = average.")
    st.plotly_chart(px.imshow(lift, text_auto=".1f", color_continuous_scale="RdBu_r", color_continuous_midpoint=1, aspect="auto"))
    st.markdown("**Post audience profiles** - posts grouped by the mix of needs in their comments")
    prof = fp.dropna(subset=["profile"]) if "profile" in fp else pd.DataFrame()
    if len(prof):
        s = prof.groupby("profile").agg(posts=("post_id", "size"), median_index=("engagement_index", "median"),
                                        top_content_type=("content_type", lambda x: x.value_counts().index[0]),
                                        top_brand=("brand", lambda x: x.value_counts().index[0])).round(2).reset_index()
        st.dataframe(s, hide_index=True)

# ------------------------------------------------------------------ 5 recommendations
with tabs[4]:
    st.subheader("What should the marketing team do next?")
    recs = recommend(fp, fcm)
    if recs.empty:
        st.info("No pattern strong enough for a recommendation under the current filters.")
    for pr, icon in [("High", "🔴"), ("Medium", "🟠"), ("Low", "🟢")]:
        for r in recs[recs["priority"] == pr].itertuples():
            with st.container(border=True):
                st.markdown(f"{icon} **{r.priority} · {r.area}** — {r.action}")
                st.caption(f"Evidence: {r.evidence}")
    if len(recs):
        csv_button("Download recommendations", recs, "recommendations.csv")

# ------------------------------------------------------------------ quality
with tabs[5]:
    st.subheader("Data quality & accuracy")
    ev = R["evaluation"]
    if ev["clustering"]:
        cl, an = ev["clustering"], ev["anomalies"]
        st.markdown("Measured against the labels in the data (for simulated data these are the generator's ground truth):")
        a, b, c = st.columns(3)
        a.metric("Need-level agreement", f"{cl['needs']['accuracy']:.1%}")
        b.metric("Anomaly precision", f"{an['precision']:.1%}" if an else "n/a")
        c.metric("Anomaly recall", f"{an['recall']:.1%}" if an else "n/a")
        st.json(ev)
        st.caption("These scores are only meaningful on labelled data. On simulated data they show the code works; "
                   "they do NOT guarantee the same accuracy on real, noisier comments.")
    else:
        st.info("No labelled 'topic' column found, so accuracy cannot be measured. Review the example comments per need manually.")
    st.markdown(f"Comment groups used: **{R['k']}** · posts: **{len(posts):,}** · comments: **{len(cm):,}**")
