"""Q5 - What should the marketing team do next?

Every recommendation is generated from a measured pattern and carries its evidence, so a
manager can see *why* it was suggested. Nothing here is hard-coded advice: if the data does
not show a pattern strong enough (see MIN_LIFT / MIN_COMMENTS), no recommendation appears.
"""
from __future__ import annotations

import pandas as pd

from .content_performance import group_difference_pvalue, performance_table

MIN_LIFT = 1.3          # a need must be at least 30% more common in a group than overall
MIN_COMMENTS = 40       # ...and the group must have enough comments to trust the share
PRIORITY = {"High": 0, "Medium": 1, "Low": 2}


def _top_lift(cm: pd.DataFrame, group_col: str, need: str):
    """Group where `need` is most over-represented: (group, lift, share, overall_share, n_comments) or None."""
    if group_col not in cm or need not in set(cm["need"]):
        return None
    overall = (cm["need"] == need).mean()
    g = cm.groupby(group_col)["need"].agg(n="size", hits=lambda s: (s == need).sum())
    g = g[g["n"] >= MIN_COMMENTS]
    if g.empty or overall == 0:
        return None
    g["share"] = g["hits"] / g["n"]
    g["lift"] = g["share"] / overall
    top = g.sort_values("lift", ascending=False).iloc[0]
    return (g.sort_values("lift", ascending=False).index[0], float(top["lift"]), float(top["share"]), float(overall), int(top["n"])) \
        if top["lift"] >= MIN_LIFT else None


def build_recommendations(posts: pd.DataFrame, cm: pd.DataFrame) -> pd.DataFrame:
    """posts: posts with engagement_index (+ anomaly columns); cm: comments joined with post metadata and `need`."""
    recs: list[dict] = []

    def add(priority, area, action, evidence):
        recs.append(dict(priority=priority, area=area, action=action, evidence=evidence))

    # 1 ---- content ------------------------------------------------------------
    if posts["content_type"].nunique() >= 2:
        t = performance_table(posts, "content_type")
        best, worst = t.iloc[0], t.iloc[-1]
        p = group_difference_pvalue(posts, "content_type")
        ev = (f"{best['content_type']}: median engagement index {best['median_index']:.2f} (95% CI {best['ci_low']:.2f}-{best['ci_high']:.2f}, "
              f"{int(best['posts'])} posts) vs {worst['content_type']}: {worst['median_index']:.2f} ({worst['ci_low']:.2f}-{worst['ci_high']:.2f}, "
              f"{int(worst['posts'])} posts). Kruskal-Wallis p={p:.3g}.")
        if best["ci_low"] > worst["ci_high"]:
            add("High", "Content", f"Shift creative effort and paid reach towards {best['content_type']} and away from {worst['content_type']}.", ev)
        else:
            add("Low", "Content", "Content types do not differ clearly yet - run a controlled test before reallocating budget.", ev)
    if {"platform", "content_type"} <= set(posts) and len(posts) >= 60:
        combo = posts.groupby(["platform", "content_type"])["engagement_index"].agg(["median", "size"])
        combo = combo[combo["size"] >= 20].sort_values("median", ascending=False)
        if len(combo) >= 2:
            (pl, ct), r = combo.index[0], combo.iloc[0]
            add("Medium", "Content", f"Prioritise {ct} on {pl} - the strongest platform/format pairing.",
                f"Median engagement index {r['median']:.2f} across {int(r['size'])} posts (overall typical = 1.0).")

    if len(cm) and "need" in cm:
        # 2 ---- promotions / pricing -----------------------------------------
        for need, col, text in [
            ("Discount seekers", "content_type", "Put the discount, sale price and promo code directly in the caption (and pin a comment) for {g} posts."),
            ("Discount seekers", "campaign", "Publish clear offer terms and end dates at the start of the {g} campaign to pre-empt discount questions."),
            ("Price questions", "content_type", "Show the price in the caption/visual of {g} posts instead of making people ask."),
            ("Price objections", "brand", "Review pricing and value messaging for {g} (fabric details, bundles, instalments) - price pushback is concentrated there."),
        ]:
            r = _top_lift(cm, col, need)
            if r:
                g, lift, share, overall, n = r
                add("High" if "objection" in need else "Medium", "Promotion",
                    text.format(g=g), f"'{need}' make up {share:.0%} of comments on {g} ({col.replace('_', ' ')}) vs {overall:.0%} overall = {lift:.1f}x ({n} comments).")

        # 3 ---- customer service -----------------------------------------------
        for col in ("brand", "campaign"):
            r = _top_lift(cm, col, "Delivery complaints")
            if r:
                g, lift, share, overall, n = r
                add("High", "Customer service",
                    f"Escalate delivery performance for {g}: audit courier SLAs, send proactive tracking updates and assign a person to answer order-status comments.",
                    f"Delivery complaints are {share:.0%} of comments for {g} ({col}) vs {overall:.0%} overall = {lift:.1f}x ({n} comments).")
        r = _top_lift(cm, "content_type", "Delivery questions")
        if r:
            g, lift, share, overall, n = r
            add("Medium", "Customer service", f"Add delivery times, charges and COD availability to {g} posts / pinned FAQ.",
                f"Delivery questions are {share:.0%} of comments on {g} vs {overall:.0%} overall = {lift:.1f}x.")
        # availability by city
        if "city" in cm:
            ask = cm[cm["need"].isin(["Availability / stock questions", "Delivery questions"]) & cm["city"].notna()]
            if len(ask) >= 20:
                top = ask["city"].value_counts()
                names = ", ".join(f"{c} ({n})" for c, n in top.head(3).items())
                add("Medium", "Customer service", f"Publish city-wise stock/delivery information and consider local stockists for {', '.join(top.head(3).index)}.",
                    f"{len(ask)} availability/delivery questions name a city; most frequent: {names}.")
        r = _top_lift(cm, "content_type", "Availability / stock questions")
        if r:
            g, lift, share, overall, n = r
            add("Low", "Content", f"Mention stock status/size availability on {g} posts and use 'few pieces left' or restock notices.",
                f"Availability questions are {share:.0%} of comments on {g} vs {overall:.0%} overall = {lift:.1f}x.")

        # 4 ---- purchase intent & testing ----------------------------------------
        r = _top_lift(cm, "content_type", "Ready-to-buy (purchase intent)")
        if r:
            g, lift, share, overall, n = r
            add("High", "Sales", f"Treat {g} comments as hot leads: add a clear 'order via WhatsApp/inbox' call-to-action and reply within the hour.",
                f"Purchase-intent comments are {share:.0%} of comments on {g} vs {overall:.0%} overall = {lift:.1f}x ({n} comments).")
            add("Medium", "Campaign testing", f"A/B test a 'link in bio / WhatsApp order' CTA on {g} content vs the current caption; success metric = inbox enquiries per 1,000 views.",
                "Derived from the purchase-intent concentration above.")
        r = _top_lift(cm, "content_type", "Price questions")
        if r:
            add("Medium", "Campaign testing", f"A/B test price-in-caption vs price-on-request for {r[0]} content; metric = price-question comments per 1,000 views and enquiries.",
                f"Price questions are {r[1]:.1f}x more common on {r[0]}.")

    # 5 ---- anomalies -----------------------------------------------------------
    if "is_anomaly" in posts:
        an = posts[posts["is_anomaly"]]
        def ids(kind, n=3):
            s = an[an["anomaly_type"] == kind].sort_values("anomaly_score", ascending=False)
            return len(s), ", ".join(f"{r.post_id} ({r.brand}, {r.content_type})" for r in s.head(n).itertuples())
        n, e = ids("Complaint storm")
        if n:
            add("High", "Customer service", f"Investigate immediately: {n} post(s) are dominated by complaints. Reply publicly and fix the underlying issue.", f"Worst: {e}.")
        n, e = ids("Viral / over-performing")
        if n:
            add("Medium", "Content", f"Study and replicate {n} over-performing post(s): reuse the format, hook and timing, and boost them with paid reach.", f"Top: {e}.")
        n, e = ids("Likes without conversation")
        if n:
            add("Medium", "Content", f"Audit {n} post(s) whose likes are not matched by comments (possible paid/bot likes) before using them as benchmarks.", f"Top: {e}.")
        n, e = ids("Under-performing")
        if n:
            add("Low", "Content", f"Review {n} under-performing post(s) for creative, timing or targeting problems.", f"Lowest: {e}.")

    out = pd.DataFrame(recs, columns=["priority", "area", "action", "evidence"])
    if out.empty:
        return out
    return out.sort_values("priority", key=lambda s: s.map(PRIORITY), kind="stable").reset_index(drop=True)
