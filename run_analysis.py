"""Command-line runner: python run_analysis.py [--posts P.csv --comments C.csv] [--k 28] [--z 3.5] [--out reports]

Without file arguments it analyses the bundled SIMULATED demo data. Writes CSV outputs and a
Markdown summary to the output folder.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from engine.content_performance import group_difference_pvalue, performance_table
from engine.data_io import load_demo
from engine.pipeline import fit, recommend


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--posts"), ap.add_argument("--comments")
    ap.add_argument("--k", type=int, default=28, help="number of fine-grained comment groups")
    ap.add_argument("--z", type=float, default=3.5, help="anomaly robust-z threshold")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    if bool(a.posts) != bool(a.comments):
        ap.error("give both --posts and --comments, or neither (demo data)")
    posts, comments = (pd.read_csv(a.posts), pd.read_csv(a.comments)) if a.posts else load_demo()

    R = fit(posts, comments, k=a.k, z_threshold=a.z)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fp, cm = R["posts"], R["comments"]
    recs = recommend(fp, cm)

    tables = {"content_performance_by_type": performance_table(fp, "content_type"),
              "content_performance_by_campaign": performance_table(fp, "campaign"),
              "audience_needs": R["needs"], "unusual_posts": fp[fp["is_anomaly"]].sort_values("anomaly_score", ascending=False),
              "recommendations": recs}
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")

    md = ["# Analysis summary", ""]
    md += [f"- Posts analysed: {len(fp):,}  |  comments: {len(cm):,}  |  unusual posts: {int(fp['is_anomaly'].sum())}"]
    md += [f"- {w}" for w in R["warnings"]]
    t = tables["content_performance_by_type"]
    md += ["", "## 1. Content (median engagement index, 1.0 = typical)", ""]
    md += [f"- {r.content_type}: {r.median_index:.2f} (95% CI {r.ci_low:.2f}-{r.ci_high:.2f}, n={r.posts})" for r in t.itertuples()]
    md += [f"- Kruskal-Wallis p = {group_difference_pvalue(fp, 'content_type'):.3g}"]
    md += ["", "## 2. What the audience wants", ""]
    md += [f"- {r.need}: {r.share:.1%} of comments (negative {r.negative_share:.0%}) e.g. {r.examples.split(' | ')[0]}" for r in R["needs"].itertuples()]
    md += ["", "## 3. Unusual posts", ""]
    md += [f"- {k}: {v}" for k, v in fp["anomaly_type"].replace("", pd.NA).dropna().value_counts().items()]
    md += ["", "## 5. Recommended actions", ""]
    md += [f"- **{r.priority} / {r.area}**: {r.action}  \n  _Evidence: {r.evidence}_" for r in recs.itertuples()]
    ev = R["evaluation"]
    if ev["clustering"]:
        md += ["", "## Accuracy against labels in the data", "", f"- Need-level agreement: {ev['clustering']['needs']['accuracy']:.1%}",
               f"- Phrase-group purity: {ev['clustering']['phrase_groups']['purity']:.1%}"]
        if ev["anomalies"]:
            md += [f"- Anomaly precision {ev['anomalies']['precision']:.1%}, recall {ev['anomalies']['recall']:.1%}"]
    (out / "analysis_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nFiles written to {out.resolve()}")


if __name__ == "__main__":
    main()
