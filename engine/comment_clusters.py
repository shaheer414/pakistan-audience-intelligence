"""Q2 / Q4 - What does the audience want, and which needs should the brand address?

Pipeline: clean text -> character n-gram TF-IDF (robust to Roman-Urdu spelling) -> SVD ->
spherical K-Means. Cluster names come from the dominant rule-based theme inside each
cluster; the clustering itself never sees labels. If the data has a human-labelled
``topic`` column, agreement metrics (ARI / NMI / purity) are reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import normalize

from .text_utils import STOPWORDS, canonical_text, find_city, rule_negative, rule_theme

TOPIC_TO_THEME = {"Price inquiry": "Price", "Price objection": "Price", "Delivery inquiry": "Delivery",
                  "Delivery complaint": "Delivery", "Discount request": "Discount"}
NEED_NAMES = {"Price": "Price questions", "Discount": "Discount seekers", "Fabric quality": "Fabric-quality questions",
              "Sizing": "Sizing questions", "Delivery": "Delivery questions", "Availability": "Availability / stock questions",
              "Purchase intent": "Ready-to-buy (purchase intent)", "Praise": "Praise / brand love", "Other": "Other / unclear"}
NEG_NAMES = {"Price": "Price objections", "Delivery": "Delivery complaints"}
_STOP = {"qmark", "num", "emojipos", "emojineg", "hai", "ye", "is", "ka", "ki", "ko", "the", "this", "of", "to", "in", "me",
         "mein", "for", "you", "it", "be", "and", "na", "kar", "ho", "se", "hain", "are", "there", "any", "can", "will", "pe"}


def _embed(clean: pd.Series, seed: int):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(clean)
    n = max(2, min(50, X.shape[1] - 1, X.shape[0] - 1))
    return normalize(TruncatedSVD(n, random_state=seed).fit_transform(X))


def choose_k(Z: np.ndarray, k_range=(16, 36), seed: int = 42) -> tuple[int, dict[int, float]]:
    """Pick k with the best silhouette on a sample (also returned so the UI can plot it)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), min(4000, len(Z)), replace=False)
    scores = {}
    for k in range(k_range[0], min(k_range[1], len(Z) - 1) + 1):
        lab = KMeans(k, n_init=3, random_state=seed).fit_predict(Z)
        scores[k] = float(silhouette_score(Z[idx], lab[idx])) if len(set(lab[idx])) > 1 else -1.0
    return max(scores, key=scores.get), scores


def _top_terms(clean: pd.Series, labels: np.ndarray, k: int, n_terms: int = 6) -> dict[int, list[str]]:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True, stop_words=list(_STOP))
    try:
        W = vec.fit_transform(clean)
    except ValueError:
        return {c: [] for c in range(k)}
    terms, overall = np.array(vec.get_feature_names_out()), np.asarray(W.mean(axis=0)).ravel()
    return {c: list(terms[np.argsort(-(np.asarray(W[labels == c].mean(axis=0)).ravel() - overall))[:n_terms]]) for c in range(k)}


def cluster_comments(comments: pd.DataFrame, k: int | None = 28, k_range=(16, 36), seed: int = 42) -> dict:
    df = comments.copy().reset_index(drop=True)
    df["canon"] = df["comment_text"].map(canonical_text)
    df["clean"] = df["canon"].map(lambda t: " ".join(w for w in t.split() if w not in STOPWORDS))
    df = df[df["clean"].str.strip() != ""].reset_index(drop=True)
    Z = _embed(df["clean"], seed)
    k_scores: dict[int, float] = {}
    if k is None:
        k, k_scores = choose_k(Z, k_range, seed)
    km = KMeans(k, n_init=10, random_state=seed).fit(Z)
    df["cluster"] = km.labels_
    df["rule_theme"] = df["canon"].map(lambda t: rule_theme(t, True))
    df["negative"] = (df["sentiment"].eq("Negative") if "sentiment" in df else df["canon"].map(lambda t: rule_negative(t, True))).astype(bool)
    df["dist_to_center"] = np.linalg.norm(Z - normalize(km.cluster_centers_)[km.labels_], axis=1)
    terms = _top_terms(df["clean"], km.labels_, k)

    df["complaint"] = df["canon"].map(lambda t: rule_negative(t, True))   # lexicon polarity (works without labels)
    rows = []
    for c in range(k):
        g = df[df.cluster == c]
        theme = g["rule_theme"].value_counts().index[0]
        purity = (g["rule_theme"] == theme).mean()
        ex = g.sort_values("dist_to_center").drop_duplicates("comment_text")["comment_text"].head(3).tolist()
        rows.append(dict(cluster=c, theme=theme if purity >= 0.5 and theme != "Other" else "Unclear", comments=len(g),
                         share=len(g) / len(df), negative_share=g["negative"].mean(), theme_purity=purity,
                         top_terms=", ".join(terms[c]), examples=" | ".join(ex),
                         languages=", ".join(f"{l} {v:.0%}" for l, v in g["language_type"].value_counts(normalize=True).items())
                         if "language_type" in g else ""))
    groups = pd.DataFrame(rows).set_index("cluster")
    df["theme"] = df["cluster"].map(groups["theme"])
    # a need = theme of the phrase group + polarity (complaint vs question) for themes where it matters
    df["need"] = [_need_name(t, neg) for t, neg in zip(df["theme"], df["complaint"])]
    groups["need"] = df.groupby("cluster")["need"].agg(lambda s: s.value_counts().index[0])
    needs = []
    for name, g in df.groupby("need"):
        sub = groups[groups["need"] == name].sort_values("comments", ascending=False)
        needs.append(dict(need=name, comments=len(g), share=len(g) / len(df), negative_share=g["negative"].mean(),
                          phrase_groups=len(sub), top_terms=" | ".join(sub["top_terms"].head(3)),
                          examples=" | ".join(g.drop_duplicates("comment_text")["comment_text"].head(3)),
                          languages=", ".join(f"{l} {v:.0%}" for l, v in g["language_type"].value_counts(normalize=True).items())
                          if "language_type" in g else ""))
    df["city"] = df["canon"].map(find_city)
    return dict(comments=df, groups=groups.reset_index().sort_values("comments", ascending=False).reset_index(drop=True),
                needs=pd.DataFrame(needs).sort_values("comments", ascending=False).reset_index(drop=True),
                k=k, k_scores=k_scores, evaluation=evaluate_clusters(df))


def _need_name(theme: str, complaint: bool) -> str:
    if complaint and theme in NEG_NAMES:
        return NEG_NAMES[theme]
    return NEED_NAMES.get(theme, "Unclear / needs review")


TOPIC_TO_NEED = {"Price inquiry": "Price questions", "Price objection": "Price objections", "Delivery inquiry": "Delivery questions",
                 "Delivery complaint": "Delivery complaints", "Discount request": "Discount seekers", "Fabric quality": "Fabric-quality questions",
                 "Sizing": "Sizing questions", "Availability": "Availability / stock questions",
                 "Purchase intent": "Ready-to-buy (purchase intent)", "Praise": "Praise / brand love"}


def evaluate_clusters(df: pd.DataFrame) -> dict:
    """Agreement with labelled comments (only if a 'topic' column exists). Honest reading:
    'phrase_groups' = pure unsupervised clustering; 'needs' = after lexicon-based merging/naming."""
    if "topic" not in df:
        return {}
    ct = pd.crosstab(df["cluster"], df["topic"])
    truth_need = df["topic"].map(TOPIC_TO_NEED).fillna(df["topic"])
    return {"phrase_groups": dict(ARI=adjusted_rand_score(df["topic"], df["cluster"]),
                                  NMI=normalized_mutual_info_score(df["topic"], df["cluster"]),
                                  purity=float(ct.max(axis=1).sum() / ct.values.sum())),
            "needs": dict(accuracy=float((df["need"] == truth_need).mean()), ARI=adjusted_rand_score(truth_need, df["need"]),
                          NMI=normalized_mutual_info_score(truth_need, df["need"])),
            "lexicon_only_theme_accuracy": float((df["rule_theme"] == df["topic"].map(lambda t: TOPIC_TO_THEME.get(t, t))).mean())}


def need_by_group(comments: pd.DataFrame, posts: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(share, lift): share of each need inside each group, and lift vs the overall share."""
    m = comments if group_col in comments else comments.merge(posts[["post_id", group_col]], on="post_id", how="inner")
    m = m[m["post_id"].isin(posts["post_id"])]
    share = pd.crosstab(m[group_col], m["need"], normalize="index")
    overall = m["need"].value_counts(normalize=True)
    return share, share.div(overall.reindex(share.columns), axis=1)


def post_audience_profiles(comments: pd.DataFrame, n_profiles: int = 4, min_comments: int = 4, seed: int = 42) -> pd.DataFrame:
    """Cluster POSTS by the mix of needs in their comments -> audience profile per post."""
    share = pd.crosstab(comments["post_id"], comments["need"], normalize="index")
    share = share[comments.groupby("post_id").size().reindex(share.index) >= min_comments]
    n_profiles = min(n_profiles, len(share))
    if n_profiles < 2:
        return pd.DataFrame(columns=["post_id", "profile"])
    lab = KMeans(n_profiles, n_init=10, random_state=seed).fit_predict(share.values)
    overall = comments["need"].value_counts(normalize=True)
    names = {}
    for p in range(n_profiles):
        lift = (share[lab == p].mean() / overall.reindex(share.columns)).sort_values(ascending=False)
        names[p] = f"{lift.index[0]}-led"
    out = pd.DataFrame({"post_id": share.index, "profile": [names[i] for i in lab]})
    d = out["profile"].duplicated(keep=False)
    out.loc[d, "profile"] = [f"{n} #{names_i}" for n, names_i in zip(out.loc[d, "profile"], lab[d.values])]
    return out
