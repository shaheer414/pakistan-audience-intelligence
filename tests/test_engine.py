import numpy as np
import pandas as pd
import pytest

from engine.anomalies import robust_z
from engine.data_generator import generate_dataset
from engine.data_io import validate
from engine.pipeline import fit, recommend
from engine.text_utils import canonical_text, clean_text, find_city, rule_theme


# ---------------------------------------------------------------- text
def test_spelling_variants_map_to_same_concept():
    assert canonical_text("Qeemat kya hai?") .split()[0] == "price"
    assert "price" in canonical_text("قیمت کیا ہے؟").split()
    assert "delivery" in canonical_text("Dilevery kitne din mein hogi").split()
    assert canonical_text("pleaseeee") == "please"


def test_urdu_and_roman_urdu_share_tokens():
    assert set(clean_text("ڈیلیوری کتنے دن میں ہوگی؟").split()) & set(clean_text("Delivery kitne din mein hogi?").split()) >= {"delivery", "howmuch", "days"}


@pytest.mark.parametrize("text,theme", [
    ("Price please?", "Price"), ("Any discount on this suit?", "Discount"), ("Is this available in Karachi?", "Availability"),
    ("Ordered last week, still waiting.", "Delivery"), ("Inbox me details send kar dein.", "Purchase intent"),
    ("Very beautiful dress!", "Praise"), ("Size chart bhej dein please", "Sizing"), ("Kapra kaisa hai?", "Fabric quality"),
    ("کیا اس پر ڈسکاؤنٹ ملے گا؟", "Discount"), ("ڈیلیوری کتنے دن میں ہوگی؟", "Delivery")])
def test_rule_theme_on_brief_examples(text, theme):
    assert rule_theme(text) == theme


def test_find_city_english_and_urdu():
    assert find_city(canonical_text("Is this in Karachi?")) == "Karachi"
    assert find_city(canonical_text("کیا یہ لاہور میں ہے؟")) == "Lahore"
    assert find_city(canonical_text("price please")) is None


def test_robust_z_ignores_nan_and_outlier():
    s = pd.Series([1.0] * 20 + [np.nan] * 10 + [50.0])
    z = robust_z(s, floor=0.05)
    assert z.iloc[-1] > 10 and (z.iloc[20:30] == 0).all()


# ---------------------------------------------------------------- data
def test_generator_schema_and_determinism():
    p1, c1 = generate_dataset(200, seed=3)
    p2, c2 = generate_dataset(200, seed=3)
    pd.testing.assert_frame_equal(p1, p2)
    pd.testing.assert_frame_equal(c1, c2)
    assert {"post_id", "brand", "platform", "campaign", "content_type", "product_category", "likes", "comments_count", "shares", "views",
            "days_since_post"} <= set(p1)
    assert {"post_id", "comment_text", "language_type", "topic", "sentiment", "purchase_intent"} <= set(c1)
    assert (p1["data_provenance"] == "simulated").all() and set(c1.language_type) == {"English", "Roman Urdu", "Urdu"}
    assert set(c1.post_id) <= set(p1.post_id)


def test_validate_cleans_and_reports():
    p, c = generate_dataset(300, seed=1)
    p["likes"] = p["likes"].map("{:,}".format)                      # "1,250" style numbers from the brief
    p = pd.concat([p, p.head(2)])                                    # duplicate post_ids
    c = pd.concat([c, pd.DataFrame({"post_id": ["NOPE", "P0001"], "comment_text": ["x", "  "]})])
    p2, c2, warn = validate(p, c)
    assert p2["likes"].dtype.kind in "fi" and p2["post_id"].is_unique
    assert "NOPE" not in set(c2.post_id) and (c2["comment_text"].str.len() > 0).all()
    assert any("duplicate" in w for w in warn) and any("removed" in w for w in warn)


def test_validate_rejects_missing_columns_and_tiny_data():
    p, c = generate_dataset(100, seed=1)
    with pytest.raises(ValueError, match="missing required"):
        validate(p.drop(columns=["likes"]), c)
    with pytest.raises(ValueError, match="comment_text"):
        validate(p, c.drop(columns=["comment_text"]))
    with pytest.raises(ValueError, match="usable posts"):
        validate(p.head(5), c)


# ---------------------------------------------------------------- end to end (unseen seed, not used for tuning)
@pytest.fixture(scope="module")
def result():
    p, c = generate_dataset(1500, seed=7)
    return fit(p, c)


def test_accuracy_floors(result):
    ev = result["evaluation"]
    assert ev["clustering"]["needs"]["accuracy"] >= 0.75
    assert ev["clustering"]["phrase_groups"]["purity"] >= 0.75
    assert ev["anomalies"]["precision"] >= 0.65 and ev["anomalies"]["recall"] >= 0.55
    assert ev["anomalies"]["recall_by_type"]["complaint_storm"] >= 0.9


def test_content_ranking_matches_simulated_truth(result):
    from engine.content_performance import performance_table
    t = performance_table(result["posts"], "content_type").set_index("content_type")
    assert t.loc["Reel", "median_index"] > t.loc["Product Photo", "median_index"]
    assert t.loc["Reel", "ci_low"] > t.loc["Product Photo", "ci_high"]


def test_recommendations_find_planted_patterns(result):
    r = recommend(result["posts"], result["comments"])
    text = " ".join(r["action"])
    assert {"priority", "area", "action", "evidence"} == set(r.columns)
    assert "Brand C" in text and "Brand D" in text
    assert (r["evidence"].str.len() > 20).all() and r["priority"].iloc[0] == "High"


def test_pipeline_works_without_labels_shares_or_views():
    p, c = generate_dataset(500, seed=11)
    p = p.drop(columns=["shares", "views", "sim_anomaly_type", "product_category", "data_provenance"])
    c = c[["post_id", "comment_text"]]
    r = fit(p, c, k=20)
    assert r["evaluation"]["clustering"] == {} and r["comments"]["need"].notna().all()
    assert len(recommend(r["posts"], r["comments"])) > 0
