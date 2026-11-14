# Pakistan Social Audience & Campaign Intelligence Engine

A clustering-based analytics engine and small web app for a Pakistani online fashion brand (demo category:
**women's lawn suits**) that runs Instagram, TikTok and Facebook campaigns such as 11.11, 12.12 and Eid.
It turns thousands of English / Roman Urdu / Urdu comments and post metrics into marketing decisions.

> ⚠️ **The bundled data is SIMULATED.** Brands are fictional ("Brand A-E"), every row has
> `data_provenance = simulated`, and the findings describe the simulation, not real audiences.
> See [Using real data](#using-real-data).

## The five questions it answers

| # | Question | Module | Method |
|---|----------|--------|--------|
| 1 | Which content attracts attention? | `engine/content_performance.py` | Age/brand/platform-adjusted **engagement index**, bootstrap 95% CIs, Kruskal-Wallis test |
| 2 | What does the audience want? | `engine/comment_clusters.py` | Code-mixed text normalisation → TF-IDF → SVD → **K-Means** → named audience needs |
| 3 | Which patterns are unusual? | `engine/anomalies.py` | Robust z-scores (median/MAD) against *comparable* posts; typed reasons |
| 4 | Which needs should the brand address? | `engine/comment_clusters.py` | Need × content/campaign/brand **lift** heat-map; post audience profiles (K-Means on need mix) |
| 5 | What should marketing do next? | `engine/recommendations.py` | Evidence-based rules; every action shows its numbers |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py                 # web app (demo data, or upload your own CSVs)
python run_analysis.py               # CLI: writes CSVs + reports/analysis_report.md
python -m engine.data_generator      # regenerate the simulated data in data/
python -m pytest -q                  # 21 tests
```

## Data schema

**Table A – posts** (`posts.csv`): required `post_id, brand, platform, campaign, content_type, likes, comments_count, days_since_post`;
optional `shares, views, product_category` (blank = not publicly available; numbers like `1,250` are accepted).

**Table B – comments** (`comments.csv`): required `post_id, comment_text`; optional `language_type, topic, sentiment, purchase_intent`.
If a `topic` column is present, the app reports how well the clusters agree with it.

The engine never reads the generator's ground-truth helper column `sim_anomaly_type`; it is used only for evaluation.

## How it works (short)

1. **Validation** – coerces numbers, drops duplicate / orphan / empty rows and tells you what it removed.
2. **Engagement index** – OLS baseline of `log(likes+comments+shares)` on brand, platform and post age (`log d`, `1/d`).
   Content type is deliberately *not* in the baseline, so its effect shows up in the index (1.0 = typical post).
3. **Comment understanding** – a small bilingual glossary (`CONCEPTS` in `engine/text_utils.py`) maps "price / qeemat / قیمت" etc. to one token;
   filler words are removed; word 1-2-gram TF-IDF → 50-d SVD → K-Means (default 28 fine phrase groups).
   Groups are merged into **needs** using their dominant theme and complaint-vs-question polarity.
   Groups the lexicon cannot name land in **"Unclear / needs review"** rather than being forced into a label.
4. **Anomalies** – four robust-z features (engagement vs expectation, comments/like, shares/like, engagement/view); flagged at |z| ≥ 3.5.
5. **Recommendations** – only emitted when a need is ≥1.3× over-represented in a group with ≥40 comments (or a significant content gap exists).

## Accuracy – what was actually measured

On the **simulated** data (labels come from the generator, so they are ground truth):

| Metric | Default data (seed 42) | 5 unseen seeds (1-5), mean (range) |
|--------|------------------------|-------------------------------------|
| Need-level agreement with true topic | 85.7 % | 84.6 % (80.0 – 89.7 %) |
| Phrase-group purity | 87.2 % | 85.4 % (83.5 – 88.2 %) |
| Anomaly precision | 87.7 % | 81.8 % (76.8 – 85.3 %) |
| Anomaly recall | 80.0 % | 71.0 % (63.7 – 78.7 %) |

Honest reading of these numbers:

* They show the code works and generalises across *simulated* seeds. **They are not a guarantee for real comments.**
  Real comments are noisier, more varied and contain slang, sarcasm and spellings the glossary has not seen.
* Complaint storms and "likes without conversation" are found reliably; mild viral posts and mild flops (≈6-8× effects) are
  below the detection limit at the default threshold. Lower the sensitivity slider to catch more, at the cost of precision.
* The glossary and lexicon were written by looking at the simulated templates. Expect to extend them for real data.
* Clustering is lexical (TF-IDF). A multilingual sentence-embedding model would handle paraphrases better; it was not used
  because it needs a model download.

## Using real data

* Collect only through **permitted** channels (official APIs, exports from your own brand accounts, licensed tools) and follow each
  platform's terms. Do not assume comments or analytics are publicly downloadable.
* Keep `source_url`, collection date and reuse permissions next to the data, and keep real and simulated records separate
  (set `data_provenance` accordingly). Remove personal identifiers before sharing.
* Upload your two CSVs in the app sidebar or run `python run_analysis.py --posts my_posts.csv --comments my_comments.csv`.
* Add the new spellings you see to `CONCEPTS` / `_LEX` in `engine/text_utils.py`, then check the "Unclear / needs review" need and the
  example comments per need.
* Optional extra context mentioned in the project brief (check each source's licence first): a 2026 study of Pakistani fashion-brand
  sentiment, Open Data Pakistan's e-commerce orders, and the Pakistani Fashion Catalogue on Hugging Face for PKR prices.

## Project layout

```
app.py                  Streamlit web app (6 tabs + filters + CSV downloads)
run_analysis.py         command-line report
engine/
  data_generator.py     simulated Table A + Table B (clearly labelled)
  data_io.py            loading + validation
  text_utils.py         normalisation, glossary, lexicon, city detection
  content_performance.py   Q1
  comment_clusters.py      Q2 / Q4
  anomalies.py             Q3
  recommendations.py       Q5
  pipeline.py           glues everything together
tests/test_engine.py    21 tests (unit + end-to-end on an unseen seed)
data/                   simulated demo CSVs
reports/                sample CLI output on the demo data
```

## Limitations

* Comments are a sample (4-12 per post), so need shares are estimates of the full comment stream.
* Engagement index assumes similar posts age the same way; very old or boosted posts may need extra care.
* Anomalies are leads to investigate, not proof of bots, complaints or virality.
* Roman Urdu has no standard spelling; coverage depends on the glossary.
