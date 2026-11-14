"""Simulated Pakistani lawn-suit social media data (Table A posts + Table B comments).

EVERYTHING produced here is SIMULATED (column ``data_provenance`` = "simulated").
Brands are fictional. Real observations must be collected through permitted
channels and kept separate from these records.

Ground-truth helper columns (``topic``, ``sentiment``, ``purchase_intent`` in
comments and ``sim_anomaly_type`` in posts) exist so the engine's accuracy can
be measured. The analysis never reads ``sim_anomaly_type``.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

BRANDS = ["Brand A", "Brand B", "Brand C", "Brand D", "Brand E"]
PLATFORMS = ["Instagram", "TikTok", "Facebook"]
CAMPAIGNS = ["Eid Collection", "11.11 Sale", "12.12 Sale", "Summer Lawn", "Winter Collection"]
CONTENT_TYPES = ["Reel", "Product Photo", "Influencer Post", "Styling Video",
                 "Sale Announcement", "Eid Campaign"]

TOPICS = ["Price inquiry", "Price objection", "Fabric quality", "Sizing", "Delivery inquiry",
          "Delivery complaint", "Discount request", "Availability", "Purchase intent", "Praise"]
BASE_TOPIC_W = dict(zip(TOPICS, [0.12, 0.06, 0.08, 0.07, 0.07, 0.04, 0.10, 0.09, 0.10, 0.27]))

# --- engagement model -------------------------------------------------------
BASE_FOLLOWERS = {"Instagram": 150_000, "TikTok": 200_000, "Facebook": 250_000}
VIEW_MULT = {"Reel": 1.8, "Styling Video": 1.4, "Influencer Post": 1.3, "Eid Campaign": 1.2,
             "Sale Announcement": 0.9, "Product Photo": 0.7}
LIKE_RATE = {"Reel": .045, "Styling Video": .040, "Influencer Post": .038, "Eid Campaign": .035,
             "Sale Announcement": .022, "Product Photo": .030}          # likes per view
COMMENT_RATIO = {"Reel": .08, "Styling Video": .09, "Influencer Post": .12, "Eid Campaign": .10,
                 "Sale Announcement": .18, "Product Photo": .07}        # comments per like
SHARE_RATIO = {"Reel": .10, "Styling Video": .08, "Influencer Post": .06, "Eid Campaign": .09,
               "Sale Announcement": .07, "Product Photo": .03}          # shares per like
CAMPAIGN_VIEW_MULT = {"11.11 Sale": 1.3, "12.12 Sale": 1.3, "Eid Collection": 1.2}

# --- who comments about what -----------------------------------------------
TOPIC_MODS = {
    ("content_type", "Sale Announcement"): {"Discount request": 2.8, "Price inquiry": 1.5, "Price objection": 1.6, "Praise": 0.5},
    ("content_type", "Eid Campaign"): {"Delivery inquiry": 2.0, "Availability": 1.8, "Sizing": 1.2},
    ("content_type", "Reel"): {"Praise": 1.4},
    ("content_type", "Styling Video"): {"Fabric quality": 1.5, "Sizing": 1.4, "Praise": 1.2},
    ("content_type", "Influencer Post"): {"Purchase intent": 2.0, "Price inquiry": 1.4, "Discount request": 1.3},
    ("content_type", "Product Photo"): {"Price inquiry": 1.5, "Availability": 1.4, "Sizing": 1.3, "Fabric quality": 1.5, "Praise": 0.7},
    ("campaign", "11.11 Sale"): {"Discount request": 1.6, "Delivery complaint": 1.6, "Delivery inquiry": 1.3},
    ("campaign", "12.12 Sale"): {"Discount request": 1.6, "Delivery complaint": 1.6, "Delivery inquiry": 1.3},
    ("campaign", "Summer Lawn"): {"Fabric quality": 1.4},
    ("platform", "Facebook"): {"Price inquiry": 1.3, "Delivery inquiry": 1.3, "Praise": 0.8},
    ("platform", "TikTok"): {"Praise": 1.4, "Purchase intent": 0.8},
    ("brand", "Brand C"): {"Delivery complaint": 3.0},      # brand with delivery problems
    ("brand", "Brand D"): {"Price objection": 2.0},         # brand perceived as expensive
}
LANG_P = {"Instagram": [.45, .45, .10], "TikTok": [.25, .60, .15], "Facebook": [.35, .40, .25]}
LANGS = ["English", "Roman Urdu", "Urdu"]

CITIES = [("Karachi", "کراچی"), ("Islamabad", "اسلام آباد"), ("Lahore", "لاہور"), ("Peshawar", "پشاور"),
          ("Multan", "ملتان"), ("Faisalabad", "فیصل آباد"), ("Quetta", "کوئٹہ"), ("Rawalpindi", "راولپنڈی"),
          ("Sialkot", "سیالکوٹ"), ("Hyderabad", "حیدرآباد")]
ITEMS = ["suit", "lawn suit", "dress", "3-piece", "kurta"]
EMOJIS = ["😍", "❤️", "🔥", "👌", "🙏", "✨"]

T = {  # topic -> language -> templates ({item} {city} {city_ur} {n} are filled in)
    "Price inquiry": {
        "English": ["Price please?", "What is the price of this {item}?", "How much for this {item}?", "Price?",
                    "Can you tell me the price", "Rate of this {item} plz"],
        "Roman Urdu": ["Price kya hai?", "iski price kitni hai?", "Rate bata dein please", "Is {item} ki qeemat kya hai?",
                       "kitne ka hai ye {item}", "price bataye plz"],
        "Urdu": ["قیمت کیا ہے؟", "اس کی قیمت بتا دیں", "یہ کتنے کا ہے؟", "ریٹ کیا ہے؟"]},
    "Price objection": {
        "English": ["Price is too high.", "Too expensive for a lawn {item}.", "Quality does not justify this price.",
                    "Prices have gone up so much this year", "Way overpriced honestly"],
        "Roman Urdu": ["Price bohat zyada hai.", "Bohat mehnga hai ye {item}", "itni mehngai mein kon kharidega",
                       "price kam karein, bohat zyada hai", "Mehnga hai yaar"],
        "Urdu": ["قیمت بہت زیادہ ہے", "بہت مہنگا ہے", "اتنی مہنگائی میں کون خریدے گا"]},
    "Fabric quality": {
        "English": ["What is the fabric of this {item}?", "Is it pure cotton lawn?", "How is the quality of the fabric?",
                    "Does the colour fade after wash?", "Is the fabric see-through?", "Which lawn is this, 90 GSM?"],
        "Roman Urdu": ["Fabric kaisa hai?", "Kapra kaisa hai is {item} ka?", "kya ye pure cotton hai?",
                       "Dhone ke baad rang nikalta hai?", "Quality kesi hai kapre ki?", "Kapra patla toh nahi hai?"],
        "Urdu": ["کپڑا کیسا ہے؟", "کیا یہ خالص کاٹن ہے؟", "کپڑے کی کوالٹی کیسی ہے؟"]},
    "Sizing": {
        "English": ["Is the size chart available?", "I am 5'4, which size should I take?", "Does it come in XL?",
                    "Is the {item} stitched or unstitched?", "Please share size chart", "Is it available in plus size?"],
        "Roman Urdu": ["Size chart bhej dein please", "Medium size available hai?", "Mera size large hai, ye fit ho jaye ga?",
                       "stitched hai ya unstitched?", "XL size mein milega?", "Length kitni hai is kameez ki?"],
        "Urdu": ["سائز چارٹ بھیج دیں", "کیا ایکس ایل سائز مل جائے گا؟", "یہ سلا ہوا ہے یا بغیر سلا؟"]},
    "Delivery inquiry": {
        "English": ["How many days for delivery?", "Do you deliver to {city}?", "What are the delivery charges?",
                    "Is cash on delivery available?", "How long will shipping take to {city}?"],
        "Roman Urdu": ["Delivery kitne din mein hogi?", "{city} mein delivery ho jaye gi?", "Delivery charges kitne hain?",
                       "COD available hai?", "Parcel kitne din mein pohnchega?"],
        "Urdu": ["ڈیلیوری کتنے دن میں ہوگی؟", "کیا {city_ur} میں ڈیلیوری ہوتی ہے؟", "ڈیلیوری چارجز کتنے ہیں؟"]},
    "Delivery complaint": {
        "English": ["Ordered last week, still waiting.", "Order not received after {n} days.",
                    "Worst delivery service, no update on my parcel.", "Still no tracking number, order placed {n} days ago",
                    "Nobody is replying to my order complaint"],
        "Roman Urdu": ["Order kiya tha {n} din pehle, abhi tak nahi mila.", "Parcel abhi tak nahi aaya, koi reply nahi karta",
                       "Delivery bohat late hai, tracking bhi nahi mil rahi", "Itna intezar kar liya order ka",
                       "Mera order kahan hai? Koi jawab nahi"],
        "Urdu": ["آرڈر کیے ہوئے دس دن ہو گئے، ابھی تک نہیں ملا", "پارسل ابھی تک نہیں پہنچا", "کوئی جواب نہیں دیتا، بہت بری سروس"]},
    "Discount request": {
        "English": ["Any discount on this suit?", "Is there any sale coming up?", "Please give discount code",
                    "Any offer for bulk orders?", "Will this be on 50% off?", "Do you have any promo code?"],
        "Roman Urdu": ["Discount milega is suit pe?", "Sale kab lag rahi hai?", "Koi discount code hai?",
                       "Kuch kam karein na, discount dein", "Sale mein kitna off hoga?"],
        "Urdu": ["کیا اس پر ڈسکاؤنٹ ملے گا؟", "سیل کب لگے گی؟", "کوئی ڈسکاؤنٹ کوڈ ہے؟"]},
    "Availability": {
        "English": ["Is this available in {city}?", "Is this in stock?", "Do you have a store in {city}?",
                    "Is this still available?", "Please restock this design"],
        "Roman Urdu": ["Kya ye {city} mein available hai?", "Stock mein hai ye?", "{city} mein outlet kahan hai?",
                       "Ye design dobara kab aayega?", "Available hai abhi?"],
        "Urdu": ["کیا یہ {city_ur} میں دستیاب ہے؟", "کیا یہ اسٹاک میں ہے؟", "یہ ڈیزائن دوبارہ کب آئے گا؟"]},
    "Purchase intent": {
        "English": ["I want to order this {item}.", "How can I place an order?", "Inbox me the details please",
                    "Interested, DM please", "Sending you a DM to order", "I'll take two of these. How do I pay?"],
        "Roman Urdu": ["Inbox me details send kar dein.", "Mujhe ye order karna hai, kaise karun?", "Interested hun, DM karein please",
                       "Order kaise place karun?", "Ye mujhe chahiye, payment kahan bhejun?"],
        "Urdu": ["ان باکس میں تفصیل بھیج دیں", "مجھے یہ آرڈر کرنا ہے", "میں خریدنا چاہتی ہوں"]},
    "Praise": {
        "English": ["Very beautiful dress!", "Love the colours", "Stunning!", "Gorgeous collection, well done",
                    "Looks so elegant", "Masha Allah beautiful"],
        "Roman Urdu": ["Bohat khoobsurat hai!", "Boht pyara design hai", "MashaAllah bohat acha lag raha hai",
                       "Kamaal ka collection hai", "Rang bohat piyare hain"],
        "Urdu": ["بہت خوبصورت ڈریس", "ماشاءاللہ بہت پیارا", "کیا کمال کا ڈیزائن ہے"]},
}

TYPOS = {
    "English": [("please", "plz"), ("price", "prise"), ("delivery", "delivary"), ("available", "availble"), ("discount", "dicount")],
    "Roman Urdu": [("kitne", "kitnay"), ("bohat", "bht"), ("hai", "h"), ("kya", "kia"), ("karein", "krein"),
                   ("please", "plz"), ("mehnga", "mahenga"), ("delivery", "dilevery"), ("kapra", "kapda")],
    "Urdu": [],
}
SENTIMENT_P = {  # topic -> (Positive, Neutral, Negative)
    "Praise": (1.0, 0, 0), "Price objection": (0, .1, .9), "Delivery complaint": (0, 0, 1.0),
    "Purchase intent": (.5, .5, 0), "Fabric quality": (.15, .70, .15),
}
INTENT = {"Purchase intent": "Strong intent", "Price inquiry": "Possible inquiry", "Discount request": "Possible inquiry",
          "Availability": "Possible inquiry", "Sizing": "Possible inquiry"}


def _make_comment(topic: str, lang: str, rng: np.random.Generator) -> str:
    def fill(t: str) -> str:
        c = CITIES[rng.integers(len(CITIES))]
        return t.format(item=rng.choice(ITEMS), city=c[0], city_ur=c[1], n=int(rng.integers(5, 21)))

    text = fill(rng.choice(T[topic][lang]))
    if rng.random() < 0.05:  # code-mixed / multi-intent comment; label stays the first topic
        other = rng.choice([t for t in TOPICS if t != topic])
        text += " " + fill(rng.choice(T[other][lang]))
    if lang != "Urdu":
        r = rng.random()
        text = text.lower() if r < 0.15 else text.upper() if r < 0.20 else text
        for a, b in TYPOS[lang]:
            if rng.random() < 0.25:
                text = re.sub(rf"\b{a}\b", b, text, flags=re.I)
        if rng.random() < 0.15:
            text = text.rstrip("?!.")
        if rng.random() < 0.05:
            w = text.split(" ")
            i = int(rng.integers(len(w)))
            w[i] = w[i] + w[i][-1:] * 2
            text = " ".join(w)
    if rng.random() < 0.25:
        text += " " + str(rng.choice(EMOJIS))
    return text


def _topic_probs(row: dict, storm: bool) -> np.ndarray:
    w = dict(BASE_TOPIC_W)
    for key, mods in TOPIC_MODS.items():
        if row[key[0]] == key[1]:
            for t, m in mods.items():
                w[t] *= m
    p = np.array([w[t] for t in TOPICS], float)
    p /= p.sum()
    if storm:  # complaint storm: most comments are delivery complaints / price anger
        s = np.zeros(len(TOPICS))
        s[TOPICS.index("Delivery complaint")], s[TOPICS.index("Price objection")] = .45, .25
        p = .30 * p + s
        p /= p.sum()
    return p


def generate_dataset(n_posts: int = 2500, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (posts, comments) following Table A and Table B of the project brief."""
    rng = np.random.default_rng(seed)
    followers = {(b, p): BASE_FOLLOWERS[p] * rng.lognormal(0, .35) for b in BRANDS for p in PLATFORMS}
    rows = []
    for i in range(n_posts):
        brand = BRANDS[rng.integers(len(BRANDS))]
        platform = str(rng.choice(PLATFORMS, p=[.45, .30, .25]))
        campaign = CAMPAIGNS[rng.integers(len(CAMPAIGNS))]
        ct = str(rng.choice(CONTENT_TYPES, p=[.25, .20, .15, .15, .15, .10]))
        if campaign == "Eid Collection" and rng.random() < .35:
            ct = "Eid Campaign"
        if campaign in ("11.11 Sale", "12.12 Sale") and rng.random() < .40:
            ct = "Sale Announcement"
        days = int(rng.integers(1, 46))
        maturity = 1 - np.exp(-days / 2)
        vm = VIEW_MULT[ct] * (1.4 if platform == "TikTok" and ct in ("Reel", "Styling Video") else 1.0)
        views = followers[(brand, platform)] * .08 * vm * CAMPAIGN_VIEW_MULT.get(campaign, 1.0) * rng.lognormal(0, .55) * maturity
        likes = views * LIKE_RATE[ct] * rng.lognormal(0, .2)
        comments = likes * COMMENT_RATIO[ct] * rng.lognormal(0, .25)
        shares = likes * SHARE_RATIO[ct] * rng.lognormal(0, .3)
        rows.append(dict(post_id=f"P{i + 1:04d}", brand=brand, platform=platform, campaign=campaign, content_type=ct,
                         product_category="Lawn suit", likes=likes, comments_count=comments, shares=shares,
                         views=views, days_since_post=days, sim_anomaly_type=""))
    posts = pd.DataFrame(rows)

    # inject known anomalies so detection quality can be measured
    n_anom = int(round(.032 * n_posts))
    pick = rng.choice(n_posts, n_anom, replace=False)
    kinds = rng.choice(["viral", "complaint_storm", "flop", "like_spike"], n_anom, p=[.30, .25, .25, .20])
    for idx, kind in zip(pick, kinds):
        if kind == "flop" and posts.at[idx, "days_since_post"] < 7:
            kind = "viral"
        if kind == "viral":
            f = rng.uniform(6, 12); posts.loc[idx, ["likes", "views", "shares", "comments_count"]] *= [f, f, f * 1.3, f]
        elif kind == "complaint_storm":
            posts.loc[idx, "comments_count"] *= rng.uniform(8, 15); posts.loc[idx, "likes"] *= .8
        elif kind == "flop":
            posts.loc[idx, ["likes", "views", "shares", "comments_count"]] *= rng.uniform(.05, .10)
        else:  # likes without conversation (e.g. bought likes)
            posts.loc[idx, "likes"] *= rng.uniform(8, 12)
        posts.at[idx, "sim_anomaly_type"] = kind

    for c in ["likes", "comments_count", "shares", "views"]:
        posts[c] = rng.poisson(posts[c].clip(lower=0)).astype(float)
    # "if available": photos usually have no view count, Instagram hides shares
    hide_v = ((posts.platform == "Facebook") & (posts.content_type == "Product Photo") & (rng.random(n_posts) < .5)) | \
             ((posts.platform == "Instagram") & (posts.content_type == "Product Photo") & (rng.random(n_posts) < .8))
    posts.loc[hide_v, "views"] = np.nan
    posts.loc[(posts.platform == "Instagram") & (rng.random(n_posts) < .6), "shares"] = np.nan
    for c in ["likes", "comments_count"]:
        posts[c] = posts[c].astype(int)
    posts["data_provenance"] = "simulated"

    crow = []
    storm = set(posts.index[posts.sim_anomaly_type == "complaint_storm"])
    for i, r in posts.iterrows():
        n = int(min(r.comments_count, rng.integers(4, 13)))
        if n == 0:
            continue
        p = _topic_probs(r.to_dict(), i in storm)
        lp = LANG_P[r.platform]
        for _ in range(n):
            topic = TOPICS[rng.choice(len(TOPICS), p=p)]
            lang = LANGS[rng.choice(3, p=lp)]
            sp = SENTIMENT_P.get(topic, (.05, .90, .05))
            crow.append(dict(post_id=r.post_id, comment_text=_make_comment(topic, lang, rng), language_type=lang,
                             topic=topic, sentiment=["Positive", "Neutral", "Negative"][rng.choice(3, p=sp)],
                             purchase_intent=INTENT.get(topic, "None"), data_provenance="simulated"))
    comments = pd.DataFrame(crow)
    return posts, comments


if __name__ == "__main__":
    p, c = generate_dataset()
    p.to_csv("data/posts.csv", index=False)
    c.to_csv("data/comments.csv", index=False)
    print(f"posts: {len(p)}  comments: {len(c)}  -> data/posts.csv, data/comments.csv")
