"""Text normalisation for English / Roman Urdu / Urdu-script comments.

Code-mixed text is hard because the same idea is written many ways ("price", "qeemat",
"قیمت"). ``CONCEPTS`` is a small bilingual glossary that maps those variants to ONE
canonical token so clustering can group by meaning instead of by script/spelling.
Extend it when you load real data (see README, "Adapting to real data").

Also contains a rule-based lexicon used only to NAME discovered clusters and as a
fallback when a dataset has no human-labelled ``topic`` column.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np

_URDU_CHARS = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ه": "ہ", "ۃ": "ہ", "ة": "ہ", "ؤ": "و"})
_DIACRITICS = re.compile("[\u064B-\u065F\u0670\u0640]")
_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"[@#]\w+")
_EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]")
_POS_EMOJI, _NEG_EMOJI = set("😍🥰😘❤💕🔥👌😊✨💖👏"), set("😡😠😤👎😢🤬😞")


def _norm_urdu(s: str) -> str:
    return _DIACRITICS.sub("", s).translate(_URDU_CHARS)


# canonical token -> spellings in English / Roman Urdu / Urdu script
CONCEPTS = {
    "price": "price prices prise qeemat keemat qimat kimat rate rates cost قیمت ریٹ",
    "expensive": "mehnga mehenga mahenga mahnga mehngi mahngi mehngai mahngai expensive costly overpriced مہنگا مہنگی مہنگائی",
    "high": "zyada ziada zyadah zayada zada high زیادہ",
    "very": "bohat bht boht bohot bahut very too بہت",
    "howmuch": "kitna kitne kitni kitnay kitny کتنے کتنا کتنی",
    "fabric": "fabric kapra kapray kapda kapre kapry کپڑا کپڑے",
    "cotton": "cotton کاٹن", "quality": "quality kwality کوالٹی", "pure": "pure خالص",
    "size": "size sizes سائز", "chart": "chart چارٹ", "xl": "xl ایکس", "stitched": "stitched سلا",
    "delivery": "delivery deliver dilivery delivary deliveri delevery dilevery dlivery shipping ڈیلیوری",
    "parcel": "parcel پارسل", "days": "days day din دن", "charges": "charges charge چارجز",
    "order": "order oder ordr آرڈر", "discount": "discount dicount discnt off ڈسکاؤنٹ",
    "sale": "sale سیل", "code": "code promo کوڈ",
    "available": "available avail availble avalible avaliable دستیاب", "stock": "stock اسٹاک",
    "restock": "restock dobara دوبارہ", "design": "design ڈیزائن",
    "beautiful": "beautiful beautifull khoobsurat khubsurat khoobsoorat khubsoorat khobsurat pyara pyari pyare piyara piyare "
                 "piyari gorgeous stunning elegant kamaal love خوبصورت پیارا کمال",
    "mashaallah": "mashaallah ماشاءاللہ", "please": "plz pls plzz plss please",
    "inbox": "inbox dm باکس", "details": "details tafseel تفصیل", "send": "send share bhej bhejo بھیج",
    "want": "chahiye chahti chahta chahun چاہتی چاہتا", "buy": "khareedna kharidega خریدنا",
    "waiting": "waiting intezar", "reply": "reply jawab جواب",
}
CANON = {_norm_urdu(w): c for c, ws in CONCEPTS.items() for w in ws.split()}

# fillers that carry no topic information (after canonicalisation)
STOPWORDS = set("""a an the is are was be to of in on for at by with this that these it its i you your we our me my do does did can
could will would should have has had there any some or and but if so up much year iska iski ye yeh wo iss ka ki ke ko se pe par mein
main hai hain ho hogi hoga hoge kya kia kar karein dein na bhi toh hi aur ek yaar kon kab kaise kaisa kahan allah raha rahi
num qmark emojipos emojineg suit dress lawn kurta piece kameez ہے ہیں ہو ہوگی ہوگا ہوتی یہ اس کی کا کے کو سے پر میں بھی اور ان کیا
دیں ہوا ہوئے کیے گا گی ملے مل جائے ہم""".split())

_HOWMUCH = re.compile(r"\bhow (much|many|long)\b")


def canonical_text(text: object) -> str:
    """Normalised tokens with canonical concepts, stopwords KEPT (used by the lexicon)."""
    s = unicodedata.normalize("NFKC", str(text)).lower()
    s = _URL.sub(" ", s)
    s = _MENTION.sub(" ", s)
    s = "".join(" emojipos " if ch in _POS_EMOJI else " emojineg " if ch in _NEG_EMOJI else ch for ch in s)
    s = _EMOJI.sub(" ", s)
    s = _norm_urdu(s)
    s = re.sub(r"([^\W\d_])\1{2,}", r"\1", s)           # pleaseee -> please
    s = _HOWMUCH.sub(" howmuch ", s)
    s = s.replace("?", " qmark ").replace("؟", " qmark ").replace("%", " percent ")
    s = re.sub(r"\d+", " num ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(h|hy|hay)\b", " hai ", s)
    return " ".join(CANON.get(t, t) for t in s.split())


def clean_text(text: object) -> str:
    """Canonical text without stopwords: the input for clustering."""
    return " ".join(t for t in canonical_text(text).split() if t not in STOPWORDS)


# ---- rule-based lexicon over canonical tokens -----------------------------------
THEMES = ["Price", "Discount", "Fabric quality", "Sizing", "Delivery", "Availability", "Purchase intent", "Praise"]
_LEX = {
    "Price": r"\bprice|howmuch (for|ka)|\bexpensive|\bhigh\b",
    "Discount": r"discount|\bsale\b|offer|promo|\bcode\b|bulk|\bpercent\b",
    "Fabric quality": r"fabric|cotton|quality|\bgsm\b|\bwash\b|dhone|see through|patla",
    "Sizing": r"\bsize\b|\bxl\b|stitched|length|\bfit\b|chart",
    "Delivery": r"deliver|\bcod\b|parcel|tracking|ordered|received|waiting|\bmila\b|aaya|\blate\b|reply|سروس|order kiya|order placed|my order|mera order|نہیں ملا|نہیں پہنچا|ابھی تک",
    "Availability": r"available|in stock|\bstock\b|restock|outlet|\bstore\b|dobara",
    "Purchase intent": r"inbox|\bdm\b|interested|want to order|place an order|order karna|order kaise|\bwant\b|payment|how do i pay|i ll take|\bbuy\b|order this|order کرنا",
    "Praise": r"beautiful|mashaallah|masha allah|\blove\b|\bmasha\b",
}
_LEX = {k: re.compile(_norm_urdu(v)) for k, v in _LEX.items()}
_NEG = re.compile(_norm_urdu(r"\bhigh\b|\bexpensive|not received|nahi mila|nahi aaya|abhi tak|waiting|still no|\blate\b|worst|"
                  r"no update|nobody|reply nahi|justify|emojineg|نہیں ملا|نہیں پہنچا|بری|nahi mil|intezar|reply نہیں"))
_DELIVERY_COMPLAINT = re.compile(_norm_urdu(r"ordered|not received|waiting|nahi mila|nahi aaya|abhi tak|still no|\blate\b|worst|"
                                            r"no update|nobody|koi reply|mera order|order kiya|ابھی تک|نہیں ملا|نہیں پہنچا|بری|reply نہیں|nahi mil"))


def rule_theme(text: str, already_canonical: bool = False) -> str:
    """Best-matching marketing theme for a comment ('Other' if nothing matches)."""
    c = text if already_canonical else canonical_text(text)
    hits = {t: len(rx.findall(c)) for t, rx in _LEX.items()}
    if hits["Delivery"] and _DELIVERY_COMPLAINT.search(c):
        hits["Delivery"] += 2        # "ordered ... still waiting" is a delivery issue, not a purchase
    best = max(hits.values())
    if best == 0:
        return "Praise" if "emojipos" in c else "Other"
    return next(t for t in THEMES if hits[t] == best)


def rule_negative(text: str, already_canonical: bool = False) -> bool:
    return bool(_NEG.search(text if already_canonical else canonical_text(text)))


# ---- city mentions (for "is this available in Karachi?" style demand) -----------
CITY_ALIASES = {
    "Karachi": ["karachi", "کراچی"], "Lahore": ["lahore", "لاہور"], "Islamabad": ["islamabad", "اسلام آباد"],
    "Rawalpindi": ["rawalpindi", "راولپنڈی"], "Peshawar": ["peshawar", "پشاور"], "Multan": ["multan", "ملتان"],
    "Faisalabad": ["faisalabad", "فیصل آباد"], "Quetta": ["quetta", "کوئٹہ"], "Sialkot": ["sialkot", "سیالکوٹ"],
    "Hyderabad": ["hyderabad", "حیدرآباد"], "Gujranwala": ["gujranwala", "گوجرانوالہ"], "Sargodha": ["sargodha", "سرگودھا"],
}
_CITY_RX = {c: re.compile("|".join(re.escape(canonical_text(a)) for a in al)) for c, al in CITY_ALIASES.items()}


def find_city(canon: str) -> str | None:
    return next((c for c, rx in _CITY_RX.items() if rx.search(canon)), None)


def rule_themes(series) -> np.ndarray:
    return np.array([rule_theme(t) for t in series])
