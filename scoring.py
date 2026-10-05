"""Story scoring for the daily digest.

Pure Python. No network, no LLM. The LLM only supplies substance ratings;
coverage, PR signals, weights, penalties and labels are all computed here so
they are transparent and tunable.

Pipeline:
  1. cluster()          group items that are about the same story
  2. features()         count independent outlets, detect PR signals, recency
  3. shortlist()        pick the clusters worth sending to the LLM for rating
  4. score()            combine local features + LLM ratings into a final score
"""

import math
import re
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlparse

# ---------------------------------------------------------------- tunables
CLUSTER_THRESHOLD = 0.28        # cosine similarity that alone joins two items
LOOSE_THRESHOLD = 0.15          # lower cosine floor when titles share named entities
MIN_SHARED_RARE = 2             # distinctive words two items must share
MIN_SHARED_TITLE_RARE = 3       # distinctive TITLE words that justify the looser rule
                                # (one shared versioned name like gpt-6 or llama-4 is enough)
COVERAGE_TABLE = [0.0, 0.35, 0.60, 0.80, 0.92, 1.0]   # by # of independent outlets
CONFIRMED_BONUS = 0.10          # company/regulator source AND independent coverage
ATTENTION_BONUS = 0.10          # also trending on an aggregator (Hacker News)
OFFICIAL_COVERAGE_FLOOR = 0.50  # a regulator/standards body is its own source of truth
WEIGHTS = {"coverage": 0.30, "impact": 0.25, "structural": 0.20,
           "substance": 0.15, "relevance": 0.10}
BREADTH_BONUS = 0.15            # impact >= 2 for both everyday people and the C-suite
MAJOR_MIN = 0.65
NOTABLE_MIN = 0.50
WATCH_MIN = 0.35
SHORTLIST_K = 12
LOOKBACK_HOURS = 36

WIRE_DOMAINS = {
    "prnewswire.com", "businesswire.com", "globenewswire.com", "einpresswire.com",
    "accesswire.com", "prweb.com", "newswire.com", "prlog.org", "send2press.com",
}

STOP = set("""
a an the and or but of to in on for with at by from as is are was were be been it its
this that these those new says say said report reports after over into about how why
what who will can could may more most than you your we our their his her they them not
no yes up out off just now also has have had get gets got via per vs
""".split())

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.\-+]*[a-z0-9]|[a-z0-9]")

BOILERPLATE_RE = re.compile(
    r"\b(today announced|is pleased to|are pleased to|proud to|thrilled to|"
    r"excited to (announce|share|introduce)|is proud|press release|sponsored|"
    r"partner content|for immediate release|named a leader|recognized as)\b", re.I)
TITLE_VERB_RE = re.compile(
    r"\b(announces?|unveils?|launch(es|ed)?|introduc(es|ing)|partners? with|"
    r"teams? up with|names|named|appoints?|achieves?|awarded|expands?|"
    r"now available|rolls? out)\b", re.I)


# ---------------------------------------------------------------- helpers
def outlet_of(url):
    """Registered-domain style outlet key: techcrunch.com, theverge.com, ..."""
    host = (urlparse(url).netloc or "").lower().split(":")[0]
    if host.startswith("www."):
        host = host[4:]
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def is_wire(link):
    return outlet_of(link) in WIRE_DOMAINS


def _tokens(text):
    out = []
    for t in TOKEN_RE.findall((text or "").lower()):
        t = t.strip(".-+")
        if len(t) < 2 or t in STOP:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


def _cos(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(t, 0.0) for t, x in a.items())


def _parse_dt(s):
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def pr_signal(item):
    """0..1 promotional-language signal for one item."""
    if is_wire(item["link"]):
        return 1.0
    text = f"{item['title']} {item['summary']}"
    if BOILERPLATE_RE.search(text):
        return 0.7
    if TITLE_VERB_RE.search(item["title"]):
        return 0.3
    return 0.0


# ---------------------------------------------------------------- clustering
def cluster(items, threshold=CLUSTER_THRESHOLD):
    """Greedy grouping of items about the same story. Returns list of index lists."""
    n = len(items)
    docs = [_tokens(it["title"]) * 3 + _tokens(it["summary"]) for it in items]
    df = Counter()
    for d in docs:
        df.update(set(d))
    vecs = []
    for d in docs:
        tf = Counter(d)
        v = {t: (1 + math.log(c)) * (math.log((n + 1) / (df[t] + 1)) + 1)
             for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({t: x / norm for t, x in v.items()})
    cap = max(8, int(0.15 * n))                       # "rare" = distinctive word
    rare = [{t for t in set(d) if df[t] <= cap} for d in docs]
    title_rare = [{t for t in set(_tokens(it["title"])) if df[t] <= cap} for it in items]

    groups = []
    for i in range(n):
        best, best_sim = None, 0.0
        for gi, g in enumerate(groups):
            sim = 0.0
            for j in g:
                if len(rare[i] & rare[j]) < MIN_SHARED_RARE:
                    continue
                c = _cos(vecs[i], vecs[j])
                # Same story, different wording: titles share distinctive names and the
                # texts are not wholly unrelated. Kept strict on purpose: wrongly merging two
                # stories overstates coverage, while wrongly splitting only understates it.
                if c < threshold and c >= LOOSE_THRESHOLD:
                    shared = title_rare[i] & title_rare[j]
                    versioned = any(any(ch.isdigit() for ch in t) and any(ch.isalpha() for ch in t)
                                    for t in shared)
                    if len(shared) >= MIN_SHARED_TITLE_RARE or versioned:
                        c = threshold
                sim = max(sim, c)
            if sim > best_sim:
                best, best_sim = gi, sim
        if best is not None and best_sim >= threshold:
            groups[best].append(i)
        else:
            groups.append([i])
    return groups


# ---------------------------------------------------------------- features
def features(items, idxs, now=None):
    now = now or datetime.now(timezone.utc)
    m = [items[i] for i in idxs]
    news = {x["outlet"] for x in m if x["type"] == "news" and not is_wire(x["link"])}
    primary = {x["outlet"] for x in m if x["type"] == "primary"}
    official = {x["outlet"] for x in m if x["type"] == "official"}
    hn = any(x["type"] == "aggregator" for x in m)
    wire = any(is_wire(x["link"]) for x in m)
    sigs = [pr_signal(x) for x in m if x["type"] != "aggregator"]
    pr = sum(sigs) / len(sigs) if sigs else 0.0

    n = len(news)
    confirmed = bool(n and (primary or official))
    cov = COVERAGE_TABLE[min(n, len(COVERAGE_TABLE) - 1)]
    cov += CONFIRMED_BONUS if confirmed else 0.0
    cov += ATTENTION_BONUS if hn else 0.0
    if official:
        cov = max(cov, OFFICIAL_COVERAGE_FLOOR)
    cov = min(1.0, cov)

    stamps = [d for d in (_parse_dt(x["published"]) for x in m) if d]
    age = (now - max(stamps)).total_seconds() / 3600 if stamps else 18.0

    return {
        "idxs": list(idxs),
        "n_news": n,
        "news_outlets": sorted(news),
        "primary_outlets": sorted(primary),
        "official": bool(official),
        "hn": hn,
        "wire": wire,
        "pr_signal": pr,
        "coverage": cov,
        "confirmed": confirmed,
        "primary_only": bool(primary) and not news and not official,
        "age_hours": max(0.0, age),
    }


def prescore(f):
    s = f["coverage"] * (1 - 0.5 * f["pr_signal"])
    s += 0.05 * max(0.0, 1 - f["age_hours"] / LOOKBACK_HOURS)
    if f["wire"] and f["n_news"] == 0:
        s *= 0.3
    return s


def shortlist(feats, k=SHORTLIST_K):
    order = sorted(range(len(feats)), key=lambda i: prescore(feats[i]), reverse=True)
    return order[:k]


# ---------------------------------------------------------------- link choice
def pick_link_item(items, idxs):
    """Best item to link in the post: independent news first, then regulator, then vendor."""
    news = [i for i in idxs if items[i]["type"] == "news" and not is_wire(items[i]["link"])]
    if news:
        return max(news, key=lambda i: len(items[i]["summary"]))
    for typ in ("official", "primary"):
        c = [i for i in idxs if items[i]["type"] == typ]
        if c:
            return c[0]
    return idxs[0]


def pick_source_items(items, idxs):
    """Up to two items to read for the post: the link item plus the source of truth."""
    first = pick_link_item(items, idxs)
    out = [first]
    for typ in ("official", "primary", "news"):
        for i in idxs:
            if i != first and items[i]["type"] == typ and (
                    typ != "news" or items[i]["outlet"] != items[first]["outlet"]):
                out.append(i)
                return out
    return out


# ---------------------------------------------------------------- scoring
def _clamp(v, lo=0, hi=3):
    try:
        return max(lo, min(hi, int(v)))
    except Exception:
        return lo


def score(f, r):
    """Combine local features f with LLM ratings r into a transparent final score."""
    r = r or {}
    cls = str(r.get("classification", "other")).lower()
    amplified = bool(r.get("pr_amplified", False))
    ev, en = _clamp(r.get("everyday_impact")), _clamp(r.get("enterprise_impact"))
    sub_raw = _clamp(r.get("independent_substance"))
    if f["official"]:
        sub_raw = max(sub_raw, 2)

    impact = max(ev, en) / 3 + (BREADTH_BONUS if min(ev, en) >= 2 else 0.0)
    comp = {
        "coverage": f["coverage"],
        "impact": min(1.0, impact),
        "structural": _clamp(r.get("structural")) / 3,
        "substance": sub_raw / 3,
        "relevance": _clamp(r.get("albert_relevance")) / 3,
    }
    raw = sum(WEIGHTS[k] * comp[k] for k in WEIGHTS)

    mult, notes = 1.0, []
    if cls == "pr" or amplified:
        mult *= 0.35
        notes.append("PR")
    if cls == "rumor":
        mult *= 0.6
        notes.append("unconfirmed")
    if cls == "opinion_analysis":
        mult *= 0.8
        notes.append("commentary")
    if f["primary_only"]:
        mult *= 0.6
        notes.append("company source only, no independent coverage")
    if f["pr_signal"] >= 0.7 and sub_raw <= 1:
        mult *= 0.6
        notes.append("promotional language, thin substance")
    final = min(1.0, raw * mult)

    n = f["n_news"]
    if cls == "pr" or amplified or cls == "rumor":
        lab = "filtered"
    elif final >= MAJOR_MIN and (n >= 3 or f["official"]):
        lab = "major"
    elif final >= NOTABLE_MIN and (n >= 2 or f["official"]
                                   or (n >= 1 and (f["primary_outlets"] or sub_raw >= 3))):
        lab = "notable"
    elif final >= WATCH_MIN:
        lab = "watch"
    else:
        lab = "skip"

    return {"final": final, "label": lab, "classification": cls,
            "components": comp, "multiplier": mult, "notes": notes}
