"""Daily AI/SaaS LinkedIn digest agent.

Flow:
  feeds -> cluster items into stories -> count independent outlets (local code)
  -> LLM rates substance and flags PR on the top clusters
  -> final score computed in scoring.py -> pick story -> write post -> fact-check -> email
"""

import html
import json
import os
import re
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from time import mktime
from zoneinfo import ZoneInfo

import feedparser
import requests
import trafilatura
from anthropic import Anthropic

import scoring

ROOT = Path(__file__).parent
MODEL = os.getenv("MODEL", "claude-sonnet-5-5")
LA = ZoneInfo("America/Los_Angeles")
SEND_HOUR = 8
LOOKBACK_HOURS = 36
MAX_PER_FEED = 15
MAX_ITEMS = 200
SEEN_DAYS = 10
LOG_DAYS = 30
FEED_TYPES = {"news", "primary", "official", "aggregator"}
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; DailyDigestBot/1.0)"}

client = Anthropic()


# ---------- helpers ----------

def read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def strip_html(text):
    text = re.sub(r"<[^>]+>", " ", text or "")
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def ask(system, user, max_tokens=2500):
    r = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in r.content if b.type == "text").strip()


def ask_json(system, user, max_tokens=2500):
    last = None
    for _ in range(2):
        raw = ask(system, user, max_tokens)
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            return json.loads(m.group(0))
        except Exception as e:  # retry once
            last = e
    raise RuntimeError(f"Model did not return valid JSON: {last}")


# ---------- state ----------

def _load_json(rel):
    try:
        return json.loads((ROOT / rel).read_text())
    except Exception:
        return {}


def _save_json(rel, data):
    p = ROOT / rel
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True))


def load_seen():
    return _load_json("state/seen.json")


def save_seen(seen, today):
    cutoff = (today - timedelta(days=SEEN_DAYS)).isoformat()
    _save_json("state/seen.json", {u: d for u, d in seen.items() if d >= cutoff})


def save_log(entries, today):
    """Scoring history so thresholds can be calibrated against real days."""
    log = _load_json("state/scoring_log.json")
    log[today.isoformat()] = entries
    cutoff = (today - timedelta(days=LOG_DAYS)).isoformat()
    _save_json("state/scoring_log.json", {d: v for d, v in log.items() if d >= cutoff})


# ---------- RSS ----------

def load_feeds():
    """feeds.txt lines: URL  or  URL | type   (type: news, primary, official, aggregator)."""
    feeds = []
    for line in read("feeds.txt").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        url, _, typ = line.partition(" | ")
        typ = typ.strip().lower() or "news"
        feeds.append((url.strip(), typ if typ in FEED_TYPES else "news"))
    return feeds


def collect_items(feeds):
    """Pull recent items. Cross-outlet duplicates are KEPT: they are the coverage signal."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    items, failed, seen_keys = [], [], set()

    for url, typ in feeds:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            if not feed.entries:
                raise ValueError("no entries")
        except Exception as e:
            failed.append(f"{url} ({type(e).__name__})")
            continue

        outlet = scoring.outlet_of(url)
        source = strip_html(feed.feed.get("title", outlet))[:60]
        count = 0
        for e in feed.entries:
            link = e.get("link")
            title = strip_html(e.get("title", ""))
            if not link or not title or (link, typ) in seen_keys:
                continue
            ts = e.get("published_parsed") or e.get("updated_parsed")
            published = datetime.fromtimestamp(mktime(ts), timezone.utc) if ts else None
            if published and published < cutoff:
                continue
            seen_keys.add((link, typ))
            items.append({
                "title": title,
                "link": link,
                "outlet": outlet,
                "source": source,
                "type": typ,
                "summary": strip_html(e.get("summary", ""))[:400],
                "published": published.isoformat() if published else "unknown",
            })
            count += 1
            if count >= MAX_PER_FEED:
                break

    items.sort(key=lambda i: i["published"] if i["published"] != "unknown" else "0", reverse=True)
    return items[:MAX_ITEMS], failed


def fetch_article(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return (trafilatura.extract(r.text) or "")[:7000]
    except Exception:
        return ""


# ---------- LLM step 1: rate substance and flag PR ----------

RATE_SYSTEM = """You are a news-desk analyst separating legitimate industry news from public relations. Each cluster below is ONE story reported by one or more sources. Local code has already counted coverage. Do not comment on coverage counts. Rate substance only, from the text provided. If the text is thin, rate conservatively.

classification (pick one):
- industry_news: an independently reported development with consequences for companies, workers, or consumers.
- pr: content whose main purpose is to promote a company, product, event, award, partnership, or funding round, with claims coming only from the company. This includes vendor blog posts restating a launch, "named a leader" announcements, customer-win stories, webinar promotions, and syndicated press releases. Many outlets repeating one press release is STILL pr: set pr_amplified true.
- opinion_analysis: commentary or explainers with no new facts.
- research: a paper or study with findings.
- rumor: unsourced or single anonymous-source claims, speculation.
- other

Ratings, integers 0 to 3:
- everyday_impact: 0 none. 1 niche or early adopters. 2 many workers or consumers affected within a year. 3 changes jobs, costs, safety, or daily life for a large population.
- enterprise_impact: 0 none. 1 one tool or function. 2 a department's budget, tooling, or vendor choice. 3 C-suite or board level: strategy, risk, M&A, regulation, pricing power.
- structural: 0 incremental. 1 notable feature. 2 shifts competitive position or buying criteria. 3 changes how software is built, bought, priced, or regulated.
- independent_substance: 0 restates company claims. 1 minor added context. 2 third-party data, customers, analysts, or regulators. 3 original reporting, critical analysis, or primary documents such as filings and rules.
- albert_relevance: how much standing Albert (see profile) has to comment: 0 none, 3 core to his work.

audience: everyday, enterprise, both, or narrow.

Return ONLY JSON:
{"clusters":[{"id":int,"classification":"...","pr_amplified":true or false,"audience":"...","everyday_impact":int,"enterprise_impact":int,"structural":int,"independent_substance":int,"albert_relevance":int,"reason":"one sentence on why this is or is not real news","pr_evidence":"short phrase or empty"}]}
Include every cluster id you were given."""


def rate_clusters(profile, items, groups, feats, order):
    blocks = []
    for cid, ci in enumerate(order):
        f = feats[ci]
        head = (
            f"[C{cid}] independent news outlets: {f['n_news']} ({', '.join(f['news_outlets']) or 'none'}); "
            f"company source: {'yes' if f['primary_outlets'] else 'no'}; "
            f"regulator/official source: {'yes' if f['official'] else 'no'}; "
            f"on Hacker News: {'yes' if f['hn'] else 'no'}; "
            f"local PR-language score: {f['pr_signal']:.2f}; "
            f"wire-service link: {'yes' if f['wire'] else 'no'}"
        )
        lines = [
            f"  - ({items[i]['outlet']}, {items[i]['type']}) {items[i]['title']}: {items[i]['summary'][:250]}"
            for i in groups[ci][:6]
        ]
        blocks.append(head + "\n" + "\n".join(lines))
    data = ask_json(RATE_SYSTEM, f"# PROFILE\n{profile}\n\n# CLUSTERS\n" + "\n\n".join(blocks), max_tokens=3500)
    ratings = {}
    for r in data.get("clusters", []):
        cid = r.get("id")
        if isinstance(cid, int) and 0 <= cid < len(order):
            ratings[cid] = r
    return ratings


# ---------- LLM step 2: angles ----------

ANGLE_SYSTEM = """You help Albert Qian choose the angle for a LinkedIn post. For each candidate story, propose the specific opinion the post should take, following the POINT OF VIEW. One or two sentences each. Take a position. Do not force the human-in-the-loop theme onto stories it does not fit.
Return ONLY JSON: {"angles":[{"id":int,"angle":"..."}]}"""


def propose_angles(profile, pov, items, picks):
    blocks = []
    for k, s in enumerate(picks):
        it = items[s["link_idx"]]
        blocks.append(f"[{k}] {it['title']}: {it['summary'][:300]}\nWhy it is news: {s['r'].get('reason', '')}")
    try:
        data = ask_json(ANGLE_SYSTEM, f"# PROFILE\n{profile}\n\n# POINT OF VIEW\n{pov}\n\n# CANDIDATES\n" + "\n\n".join(blocks), max_tokens=900)
        return {a["id"]: a["angle"] for a in data.get("angles", []) if isinstance(a.get("id"), int)}
    except Exception:
        return {}


# ---------- LLM step 3: write and check ----------

WRITE_RULES = """You write LinkedIn posts for Albert Qian that he will paste directly, with no editing. Match his voice exactly using the style guide and samples below.

Hard rules:
- 120 to 200 words. Line 1 is the hook, under 140 characters.
- Short paragraphs separated by a blank line. No headers, bullets, bold, hashtags, or emojis.
- State a real position: what changes, who is affected, what Albert expects next. Express the stance in the POINT OF VIEW section: pro-AI, deliberate about where it decides, balanced but decisive. Never end on "it depends." Do not force the human-in-the-loop theme onto stories it does not fit.
- Facts about the news come ONLY from the article text provided. If only a summary is available, do not assert details beyond it.
- Any first-person claim about Albert's experience comes ONLY from the profile's verifiable facts. Never invent stats, quotes, customers, or anecdotes.
- Never reuse quotes, names, statistics, or product claims that appear in the writing samples. They show voice only.
- Albert speaks as an individual, not for his employer. No confidential information, no disparaging named companies.
- Last line is the article link, alone.
- Output the post text only. No preamble, no quotes around it."""


def write_post(profile, style, pov, samples, pick_item, basis, angle):
    system = f"{WRITE_RULES}\n\n# POINT OF VIEW\n{pov}\n\n# STYLE GUIDE\n{style}\n\n# WRITING SAMPLES\n{samples}"
    user = (
        f"# PROFILE\n{profile}\n\n# ARTICLE\nLink: {pick_item['link']}\n\n{basis}\n\n"
        f"# ANGLE TO TAKE\n{angle}"
    )
    return ask(system, user, max_tokens=1200)


CHECK_SYSTEM = """You are a strict fact-checker and editor for a LinkedIn post. Check:
1. Every claim about the news is supported by the article text.
2. Every first-person claim about the author is supported by the profile's verifiable facts.
3. Banned phrasing is absent: "game-changer", "landscape", "delve", "in today's", "unlock", "revolutionize", "excited to share", "thrilled"; the "It's not X, it's Y" construction; more than one em dash; hashtags; emojis.
4. Length is 120 to 200 words and the hook is under 140 characters.
5. Stance matches the POINT OF VIEW: takes a clear position, balanced, no hype, no doom, no "time will tell," and does not force a human-in-the-loop angle where it does not fit.
Fix every problem with the smallest edit that preserves voice. Do not add new facts.
Return ONLY JSON: {"post": "final post text, ending with the article link alone on the last line", "issues": ["each problem found and fixed, or empty list"]}"""


def check_post(profile, pov, basis, post, link):
    data = ask_json(
        CHECK_SYSTEM,
        f"# PROFILE\n{profile}\n\n# POINT OF VIEW\n{pov}\n\n# ARTICLE\n{basis}\n\n# LINK\n{link}\n\n# POST\n{post}",
    )
    final = (data.get("post") or post).strip()
    if link not in final:
        final += f"\n\n{link}"
    return final, data.get("issues", [])


# ---------- scoring glue ----------

def build_basis(items, idxs):
    """Article text from the link item plus the source of truth; fall back to RSS summaries."""
    parts = []
    for i in scoring.pick_source_items(items, idxs):
        t = fetch_article(items[i]["link"])
        if len(t) > 400:
            parts.append(f"SOURCE: {items[i]['source']} ({items[i]['type']})\n{t[:3500]}")
    if parts:
        return "\n\n".join(parts)
    sums = "\n".join(f"- {items[i]['source']}: {items[i]['summary']}" for i in idxs[:3])
    return f"(RSS summaries only. Do not assert details beyond these.)\n{sums}"


def score_all(profile, items, groups, feats):
    order = scoring.shortlist(feats)
    ratings = rate_clusters(profile, items, groups, feats, order)
    scored = []
    for cid, ci in enumerate(order):
        r = ratings.get(cid, {})
        sc = scoring.score(feats[ci], r)
        li = scoring.pick_link_item(items, groups[ci])
        scored.append({"ci": ci, "idxs": groups[ci], "f": feats[ci], "r": r, "sc": sc,
                       "link_idx": li, "title": items[li]["title"]})
    scored.sort(key=lambda s: s["sc"]["final"], reverse=True)
    return scored


# ---------- email ----------

TIER = {"major": "MAJOR", "notable": "NOTABLE"}


def _meta(s):
    f = s["f"]
    bits = [f"{f['n_news']} independent outlet{'s' if f['n_news'] != 1 else ''}"]
    bits.append("company source" if f["primary_outlets"] else "no company source")
    if f["official"]:
        bits.append("regulator source")
    if f["hn"]:
        bits.append("trending on HN")
    bits.append(f"audience: {s['r'].get('audience', 'n/a')}")
    bits.append(f"score {s['sc']['final']:.2f}")
    return " · ".join(bits)


def build_email(now, items, snapshot, watch, filtered, post, issues, failed, major, pick, angles, runners):
    e = html.escape

    def story(s, tier=True):
        it = items[s["link_idx"]]
        tag = f'<b>{TIER.get(s["sc"]["label"], "")}</b> · ' if tier and s["sc"]["label"] in TIER else ""
        notes = f' <span style="color:#a60">({e("; ".join(s["sc"]["notes"]))})</span>' if s["sc"]["notes"] else ""
        return (
            f'<li style="margin-bottom:12px">{tag}<a href="{e(it["link"])}">{e(it["title"])}</a>{notes}<br>'
            f'<span style="color:#666;font-size:13px">{e(_meta(s))}</span><br>{e(s["r"].get("reason", ""))}</li>'
        )

    snap = "".join(story(s) for s in snapshot) or "<li>Nothing cleared the newsworthiness bar today.</li>"
    wl = "".join(story(s, tier=False) for s in watch)
    flt = "".join(story(s, tier=False) for s in filtered)
    alts = "".join(
        f'<li style="margin-bottom:8px">{e(angles.get(k + 1, ""))} '
        f'<a href="{e(items[s["link_idx"]]["link"])}">{e(items[s["link_idx"]]["title"])}</a></li>'
        for k, s in enumerate(runners)
    )
    flag = "" if major else (
        '<p style="background:#fff3cd;padding:8px;border-radius:4px"><b>Slow news day.</b> '
        "No story met the MAJOR bar (broad independent coverage plus real impact). "
        "Consider skipping the post.</p>"
    )
    notes = ""
    if issues:
        notes += "<p style='color:#666;font-size:13px'><b>Fact-check edits:</b> " + e("; ".join(issues)) + "</p>"
    if failed:
        notes += "<p style='color:#a00;font-size:13px'><b>Feeds failed:</b> " + e("; ".join(failed)) + "</p>"

    body = f"""<div style="font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:640px;margin:auto;color:#111;line-height:1.5">
<h2 style="margin-bottom:4px">AI &amp; SaaS snapshot</h2>
<div style="color:#666;margin-bottom:16px">{e(now.strftime('%A, %B %d, %Y'))}</div>
{flag}
<h3>Verified news, ranked</h3><ol style="padding-left:20px">{snap}</ol>
<h3>LinkedIn post (copy and paste)</h3>
<div style="white-space:pre-wrap;border:1px solid #ccc;border-radius:6px;padding:14px;background:#fafafa">{e(post)}</div>
<p style="color:#666;font-size:13px"><b>Why this story:</b> {e(pick["r"].get("reason", ""))} ({e(_meta(pick))})</p>
{"<h3>Alternate angles</h3><ul style='padding-left:20px'>" + alts + "</ul>" if alts else ""}
{"<h3>Watch list</h3><ul style='padding-left:20px'>" + wl + "</ul>" if wl else ""}
{"<h3>Filtered out (PR or unconfirmed)</h3><ul style='padding-left:20px;color:#555'>" + flt + "</ul>" if flt else ""}
{notes}</div>"""
    text = f"AI & SaaS snapshot, {now.strftime('%A, %B %d, %Y')}\n\nLINKEDIN POST\n\n{post}\n"
    return body, text


def send_email(subject, body_html, body_text):
    addr = os.environ["GMAIL_ADDRESS"]  # login account that owns the app password
    pw = os.environ["GMAIL_APP_PASSWORD"]
    sender = os.getenv("EMAIL_FROM") or addr  # empty or unset falls back to the login account
    to = os.getenv("EMAIL_TO") or addr
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, sender, to
    msg.attach(MIMEText(body_text, "plain", "utf-8"))
    msg.attach(MIMEText(body_html, "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(addr, pw)
        s.sendmail(addr, [to], msg.as_string())


# ---------- main ----------

def main():
    now = datetime.now(LA)
    # GitHub cron runs in UTC with no DST. Two crons fire; only the one landing at 8am LA proceeds.
    if os.getenv("GITHUB_EVENT_NAME") == "schedule" and now.hour != SEND_HOUR:
        print(f"Skipping: LA hour is {now.hour}, not {SEND_HOUR}.")
        return

    profile = read("profile.md")
    style = read("style_guide.md")
    pov = read("point_of_view.md")
    samples = "\n\n---\n\n".join(
        p.read_text(encoding="utf-8") for p in sorted((ROOT / "samples").glob("*.md"))
    )

    seen = load_seen()
    items, failed = collect_items(load_feeds())
    if not items:
        body = "<p>No new items found in the last 36 hours.</p>"
        if failed:
            body += f"<p><b>Feeds failed:</b> {html.escape('; '.join(failed))}</p>"
        send_email("LinkedIn digest: no new items", body, "No new items found.")
        return

    # 1. cluster, drop stories already featured, compute local features
    groups = scoring.cluster(items)
    groups = [g for g in groups if not any(items[i]["link"] in seen for i in g)]
    now_utc = datetime.now(timezone.utc)
    feats = [scoring.features(items, g, now_utc) for g in groups]

    # 2. rate + score
    scored = score_all(profile, items, groups, feats)
    snapshot = [s for s in scored if s["sc"]["label"] in ("major", "notable")][:5]
    watch = [s for s in scored if s["sc"]["label"] == "watch"][:3]
    filtered = [s for s in scored if s["sc"]["label"] == "filtered"][:3]
    candidates = [s for s in scored if s["sc"]["label"] != "filtered"]

    # 3. calibration log (always written)
    entries = [{
        "title": s["title"], "n_news": s["f"]["n_news"], "company_source": bool(s["f"]["primary_outlets"]),
        "official": s["f"]["official"], "hn": s["f"]["hn"], "class": s["sc"]["classification"],
        "audience": s["r"].get("audience", ""), "final": round(s["sc"]["final"], 3), "label": s["sc"]["label"],
    } for s in scored[:12]]
    save_log(entries, now.date())

    if not candidates:
        body = ("<p><b>No real news today.</b> Every shortlisted story was PR or unconfirmed. "
                "No post drafted.</p>")
        if filtered:
            body += "<ul>" + "".join(
                f"<li>{html.escape(s['title'])} ({html.escape('; '.join(s['sc']['notes']))})</li>" for s in filtered
            ) + "</ul>"
        if failed:
            body += f"<p><b>Feeds failed:</b> {html.escape('; '.join(failed))}</p>"
        send_email("LinkedIn digest: no qualifying news today", body, "No qualifying news today.")
        return

    # 4. choose story + angle
    pool = snapshot or candidates[:3]
    pick, runners = pool[0], pool[1:3]
    angles = propose_angles(profile, pov, items, [pick] + runners)
    angle = angles.get(0) or pick["r"].get("reason", "")

    # 5. write, check, send
    pick_item = items[pick["link_idx"]]
    basis = build_basis(items, pick["idxs"])
    post = write_post(profile, style, pov, samples, pick_item, basis, angle)
    post, issues = check_post(profile, pov, basis, post, pick_item["link"])

    major = any(s["sc"]["label"] == "major" for s in snapshot)
    body_html, body_text = build_email(
        now, items, snapshot, watch, filtered, post, issues, failed, major, pick, angles, runners)
    subject = f"LinkedIn digest {now.strftime('%b %d')}: {pick_item['title'][:70]}"
    send_email(subject, body_html, body_text)

    # 6. remember what was featured (whole clusters, so follow-up coverage is not re-featured)
    today = now.date().isoformat()
    for s in snapshot + [pick]:
        for i in s["idxs"]:
            seen[items[i]["link"]] = today
    save_seen(seen, now.date())
    print("Sent:", subject)


if __name__ == "__main__":
    main()
