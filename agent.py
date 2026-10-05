"""Daily AI/SaaS LinkedIn digest agent.

Flow: read profile + style -> pull RSS -> pick major story (LLM) ->
fetch article -> write post in Albert's voice (LLM) -> fact-check (LLM) -> email.
"""

import html
import json
import os
import re
import smtplib
import sys
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

ROOT = Path(__file__).parent
MODEL = os.getenv("MODEL", "claude-sonnet-5-5")
LA = ZoneInfo("America/Los_Angeles")
SEND_HOUR = 8
LOOKBACK_HOURS = 36
MAX_PER_FEED = 8
MAX_ITEMS = 60
SEEN_DAYS = 10
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

def load_seen():
    p = ROOT / "state" / "seen.json"
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def save_seen(seen, today):
    cutoff = (today - timedelta(days=SEEN_DAYS)).isoformat()
    seen = {u: d for u, d in seen.items() if d >= cutoff}
    p = ROOT / "state" / "seen.json"
    p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(seen, indent=2, sort_keys=True))


# ---------- RSS ----------

def collect_items(seen):
    feeds = [
        l.strip() for l in read("feeds.txt").splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    items, failed, seen_titles = [], [], set()

    for url in feeds:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            if not feed.entries:
                raise ValueError("no entries")
        except Exception as e:
            failed.append(f"{url} ({type(e).__name__})")
            continue

        source = strip_html(feed.feed.get("title", url))[:60]
        count = 0
        for e in feed.entries:
            link = e.get("link")
            title = strip_html(e.get("title", ""))
            if not link or not title or link in seen:
                continue
            ts = e.get("published_parsed") or e.get("updated_parsed")
            published = (
                datetime.fromtimestamp(mktime(ts), timezone.utc) if ts else None
            )
            if published and published < cutoff:
                continue
            key = re.sub(r"\W+", "", title.lower())
            if key in seen_titles:
                continue
            seen_titles.add(key)
            items.append({
                "title": title,
                "link": link,
                "source": source,
                "summary": strip_html(e.get("summary", ""))[:400],
                "published": published.isoformat() if published else "unknown",
            })
            count += 1
            if count >= MAX_PER_FEED:
                break

    items.sort(key=lambda i: i["published"], reverse=True)
    return items[:MAX_ITEMS], failed


def fetch_article(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return (trafilatura.extract(r.text) or "")[:7000]
    except Exception:
        return ""


# ---------- LLM steps ----------

SELECT_SYSTEM = """You are the editorial analyst for Albert Qian, a SaaS and AI product marketer. From today's feed items, decide what actually matters in AI and SaaS, then choose the single story Albert is best positioned to comment on.

Major news means it changes how software is built, bought, priced, regulated, or distributed, or shifts competitive position among platform vendors. Not major: routine funding rounds, minor feature releases, opinion pieces, listicles, unsourced rumors, rehashes of older news.

Pick rules: (1) major or close to it, (2) Albert has credible standing to take a position given his profile, (3) a non-obvious opinion is possible, not just "this is exciting," (4) Albert can apply his point of view (where AI should and should not decide). Lightly prefer stories where that lens adds something, but never skip a genuinely major story because the lens does not fit. Prefer primary sources over commentary about them.

Return ONLY JSON in this exact shape:
{
  "major_news_today": true or false,
  "snapshot": [{"index": int, "why_it_matters": "one sentence"}],
  "pick": {"index": int, "reason": "why this story and why Albert", "angle": "the specific opinion the post should take"},
  "runner_ups": [{"index": int, "angle": "one-sentence alternate angle"}]
}
snapshot: exactly 5 items, most important first. runner_ups: exactly 2. If nothing is major, set major_news_today false and still pick the best candidate."""


def select_story(profile, pov, items):
    listing = "\n".join(
        f"[{i}] ({it['source']}, {it['published'][:16]}) {it['title']}: {it['summary']}"
        for i, it in enumerate(items)
    )
    data = ask_json(SELECT_SYSTEM, f"# PROFILE\n{profile}\n\n# POINT OF VIEW\n{pov}\n\n# TODAY'S ITEMS\n{listing}")
    n = len(items)
    ok = lambda i: isinstance(i, int) and 0 <= i < n
    data["snapshot"] = [s for s in data.get("snapshot", []) if ok(s.get("index"))][:5]
    data["runner_ups"] = [s for s in data.get("runner_ups", []) if ok(s.get("index"))][:2]
    if not ok(data.get("pick", {}).get("index")):
        raise RuntimeError("Model returned an invalid pick index")
    return data


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


def write_post(profile, style, pov, samples, pick_item, article, angle):
    system = f"{WRITE_RULES}\n\n# POINT OF VIEW\n{pov}\n\n# STYLE GUIDE\n{style}\n\n# WRITING SAMPLES\n{samples}"
    basis = article if len(article) > 400 else f"(Full text unavailable. RSS summary only.) {pick_item['summary']}"
    user = (
        f"# PROFILE\n{profile}\n\n# ARTICLE\nSource: {pick_item['source']}\n"
        f"Title: {pick_item['title']}\nLink: {pick_item['link']}\n\n{basis}\n\n"
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


def check_post(profile, pov, article_basis, post, link):
    data = ask_json(
        CHECK_SYSTEM,
        f"# PROFILE\n{profile}\n\n# POINT OF VIEW\n{pov}\n\n# ARTICLE\n{article_basis}\n\n# LINK\n{link}\n\n# POST\n{post}",
    )
    final = (data.get("post") or post).strip()
    if link not in final:
        final += f"\n\n{link}"
    return final, data.get("issues", [])


# ---------- email ----------

def build_email(now, items, sel, post, issues, failed, major, pick_item):
    e = html.escape
    snap = "".join(
        f'<li style="margin-bottom:10px"><a href="{e(items[s["index"]]["link"])}">{e(items[s["index"]]["title"])}</a>'
        f' <span style="color:#666">({e(items[s["index"]]["source"])})</span><br>{e(s["why_it_matters"])}</li>'
        for s in sel["snapshot"]
    )
    alts = "".join(
        f'<li style="margin-bottom:8px">{e(r["angle"])} '
        f'<a href="{e(items[r["index"]]["link"])}">{e(items[r["index"]]["title"])}</a></li>'
        for r in sel["runner_ups"]
    )
    flag = "" if major else (
        '<p style="background:#fff3cd;padding:8px;border-radius:4px">'
        "<b>Slow news day.</b> Nothing cleared the major-news bar. "
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
<h3>Today's top 5</h3><ol style="padding-left:20px">{snap}</ol>
<h3>LinkedIn post (copy and paste)</h3>
<div style="white-space:pre-wrap;border:1px solid #ccc;border-radius:6px;padding:14px;background:#fafafa">{e(post)}</div>
<p style="color:#666;font-size:13px"><b>Why this story:</b> {e(sel["pick"].get("reason", ""))}</p>
<h3>Alternate angles</h3><ul style="padding-left:20px">{alts}</ul>
{notes}</div>"""
    text = f"AI & SaaS snapshot, {now.strftime('%A, %B %d, %Y')}\n\nLINKEDIN POST\n\n{post}\n"
    return body, text


def send_email(subject, body_html, body_text):
    addr = os.environ["GMAIL_ADDRESS"]
    pw = os.environ["GMAIL_APP_PASSWORD"]
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, addr, addr
    msg.attach(MIMEText(body_text, "plain", "utf-8"))
    msg.attach(MIMEText(body_html, "html", "utf-8"))
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(addr, pw)
        s.sendmail(addr, [addr], msg.as_string())


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
    items, failed = collect_items(seen)

    if not items:
        body = "<p>No new items found in the last 36 hours.</p>"
        if failed:
            body += f"<p><b>Feeds failed:</b> {html.escape('; '.join(failed))}</p>"
        send_email("LinkedIn digest: no new items", body, "No new items found.")
        return

    sel = select_story(profile, pov, items)
    pick_item = items[sel["pick"]["index"]]
    article = fetch_article(pick_item["link"])
    basis = article if len(article) > 400 else f"(RSS summary only) {pick_item['summary']}"

    post = write_post(profile, style, pov, samples, pick_item, article, sel["pick"].get("angle", ""))
    post, issues = check_post(profile, pov, basis, post, pick_item["link"])

    major = bool(sel.get("major_news_today", True))
    body_html, body_text = build_email(now, items, sel, post, issues, failed, major, pick_item)
    subject = f"LinkedIn digest {now.strftime('%b %d')}: {pick_item['title'][:70]}"
    send_email(subject, body_html, body_text)

    today = now.date().isoformat()
    for s in sel["snapshot"]:
        seen[items[s["index"]]["link"]] = today
    seen[pick_item["link"]] = today
    save_seen(seen, now.date())
    print("Sent:", subject)


if __name__ == "__main__":
    main()
