#!/usr/bin/env python3
"""Daily content digest — fetch new items, summarize with Groq, publish to GitHub Pages."""

import hashlib
import json
import os
import smtplib
import sys
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq

BASE_DIR = Path(__file__).parent
SOURCES_FILE = BASE_DIR / "sources.json"
STATE_FILE = BASE_DIR / "state.json"
INDEX_FILE = BASE_DIR / "index.html"
ARCHIVE_DIR = BASE_DIR / "archive"

load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO = os.getenv("EMAIL_TO")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
LOOKBACK_HOURS = int(os.getenv("LOOKBACK_HOURS", "26"))
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# -- State ----------------------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text())
        # Migrate old format (just a list of seen IDs)
        if isinstance(data, list):
            return {"seen": set(data), "last_items": {}, "summaries": {}}
        return {
            "seen": set(data.get("seen", [])),
            "last_items": data.get("last_items", {}),
            "summaries": data.get("summaries", {}),
        }
    return {"seen": set(), "last_items": {}, "summaries": {}}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps({
        "seen": list(state["seen"])[-15000:],
        "last_items": state["last_items"],
        "summaries": state["summaries"],
    }, indent=2))


# -- Fetching -------------------------------------------------------------------------------------

def _strip_html(html: str, max_chars: int = 2500) -> str:
    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)[:max_chars]


def _parse_pub_date(entry) -> tuple:
    """Returns (datetime_utc_or_None, formatted_string_or_None)."""
    pub = entry.get("published_parsed") or entry.get("updated_parsed")
    if pub:
        try:
            dt = datetime(*pub[:6], tzinfo=timezone.utc)
            return dt, dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    return None, None


def _clean_xml(raw: bytes) -> bytes:
    """Strip invalid XML 1.0 characters and fix unescaped ampersands."""
    import re
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        text = raw.decode("latin-1", errors="replace")
    # Remove invalid XML 1.0 control characters (keep tab=0x9, LF=0xA, CR=0xD)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    # Fix unescaped ampersands (not already part of a valid XML entity)
    text = re.sub(r"&(?![a-zA-Z#][a-zA-Z0-9#]*;)", "&amp;", text)
    return text.encode("utf-8")


def _bs4_parse_feed(raw: bytes, source_url: str) -> list:
    """
    Fallback: manually extract Atom/RSS entries using BeautifulSoup.
    Uses html.parser first (most lenient — ignores XML namespaces and bad chars),
    then lxml/lxml-xml as alternatives.
    Returns a list of raw item dicts: id, title, link, content, pub_dt, pub_date.
    """
    soup = None
    for parser in ("html.parser", "lxml", "lxml-xml"):
        try:
            candidate = BeautifulSoup(raw, parser)
            # Validate this parser actually found feed entries
            if candidate.find(["entry", "item"]):
                soup = candidate
                break
        except Exception:
            continue
    if not soup:
        return []

    entries = soup.find_all(["entry", "item"])
    items = []
    for entry in entries:
        # Use simple find with multiple candidate names
        def _first(*names):
            for n in names:
                t = entry.find(n)
                if t:
                    return t
            return None

        title_tag   = _first("title")
        link_tag    = _first("link")
        id_tag      = _first("id", "guid")
        pub_tag     = _first("updated", "published", "pubdate", "pubDate", "dc:date")
        content_tag = _first("content", "summary", "description")

        link = ""
        if link_tag:
            link = link_tag.get("href") or link_tag.get_text(strip=True)
        item_id = (id_tag.get_text(strip=True) if id_tag else "") or link

        pub_str_raw = pub_tag.get_text(strip=True) if pub_tag else ""
        pub_dt = None
        pub_date = None
        for fmt in (
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S GMT",
            "%Y-%m-%d",
        ):
            try:
                pub_dt = datetime.strptime(pub_str_raw[:25], fmt)
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                pub_date = pub_dt.strftime("%Y-%m-%d")
                break
            except ValueError:
                continue

        content = ""
        if content_tag:
            inner = content_tag.get_text(separator=" ", strip=True)
            content = _strip_html(inner) if "<" in inner else inner[:2500]

        if not item_id:
            continue

        items.append({
            "id": item_id,
            "title": title_tag.get_text(strip=True) if title_tag else "Untitled",
            "link": link or source_url,
            "content": content,
            "pub_dt": pub_dt,
            "pub_date": pub_date,
        })
    return items


def fetch_rss(source: dict, state: dict, cutoff: datetime) -> tuple:
    """
    Returns (new_items, fallback_item, error).
    - new_items: list of items published after cutoff (not yet seen)
    - fallback_item: the most recent item overall (for "last available" display)
    - error: string or None
    Three-level strategy:
      1. feedparser directly on the URL
      2. feedparser on cleaned raw bytes (strips invalid XML chars)
      3. BeautifulSoup lxml-xml recovery mode (handles mismatched tags)
    """
    seen_ids = state["seen"]
    ua = "MorningDigest/1.0 contact@morningdigest.dev"
    try:
        # Level 1: feedparser with a browser-like User-Agent
        # (SEC EDGAR and Cloudflare-protected sites block feedparser's default UA)
        feed = feedparser.parse(source["url"], request_headers={"User-Agent": ua})
        raw = None

        if feed.bozo and not feed.entries:
            # Level 2: fetch raw bytes, strip bad chars, retry feedparser
            try:
                import urllib.request
                raw = urllib.request.urlopen(
                    urllib.request.Request(source["url"], headers={"User-Agent": ua}),
                    timeout=15,
                ).read()
                feed = feedparser.parse(_clean_xml(raw))
                if feed.bozo and not feed.entries:
                    # Also try without cleaning (encoding-declaration mismatch)
                    feed = feedparser.parse(raw)
            except Exception:
                pass

        # Level 3: if feedparser is still bozo with no entries, use BS4/lxml recovery
        if feed.bozo and not feed.entries:
            if raw is None:
                try:
                    import urllib.request
                    raw = urllib.request.urlopen(
                        urllib.request.Request(source["url"], headers={"User-Agent": ua}),
                        timeout=15,
                    ).read()
                except Exception:
                    pass
            if raw:
                # Try cleaned bytes first, then raw (handles encoding edge cases)
                bs4_items = _bs4_parse_feed(_clean_xml(raw), source["url"])
                if not bs4_items:
                    bs4_items = _bs4_parse_feed(raw, source["url"])
                if bs4_items:
                    # Process the manually-parsed items the same way
                    new_items = []
                    fallback_item = None
                    fallback_pub_dt = None
                    for item in bs4_items:
                        pub_dt = item.pop("pub_dt", None)
                        if pub_dt and (fallback_pub_dt is None or pub_dt > fallback_pub_dt):
                            fallback_pub_dt = pub_dt
                            fallback_item = item
                        elif fallback_item is None:
                            fallback_item = item
                        if not item["id"] or item["id"] in seen_ids:
                            continue
                        if pub_dt and pub_dt < cutoff:
                            continue
                        new_items.append(item)
                        seen_ids.add(item["id"])
                    return new_items, fallback_item, None

            # Level 4: Feedly public stream API fallback
            # Works for any Cloudflare-blocked feed — Feedly caches feeds from
            # its own servers, so datacenter IP blocks don't apply.
            try:
                import urllib.parse
                feedly_stream = f"feed/{source['url']}"
                feedly_url = (
                    "https://cloud.feedly.com/v3/streams/contents?"
                    f"streamId={urllib.parse.quote(feedly_stream, safe='')}&count=20"
                )
                resp = requests.get(feedly_url, headers={"User-Agent": ua}, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("items", [])
                    if items:
                        print(f"    [Feedly fallback] Got {len(items)} items")
                        new_items = []
                        fallback_item = None
                        fallback_pub_dt = None
                        for entry in items:
                            item_id = entry.get("originId") or entry.get("id", "")
                            title = entry.get("title") or "Untitled"
                            link = (
                                entry.get("canonicalUrl")
                                or entry.get("alternate", [{}])[0].get("href", "")
                                or item_id
                            )
                            # Feedly provides summary or content
                            content_obj = entry.get("summary") or entry.get("content") or {}
                            content = (content_obj.get("content") or "")[:2500]
                            pub_dt = None
                            pub_date = None
                            ts = entry.get("published")
                            if ts:
                                try:
                                    pub_dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
                                    pub_date = pub_dt.strftime("%Y-%m-%d")
                                except (ValueError, OSError):
                                    pass
                            item = {"id": item_id, "title": title, "link": link,
                                    "content": content, "pub_date": pub_date}
                            if pub_dt and (fallback_pub_dt is None or pub_dt > fallback_pub_dt):
                                fallback_pub_dt = pub_dt
                                fallback_item = item
                            elif fallback_item is None:
                                fallback_item = item
                            if not item_id or item_id in seen_ids:
                                continue
                            if pub_dt and pub_dt < cutoff:
                                continue
                            new_items.append(item)
                            seen_ids.add(item_id)
                        return new_items, fallback_item, None
            except Exception:
                pass

            if feed.bozo and not feed.entries:
                return [], None, f"Feed parse error: {feed.bozo_exception}"

        # Process feedparser entries normally
        new_items = []
        fallback_item = None
        fallback_pub_dt = None

        for entry in feed.entries:
            item_id = entry.get("id") or entry.get("link") or ""
            if not item_id:
                continue

            pub_dt, pub_str = _parse_pub_date(entry)

            content = ""
            if entry.get("content"):
                content = entry.content[0].value
            elif entry.get("summary"):
                content = entry.summary
            if content:
                content = _strip_html(content)

            item = {
                "id": item_id,
                "title": entry.get("title", "Untitled"),
                "link": entry.get("link", source["url"]),
                "content": content,
                "pub_date": pub_str,
            }

            # Track the most recent entry as fallback (regardless of seen/cutoff)
            if pub_dt and (fallback_pub_dt is None or pub_dt > fallback_pub_dt):
                fallback_pub_dt = pub_dt
                fallback_item = item
            elif fallback_item is None:
                fallback_item = item  # no date, use first entry

            # Skip if already seen or too old
            if item_id in seen_ids:
                continue
            if pub_dt and pub_dt < cutoff:
                continue

            new_items.append(item)
            seen_ids.add(item_id)

        return new_items, fallback_item, None
    except Exception as e:
        return [], None, str(e)


def fetch_web(source: dict, state: dict, cutoff: datetime) -> tuple:
    """
    Returns (new_items, fallback_item, error).
    For web sources, fallback_item is the stored last-seen snapshot info.
    """
    seen_ids = state["seen"]
    last_items = state["last_items"]
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; DailyDigestBot/1.0)"}
        resp = requests.get(source["url"], timeout=15, headers=headers)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Auto-detect RSS
        rss_link = (soup.find("link", type="application/rss+xml")
                    or soup.find("link", type="application/atom+xml"))
        if rss_link and rss_link.get("href"):
            rss_url = rss_link["href"]
            if not rss_url.startswith("http"):
                rss_url = urljoin(source["url"], rss_url)
            new_items, fallback, err = fetch_rss({**source, "url": rss_url}, state, cutoff)
            if new_items or err is None:
                return new_items, fallback, None

        # Content-hash fallback
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.find(id="content") or soup.body
        text = (main or soup).get_text(separator="\n", strip=True)
        text = "\n".join(l for l in text.splitlines() if l.strip())[:3000]
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        item_id = f"{source['url']}#{content_hash}"
        title = soup.title.string.strip() if soup.title and soup.title.string else source["name"]
        today = datetime.now().strftime("%Y-%m-%d")

        item = {"id": item_id, "title": title, "link": source["url"], "content": text, "pub_date": today}

        if item_id in seen_ids:
            # No change — build fallback from stored info
            stored = last_items.get(source["url"], {})
            fallback = {
                "title": stored.get("title", title),
                "link": source["url"],
                "pub_date": stored.get("date"),
            }
            return [], fallback, None

        # New/changed content
        seen_ids.add(item_id)
        last_items[source["url"]] = {"title": title, "date": today}
        return [item], None, None

    except Exception as e:
        return [], None, str(e)


# -- Summarization --------------------------------------------------------------------------------

READER_PROFILE = """You are a research analyst writing a morning briefing for a hedge fund portfolio manager.
The reader runs quantitative macro strategies and also makes personal investments concentrated in AI/semiconductor stocks.

When summarizing, always surface:
- Market-moving signals: earnings, guidance changes, capacity/capex announcements, supply chain shifts
- Positioning implications: what this means for long/short theses on specific names (NVDA, MSFT, GOOGL, META, AMD, ASML, TSM, etc.)
- Macro read-throughs: interest rate sensitivity, trade policy, datacenter capex cycles, power/energy constraints
- AI ecosystem shifts: model capability jumps, inference cost curves, open vs. closed dynamics, regulatory moves
- Quantitative angles: new alpha signals, market microstructure changes, data availability

Be direct. No filler. Lead with what matters for positioning."""


def summarize(client: Groq, source_name: str, items: list) -> str:
    blocks = []
    for item in items:
        pub = f" ({item['pub_date']})" if item.get("pub_date") else ""
        block = f"### {item['title']}{pub}\nURL: {item['link']}"
        if item.get("content"):
            block += f"\n{item['content']}"
        blocks.append(block)
    combined = "\n\n---\n\n".join(blocks)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": READER_PROFILE},
            {"role": "user", "content": f"""Summarize these new items from "{source_name}".

Use this exact format:

**New this period:**
- [one-line bullet per item — what it is and why it matters for positioning]

**Details:**
**[Title](url)**
2–3 sentences: key finding, market implication, and any actionable read-through.

Items:
{combined}"""}
        ]
    )
    return resp.choices[0].message.content


# -- HTML output ----------------------------------------------------------------------------------

def md_to_html_simple(text: str) -> str:
    """Very lightweight markdown-to-HTML for the summary output."""
    import re
    lines = text.split("\n")
    html_lines = []
    in_ul = False
    for line in lines:
        # Bold links: **[text](url)**
        line = re.sub(r'\*\*\[(.+?)\]\((.+?)\)\*\*', r'<strong><a href="\2">\1</a></strong>', line)
        # Regular links: [text](url)
        line = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', line)
        # Bold: **text**
        line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
        if line.startswith("- "):
            if not in_ul:
                html_lines.append("<ul>")
                in_ul = True
            html_lines.append(f"  <li>{line[2:]}</li>")
        else:
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if line.strip():
                html_lines.append(f"<p>{line}</p>")
    if in_ul:
        html_lines.append("</ul>")
    return "\n".join(html_lines)



def build_html(date_str: str, results: list) -> str:
    """
    results is a list of dicts with category field.
    Groups results by category for display.
    """
    from collections import OrderedDict

    CATEGORY_META = OrderedDict([
        ("AI Opinion Leaders",   {"icon": "\U0001f9e0", "color": "#6c5ce7"}),
        ("Investor Positioning", {"icon": "\U0001f4b0", "color": "#e17055"}),
        ("Institutional Views",  {"icon": "\U0001f3db️", "color": "#0984e3"}),
        ("Tech & AI Podcasts",   {"icon": "\U0001f399️", "color": "#00b894"}),
        ("Research",             {"icon": "\U0001f4c4", "color": "#636e72"}),
    ])

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Group results by category, preserving order
    grouped = OrderedDict()
    for cat in CATEGORY_META:
        grouped[cat] = []
    grouped["Other"] = []
    for r in results:
        cat = r.get("category", "Other")
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(r)

    # Count fresh items per category for the nav
    nav_parts = []
    for cat, items in grouped.items():
        if not items:
            continue
        meta = CATEGORY_META.get(cat, {"icon": "•", "color": "#666"})
        fresh_count = sum(1 for r in items if r.get("is_fresh"))
        label = f'{meta["icon"]} {cat}'
        if fresh_count:
            label += f' <span class="nav-count">{fresh_count}</span>'
        cat_id = cat.lower().replace(" ", "-").replace("&", "and")
        nav_parts.append(f'<a href="#{cat_id}" class="nav-pill" style="border-color:{meta["color"]}">{label}</a>')

    nav_html = "\n    ".join(nav_parts)

    # Build sections
    sections = []
    for cat, items in grouped.items():
        if not items:
            continue
        meta = CATEGORY_META.get(cat, {"icon": "•", "color": "#666"})
        cat_id = cat.lower().replace(" ", "-").replace("&", "and")

        source_blocks = []
        for r in items:
            name = r["name"]
            summary = r.get("summary")
            error = r.get("error")
            is_fresh = r.get("is_fresh", False)
            latest_date = r.get("latest_date")
            latest_title = r.get("latest_title")
            latest_link = r.get("latest_link")

            if is_fresh and latest_date:
                badge = f'<span class="badge fresh">{latest_date}</span>'
            elif latest_date:
                badge = f'<span class="badge stale">{latest_date}</span>'
            else:
                badge = ''

            block = f'<div class="source {"fresh" if is_fresh else "quiet"}">\n'
            block += f'<h3>{name} {badge}</h3>\n'

            if error:
                block += f'<p class="error">{error}</p>\n'

            if summary:
                block += md_to_html_simple(summary) + "\n"
            elif not error and not is_fresh:
                if latest_title and latest_link:
                    block += f'<p class="quiet-note">Latest: <a href="{latest_link}">{latest_title}</a></p>\n'
                elif latest_link:
                    block += f'<p class="quiet-note"><a href="{latest_link}">View source</a></p>\n'
                else:
                    block += '<p class="quiet-note">No new items.</p>\n'

            block += "</div>"
            source_blocks.append(block)

        section = f'<section id="{cat_id}" class="category">\n'
        section += f'<h2 style="border-left-color:{meta["color"]}">{meta["icon"]} {cat}</h2>\n'
        section += "\n".join(source_blocks)
        section += "\n</section>"
        sections.append(section)

    content_html = "\n".join(sections) if sections else '<p class="empty">No sources configured.</p>'

    css = """
    :root { --bg: #fafafa; --card: #fff; --text: #1a1a2e; --muted: #6b7280; --border: #e5e7eb; --accent: #2563eb; }
    @media (prefers-color-scheme: dark) {
      :root { --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --muted: #94a3b8; --border: #334155; --accent: #60a5fa; }
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif; background: var(--bg); color: var(--text); line-height: 1.65; }
    .container { max-width: 820px; margin: 0 auto; padding: 2rem 1.5rem; }
    header { margin-bottom: 2rem; }
    header h1 { font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; }
    header .subtitle { color: var(--muted); font-size: 0.85rem; margin-top: 0.3rem; }
    header .subtitle a { color: var(--accent); text-decoration: none; }
    header .subtitle a:hover { text-decoration: underline; }
    .nav { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1.5rem 0 2rem; }
    .nav-pill { display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.35rem 0.85rem; font-size: 0.8rem; font-weight: 500; color: var(--text); background: var(--card); border: 1.5px solid var(--border); border-left-width: 3px; border-radius: 8px; text-decoration: none; transition: all 0.15s; }
    .nav-pill:hover { background: var(--bg); box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
    .nav-count { background: #22c55e; color: #fff; font-size: 0.68rem; padding: 0.1rem 0.45rem; border-radius: 10px; font-weight: 600; }
    .category { margin-bottom: 2.5rem; }
    .category h2 { font-size: 1.05rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: var(--muted); border-left: 4px solid; padding-left: 0.75rem; margin-bottom: 1rem; }
    .source { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 0.75rem; transition: box-shadow 0.15s; }
    .source.fresh { border-left: 3px solid #22c55e; }
    .source.quiet { opacity: 0.6; }
    .source.quiet:hover { opacity: 0.85; }
    .source h3 { font-size: 0.95rem; font-weight: 600; display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.4rem; }
    .source p { margin: 0.4em 0; font-size: 0.9rem; }
    .source ul { padding-left: 1.3em; margin: 0.4em 0; font-size: 0.9rem; }
    .source li { margin-bottom: 0.25rem; }
    .source a { color: var(--accent); text-decoration: none; }
    .source a:hover { text-decoration: underline; }
    .badge { font-size: 0.7rem; font-weight: 500; padding: 0.15rem 0.55rem; border-radius: 6px; white-space: nowrap; }
    .badge.fresh { background: #dcfce7; color: #166534; }
    .badge.stale { background: var(--bg); color: var(--muted); }
    @media (prefers-color-scheme: dark) {
      .badge.fresh { background: #14532d; color: #86efac; }
    }
    .quiet-note { color: var(--muted); font-size: 0.85rem; font-style: italic; }
    .error { color: #ef4444; font-size: 0.85rem; }
    .empty { color: var(--muted); font-style: italic; }
    footer { border-top: 1px solid var(--border); padding-top: 1.5rem; margin-top: 1rem; color: var(--muted); font-size: 0.8rem; }
    footer a { color: var(--accent); text-decoration: none; }
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Morning Digest &ndash; {date_str}</title>
  <style>{css}</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Morning Digest</h1>
    <p class="subtitle">{date_str} &middot; Generated {now_str} &middot; <a href="https://github.com/2310renhe/morning-digest/blob/main/sources.json">edit sources</a> &middot; <a href="archive/">archive</a></p>
  </header>
  <nav class="nav">
    {nav_html}
  </nav>
  {content_html}
  <footer>
    <p>Powered by <a href="https://github.com/2310renhe/morning-digest">morning-digest</a></p>
  </footer>
</div>
</body>
</html>"""


def build_archive_index(dates: list) -> str:
    items = "\n".join(f'  <li><a href="{d}.html">{d}</a></li>' for d in sorted(dates, reverse=True))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Morning Digest — Archive</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 500px; margin: 40px auto; padding: 0 20px; }}
    a {{ color: #0066cc; }}
    li {{ margin-bottom: 6px; }}
  </style>
</head>
<body>
  <h1>☕ Digest Archive</h1>
  <ul>
{items}
  </ul>
  <p><a href="../index.html">← Today's digest</a></p>
</body>
</html>"""


# -- Email ----------------------------------------------------------------------------------------

def send_email(subject: str, body: str):
    if not all([EMAIL_FROM, EMAIL_TO, EMAIL_PASSWORD]):
        print("  Email not configured — skipping.")
        return
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(EMAIL_FROM, EMAIL_PASSWORD)
        s.send_message(msg)
    print("  Email sent.")


# -- Main -----------------------------------------------------------------------------------------

def main():
    ARCHIVE_DIR.mkdir(exist_ok=True)

    if not SOURCES_FILE.exists():
        print("sources.json not found.")
        sys.exit(1)

    sources = json.loads(SOURCES_FILE.read_text())
    if not sources:
        print("No sources configured.")
        sys.exit(0)

    state = load_state()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    client = Groq(api_key=GROQ_API_KEY)
    date_str = datetime.now().strftime("%Y-%m-%d")

    results = []

    for source in sources:
        stype = source.get("type", "rss").lower()
        print(f"[{stype}] {source['name']} ...")

        if stype in ("rss", "podcast"):
            new_items, fallback, err = fetch_rss(source, state, cutoff)
        else:
            new_items, fallback, err = fetch_web(source, state, cutoff)

        cat = source.get("category", "Other")

        if err and not new_items and not fallback:
            print(f"  ERROR: {err}")
            results.append({"name": source["name"], "category": cat, "error": err, "is_fresh": False})
            continue

        if new_items:
            print(f"  {len(new_items)} new item(s) — summarizing...")
            # Collect publish dates for the badge
            pub_dates = [i["pub_date"] for i in new_items if i.get("pub_date")]
            latest_date = max(pub_dates) if pub_dates else date_str
            try:
                summary = summarize(client, source["name"], new_items)
                # Persist summary so stale items still show context on future runs
                state["summaries"][source["name"]] = {
                    "text": summary,
                    "date": latest_date,
                }
                results.append({
                    "name": source["name"],
                    "category": cat,
                    "summary": summary,
                    "is_fresh": True,
                    "latest_date": latest_date,
                    "error": err,  # non-fatal warning if any
                })
                save_state(state)
            except Exception as e:
                print(f"  Summarization error: {e}")
                results.append({
                    "name": source["name"],
                    "category": cat,
                    "error": f"Summarization failed: {e}",
                    "is_fresh": True,
                    "latest_date": latest_date,
                })
        else:
            # No new items — show cached summary if available, otherwise just title
            cached = state["summaries"].get(source["name"])
            fb_title = fallback['title'] if fallback else 'unknown'
            fb_date = fallback.get('pub_date', '?') if fallback else '?'
            print(f"  No new items. Latest: {fb_title} ({fb_date})"
                  + (" [cached summary]" if cached else ""))
            result = {
                "name": source["name"],
                "category": cat,
                "is_fresh": False,
                "latest_date": cached["date"] if cached else (fallback.get("pub_date") if fallback else None),
                "latest_title": fallback.get("title") if fallback else None,
                "latest_link": fallback.get("link") if fallback else source.get("url"),
                "error": err,
            }
            if cached:
                result["summary"] = cached["text"]
            results.append(result)

    # Always write index.html
    html = build_html(date_str, results)
    INDEX_FILE.write_text(html)
    print(f"\nWrote index.html")

    # Archive today's digest
    archive_path = ARCHIVE_DIR / f"{date_str}.html"
    archive_path.write_text(html)

    # Update archive index
    dates = [p.stem for p in ARCHIVE_DIR.glob("????-??-??.html")]
    (ARCHIVE_DIR / "index.html").write_text(build_archive_index(dates))

    # Email: only mention sources with fresh content
    fresh = [r for r in results if r.get("is_fresh") and r.get("summary")]
    if fresh:
        subject = f"☕ Morning Digest — {date_str} | {len(fresh)} source{'s' if len(fresh) != 1 else ''} updated"
        bullets = []
        for r in fresh:
            bullet_lines = [l.strip() for l in r["summary"].splitlines() if l.strip().startswith("- ")]
            bullets.append(f"• {r['name']} ({len(bullet_lines)} items)")
        body = (
            f"New content in today's digest:\n\n"
            + "\n".join(bullets)
            + f"\n\nhttps://2310renhe.github.io/morning-digest/\n"
        )
        send_email(subject, body)
    else:
        print("Nothing new today.")


if __name__ == "__main__":
    main()
