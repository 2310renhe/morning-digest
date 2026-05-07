#!/usr/bin/env python3
"""Daily content digest â fetch new items, summarize with Groq, publish to GitHub Pages."""

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


# ââ State âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def load_state() -> dict:
    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text())
        # Migrate old format (just a list of seen IDs)
        if isinstance(data, list):
            return {"seen": set(data), "last_items": {}}
        return {
            "seen": set(data.get("seen", [])),
            "last_items": data.get("last_items", {}),
        }
    return {"seen": set(), "last_items": {}}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps({
        "seen": list(state["seen"])[-15000:],
        "last_items": state["last_items"],
    }, indent=2))


# ââ Fetching ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

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
    Uses html.parser first (most lenient â ignores XML namespaces and bad chars),
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

            # Level 4: Substack JSON API fallback
            # Cloudflare blocks GitHub Actions IPs on /feed but not /api/v1/posts
            if "substack.com" in source["url"]:
                try:
                    api_url = source["url"].replace("/feed", "/api/v1/posts?limit=20")
                    print(f"    [Level 4] Trying Substack JSON API: {api_url}")
                    resp = requests.get(api_url, headers={"User-Agent": ua}, timeout=15)
                    print(f"    [Level 4] Status: {resp.status_code}, Content-Type: {resp.headers.get('content-type', 'unknown')}")
                    if resp.status_code == 200:
                        posts = resp.json()
                        print(f"    [Level 4] Got {len(posts) if isinstance(posts, list) else type(posts).__name__} posts")
                        if isinstance(posts, list) and posts:
                            new_items = []
                            fallback_item = None
                            fallback_pub_dt = None
                            for p in posts:
                                item_id = p.get("canonical_url") or str(p.get("id", ""))
                                title = p.get("title") or "Untitled"
                                link = p.get("canonical_url") or source["url"]
                                content = (p.get("truncated_body_text") or p.get("description") or "")[:2500]
                                pub_dt = None
                                pub_date = None
                                if p.get("post_date"):
                                    try:
                                        pub_dt = datetime.strptime(p["post_date"][:19], "%Y-%m-%dT%H:%M:%S")
                                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                                        pub_date = pub_dt.strftime("%Y-%m-%d")
                                    except ValueError:
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
                    else:
                        print(f"    [Level 4] Non-200 response, first 200 chars: {resp.text[:200]}")
                except Exception as e:
                    print(f"    [Level 4] Exception: {e}")

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
            # No change â build fallback from stored info
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


# ââ Summarization âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

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
            {"role": "system", "content": "You are a concise research assistant writing a morning digest. Be direct and informative."},
            {"role": "user", "content": f"""Summarize these new items from "{source_name}".

Use this exact format:

**New this period:**
- [one-line bullet per item â what it is and why it matters]

**Details:**
**[Title](url)**
2â3 sentences on the key point and takeaway.

Items:
{combined}"""}
        ]
    )
    return resp.choices[0].message.content


# ââ HTML output âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

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
    results is a list of dicts:
      { name, summary, error, is_fresh, latest_date, latest_title, latest_link }
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    content_parts = []

    for r in results:
        name = r["name"]
        summary = r.get("summary")
        error = r.get("error")
        is_fresh = r.get("is_fresh", False)
        latest_date = r.get("latest_date")
        latest_title = r.get("latest_title")
        latest_link = r.get("latest_link")

        # Header badge
        if is_fresh and latest_date:
            badge = f'<span class="badge fresh">ð {latest_date}</span>'
        elif latest_date:
            badge = f'<span class="badge stale">last: {latest_date}</span>'
        else:
            badge = ''

        block = f'<div class="source-block {"fresh" if is_fresh else "quiet"}">\n'
        block += f'<h2>{name} {badge}</h2>\n'

        if error:
            block += f'<p class="error">â ï¸ {error}</p>\n'

        if summary:
            block += md_to_html_simple(summary) + "\n"
        elif not error and not is_fresh:
            # Show latest available
            if latest_title and latest_link:
                block += f'<p class="quiet-note">No new posts. Latest: <a href="{latest_link}">{latest_title}</a></p>\n'
            elif latest_link:
                block += f'<p class="quiet-note">No changes detected. <a href="{latest_link}">View source â</a></p>\n'
            else:
                block += '<p class="quiet-note">No new items.</p>\n'

        block += "</div>"
        content_parts.append(block)

    content_html = "\n".join(content_parts) if content_parts else '<p class="empty">No sources configured.</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Morning Digest â {date_str}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 760px; margin: 40px auto; padding: 0 20px; color: #222; line-height: 1.7; }}
    h1 {{ font-size: 1.8em; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
    h2 {{ font-size: 1.1em; margin-top: 2em; background: #f5f5f5; padding: 6px 14px; border-radius: 4px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
    a {{ color: #0066cc; }}
    ul {{ padding-left: 1.4em; }}
    li {{ margin-bottom: 4px; }}
    p {{ margin: 0.6em 0; }}
    .source-block {{ border-left: 3px solid #ddd; padding-left: 16px; margin-bottom: 2.5em; }}
    .source-block.fresh {{ border-left-color: #2a9d2a; }}
    .source-block.quiet {{ border-left-color: #ddd; opacity: 0.75; }}
    .badge {{ font-size: 0.72em; font-weight: normal; padding: 2px 8px; border-radius: 10px; white-space: nowrap; }}
    .badge.fresh {{ background: #d4edda; color: #155724; }}
    .badge.stale {{ background: #f0f0f0; color: #666; }}
    .quiet-note {{ color: #888; font-size: 0.92em; font-style: italic; }}
    .meta {{ color: #888; font-size: 0.9em; }}
    .error {{ color: #c00; font-size: 0.9em; }}
    .empty {{ color: #888; font-style: italic; }}
    hr {{ border: none; border-top: 1px solid #eee; margin: 2em 0; }}
  </style>
</head>
<body>
  <h1>â Morning Digest â {date_str}</h1>
  <p class="meta">Generated {now_str} &nbsp;Â·&nbsp; <a href="https://github.com/2310renhe/morning-digest/blob/main/sources.json">edit sources</a> &nbsp;Â·&nbsp; <a href="archive/">archive</a></p>
  {content_html}
  <hr>
  <p class="meta">Powered by <a href="https://github.com/2310renhe/morning-digest">morning-digest</a></p>
</body>
</html>"""


def build_archive_index(dates: list) -> str:
    items = "\n".join(f'  <li><a href="{d}.html">{d}</a></li>' for d in sorted(dates, reverse=True))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Morning Digest â Archive</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 500px; margin: 40px auto; padding: 0 20px; }}
    a {{ color: #0066cc; }}
    li {{ margin-bottom: 6px; }}
  </style>
</head>
<body>
  <h1>â Digest Archive</h1>
  <ul>
{items}
  </ul>
  <p><a href="../index.html">â Today's digest</a></p>
</body>
</html>"""


# ââ Email âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

def send_email(subject: str, body: str):
    if not all([EMAIL_FROM, EMAIL_TO, EMAIL_PASSWORD]):
        print("  Email not configured â skipping.")
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


# ââ Main ââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ

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

        if err and not new_items and not fallback:
            print(f"  ERROR: {err}")
            results.append({"name": source["name"], "error": err, "is_fresh": False})
            continue

        if new_items:
            print(f"  {len(new_items)} new item(s) â summarizing...")
            # Collect publish dates for the badge
            pub_dates = [i["pub_date"] for i in new_items if i.get("pub_date")]
            latest_date = max(pub_dates) if pub_dates else date_str
            try:
                summary = summarize(client, source["name"], new_items)
                results.append({
                    "name": source["name"],
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
                    "error": f"Summarization failed: {e}",
                    "is_fresh": True,
                    "latest_date": latest_date,
                })
        else:
            # No new items â show latest available
            print(f"  No new items. Latest: {fallback['title'] if fallback else 'unknown'} ({fallback.get('pub_date', '?') if fallback else '?'})")
            results.append({
                "name": source["name"],
                "is_fresh": False,
                "latest_date": fallback.get("pub_date") if fallback else None,
                "latest_title": fallback.get("title") if fallback else None,
                "latest_link": fallback.get("link") if fallback else source.get("url"),
                "error": err,
            })

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
        subject = f"â Morning Digest â {date_str} | {len(fresh)} source{'s' if len(fresh) != 1 else ''} updated"
        bullets = []
        for r in fresh:
            bullet_lines = [l.strip() for l in r["summary"].splitlines() if l.strip().startswith("- ")]
            bullets.append(f"â¢ {r['name']} ({len(bullet_lines)} items)")
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
