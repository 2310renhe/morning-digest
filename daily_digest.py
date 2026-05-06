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


# ── State ─────────────────────────────────────────────────────────────────────

def load_state() -> set:
    if STATE_FILE.exists():
        return set(json.loads(STATE_FILE.read_text()).get("seen", []))
    return set()


def save_state(seen_ids: set):
    STATE_FILE.write_text(json.dumps({"seen": list(seen_ids)[-15000:]}, indent=2))


# ── Fetching ──────────────────────────────────────────────────────────────────

def _strip_html(html: str, max_chars: int = 2500) -> str:
    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)[:max_chars]


def fetch_rss(source: dict, seen_ids: set, cutoff: datetime) -> tuple:
    try:
        feed = feedparser.parse(source["url"])
        if feed.bozo and not feed.entries:
            return [], f"Feed parse error: {feed.bozo_exception}"
        new_items = []
        for entry in feed.entries:
            item_id = entry.get("id") or entry.get("link") or ""
            if not item_id or item_id in seen_ids:
                continue
            pub = entry.get("published_parsed") or entry.get("updated_parsed")
            if pub:
                pub_dt = datetime(*pub[:6], tzinfo=timezone.utc)
                if pub_dt < cutoff:
                    continue
            content = ""
            if entry.get("content"):
                content = entry.content[0].value
            elif entry.get("summary"):
                content = entry.summary
            if content:
                content = _strip_html(content)
            new_items.append({
                "id": item_id,
                "title": entry.get("title", "Untitled"),
                "link": entry.get("link", source["url"]),
                "content": content,
            })
            seen_ids.add(item_id)
        return new_items, None
    except Exception as e:
        return [], str(e)


def fetch_web(source: dict, seen_ids: set, cutoff: datetime) -> tuple:
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
            items, err = fetch_rss({**source, "url": rss_url}, seen_ids, cutoff)
            if items or err is None:
                return items, None
        # Content-hash fallback
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        main = soup.find("main") or soup.find("article") or soup.find(id="content") or soup.body
        text = (main or soup).get_text(separator="\n", strip=True)
        text = "\n".join(l for l in text.splitlines() if l.strip())[:3000]
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        item_id = f"{source['url']}#{content_hash}"
        if item_id in seen_ids:
            return [], None
        seen_ids.add(item_id)
        title = soup.title.string.strip() if soup.title and soup.title.string else source["name"]
        return [{"id": item_id, "title": title, "link": source["url"], "content": text}], None
    except Exception as e:
        return [], str(e)


# ── Summarization ─────────────────────────────────────────────────────────────

def summarize(client: Groq, source_name: str, items: list) -> str:
    blocks = []
    for item in items:
        block = f"### {item['title']}\nURL: {item['link']}"
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
- [one-line bullet per item — what it is and why it matters]

**Details:**
**[Title](url)**
2–3 sentences on the key point and takeaway.

Items:
{combined}"""}
        ]
    )
    return resp.choices[0].message.content


# ── HTML output ───────────────────────────────────────────────────────────────

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
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    content_parts = []
    for source_name, summary, error in results:
        block = f'<div class="source-block">\n<h2>{source_name}</h2>\n'
        if error:
            block += f'<p class="error">⚠️ {error}</p>\n'
        if summary:
            block += md_to_html_simple(summary) + "\n"
        block += "</div>"
        content_parts.append(block)
    content_html = "\n".join(content_parts) if content_parts else '<p class="empty">Nothing new today.</p>'
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Morning Digest — {date_str}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 760px; margin: 40px auto; padding: 0 20px; color: #222; line-height: 1.7; }}
    h1 {{ font-size: 1.8em; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
    h2 {{ font-size: 1.1em; margin-top: 2em; background: #f5f5f5; padding: 6px 14px; border-radius: 4px; }}
    a {{ color: #0066cc; }}
    ul {{ padding-left: 1.4em; }}
    li {{ margin-bottom: 4px; }}
    p {{ margin: 0.6em 0; }}
    .source-block {{ border-left: 3px solid #ddd; padding-left: 16px; margin-bottom: 2.5em; }}
    .meta {{ color: #888; font-size: 0.9em; }}
    .error {{ color: #c00; font-size: 0.9em; }}
    .empty {{ color: #888; font-style: italic; }}
    hr {{ border: none; border-top: 1px solid #eee; margin: 2em 0; }}
  </style>
</head>
<body>
  <h1>☕ Morning Digest — {date_str}</h1>
  <p class="meta">Generated {now_str} &nbsp;·&nbsp; <a href="https://github.com/2310renhe/morning-digest/blob/main/sources.json">edit sources</a> &nbsp;·&nbsp; <a href="archive/">archive</a></p>
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


# ── Email ─────────────────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ARCHIVE_DIR.mkdir(exist_ok=True)

    if not SOURCES_FILE.exists():
        print("sources.json not found.")
        sys.exit(1)

    sources = json.loads(SOURCES_FILE.read_text())
    if not sources:
        print("No sources configured.")
        sys.exit(0)

    seen_ids = load_state()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)
    client = Groq(api_key=GROQ_API_KEY)
    date_str = datetime.now().strftime("%Y-%m-%d")

    results = []

    for source in sources:
        stype = source.get("type", "rss").lower()
        print(f"[{stype}] {source['name']} ...")
        if stype in ("rss", "podcast"):
            items, err = fetch_rss(source, seen_ids, cutoff)
        else:
            items, err = fetch_web(source, seen_ids, cutoff)

        if err and not items:
            print(f"  ERROR: {err}")
            results.append((source["name"], None, err))
            continue
        if not items:
            print("  No new items.")
            continue

        print(f"  {len(items)} new item(s) — summarizing...")
        try:
            summary = summarize(client, source["name"], items)
            results.append((source["name"], summary, None))
            save_state(seen_ids)
        except Exception as e:
            print(f"  Summarization error: {e}")
            results.append((source["name"], None, f"Summarization failed: {e}"))

    # Always write index.html (even if nothing new)
    html = build_html(date_str, results)
    INDEX_FILE.write_text(html)
    print(f"\nWrote index.html")

    # Archive today's digest
    archive_path = ARCHIVE_DIR / f"{date_str}.html"
    archive_path.write_text(html)

    # Update archive index
    dates = [p.stem for p in ARCHIVE_DIR.glob("????-??-??.html")]
    (ARCHIVE_DIR / "index.html").write_text(build_archive_index(dates))

    active = [(n, s) for n, s, _ in results if s]
    if active:
        subject = f"☕ Morning Digest — {date_str} | {len(active)} source{'s' if len(active) != 1 else ''} updated"
        bullets = []
        for name, summary in active:
            bullet_lines = [l.strip() for l in summary.splitlines() if l.strip().startswith("- ")]
            bullets.append(f"• {name} ({len(bullet_lines)} items)")
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
