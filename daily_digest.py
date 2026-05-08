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

def _strip_html(html: str, max_chars: int = 20000) -> str:
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
            content = _strip_html(inner) if "<" in inner else inner[:20000]

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


def _fetch_podcast_transcript(feed_url: str, episode_id: str, ua: str) -> str:
    """
    Try to fetch a podcast transcript from <podcast:transcript> tags in the feed.
    Returns plain text transcript (up to 20k chars) or empty string.
    """
    import re
    try:
        raw = requests.get(feed_url, headers={"User-Agent": ua}, timeout=15).text
        # Find the <item> block containing this episode (match by guid/id)
        # We search for transcript tags near the episode ID
        # Strategy: find all items, match by episode_id, extract transcript URL
        items = re.split(r'<item\b', raw)
        for item_block in items:
            if episode_id not in item_block:
                continue
            # Prefer text/plain format, fall back to SRT
            # Attribute order varies, so match the full tag then extract url + type
            all_tags = re.findall(r'<podcast:transcript([^>]+)/>', item_block)
            txt_urls = []
            srt_urls = []
            for attrs in all_tags:
                url_m = re.search(r'url="([^"]+)"', attrs)
                type_m = re.search(r'type="([^"]+)"', attrs)
                if not url_m:
                    continue
                t = type_m.group(1) if type_m else ""
                u = url_m.group(1)
                if "text/plain" in t:
                    txt_urls.append(u)
                elif "application/srt" in t:
                    srt_urls.append(u)
            transcript_url = (txt_urls[0] if txt_urls else srt_urls[0] if srt_urls else "")
            if not transcript_url:
                continue
            transcript_url = transcript_url.replace("&amp;", "&")
            resp = requests.get(transcript_url, headers={"User-Agent": ua}, timeout=20)
            if resp.status_code != 200:
                continue
            text = resp.text
            # Strip SRT timestamps if present (lines like "00:01:23,456 --> 00:01:25,789")
            text = re.sub(r'\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,\.]\d{3}\s*', '', text)
            # Strip plain-text timestamps (lines like "00:01:23")
            text = re.sub(r'^\d{2}:\d{2}:\d{2}\s*$', '', text, flags=re.MULTILINE)
            # Strip SRT sequence numbers (standalone digits on a line)
            text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
            # Collapse blank lines
            text = re.sub(r'\n{3,}', '\n\n', text).strip()
            if len(text) > 500:  # sanity check — must be a real transcript
                print(f"    [transcript] Got {len(text)} chars")
                return text[:20000]
    except Exception:
        pass
    return ""


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
                            content = _strip_html((content_obj.get("content") or ""))
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


def fetch_13f(source: dict, state: dict) -> tuple:
    """
    Parse SEC EDGAR 13F-HR filing for a fund.
    Returns (holdings_text, filing_date, filing_url, error).
    holdings_text is a markdown table of top holdings with % of portfolio.
    """
    ua = "MorningDigest/1.0 contact@morningdigest.dev"
    try:
        # Step 1: Get latest filing link from EDGAR RSS
        resp = requests.get(source["url"], headers={"User-Agent": ua}, timeout=15)
        soup = BeautifulSoup(resp.text, "xml")
        entry = soup.find("entry")
        if not entry:
            return None, None, None, "No filings found"

        filing_url = entry.find("link")["href"]
        filing_date = entry.find("updated").text[:10] if entry.find("updated") else None

        # Check if we already processed this filing
        cache_key = f"13f_{source['name']}"
        cached = state.get("summaries", {}).get(cache_key)
        if cached and cached.get("filing_url") == filing_url:
            return cached["text"], filing_date, filing_url, None

        # Step 2: Find the XML holdings file from the filing index
        # Collect all candidate XML links, prefer the plain one (not XSL-wrapped)
        resp2 = requests.get(filing_url, headers={"User-Agent": ua}, timeout=15)
        soup2 = BeautifulSoup(resp2.text, "html.parser")
        xml_candidates = []
        table = soup2.find("table", class_="tableFile")
        if table:
            for a in table.find_all("a", href=True):
                href = a["href"]
                fname = href.split("/")[-1].lower()
                if fname.endswith(".xml") and "primary_doc" not in fname:
                    xml_candidates.append("https://www.sec.gov" + href)
        # Prefer the plain XML (not inside xslForm13F_X02/ subdir)
        xml_url = None
        for url in xml_candidates:
            if "xslForm13F" not in url:
                xml_url = url
                break
        if not xml_url and xml_candidates:
            xml_url = xml_candidates[0]

        if not xml_url:
            return None, filing_date, filing_url, "Could not find holdings XML"

        # Step 3: Parse holdings XML
        # Strip namespace prefixes (some filers use ns1:, n1:, etc.)
        # then use html.parser which lowercases tags
        import re as _re
        resp3 = requests.get(xml_url, headers={"User-Agent": ua}, timeout=15)
        raw_xml = _re.sub(r'<(/?)[a-zA-Z]\w*:', r'<\1', resp3.text)
        soup3 = BeautifulSoup(raw_xml, "html.parser")

        holdings = []
        for info in soup3.find_all("infotable"):
            name = info.find("nameofissuer")
            value = info.find("value")
            shares = info.find("sshprnamt")
            if name and value:
                try:
                    raw_val = int(value.text.strip().replace(",", ""))
                except ValueError:
                    continue
                holdings.append({
                    "name": name.text.strip().title(),
                    "raw_val": raw_val,
                    "shares": int(shares.text.strip().replace(",", "")) if shares else 0,
                })

        if not holdings:
            return None, filing_date, filing_url, "No holdings found in XML"

        # Auto-detect value scale.
        # Some 13F filers (Druckenmiller) report <value> in $thousands (SEC standard).
        # Others (Coatue, Tiger Global, SALP) report in raw dollars.
        # Heuristic: if median raw_val > 1,000,000, positions are in dollars
        # (a $1B median in thousands would be unrealistically large for a single position).
        # Normalize everything to $thousands for uniform downstream math.
        median_val = sorted(h["raw_val"] for h in holdings)[len(holdings)//2]
        in_dollars = median_val > 1_000_000
        scale = 1000 if in_dollars else 1  # divide dollars→thousands; keep thousands as-is
        for h in holdings:
            h["value_k"] = h["raw_val"] // scale  # now in $thousands for all filers

        holdings.sort(key=lambda x: x["value_k"], reverse=True)
        total_k = sum(h["value_k"] for h in holdings)
        total_b = total_k / 1_000_000  # thousands / 1M = billions

        def _fmt_value(value_k: int) -> str:
            """Format a $thousands value as $XB or $XM."""
            if value_k >= 1_000_000:
                return f"${value_k / 1_000_000:.2f}B"
            return f"${value_k / 1_000:.1f}M"

        # Build markdown table — top 20 holdings
        lines = [
            f"**{len(holdings)} positions · ${total_b:.1f}B AUM · as of {filing_date}**\n",
            "| # | Name | Value | % of Portfolio |",
            "|---|------|-------|---------------|",
        ]
        for i, h in enumerate(holdings[:20], 1):
            pct = h["value_k"] / total_k * 100
            val_str = _fmt_value(h["value_k"])
            lines.append(f"| {i} | {h['name']} | {val_str} | {pct:.1f}% |")

        text = "\n".join(lines)

        # Cache in state
        state.setdefault("summaries", {})[cache_key] = {
            "text": text,
            "date": filing_date,
            "filing_url": filing_url,
        }

        return text, filing_date, filing_url, None

    except Exception as e:
        return None, None, None, str(e)


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
        text = "\n".join(l for l in text.splitlines() if l.strip())[:20000]
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

READER_PROFILE = """You are a research assistant producing a morning digest.
The reader is a hedge fund PM (quant macro + AI/semiconductor stocks).

Rules:
- EXTRACT only. State what the source actually says. Never infer, speculate, or editorialize beyond the source material.
- If the source does not mention a company, ticker, or data point, do not mention it.
- Be thorough — capture ALL key arguments, data points, and conclusions from the source. Do not stop at the first point.
- For academic papers: focus on methodology, key results, and datasets used. Skip papers that have no relevance to quantitative investing, financial markets, or AI/ML applied to finance."""


def summarize(client: Groq, source_name: str, items: list) -> str:
    blocks = []
    for item in items:
        pub = f" ({item['pub_date']})" if item.get("pub_date") else ""
        block = f"### {item['title']}{pub}\nURL: {item['link']}"
        if item.get("content"):
            block += f"\n{item['content']}"
        blocks.append(block)
    combined = "\n\n---\n\n".join(blocks)
    # Cap total content sent to LLM (~12k chars ≈ ~3k tokens) to stay within rate limits
    # while still capturing the bulk of most articles
    if len(combined) > 12000:
        combined = combined[:12000] + "\n\n[content truncated]"
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=3000,
        messages=[
            {"role": "system", "content": READER_PROFILE},
            {"role": "user", "content": f"""Summarize these new items from "{source_name}".

Rules:
- Only state what the source actually contains. Do not add interpretation or connect to topics not in the source.
- Be comprehensive — extract every key claim, data point, and named entity (companies, people, products, figures). Do not omit major arguments or sections.
- For academic paper feeds: skip papers unrelated to quant investing or AI/ML for finance. For relevant ones, state the research question, method, and key finding.

Use this exact format:

**New this period:**
- [one-line bullet per item — what it actually covers]

**Details:**
**[Title](url)**
Extract ALL key points from the source. Cover every major argument, data point, and conclusion. Use bullet points for clarity:
- [key point 1]
- [key point 2]
- ...

Items:
{combined}"""}
        ]
    )
    return resp.choices[0].message.content


# -- HTML output ----------------------------------------------------------------------------------

def md_to_html_simple(text: str) -> str:
    """Very lightweight markdown-to-HTML for the summary output."""
    import re

    def _inline(line: str) -> str:
        line = re.sub(r'\*\*\[(.+?)\]\((.+?)\)\*\*', r'<strong><a href="\2">\1</a></strong>', line)
        line = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', line)
        line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
        return line

    lines = text.split("\n")
    html_lines = []
    in_ul = False
    in_table = False
    table_rows = []

    def _flush_table():
        nonlocal in_table, table_rows
        if not table_rows:
            in_table = False
            return
        html_lines.append('<table class="holdings-table">')
        for i, row in enumerate(table_rows):
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            tag = "th" if i == 0 else "td"
            html_lines.append("  <tr>" + "".join(f"<{tag}>{_inline(c)}</{tag}>" for c in cells) + "</tr>")
        html_lines.append("</table>")
        table_rows = []
        in_table = False

    for line in lines:
        is_table_row = line.strip().startswith("|") and line.strip().endswith("|")
        is_separator = is_table_row and re.match(r"^\s*\|[\s\-:|]+\|\s*$", line)

        if is_table_row and not is_separator:
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            in_table = True
            table_rows.append(line)
        elif is_separator:
            pass  # skip the |---|---| divider line
        else:
            if in_table:
                _flush_table()
            if line.startswith("- "):
                if not in_ul:
                    html_lines.append("<ul>")
                    in_ul = True
                html_lines.append(f"  <li>{_inline(line[2:])}</li>")
            else:
                if in_ul:
                    html_lines.append("</ul>")
                    in_ul = False
                if line.strip():
                    html_lines.append(f"<p>{_inline(line)}</p>")

    if in_table:
        _flush_table()
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
                # Split into bullets (always visible) and details (collapsible)
                details_split = summary.split("**Details:**")
                if len(details_split) == 2:
                    bullets_md, details_md = details_split
                    block += md_to_html_simple(bullets_md) + "\n"
                    block += '<details class="details-fold"><summary>Details ▸</summary>\n'
                    block += md_to_html_simple(details_md) + "\n"
                    block += '</details>\n'
                else:
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
    .details-fold { margin-top: 0.5rem; }
    .details-fold > summary { cursor: pointer; font-size: 0.8rem; font-weight: 500; color: var(--muted); padding: 0.3rem 0; user-select: none; }
    .details-fold > summary:hover { color: var(--accent); }
    .details-fold[open] > summary { margin-bottom: 0.3rem; }
    .details-fold > summary::-webkit-details-marker { display: none; }
    .details-fold > summary::marker { content: ""; }
    .badge { font-size: 0.7rem; font-weight: 500; padding: 0.15rem 0.55rem; border-radius: 6px; white-space: nowrap; }
    .badge.fresh { background: #dcfce7; color: #166534; }
    .badge.stale { background: var(--bg); color: var(--muted); }
    @media (prefers-color-scheme: dark) {
      .badge.fresh { background: #14532d; color: #86efac; }
    }
    .quiet-note { color: var(--muted); font-size: 0.85rem; font-style: italic; }
    .error { color: #ef4444; font-size: 0.85rem; }
    .empty { color: var(--muted); font-style: italic; }
    .holdings-table { border-collapse: collapse; width: 100%; font-size: 0.82rem; margin: 0.5rem 0; }
    .holdings-table th { background: var(--bg); color: var(--muted); font-weight: 600; text-align: left; padding: 0.3rem 0.6rem; border-bottom: 2px solid var(--border); }
    .holdings-table td { padding: 0.25rem 0.6rem; border-bottom: 1px solid var(--border); }
    .holdings-table tr:hover td { background: var(--bg); }
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

        if stype == "sec13f":
            holdings_text, filing_date, filing_url, err = fetch_13f(source, state)
            cat = source.get("category", "Other")
            if err and not holdings_text:
                print(f"  ERROR: {err}")
                results.append({"name": source["name"], "category": cat, "error": err, "is_fresh": False})
            else:
                save_state(state)
                results.append({
                    "name": source["name"],
                    "category": cat,
                    "summary": holdings_text,
                    "is_fresh": False,
                    "latest_date": filing_date,
                    "latest_link": filing_url,
                })
            continue
        elif stype in ("rss", "podcast"):
            new_items, fallback, err = fetch_rss(source, state, cutoff)
        else:
            new_items, fallback, err = fetch_web(source, state, cutoff)

        # For podcasts, try to enrich items with full transcripts
        if stype == "podcast":
            ua = "MorningDigest/1.0 contact@morningdigest.dev"
            targets = new_items if new_items else ([fallback] if fallback else [])
            for item in targets:
                if item and item.get("id") and len(item.get("content", "")) < 2000:
                    transcript = _fetch_podcast_transcript(source["url"], item["id"], ua)
                    if transcript:
                        item["content"] = transcript

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
            # No new items — use cached summary, or generate one from fallback item
            cached = state["summaries"].get(source["name"])
            fb_title = fallback['title'] if fallback else 'unknown'
            fb_date = fallback.get('pub_date', '?') if fallback else '?'

            if not cached and fallback and fallback.get("content"):
                # No cached summary yet — summarize the latest available item
                print(f"  No new items. Summarizing latest: {fb_title} ({fb_date})")
                try:
                    summary = summarize(client, source["name"], [fallback])
                    cached = {"text": summary, "date": fb_date}
                    state["summaries"][source["name"]] = cached
                    save_state(state)
                except Exception as e:
                    print(f"  Summarization error: {e}")
            else:
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
