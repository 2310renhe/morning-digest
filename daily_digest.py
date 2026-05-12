#!/usr/bin/env python3
"""Daily content digest — fetch new items, summarize with Groq, publish to GitHub Pages."""

import hashlib
import io
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


def _fetch_market_data(tickers: list, session: requests.Session) -> dict:
    """Batch-fetch YTD return + forward P/E for a list of tickers.

    Returns dict: ticker -> {"ytd": "+12.3%", "ytd_pct": 12.3, "fwd_pe": "24.5x"}
    Uses Yahoo Finance quote API (batch, for fwd P/E) + chart API (per-ticker, for YTD).
    """
    from datetime import datetime as _dt
    import time as _time

    result = {t: {"ytd": "—", "ytd_pct": None, "fwd_pe": "—"} for t in tickers}
    if not tickers:
        return result

    # ── Step 1: forward P/E via yfinance (handles Yahoo auth transparently) ──
    try:
        import yfinance as _yf
        for ticker in tickers:
            try:
                info = _yf.Ticker(ticker).info
                fpe = info.get("forwardPE")
                # Ignore negative forward P/E (loss-making companies)
                if fpe and fpe > 0:
                    result[ticker]["fwd_pe"] = f"{fpe:.1f}x"
            except Exception:
                pass
            _time.sleep(0.05)
    except ImportError:
        pass  # yfinance not installed; fwd_pe stays "—"

    # ── Step 2: chart API → YTD (Dec 31 close vs today) ─────────────────────
    now = _dt.now()
    period1 = int(_dt(now.year, 1, 1).timestamp())
    period2 = int(now.timestamp())
    for ticker in tickers:
        try:
            r = session.get(
                f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                params={"period1": period1, "period2": period2, "interval": "1mo"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=8,
            )
            meta = r.json()["chart"]["result"][0]["meta"]
            prev = meta.get("chartPreviousClose") or meta.get("previousClose")
            cur  = meta.get("regularMarketPrice")
            if prev and cur and prev > 0:
                pct = (cur - prev) / prev * 100
                result[ticker]["ytd_pct"] = pct
                result[ticker]["ytd"] = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"
        except Exception:
            pass
        _time.sleep(0.1)

    return result


def _cusips_to_tickers(cusips: list, session: requests.Session) -> dict:
    """Batch-resolve CUSIPs to primary US equity tickers via OpenFIGI API (no key needed).
    Returns dict of cusip -> ticker string. Missing/failed entries are absent.
    """
    import time as _time
    _US_EXCH = {"US", "UN", "UA", "UQ", "UR", "UP", "UW", "UT"}  # US equity exchange codes
    result = {}
    batch_size = 10  # OpenFIGI free-tier limit per request
    for i in range(0, len(cusips), batch_size):
        batch = cusips[i:i + batch_size]
        body = [{"idType": "ID_CUSIP", "idValue": c} for c in batch]
        try:
            r = session.post(
                "https://api.openfigi.com/v3/mapping",
                json=body,
                headers={"Content-Type": "application/json"},
                timeout=12,
            )
            for cusip, item in zip(batch, r.json()):
                data = item.get("data", [])
                # Prefer US exchange + Equity
                for d in data:
                    if (d.get("marketSector") == "Equity"
                            and d.get("exchCode", "") in _US_EXCH
                            and d.get("ticker")):
                        result[cusip] = d["ticker"]
                        break
                else:
                    # Fallback: any equity ticker
                    for d in data:
                        if d.get("marketSector") == "Equity" and d.get("ticker"):
                            result[cusip] = d["ticker"]
                            break
        except Exception:
            pass
        _time.sleep(0.4)
    return result


def _name_to_ticker(name: str, session: requests.Session) -> str:
    """Resolve a company name to its primary US-listed ticker via Yahoo Finance search.
    Strips "– CLASS DESCRIPTION" suffixes added by _holding_name before searching.
    Prefers plain symbols without exchange suffixes (no '.') on NASDAQ/NYSE/NYSEArca.
    Returns '' on failure or if no clean US ticker found.
    """
    import re as _re
    _US_EXCHANGES = {"NMS", "NYQ", "PCX", "NGM", "ASE", "BTS"}
    # Strip "– SUFFIX" appended by _holding_name for non-generic class labels
    clean = _re.sub(r'\s*[–\-]\s*.+$', '', name).strip()
    for query in ([clean, name] if clean != name else [name]):
        try:
            r = session.get(
                "https://query2.finance.yahoo.com/v1/finance/search",
                params={"q": query, "quotesCount": 5, "newsCount": 0, "enableFuzzyQuery": "false"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=8,
            )
            quotes = r.json().get("quotes", [])
            for q in quotes:
                sym = q.get("symbol", "")
                exch = q.get("exchange", "")
                if sym and "." not in sym and (exch in _US_EXCHANGES or not exch):
                    return sym
            for q in quotes:
                sym = q.get("symbol", "")
                if sym and "." not in sym:
                    return sym
        except Exception:
            pass
    return ""


def fetch_13f(source: dict, state: dict) -> tuple:
    """
    Parse SEC EDGAR 13F-HR filing for a fund.
    Returns (holdings_text, filing_date, filing_url, error).
    holdings_text is a markdown table of top holdings with % of portfolio + YTD return.
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

        # Check if we already processed this filing AND YTD is fresh (today)
        from datetime import date as _date
        today_str = str(_date.today())
        cache_key = f"13f_{source['name']}"
        cached = state.get("summaries", {}).get(cache_key)
        if (cached and cached.get("filing_url") == filing_url
                and cached.get("ytd_date") == today_str):
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

        # Patterns that are generic share-class labels, not meaningful fund names
        import re as _re2
        _GENERIC_CLASS = _re2.compile(
            r'^(COM( NEW| STK)?|SHS( NEW)?|CAP STK( CL [A-Z])?|CL [A-Z] (COM|SHS)|'
            r'ORD( SHS)?|ADR|UNIT|DEPOSITARY SHS?|TR UNIT|NEW)$', _re2.I
        )

        def _holding_name(issuer: str, cls: str) -> str:
            """Combine issuer + titleOfClass when class carries real information (ETF series etc.)."""
            issuer = issuer.strip().title()
            cls = cls.strip().upper() if cls else ""
            if cls and not _GENERIC_CLASS.match(cls):
                return f"{issuer} – {cls}"
            return issuer

        holdings = []
        for info in soup3.find_all("infotable"):
            name = info.find("nameofissuer")
            value = info.find("value")
            shares = info.find("sshprnamt")
            cls_tag = info.find("titleofclass")
            cusip_tag = info.find("cusip")
            if name and value:
                try:
                    raw_val = int(value.text.strip().replace(",", ""))
                except ValueError:
                    continue
                holdings.append({
                    "name": _holding_name(name.text, cls_tag.text if cls_tag else ""),
                    "raw_val": raw_val,
                    "shares": int(shares.text.strip().replace(",", "")) if shares else 0,
                    "cusip": cusip_tag.get_text(strip=True) if cusip_tag else "",
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
        n_all = len(holdings)
        top = holdings[:20]

        # ── Ticker resolution ─────────────────────────────────────────────────
        import time as _time

        # Load cached ticker map if this is the same filing (avoids re-querying OpenFIGI)
        cached_tmap  = cached.get("ticker_map", {}) if cached and cached.get("filing_url") == filing_url else {}
        cached_mdata = cached.get("market_data", {}) if cached and cached.get("ytd_date") == today_str else {}

        mkt_session = requests.Session()

        # CUSIP → ticker via OpenFIGI for uncached holdings
        uncached_top = [h for h in top if h["name"] not in cached_tmap]
        cusip_map = {}
        if uncached_top:
            cusips = [h["cusip"] for h in uncached_top if h.get("cusip")]
            if cusips:
                cusip_map = _cusips_to_tickers(cusips, mkt_session)

        ticker_map = {}
        for h in top:
            name_key = h["name"]
            if name_key in cached_tmap:
                ticker = cached_tmap[name_key]
            else:
                ticker = cusip_map.get(h.get("cusip", ""), "")
                if not ticker:
                    ticker = _name_to_ticker(name_key, mkt_session)
                    _time.sleep(0.15)
            ticker_map[name_key] = ticker
            h["ticker"] = ticker

        # ── Market data (YTD + fwd P/E), refreshed daily ─────────────────────
        all_tickers = [h["ticker"] for h in top if h.get("ticker")]
        missing_tickers = [t for t in all_tickers if t not in cached_mdata]
        market_data = dict(cached_mdata)
        if missing_tickers:
            market_data.update(_fetch_market_data(missing_tickers, mkt_session))

        def _fmt_value(value_k: int) -> str:
            if value_k >= 1_000_000:
                return f"${value_k / 1_000_000:.2f}B"
            return f"${value_k / 1_000:.1f}M"

        # ── Build markdown table ───────────────────────────────────────────────
        lines = [
            f"**{n_all} positions · ${total_b:.1f}B AUM · as of {filing_date}**\n",
            "| # | Name | Ticker | Value | % of Portfolio | YTD | Fwd P/E |",
            "|---|------|--------|-------|----------------|-----|---------|",
        ]
        positions_data = []
        for i, h in enumerate(top, 1):
            pct     = h["value_k"] / total_k * 100
            val_str = _fmt_value(h["value_k"])
            ticker  = h.get("ticker", "")
            mdata   = market_data.get(ticker, {}) if ticker else {}
            ytd_str = mdata.get("ytd", "—")
            fpe_str = mdata.get("fwd_pe", "—")
            lines.append(f"| {i} | {h['name']} | {ticker} | {val_str} | {pct:.1f}% | {ytd_str} | {fpe_str} |")
            positions_data.append({
                "ticker": ticker, "name": h["name"], "value_k": h["value_k"],
                "ytd": ytd_str, "ytd_pct": mdata.get("ytd_pct"), "fwd_pe": fpe_str,
            })

        text = "\n".join(lines)

        state.setdefault("summaries", {})[cache_key] = {
            "text": text, "date": filing_date, "filing_url": filing_url,
            "ticker_map": ticker_map, "market_data": market_data,
            "ytd_date": today_str, "positions": positions_data,
        }

        return text, filing_date, filing_url, None

    except Exception as e:
        return None, None, None, str(e)


def fetch_congress_trades(source: dict, state: dict) -> tuple:
    """
    Fetch House PTR + Annual FD filings and infer real-time holdings.
    Returns (text, latest_date, filing_url, error).

    Strategy:
      1. Fetch most recent Annual FD → baseline holdings as of Dec 31 of that year.
      2. Fetch all PTR filings from that year+1 onwards → transaction flow.
      3. Apply PTRs chronologically to infer current position status per ticker.
      4. Show inferred holdings table + recent transaction log.
    Caches separately for FD doc IDs and PTR doc IDs; re-parses only on new filings.
    """
    import re as _re
    from datetime import datetime as _dt, date as _date
    try:
        import pypdf as _pypdf
    except ImportError:
        return None, None, None, "pypdf not installed (add to requirements.txt)"

    ua = "MorningDigest/1.0 contact@morningdigest.dev"
    base = "https://disclosures-clerk.house.gov"
    cache_key = f"congress_{source['name']}"
    cached = state.get("summaries", {}).get(cache_key, {})

    # ── helpers ──────────────────────────────────────────────────────────────

    def _parse_date(d: str):
        try:
            return _dt.strptime(d, "%m/%d/%Y")
        except Exception:
            return _dt.min

    def _amount_midpoint(s: str) -> float:
        """Return midpoint dollar value for a House disclosure amount range string."""
        nums = [float(x.replace(",", "")) for x in _re.findall(r"[\d,]+", s)]
        if len(nums) >= 2:
            return (nums[0] + nums[1]) / 2
        if len(nums) == 1:
            return nums[0]
        return 0.0

    def _fmt_dollars(v: float) -> str:
        if v >= 1_000_000:
            return f"${v/1_000_000:.1f}M"
        if v >= 1_000:
            return f"${v/1_000:.0f}K"
        return f"${v:.0f}"

    def _parse_ptr_pdf(pdf_bytes: bytes) -> list:
        """Parse PTR PDF → list of trade dicts."""
        reader = _pypdf.PdfReader(io.BytesIO(pdf_bytes))
        text = _re.sub(r'[ \t]+', ' ', "".join(p.extract_text() or "" for p in reader.pages))
        pat = _re.compile(
            r'(?:SP\s+)?(.+?)\(([A-Z0-9./: ^]+)\)\s*\[([A-Z]+)\]'
            r'\s*(P|S)(\s*\(partial\))?'
            r'\s+(\d{2}/\d{2}/\d{4})\d{2}/\d{2}/\d{4}'
            r'\s*(\$[\d,]+\s*-\s*\$[\d,]+)',
            _re.DOTALL,
        )
        trades = []
        for m in pat.finditer(text):
            code = m.group(3)
            raw_action = m.group(4)
            partial = bool(m.group(5))
            kind = "Options" if code == "OP" else "Stock" if code == "ST" else code
            action = ("Purchase" if raw_action == "P"
                      else ("Partial Sale" if partial else "Sale"))
            trades.append({
                "ticker": m.group(2).strip(),
                "kind": kind, "action": action,
                "date": m.group(6),
                "amount": _re.sub(r'\s+', ' ', m.group(7)).strip(),
            })
        return trades

    def _parse_fd_pdf(pdf_bytes: bytes) -> dict:
        """Parse Annual FD PDF → {ticker: {name, lo_str, hi_str, mid, kind}} for equities/options."""
        reader = _pypdf.PdfReader(io.BytesIO(pdf_bytes))
        text = _re.sub(r'[ \t]+', ' ', "".join(p.extract_text() or "" for p in reader.pages))
        pat = _re.compile(
            r'(.+?)\(([A-Z0-9./: ^]+)\)\s*\[([A-Z]+)\]\s*(?:SP|JT)\s+'
            r'(\$[\d,]+)\s*-\s*\n?\s*(\$[\d,]+)',
            _re.MULTILINE,
        )
        holdings = {}
        for m in pat.finditer(text):
            code = m.group(3)
            if code not in ("ST", "OP"):
                continue
            ticker = m.group(2).strip()
            name = _re.sub(r'\s+', ' ', m.group(1)).strip().rstrip('-').strip()
            lo_str, hi_str = m.group(4), m.group(5)
            mid = (_amount_midpoint(lo_str) + _amount_midpoint(hi_str)) / 2
            kind = "Options" if code == "OP" else "Stock"
            holdings[ticker] = {"name": name, "lo": lo_str, "hi": hi_str,
                                 "mid": mid, "kind": kind}
        return holdings

    def _infer_holdings(fd_holdings: dict, ptr_trades: list, fd_cutoff: _dt) -> list:
        """
        Apply PTR transactions after fd_cutoff to FD baseline.
        Returns list of position dicts sorted by estimated value desc.
        """
        # positions: {ticker: {kind, fd_mid, fd_range, adjustments[], status}}
        positions = {}
        for ticker, h in fd_holdings.items():
            positions[ticker] = {
                "ticker": ticker, "name": h["name"], "kind": h["kind"],
                "fd_mid": h["mid"], "fd_range": f"{h['lo']}–{h['hi']}",
                "adjustments": [],   # list of (date, action, amount_str, mid)
                "closed": False,
            }

        # apply PTRs chronologically
        for t in sorted(ptr_trades, key=lambda x: _parse_date(x["date"])):
            if _parse_date(t["date"]) <= fd_cutoff:
                continue
            ticker = t["ticker"]
            if ticker not in positions:
                positions[ticker] = {
                    "ticker": ticker, "name": ticker, "kind": t["kind"],
                    "fd_mid": 0.0, "fd_range": "—",
                    "adjustments": [], "closed": False,
                }
            pos = positions[ticker]
            mid = _amount_midpoint(t["amount"])
            pos["adjustments"].append((t["date"], t["action"], t["amount"], mid))
            if t["action"] == "Sale":
                pos["closed"] = True
            elif t["action"] in ("Purchase",):
                pos["closed"] = False   # re-opened or added

        # compute inferred value and status label for each position
        results = []
        for ticker, pos in positions.items():
            adj = pos["adjustments"]
            buys   = sum(a[3] for a in adj if "Purchase" in a[1])
            sells  = sum(a[3] for a in adj if "Sale" in a[1])
            latest_date = adj[-1][0] if adj else None

            inferred_mid = pos["fd_mid"] + buys - sells

            if pos["closed"] and inferred_mid <= 0:
                status = "✗ Closed"
            elif pos["fd_mid"] == 0:
                status = "↑ New (PTR)"
            elif sells > 0 and buys > 0:
                status = "~ Active trading"
            elif sells > 0:
                status = "↓ Reduced" if not pos["closed"] else "✗ Closed"
            elif buys > 0:
                status = "↑ Added"
            else:
                status = "• Unchanged"

            results.append({
                "ticker": ticker,
                "name": pos["name"],
                "kind": pos["kind"],
                "fd_range": pos["fd_range"],
                "inferred_mid": max(inferred_mid, 0.0),
                "status": status,
                "latest_date": latest_date,
                "adj_count": len(adj),
            })

        results.sort(key=lambda x: x["inferred_mid"], reverse=True)
        return results

    # ── main logic ───────────────────────────────────────────────────────────
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": ua})

        # Get CSRF token
        resp = session.get(f"{base}/FinancialDisclosure/ViewSearch", timeout=15)
        token = (BeautifulSoup(resp.text, "html.parser")
                 .find("input", {"name": "__RequestVerificationToken"}) or {})
        token = token.get("value", "") if hasattr(token, "get") else ""

        current_year = _date.today().year

        def _search(doc_type, years):
            links = {}
            for year in years:
                r = session.post(
                    f"{base}/FinancialDisclosure/ViewMemberSearchResult",
                    data={"LastName": source.get("politician_last", ""),
                          "FirstName": source.get("politician_first", ""),
                          "State": source.get("politician_state", ""),
                          "District": source.get("politician_district", ""),
                          "FilingYear": year, "DocType": doc_type,
                          "__RequestVerificationToken": token},
                    timeout=15,
                )
                for a in BeautifulSoup(r.text, "html.parser").find_all("a", href=True):
                    href = a["href"]
                    if doc_type == "P" and "ptr-pdfs" in href:
                        doc_id = href.split("/")[-1].replace(".pdf", "")
                        links[doc_id] = href
                    elif doc_type == "" and "financial-pdfs" in href:
                        doc_id = href.split("/")[-1].replace(".pdf", "")
                        links[doc_id] = href
            return links

        # Fetch FD list (search last 3 years to find the most recent one)
        fd_links = _search("", [str(y) for y in range(current_year - 2, current_year + 1)])
        # Use the most recent FD
        fd_doc_id = sorted(fd_links.keys())[-1] if fd_links else None

        # Fetch PTR list (current + prior two years)
        ptr_links = _search("P", [str(y) for y in range(current_year - 2, current_year + 1)])

        # Check cache — skip PDF re-parse only if filings unchanged AND YTD is fresh (today)
        cached_fd   = cached.get("fd_doc_id")
        cached_ptrs = set(cached.get("ptr_ids", []))
        new_ptrs    = set(ptr_links) - cached_ptrs
        ytd_fresh   = cached.get("ytd_date") == str(_date.today())
        if (not new_ptrs and fd_doc_id == cached_fd and cached.get("text") and ytd_fresh):
            return cached["text"], cached.get("date"), None, None

        # Parse FD
        fd_holdings = {}
        fd_cutoff = _dt(2000, 1, 1)  # default: ancient date (apply all PTRs)
        fd_year_str = "unknown"
        if fd_doc_id:
            try:
                r = session.get(f"{base}/{fd_links[fd_doc_id]}", timeout=20)
                fd_holdings = _parse_fd_pdf(r.content)
                # Determine FD snapshot date = Dec 31 of the year BEFORE the filing year
                # (FD filed in year Y covers holdings as of Dec 31 of year Y-1)
                # Extract year from href path e.g. "public_disc/financial-pdfs/2025/..."
                m = _re.search(r'financial-pdfs/(\d{4})/', fd_links[fd_doc_id])
                filing_year = int(m.group(1)) if m else current_year
                fd_year = filing_year - 1
                fd_cutoff = _dt(fd_year, 12, 31)
                fd_year_str = str(fd_year)
                print(f"    FD parsed: {len(fd_holdings)} equity positions (snapshot: Dec 31 {fd_year})")
            except Exception as e:
                print(f"    FD parse error: {e}")

        # Parse all PTRs
        all_trades = []
        for doc_id, href in sorted(ptr_links.items()):
            try:
                r = session.get(f"{base}/{href}", timeout=20)
                all_trades.extend(_parse_ptr_pdf(r.content))
            except Exception:
                continue

        # Deduplicate PTR trades
        seen = set()
        unique_trades = []
        for t in all_trades:
            key = (t["ticker"], t["action"], t["date"], t["amount"])
            if key not in seen:
                seen.add(key)
                unique_trades.append(t)
        unique_trades.sort(key=lambda t: _parse_date(t["date"]), reverse=True)
        latest_ptr_date = unique_trades[0]["date"] if unique_trades else None

        # Infer current holdings
        positions = _infer_holdings(fd_holdings, unique_trades, fd_cutoff)

        # ── Market data: YTD + fwd P/E (refreshed daily) ─────────────────────
        today_str = str(_date.today())
        cached_mdata = cached.get("market_data", {}) if cached and cached.get("ytd_date") == today_str else {}

        active_positions = [p for p in positions if p["inferred_mid"] > 0]
        valid_tickers = [p["ticker"] for p in active_positions if p.get("ticker") and len(p["ticker"]) <= 6]
        missing_tickers = [t for t in valid_tickers if t not in cached_mdata]
        market_data = dict(cached_mdata)
        if missing_tickers:
            market_data.update(_fetch_market_data(missing_tickers, requests.Session()))

        # ── Build output (13F-style) ──────────────────────────────────────────
        name = (source.get("politician_first", "") + " " + source.get("politician_last", "")).strip()

        total_est = sum(p["inferred_mid"] for p in active_positions)
        total_str = _fmt_dollars(total_est)
        n_pos = len(active_positions)

        text_parts = [
            f"**{n_pos} positions · {total_str} est. portfolio · "
            f"FD: Dec 31 {fd_year_str} + PTR through {latest_ptr_date}**\n"
        ]

        text_parts.append("| # | Ticker | Name | Type | Est. Value | % | YTD | Fwd P/E | Status | Last PTR |")
        text_parts.append("|---|--------|------|------|------------|---|-----|---------|--------|----------|")
        positions_data = []
        for i, p in enumerate(active_positions, 1):
            pct       = p["inferred_mid"] / total_est * 100 if total_est else 0
            est       = _fmt_dollars(p["inferred_mid"])
            short_name = p["name"][:28] + "…" if len(p["name"]) > 28 else p["name"]
            last      = p["latest_date"] or "—"
            ticker    = p["ticker"]
            mdata     = market_data.get(ticker, {}) if ticker else {}
            ytd_str   = mdata.get("ytd", "—")
            fpe_str   = mdata.get("fwd_pe", "—")
            text_parts.append(
                f"| {i} | {ticker} | {short_name} | {p['kind']} "
                f"| {est} | {pct:.1f}% | {ytd_str} | {fpe_str} | {p['status']} | {last} |"
            )
            positions_data.append({
                "ticker": ticker, "name": p["name"],
                "value_k": int(p["inferred_mid"] / 1000),
                "ytd": ytd_str, "ytd_pct": mdata.get("ytd_pct"), "fwd_pe": fpe_str,
            })

        closed = [p for p in positions if p["inferred_mid"] <= 0 and p["status"].startswith("✗")]
        if closed:
            text_parts.append(f"\n*Closed since FD baseline: {', '.join(p['ticker'] for p in closed)}*")

        text = "\n".join(text_parts)

        state.setdefault("summaries", {})[cache_key] = {
            "text": text, "date": latest_ptr_date,
            "fd_doc_id": fd_doc_id, "ptr_ids": list(ptr_links.keys()),
            "market_data": market_data, "ytd_date": today_str,
            "positions": positions_data,
        }
        return text, latest_ptr_date, None, None

    except Exception as e:
        return None, None, None, str(e)


def fetch_investor_summary(state: dict, all_sources: list) -> str:
    """
    Generate a cross-investor summary card for the Investor Positioning section.
    Reads structured 'positions' lists from each investor's cached data in state.

    Returns markdown with two tables:
      1. Top 10 consensus positions (held by most investors, by $ weight)
      2. Top 10 best YTD performers across all tracked positions
    """
    from collections import defaultdict

    # Short display names for each source
    _SHORT = {
        "Situational Awareness LP – 13F Filings (SEC)": "SALP",
        "Stanley Druckenmiller – Duquesne Family Office 13F (SEC)": "Druckenmiller",
        "Philippe Laffont – Coatue Management 13F (SEC)": "Coatue",
        "Chase Coleman – Tiger Global Management 13F (SEC)": "Tiger Global",
        "Nancy Pelosi – House PTR Trades": "Pelosi",
    }

    # Collect all positions from cached investor data
    investor_data = {}   # short_name -> list of position dicts
    for source in all_sources:
        stype = source.get("type", "")
        if stype not in ("sec13f", "congress_trades"):
            continue
        prefix = "13f" if stype == "sec13f" else "congress"
        key = f"{prefix}_{source['name']}"
        cached = state.get("summaries", {}).get(key, {})
        positions = cached.get("positions", [])
        if positions:
            short = _SHORT.get(source["name"], source["name"].split("–")[0].strip())
            investor_data[short] = positions

    if not investor_data:
        return "*(Investor data not yet available — will populate after first full run)*"

    # Aggregate by ticker across all investors
    ticker_agg = defaultdict(lambda: {
        "names": [], "investors": [], "total_value_k": 0,
        "ytd_pct": None, "ytd": "—", "fwd_pe": "—",
    })
    for investor, positions in investor_data.items():
        for p in positions:
            t = p.get("ticker", "")
            if not t:
                continue
            agg = ticker_agg[t]
            agg["names"].append(p.get("name", t))
            agg["investors"].append(investor)
            agg["total_value_k"] += p.get("value_k", 0)
            if agg["ytd_pct"] is None and p.get("ytd_pct") is not None:
                agg["ytd_pct"] = p["ytd_pct"]
                agg["ytd"] = p.get("ytd", "—")
            if agg["fwd_pe"] == "—" and p.get("fwd_pe", "—") != "—":
                agg["fwd_pe"] = p["fwd_pe"]

    def _fmt_k(value_k: int) -> str:
        if value_k >= 1_000_000:
            return f"${value_k / 1_000_000:.1f}B"
        return f"${value_k / 1_000:.0f}M"

    n_investors = len(investor_data)
    tracked = ", ".join(investor_data.keys())
    lines = [f"**Cross-Investor Summary** · {n_investors} investors tracked: {tracked}\n"]

    # Table 1: Consensus positions (held by 2+ investors), ranked by # holders then $ weight
    consensus = sorted(
        ticker_agg.items(),
        key=lambda x: (-len(set(x[1]["investors"])), -x[1]["total_value_k"])
    )
    lines += [
        "**Top 10 Consensus Positions** *(ranked by number of investors holding)*\n",
        "| # | Ticker | Name | Holders | Combined $ | YTD | Fwd P/E |",
        "|---|--------|------|---------|------------|-----|---------|",
    ]
    count = 0
    for ticker, agg in consensus:
        holders = sorted(set(agg["investors"]))
        if len(holders) < 2:
            continue
        name = agg["names"][0][:28] + "…" if len(agg["names"][0]) > 28 else agg["names"][0]
        holders_str = ", ".join(holders)
        lines.append(
            f"| {count+1} | {ticker} | {name} | {len(holders)} ({holders_str}) "
            f"| {_fmt_k(agg['total_value_k'])} | {agg['ytd']} | {agg['fwd_pe']} |"
        )
        count += 1
        if count >= 10:
            break
    if count == 0:
        lines.append("| — | *(no overlapping positions yet)* | | | | | |")

    # Table 2: Top 10 YTD performers across all tracked positions
    with_ytd = [(t, agg) for t, agg in ticker_agg.items() if agg["ytd_pct"] is not None]
    with_ytd.sort(key=lambda x: x[1]["ytd_pct"], reverse=True)
    lines += [
        "\n**Top 10 YTD Performers** *(across all tracked positions)*\n",
        "| # | Ticker | Name | YTD | Fwd P/E | Held by |",
        "|---|--------|------|-----|---------|---------|",
    ]
    for rank, (ticker, agg) in enumerate(with_ytd[:10], 1):
        name = agg["names"][0][:28] + "…" if len(agg["names"][0]) > 28 else agg["names"][0]
        holders_str = ", ".join(sorted(set(agg["investors"])))
        lines.append(
            f"| {rank} | {ticker} | {name} | {agg['ytd']} | {agg['fwd_pe']} | {holders_str} |"
        )

    return "\n".join(lines)


def fetch_ai_trade_ideas(state: dict, all_sources: list, client) -> str:
    """
    Synthesize cached summaries from AI opinion leaders, podcasts, and institutional
    views into high-conviction trade ideas, using the Groq LLM.

    Input: all cached summaries for sources in the target categories.
    Output: markdown table of trade ideas + brief supporting notes per idea.
    Caches by SHA-256 of the combined input text; re-runs only when content changes.
    """
    import hashlib as _hashlib

    _TARGET_CATEGORIES = {"AI Opinion Leaders", "Tech & AI Podcasts", "Institutional Views"}

    # Collect all cached summaries for target categories, freshest sources first
    snippets = []
    for source in all_sources:
        if source.get("category") not in _TARGET_CATEGORIES:
            continue
        stype = source.get("type", "")
        if stype in ("investor_summary", "ai_synthesis"):
            continue
        name = source["name"]
        cached = state.get("summaries", {}).get(name, {})
        text = cached.get("text", "")
        if not text:
            continue
        date = cached.get("date", "")
        snippets.append((name, date, text))

    if not snippets:
        return "*(No source summaries available yet — will populate after first full run)*"

    # Build combined input, cap at 10k chars to stay within Groq rate limits
    combined_parts = []
    total = 0
    for name, date, text in snippets:
        header = f"### {name}{f' ({date})' if date else ''}\n"
        chunk = header + text[:1500]   # cap per-source at 1500 chars
        if total + len(chunk) > 10000:
            break
        combined_parts.append(chunk)
        total += len(chunk)
    combined = "\n\n---\n\n".join(combined_parts)

    # Cache check
    input_hash = _hashlib.sha256(combined.encode()).hexdigest()[:16]
    cache_key = "ai_trade_ideas"
    cached_out = state.get("summaries", {}).get(cache_key, {})
    if cached_out.get("input_hash") == input_hash and cached_out.get("text"):
        return cached_out["text"]

    # Call LLM
    TRADE_PROMPT = """You are a senior analyst at a quant macro hedge fund specialising in AI and semiconductor stocks.

Below are today's summaries from AI thought leaders, tech podcasts, and institutional research.

Your task: identify HIGH-CONVICTION LONG or SHORT trade ideas that are DIRECTLY AND SPECIFICALLY supported by the content below. Do not add outside knowledge or opinions not present in the source material.

Rules:
- Cite only what the sources explicitly state or strongly imply
- Prefer specific tickers over vague sector calls; include sector ETF if no single ticker fits
- Assign conviction: HIGH (multiple sources agree, or explicit bullish/bearish statement), MEDIUM (single strong source), LOW (inferential)
- 5–8 ideas maximum, ordered by conviction then expected magnitude
- For each idea: provide the specific evidence from the summaries that supports it

Output format — first a summary table, then one short supporting paragraph per idea:

**Table:**
| # | Asset | Ticker | Direction | Conviction | Source(s) |
|---|-------|--------|-----------|------------|-----------|
| 1 | ... | ... | LONG/SHORT | HIGH/MEDIUM/LOW | ... |

**Supporting notes:**
**1. [Asset]** — [one paragraph citing the specific evidence]
...
"""

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": TRADE_PROMPT},
                {"role": "user", "content": f"Source summaries:\n\n{combined}"},
            ],
        )
        text = resp.choices[0].message.content
    except Exception as e:
        text = f"*(Trade idea synthesis failed: {e})*"

    state.setdefault("summaries", {})[cache_key] = {
        "text": text,
        "input_hash": input_hash,
        "date": __import__("datetime").date.today().isoformat(),
    }
    return text


def fetch_github_repos(source: dict, state: dict, cutoff: datetime) -> tuple:
    """
    Specialised fetcher for GitHub profile ?tab=repositories pages.
    Parses individual repo cards (name, description, last-pushed datetime).
    Returns (new_items, fallback_item, error) where:
      - new_items contains repos that are brand-new OR were pushed within cutoff window
      - Each item's title distinguishes "New repo" vs "Updated repo"
    State key: last_items[url] = {repo_path: {"updated": iso_str, "desc": str}}
    """
    import re as _re
    headers = {"User-Agent": "Mozilla/5.0 (compatible; DailyDigestBot/1.0)"}
    try:
        resp = requests.get(source["url"], headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Parse all repo cards
        current: dict = {}
        for li in soup.find_all("li", attrs={"itemprop": "owns"}):
            a_tag = li.find("a", attrs={"itemprop": "name codeRepository"})
            desc_tag = li.find("p", attrs={"itemprop": "description"})
            time_tag = li.find("relative-time")
            if not a_tag:
                continue
            repo_path = a_tag.get("href", "").strip("/")   # e.g. "karpathy/nanochat"
            repo_name = a_tag.get_text(strip=True)
            desc = desc_tag.get_text(strip=True) if desc_tag else ""
            updated_iso = time_tag.get("datetime", "") if time_tag else ""
            if repo_path:
                current[repo_path] = {"name": repo_name, "desc": desc, "updated": updated_iso}

        if not current:
            return [], None, "No repos found on page"

        stored: dict = state["last_items"].get(source["url"] + ":repos", {})

        new_items = []
        most_recent_item = None
        most_recent_dt = None

        for repo_path, info in current.items():
            repo_url = f"https://github.com/{repo_path}"
            updated_iso = info["updated"]
            pub_dt = None
            pub_date = None
            if updated_iso:
                try:
                    pub_dt = datetime.fromisoformat(updated_iso.replace("Z", "+00:00"))
                    pub_date = pub_dt.strftime("%Y-%m-%d")
                except ValueError:
                    pass

            # Track most-recent for fallback display
            if pub_dt and (most_recent_dt is None or pub_dt > most_recent_dt):
                most_recent_dt = pub_dt
                most_recent_item = {
                    "title": info["name"],
                    "link": repo_url,
                    "pub_date": pub_date,
                    "content": info["desc"],
                }

            # "New repo" only meaningful if we have prior state to compare against.
            # On first run (stored empty), only report recently-updated repos.
            is_new_repo = bool(stored) and repo_path not in stored
            is_recently_updated = pub_dt and pub_dt >= cutoff

            if not is_new_repo and not is_recently_updated:
                continue

            kind = "New repo" if is_new_repo else "Updated"
            item_id = f"github:{repo_path}:{pub_date or 'unknown'}"
            if item_id in state["seen"]:
                continue

            new_items.append({
                "id": item_id,
                "title": f"[{kind}] {info['name']}",
                "link": repo_url,
                "pub_date": pub_date,
                "content": info["desc"],
            })
            state["seen"].add(item_id)

        # Persist current snapshot so next run can detect new repos
        state["last_items"][source["url"] + ":repos"] = {
            path: {"updated": d["updated"]} for path, d in current.items()
        }

        fallback = most_recent_item
        return new_items, fallback, None

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
        if stype in ("investor_summary", "ai_synthesis"):
            continue  # deferred — processed after all other sources
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

        if stype == "congress_trades":
            trades_text, latest_date, _, err = fetch_congress_trades(source, state)
            cat = source.get("category", "Other")
            if err and not trades_text:
                print(f"  ERROR: {err}")
                results.append({"name": source["name"], "category": cat, "error": err, "is_fresh": False})
            else:
                save_state(state)
                cache_key = f"congress_{source['name']}"
                cached_ids = state.get("summaries", {}).get(cache_key, {}).get("filing_ids", [])
                print(f"  {len(cached_ids)} PTR filings, latest: {latest_date}")
                results.append({
                    "name": source["name"],
                    "category": cat,
                    "summary": trades_text,
                    "is_fresh": False,
                    "latest_date": latest_date,
                })
            continue
        elif stype in ("rss", "podcast"):
            new_items, fallback, err = fetch_rss(source, state, cutoff)
        elif stype == "web" and "github.com" in source["url"] and "tab=repositories" in source["url"]:
            new_items, fallback, err = fetch_github_repos(source, state, cutoff)
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

    # ── Deferred pass: investor_summary + ai_synthesis ────────────────────────
    for source in sources:
        stype = source.get("type", "").lower()

        if stype == "investor_summary":
            print(f"[investor_summary] {source['name']} ...")
            summary_text = fetch_investor_summary(state, sources)
            cat = source.get("category", "Investor Positioning")
            results.insert(
                next((i for i, r in enumerate(results) if r.get("category") == cat), 0),
                {"name": source["name"], "category": cat,
                 "summary": summary_text, "is_fresh": True, "latest_date": date_str},
            )

        elif stype == "ai_synthesis":
            print(f"[ai_synthesis] {source['name']} ...")
            ideas_text = fetch_ai_trade_ideas(state, sources, client)
            save_state(state)
            cat = source.get("category", "AI Opinion Leaders")
            results.insert(
                next((i for i, r in enumerate(results) if r.get("category") == cat), 0),
                {"name": source["name"], "category": cat,
                 "summary": ideas_text, "is_fresh": True, "latest_date": date_str},
            )

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
