# Morning Digest Pipeline

## Overview

Automated daily content aggregator that fetches RSS feeds, web pages, podcast feeds, SEC 13F filings, and House financial disclosures, summarizes new items with an LLM, and publishes a static HTML digest to GitHub Pages. Optionally sends an email summary.

**Live page:** https://2310renhe.github.io/morning-digest/
**Repo:** https://github.com/2310renhe/morning-digest

## Architecture

```
sources.json          -- list of 30 sources with name, url, type, category
    |
daily_digest.py       -- main script: fetch -> dedupe -> summarize -> render -> email
    |
    |-- state.json    -- persistent seen-IDs + last-item-per-source (committed by CI)
    |-- index.html    -- generated digest page (committed by CI)
    |-- archive/      -- one HTML file per day + index.html
    |
.github/workflows/daily-digest.yml  -- cron at 6AM PDT, or manual dispatch
```

## Key Files

| File | Purpose |
|------|---------|
| `daily_digest.py` | Main pipeline (~1200+ lines). Fetching, summarization, HTML generation, email. |
| `sources.json` | Source definitions. Each entry has `name`, `url`, `type`, `category`. |
| `state.json` | Persisted between runs. Contains `seen` (set of item IDs, capped at 15k), `last_items` (per-source metadata), and `summaries` (cached rendered text + market data). |
| `index.html` | Generated output. Categorized card layout with dark mode support. |
| `.github/workflows/daily-digest.yml` | GitHub Actions workflow. Runs daily at 13:00 UTC (6AM PDT). |
| `requirements.txt` | Python deps: groq, feedparser, requests, beautifulsoup4, lxml, python-dotenv, pypdf, yfinance. |

## Source Categories

| Category | Sources | Description |
|----------|---------|-------------|
| AI Opinion Leaders | 12 | Blogs, newsletters, GitHub repos (Karpathy, SemiAnalysis, Zvi, Scott Alexander, Ethan Mollick, Simon Willison, Dario Amodei, Jack Clark, Andrew Ng, Garry Tan) |
| Investor Positioning | 7 | Cross-investor summary card + SEC 13F filings (SALP, Druckenmiller, Coatue, Tiger Global) + House PTR/FD (Pelosi) |
| Institutional Views | 4 | Podcasts (Odd Lots, Goldman Sachs Exchanges, BlackRock The Bid, Money Stuff) |
| Tech & AI Podcasts | 4 | Podcasts (All-In, AI+a16z, Lex Fridman, Latent Space) |
| Research | 4 | Arxiv feeds (q-fin.TR, q-fin.PM, q-fin.ST, q-fin.CP) |

## Investor Positioning Pipeline

### Source types

| Type | Handler | Description |
|------|---------|-------------|
| `sec13f` | `fetch_13f()` | Parses SEC EDGAR 13F-HR XML. Detects value scale (dollars vs thousands). Resolves tickers via CUSIP→OpenFIGI, then name search fallback. Fetches YTD + fwd P/E daily via `_fetch_market_data()`. |
| `congress_trades` | `fetch_congress_trades()` | Scrapes House clerk (CSRF-token POST). Parses Annual FD PDF (Dec 31 portfolio snapshot) + all PTR PDFs (transactions). Infers current position per ticker with status labels. |
| `investor_summary` | `fetch_investor_summary()` | Deferred pass — runs after all individual investor sources. Reads structured `positions` lists from state cache. Outputs two tables: top 10 consensus positions + top 10 YTD performers across all investors. |

### Ticker resolution for 13F

1. **CUSIP → OpenFIGI** (`_cusips_to_tickers`): Batch POST to `api.openfigi.com/v3/mapping` (10 CUSIPs per request, no API key needed). Prefers US-exchange equity tickers. Most reliable for ADRs, ETFs, class-labeled holdings.
2. **Name search → Yahoo Finance** (`_name_to_ticker`): Strips `"– CLASS DESCRIPTION"` suffix, queries `query2.finance.yahoo.com/v1/finance/search`. Prefers symbols without exchange suffixes on NASDAQ/NYSE.

### Market data (`_fetch_market_data`)

- **YTD return**: Yahoo Finance chart API (`v8/finance/chart/{ticker}?interval=1mo`) — `chartPreviousClose` (Dec 31) vs `regularMarketPrice`. One request per ticker.
- **Forward P/E**: `yfinance` library (`Ticker.info["forwardPE"]`). Handles Yahoo auth transparently. Negative forward P/E (loss-making companies) shown as `—`.
- Both are cached in `state["summaries"][key]["market_data"]` keyed by `ytd_date`. Refreshed daily even when the underlying filing is unchanged.

### House PTR / Annual FD (`fetch_congress_trades`)

- **CSRF**: GET `disclosures-clerk.house.gov/FinancialDisclosure/ViewSearch` to extract `__RequestVerificationToken`, then POST to `ViewMemberSearchResult`.
- **Annual FD** (type `""`): Filed by May 15 each year, covers holdings as of Dec 31 of the prior year. Parsed via `pypdf` + regex to extract ticker, asset code, value range.
- **PTR** (type `"P"`): Filed within 45 days of a trade. Parsed for ticker, action (Purchase / Sale / Partial Sale), date, amount range.
- **Inference**: FD holdings = Dec 31 baseline. PTRs since that date applied chronologically. Status per ticker: `• Unchanged / ↑ Added / ↓ Reduced / ✗ Closed / ↑ New (PTR) / ~ Active trading`.
- **Amount math**: House disclosures give value ranges (e.g. `$1,000,001–$5,000,000`). Midpoints used for arithmetic; original range strings displayed.
- **Cache key**: `congress_{source_name}`. Early return skipped if `ytd_date != today` so market data refreshes daily.

### State cache schema (investor sources)

```json
{
  "summaries": {
    "13f_Stanley Druckenmiller – Duquesne Family Office 13F (SEC)": {
      "text": "..rendered markdown table..",
      "date": "2026-02-17",
      "filing_url": "https://www.sec.gov/...",
      "ticker_map": {"Natera Inc": "NTRA", ...},
      "market_data": {"NTRA": {"ytd": "-13.1%", "ytd_pct": -13.1, "fwd_pe": "—"}, ...},
      "ytd_date": "2026-05-11",
      "positions": [{"ticker": "NTRA", "name": "Natera Inc", "value_k": 575300, "ytd": "-13.1%", "ytd_pct": -13.1, "fwd_pe": "—"}, ...]
    },
    "congress_Nancy Pelosi – House PTR Trades": {
      "text": "..rendered markdown table..",
      "date": "01/16/2026",
      "fd_doc_id": "20250081",
      "ptr_ids": ["20250123", ...],
      "market_data": {...},
      "ytd_date": "2026-05-11",
      "positions": [{"ticker": "NVDA", "name": "NVIDIA Corporation", "value_k": 16700, ...}, ...]
    }
  }
}
```

## RSS Fetch Strategy (4 levels)

The `fetch_rss()` function has a 4-level fallback chain to handle various feed issues:

1. **Level 1 — feedparser + custom UA:** Standard feedparser with `MorningDigest/1.0 contact@morningdigest.dev` User-Agent. Needed because SEC EDGAR requires email-format UA, and Cloudflare blocks feedparser's default.

2. **Level 2 — raw bytes + `_clean_xml()`:** Fetches raw bytes, strips invalid XML 1.0 control chars and fixes unescaped ampersands, then re-parses with feedparser.

3. **Level 3 — BeautifulSoup fallback (`_bs4_parse_feed`):** Manually extracts `<entry>` / `<item>` tags using html.parser, lxml, or lxml-xml parsers. Handles malformed XML that feedparser can't recover.

4. **Level 4 — Feedly public API:** Calls `https://cloud.feedly.com/v3/streams/contents?streamId=feed%2F{url}&count=20` (no auth needed). Feedly caches feeds from its own servers, so Cloudflare IP blocks on GitHub Actions datacenter IPs don't apply. Timestamps are in milliseconds.

## Known Issues & Workarounds

### Yahoo Finance API rate limits
- **Problem:** `v7/finance/quote` and `v10/finance/quoteSummary` return 401/429 from GitHub Actions IPs.
- **Solution:** Use `yfinance` library for forward P/E (handles cookie/crumb auth internally). Use `v8/finance/chart` directly for YTD (still works unauthenticated).

### Cloudflare blocks on GitHub Actions
- **Problem:** Substack (e.g., Zvi Mowshowitz) blocks datacenter IPs with Cloudflare challenge pages, returning 403 with "Just a moment..." HTML. This affects both `/feed` and `/api/v1/posts` endpoints.
- **Solution:** Level 4 Feedly API fallback. Feedly fetches from its own non-datacenter IPs.
- **Exception:** SemiAnalysis redirects to custom domain `newsletter.semianalysis.com` which does NOT have Cloudflare blocking, so it works directly at Level 1.

### SEC EDGAR 403s
- **Problem:** SEC EDGAR requires a User-Agent with contact email; blocks generic UAs.
- **Solution:** Custom UA string `MorningDigest/1.0 contact@morningdigest.dev` passed via `feedparser.parse(url, request_headers={"User-Agent": ua})`.

### Web source deduplication
- Web sources (type `"web"`) use a SHA-256 content hash of the page text. If the hash hasn't changed, no new item is generated and the stored fallback is shown instead.

## HTML Output

The `build_html()` function generates a categorized page with:
- **Nav pills** at top with anchor links to each category, showing fresh-item counts as green badges
- **Category sections** with colored left-border headers
- **Source cards** — fresh sources get a green left border; stale sources are dimmed (0.6 opacity)
- **Dark mode** via `prefers-color-scheme: dark` CSS media query with custom properties
- Lightweight markdown-to-HTML conversion (`md_to_html_simple()`) for LLM summaries

## Summarization

Uses Groq API (`llama-3.3-70b-versatile` by default). Max 3000 output tokens per source.

**Reader profile (in `READER_PROFILE` constant):** Tailored for a hedge fund PM (quant macro + AI/semiconductor stocks). The LLM is prompted to:
- EXTRACT only — state what the source actually says, never infer or speculate
- Be comprehensive — capture ALL key arguments, data points, and named entities
- For academic papers: skip those unrelated to quant investing or AI/ML for finance; for relevant ones, state research question, method, and key finding

**Content limits:** Full content stored at fetch time (up to 20k chars), but capped at 12k chars when sent to the LLM to stay within Groq's free-tier rate limit (100k tokens/day across all 30 sources).

**Summary persistence:** Summaries are cached in `state.json` under `summaries[source_name]`. When a source has no new items, the cached summary is displayed (dimmed). If no cached summary exists, the most recent available item is summarized on the spot, so every source always shows context.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Groq API key for LLM summarization |
| `EMAIL_FROM` | No | Gmail sender address |
| `EMAIL_TO` | No | Recipient address |
| `EMAIL_PASSWORD` | No | Gmail app password |
| `SMTP_HOST` | No | Default: `smtp.gmail.com` |
| `SMTP_PORT` | No | Default: `587` |
| `LOOKBACK_HOURS` | No | Default: `26` (hours to look back for new items) |
| `GROQ_MODEL` | No | Default: `llama-3.3-70b-versatile` |

## Adding a New Congress Member

Add an entry to `sources.json` in the `"Investor Positioning"` category:

```json
{
  "name": "Jane Smith – House PTR Trades",
  "url": "https://disclosures-clerk.house.gov/FinancialDisclosure/ViewSearch",
  "type": "congress_trades",
  "category": "Investor Positioning",
  "politician_last": "Smith",
  "politician_first": "Jane",
  "politician_state": "NY",
  "politician_district": "12"
}
```

The `_SHORT` dict in `fetch_investor_summary()` should also be updated with a display name for the summary card.

## Local Development

```bash
# Install deps
pip install -r requirements.txt

# Create .env with at minimum GROQ_API_KEY
cp .env.example .env

# Run locally
python daily_digest.py

# Preview the generated page
python3 -m http.server 8123
# Then open http://localhost:8123
```

## CLI Shortcuts

```bash
# Trigger a CI run manually
gh workflow run daily-digest.yml

# Watch the latest run
gh run watch

# Check recent runs
gh run list -L 5
```
