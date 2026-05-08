# Morning Digest Pipeline

## Overview

Automated daily content aggregator that fetches RSS feeds, web pages, and podcast feeds, summarizes new items with an LLM, and publishes a static HTML digest to GitHub Pages. Optionally sends an email summary.

**Live page:** https://2310renhe.github.io/morning-digest/
**Repo:** https://github.com/2310renhe/morning-digest

## Architecture

```
sources.json          -- list of 28 sources with name, url, type, category
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
| `daily_digest.py` | Main pipeline (~785 lines). Fetching, summarization, HTML generation, email. |
| `sources.json` | Source definitions. Each entry has `name`, `url`, `type` (rss/podcast/web), `category`. |
| `state.json` | Persisted between runs. Contains `seen` (set of item IDs, capped at 15k) and `last_items` (per-source metadata for fallback display). |
| `index.html` | Generated output. Categorized card layout with dark mode support. |
| `.github/workflows/daily-digest.yml` | GitHub Actions workflow. Runs daily at 13:00 UTC (6AM PDT). |
| `requirements.txt` | Python deps: groq, feedparser, requests, beautifulsoup4, lxml, python-dotenv. |

## Source Categories

| Category | Sources | Description |
|----------|---------|-------------|
| AI Opinion Leaders | 12 | Blogs, newsletters, GitHub repos (Karpathy, SemiAnalysis, Zvi, Scott Alexander, Ethan Mollick, Simon Willison, Dario Amodei, Jack Clark, Andrew Ng, Garry Tan) |
| Investor Positioning | 4 | SEC 13F filings (Situational Awareness LP, Druckenmiller, Coatue, Tiger Global) |
| Institutional Views | 4 | Podcasts (Odd Lots, Goldman Sachs Exchanges, BlackRock The Bid, Money Stuff) |
| Tech & AI Podcasts | 4 | Podcasts (All-In, AI+a16z, Lex Fridman, Latent Space) |
| Research | 4 | Arxiv feeds (q-fin.TR, q-fin.PM, q-fin.ST, q-fin.CP) |

## RSS Fetch Strategy (4 levels)

The `fetch_rss()` function has a 4-level fallback chain to handle various feed issues:

1. **Level 1 — feedparser + custom UA:** Standard feedparser with `MorningDigest/1.0 contact@morningdigest.dev` User-Agent. Needed because SEC EDGAR requires email-format UA, and Cloudflare blocks feedparser's default.

2. **Level 2 — raw bytes + `_clean_xml()`:** Fetches raw bytes, strips invalid XML 1.0 control chars and fixes unescaped ampersands, then re-parses with feedparser.

3. **Level 3 — BeautifulSoup fallback (`_bs4_parse_feed`):** Manually extracts `<entry>` / `<item>` tags using html.parser, lxml, or lxml-xml parsers. Handles malformed XML that feedparser can't recover.

4. **Level 4 — Feedly public API:** Calls `https://cloud.feedly.com/v3/streams/contents?streamId=feed%2F{url}&count=20` (no auth needed). Feedly caches feeds from its own servers, so Cloudflare IP blocks on GitHub Actions datacenter IPs don't apply. Timestamps are in milliseconds.

## Known Issues & Workarounds

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

Uses Groq API (`llama-3.3-70b-versatile` by default) to summarize new items per source. Max 1500 tokens per source. Content is truncated to 2500 chars before sending.

**Reader profile (in `READER_PROFILE` constant):** Tailored for a hedge fund PM running quant macro strategies with personal AI/semiconductor stock investments. The LLM is prompted to surface:
- Market-moving signals (earnings, guidance, capex, supply chain)
- Positioning implications for specific names (NVDA, MSFT, GOOGL, META, AMD, ASML, TSM, etc.)
- Macro read-throughs (rates, trade policy, datacenter capex cycles, power constraints)
- AI ecosystem shifts (model capabilities, inference costs, open/closed dynamics, regulation)
- Quantitative angles (new alpha signals, microstructure, data availability)

**Summary persistence:** Summaries are cached in `state.json` under `summaries[source_name]`. When a source has no new items, the cached summary from its last update is displayed (dimmed) so every source always shows context, not just a bare title.

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
