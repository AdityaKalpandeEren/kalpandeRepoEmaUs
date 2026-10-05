"""
Live news-catalyst reader for model L_ML_META_V2.

Answers one question for a symbol at a moment in time: "is there fresh,
clearly good or clearly bad news for this stock - or for the market as a
whole - right now?" as a score in [-1, +1] (negative = bearish).

Used in two places, LIVE ONLY:
  - entry: a strong opposing catalyst vetoes the trade (bad news vetoes
    a long, good news vetoes a short); an aligned catalyst lowers the
    V2 probability bar slightly (config.ML_V2_NEWS_*).
  - open-trade monitoring: fresh opposing news on a symbol you are
    already in raises an exit alert (see live_ml_v2.py).

Why live only: Yahoo serves just the latest ~10 headlines per ticker,
with no archive. There is no way to know what the headlines said on a
past date, so this cannot be backtested or trained on - see the
config.ML_V2_* block comment. Every backtest number reported for V2 is
therefore WITHOUT the news overlay; its live effect is unmeasured until
paper/live results accumulate.

Scoring:
  - default: a finance keyword/phrase lexicon (no API key, no cost).
    Deliberately conservative - it only reacts to unambiguous catalyst
    language (beats/misses, guidance, upgrades/downgrades, probes,
    offerings, bankruptcy...) and ignores opinion-piece noise.
  - optional (config.ML_V2_NEWS_LLM_ENABLED): an LLM - Google Gemini
    (free tier, default) or Claude - reads the same headlines and returns
    a structured score. Falls back to the lexicon on a missing key or any
    error, so an API problem can never block the bot.
"""
import hashlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field

import config


@dataclass
class CatalystRead:
    score: float = 0.0            # -1 (very bearish) .. +1 (very bullish)
    n_items: int = 0              # fresh headlines considered
    top_headline: str = ""        # the one that moved the score most
    method: str = "none"          # "lexicon" | "llm" | "none"
    headlines: list = field(default_factory=list)

    @property
    def summary(self) -> str:
        if not self.n_items:
            return "no fresh news"
        return f"{self.score:+.2f} ({self.n_items} items, {self.method}): {self.top_headline[:90]}"


# ═══════════════════════════════════════════════════════════════════
# Lexicon. Weights are per-phrase; a headline's score is the clipped
# sum. Multi-word phrases are listed so e.g. "cuts guidance" isn't read
# as neutral, and "beats" isn't matched inside unrelated words.
# ═══════════════════════════════════════════════════════════════════
_BULLISH = {
    r"beats? (estimates|expectations|forecasts?|consensus)": 0.8,
    r"tops? (estimates|expectations|forecasts?)": 0.7,
    r"(raises|lifts|boosts|hikes) (its )?(guidance|forecast|outlook)": 0.9,
    r"record (revenue|sales|profit|quarter)": 0.6,
    r"upgrade[sd]? (to )?(buy|outperform|overweight)": 0.7,
    r"\bupgrade[sd]?\b": 0.4,
    r"price target (raised|hiked|increased|boosted)": 0.4,
    r"(raises|lifts|hikes) price target": 0.4,
    r"(fda|regulatory) approv": 0.8,
    r"\bapproval\b": 0.3,
    r"(buyback|share repurchase)": 0.5,
    r"(dividend (hike|increase)|raises dividend)": 0.4,
    r"(to acquire|agrees to buy|takeover bid|buyout offer)": 0.5,
    r"(wins|awarded|secures|lands) .{0,30}(contract|deal|order)": 0.5,
    r"(strategic )?partnership with": 0.3,
    r"\b(surges?|soars?|jumps?|rall(y|ies)|skyrockets?)\b": 0.4,
    r"\bstrong demand\b": 0.4,
    r"(rate cut|cuts rates)": 0.4,
    r"(cooler|softer)[- ]than[- ]expected (inflation|cpi)": 0.5,
    r"\bstimulus\b": 0.3,
    r"(trade deal|tariff relief|tariffs? (paused|lifted|eased))": 0.5,
}
_BEARISH = {
    r"miss(es|ed)? (estimates|expectations|forecasts?|consensus)": -0.8,
    r"(cuts|lowers|slashes|trims|withdraws) (its )?(guidance|forecast|outlook)": -0.9,
    r"(weak|disappointing|soft) (guidance|outlook|forecast|results|quarter)": -0.7,
    r"profit warning": -0.9,
    r"downgrade[sd]? (to )?(sell|underperform|underweight|neutral|hold)": -0.7,
    r"\bdowngrade[sd]?\b": -0.4,
    r"price target (cut|lowered|reduced|slashed)": -0.4,
    r"(cuts|lowers|slashes) price target": -0.4,
    r"(sec|doj|ftc|antitrust) (probe|investigation|lawsuit|charges|sues)": -0.8,
    r"\b(probe|investigation|subpoena)\b": -0.5,
    r"\b(lawsuit|sued|class action)\b": -0.4,
    r"\brecall(s|ed)?\b": -0.5,
    r"\b(fda rejects?|rejection|complete response letter|clinical hold)\b": -0.8,
    r"(secondary|stock|share|equity) offering": -0.6,
    r"\b(dilution|dilutive)\b": -0.5,
    r"(bankruptcy|chapter 11|going concern|default(s|ed)? on)": -1.0,
    r"\b(layoffs?|job cuts)\b": -0.3,
    r"(ceo|cfo) (resigns|steps down|ousted|departs)": -0.5,
    r"\b(accounting|restatement|fraud|short seller|short report)\b": -0.8,
    r"\b(plunges?|plummets?|tumbles?|sinks?|crash(es)?|sell-?off|slumps?)\b": -0.5,
    r"\bdata breach\b|\bhack(ed)?\b|\boutage\b": -0.4,
    r"(hotter|higher)[- ]than[- ]expected (inflation|cpi|ppi)": -0.6,
    r"(rate hike|hikes rates)": -0.4,
    r"(new |more )?tariffs?\b(?! relief)": -0.3,
    r"\b(export ban|export controls?|sanctions?)\b": -0.5,
    r"\b(recession|shutdown|downturn)\b": -0.4,
    r"\b(war|missiles?|airstrikes?|(drone|missile) strikes?|invasion|escalat\w*)\b": -0.4,
}
_BULL_RX = [(re.compile(p, re.I), w) for p, w in _BULLISH.items()]
_BEAR_RX = [(re.compile(p, re.I), w) for p, w in _BEARISH.items()]


# Headlines that are commentary, listicles or explainers rather than a
# new event - scored 0 whatever words they contain ("60 years of market
# crashes" is not a crash).
_OPINION_RX = re.compile(
    r"^(why|here'?s|is|are|should|what|how|can|could|will|would|do|does|\d+ )\b"
    r"|\?|stocks? to (buy|sell|watch)|should you|buy (now|today)|millionaire|forever"
    r"|(my|our) (top|favorite)|prediction|opinion|here'?s (why|what|how)",
    re.I)

# A market-feed headline only counts if it is actually about the market
# or macro backdrop, not one stock that happened to land in the SPY feed.
_MACRO_RX = re.compile(
    r"\b(stocks|markets?|wall street|s&p|nasdaq|dow|fed|fomc|powell|inflation|cpi|ppi|"
    r"tariffs?|treasur(y|ies)|yields?|economy|economic|recession|jobs report|payrolls|"
    r"unemployment|gdp|shutdown|rate (cut|hike)s?|geopolitic\w*|war|oil prices?)\b",
    re.I)

_NAME_SUFFIX_RX = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|ltd|plc|holdings?|group|"
    r"platforms|technologies|technology|systems|class [a-c]|the)\b\.?|[,.]", re.I)
_names = {}


def _company_tokens(ticker: str) -> list:
    """Words that identify a company in a headline: its ticker plus the
    distinctive part of its name (e.g. META -> ['META', 'Meta'])."""
    if ticker in _names:
        return _names[ticker]
    tokens = [ticker]
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).get_info() or {}
        name = info.get("shortName") or info.get("longName") or ""
        core = _NAME_SUFFIX_RX.sub(" ", name).split()
        if core and len(core[0]) >= 3:
            tokens.append(core[0])
    except Exception:
        pass
    _names[ticker] = tokens
    return tokens


def is_relevant(text: str, ticker: str) -> bool:
    for tok in _company_tokens(ticker):
        if tok == ticker:
            # Bare tickers are matched case-sensitively and only when long
            # enough not to be an ordinary word ('ON', 'IT', 'ALL'); short
            # ones must appear as (T) or $T.
            if re.search(rf"[($]{re.escape(tok)}\b", text):
                return True
            if len(tok) >= 3 and re.search(rf"\b{re.escape(tok)}\b", text):
                return True
        elif re.search(rf"\b{re.escape(tok)}", text, re.I):
            return True
    return False


def score_headline(text: str) -> float:
    """Lexicon score of one headline TITLE, clipped to [-1, 1]. Titles
    only: summaries add more unrelated vocabulary than signal."""
    if not text:
        return 0.0
    if _OPINION_RX.search(text):
        return 0.0
    s = 0.0
    for rx, w in _BULL_RX:
        if rx.search(text):
            s += w
    for rx, w in _BEAR_RX:
        if rx.search(text):
            s += w
    return max(-1.0, min(1.0, s))


# ═══════════════════════════════════════════════════════════════════
# Fetch
# ═══════════════════════════════════════════════════════════════════

def _parse_item(item: dict):
    """Yahoo's news payload has changed shape before; handle both the
    current {'content': {...}} form and the older flat form."""
    c = item.get("content", item) if isinstance(item, dict) else {}
    title = c.get("title") or ""
    summary = c.get("summary") or c.get("description") or ""
    pub = c.get("pubDate") or c.get("displayTime")
    ts = None
    if pub:
        try:
            ts = datetime.fromisoformat(str(pub).replace("Z", "+00:00"))
        except ValueError:
            ts = None
    elif c.get("providerPublishTime"):
        ts = datetime.fromtimestamp(int(c["providerPublishTime"]), tz=timezone.utc)
    return title, summary, ts


_GNEWS_SOURCE_RX = re.compile(r"\s+-\s+[^-]{2,60}$")   # "Title - Publisher"


def _yahoo_search_items(ticker: str) -> list:
    """Yahoo news via the search endpoint. yf.Ticker(t).news (the
    /xhr/ncp endpoint) has returned HTTP 500 on every call since
    ~2026-09-28 midday ET, so every read came back empty."""
    import yfinance as yf
    try:
        raw = yf.Search(ticker, news_count=config.ML_V2_NEWS_FETCH_COUNT,
                        max_results=0, raise_errors=False).news or []
    except Exception as e:
        print(f"[news] Yahoo search failed for {ticker}: {e!r}"[:200])
        return []
    return [_parse_item(item) for item in raw]


def _google_news_items(ticker: str, lookback_hours: float, market: bool) -> list:
    """Google News RSS search - the fallback when Yahoo returns nothing."""
    import requests
    from email.utils import parsedate_to_datetime
    from urllib.parse import quote_plus
    import xml.etree.ElementTree as ET
    days = max(1, math.ceil(lookback_hours / 24))
    q = f"stock market when:{days}d" if market else f"{ticker} stock when:{days}d"
    url = f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en-US&gl=US&ceid=US:en"
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        print(f"[news] Google News failed for {ticker}: {e!r}"[:200])
        return []
    out = []
    for item in root.iter("item"):
        title = _GNEWS_SOURCE_RX.sub("", (item.findtext("title") or "").strip())
        try:
            ts = parsedate_to_datetime(item.findtext("pubDate") or "")
            ts = ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        except Exception:
            ts = None
        out.append((title, "", ts))
    return out[: config.ML_V2_NEWS_FETCH_COUNT]


def fetch_headlines(ticker: str, lookback_hours: float, now: datetime = None,
                    market: bool = False) -> list:
    """[(published_utc, title, summary)] newer than lookback_hours and
    relevant: about this company (symbol feeds) or about the market /
    macro backdrop (market feeds). Yahoo search first, Google News RSS
    when Yahoo has nothing fresh and relevant."""
    now = now or datetime.now(timezone.utc)
    out, source = _filter_headlines(_yahoo_search_items(ticker), ticker, lookback_hours, now, market), "yahoo"
    if not out and config.ML_V2_NEWS_GOOGLE_FALLBACK:
        out, source = _filter_headlines(_google_news_items(ticker, lookback_hours, market),
                                        ticker, lookback_hours, now, market), "google"
    print(f"[news] {ticker}: {len(out)} fresh headline(s) ({source if out else 'none found'})")
    return out


def _filter_headlines(parsed: list, ticker: str, lookback_hours: float, now: datetime,
                      market: bool) -> list:
    out, seen = [], set()
    for title, summary, ts in parsed:
        if not title or ts is None or title in seen:
            continue
        seen.add(title)
        age_h = (now - ts).total_seconds() / 3600.0
        if not (0 <= age_h <= lookback_hours):
            continue
        # Relevance is judged on the TITLE only: Yahoo summaries routinely
        # name other companies ("...alongside Meta and Nvidia"), which let
        # unrelated stories leak into a symbol's read.
        if market:
            if not _MACRO_RX.search(title):
                continue
        elif not is_relevant(title, ticker):
            continue
        out.append((ts, title, summary))
    return sorted(out, key=lambda x: x[0], reverse=True)


# ═══════════════════════════════════════════════════════════════════
# Optional LLM scoring (Claude)
# ═══════════════════════════════════════════════════════════════════
_llm_client = None

_LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "score": {"type": "number"},
                },
                "required": ["index", "score"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["scores"],
    "additionalProperties": False,
}

_LLM_SYSTEM = (
    "You score financial news headlines for their likely effect on a stock's price over "
    "the next few trading hours. For each numbered headline return a score from -1.0 "
    "(clearly bearish catalyst: earnings miss, guidance cut, downgrade, investigation, "
    "offering, recall, macro shock) to +1.0 (clearly bullish catalyst: beat-and-raise, "
    "upgrade, approval, big contract, buyback). Opinion pieces, listicles, recaps of old "
    "moves, and anything not about a new event score 0. Be conservative: only a genuine, "
    "new, material event deserves a magnitude above 0.5."
)


_GEMINI_SCHEMA = {   # Gemini's responseSchema is an OpenAPI subset (no additionalProperties)
    "type": "OBJECT",
    "properties": {
        "scores": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {"index": {"type": "INTEGER"}, "score": {"type": "NUMBER"}},
                "required": ["index", "score"],
            },
        },
    },
    "required": ["scores"],
}


def _parse_scores(text: str, n: int) -> list:
    data = json.loads(text)
    scores = [0.0] * n
    for s in data.get("scores", []):
        i = int(s.get("index", -1))
        if 0 <= i < n:
            scores[i] = max(-1.0, min(1.0, float(s.get("score", 0.0))))
    return scores


def _llm_scores(ticker: str, titles: list):
    """Per-headline scores from the configured LLM, or None on any failure
    (the caller then uses the lexicon)."""
    if config.ML_V2_NEWS_LLM_PROVIDER == "gemini":
        return _gemini_scores(ticker, titles)
    return _claude_scores(ticker, titles)


def _llm_error_detail(resp) -> str:
    """Compact, complete error: status, message and which quota was hit
    (per-minute vs per-day) with Google's suggested retry delay."""
    try:
        err = resp.json().get("error", {})
    except Exception:
        return resp.text[:1000]
    parts = [f"{err.get('status', '')}: {err.get('message', '')}"]
    for d in err.get("details", []):
        for v in d.get("violations", []):
            parts.append(f"quota={v.get('quotaId') or v.get('quotaMetric')} limit={v.get('quotaValue', '?')}")
        if d.get("retryDelay"):
            parts.append(f"retryDelay={d['retryDelay']}")
    return " | ".join(parts)[:1000]


def _gemini_scores(ticker: str, titles: list):
    """Google Gemini via its REST API (free tier key from aistudio.google.com)."""
    global _llm_blocked
    import requests
    key = config.GEMINI_API_KEY
    if not key:
        return None
    listing = "\n".join(f"{i}. {t}" for i, t in enumerate(titles))
    body = {
        "systemInstruction": {"parts": [{"text": _LLM_SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": f"Ticker: {ticker}\nHeadlines:\n{listing}"}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json",
                             "responseSchema": _GEMINI_SCHEMA},
    }
    # Main model first; on an overload (500/503) the fallback model once.
    fallback = config.ML_V2_NEWS_LLM_FALLBACK_MODEL
    models = [config.ML_V2_NEWS_LLM_MODEL] + ([fallback] if fallback and fallback != config.ML_V2_NEWS_LLM_MODEL else [])
    for i, model in enumerate(models):
        try:
            resp = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                 headers={"x-goog-api-key": key}, timeout=20, json=body)
            if resp.status_code == 200:
                parts = resp.json()["candidates"][0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                return _parse_scores(text, len(titles)) if text else None
        except Exception as e:
            print(f"[news] Gemini scoring failed for {ticker} ({model}), using lexicon: {e!r}")
            return None
        if resp.status_code in (500, 503) and i + 1 < len(models):
            print(f"[news] Gemini {resp.status_code} on {model} for {ticker} - trying {models[i + 1]}")
            continue
        if resp.status_code in (429, 500, 503):
            # Rate limit / overload on every model: more calls this run would
            # fail the same way - lexicon for the rest of this process.
            _llm_blocked = True
            print(f"[news] Gemini {resp.status_code} on {model} for {ticker} - LLM paused for the rest "
                  f"of this run, using lexicon. {_llm_error_detail(resp)}")
        else:
            print(f"[news] Gemini {resp.status_code} on {model} for {ticker}, using lexicon: "
                  f"{_llm_error_detail(resp)}")
        return None
    return None


def _claude_scores(ticker: str, titles: list):
    """Per-headline scores from Claude, or None on any failure."""
    global _llm_client
    try:
        import anthropic
    except ImportError:
        return None
    try:
        if _llm_client is None:
            _llm_client = anthropic.Anthropic(timeout=20.0, max_retries=1)
        listing = "\n".join(f"{i}. {t}" for i, t in enumerate(titles))
        response = _llm_client.beta.messages.create(
            model=config.ML_V2_NEWS_LLM_MODEL,
            max_tokens=2048,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": _LLM_SCHEMA},
            },
            system=_LLM_SYSTEM,
            messages=[{"role": "user", "content": f"Ticker: {ticker}\nHeadlines:\n{listing}"}],
        )
        if response.stop_reason == "refusal":
            return None
        text = next((b.text for b in response.content if b.type == "text"), None)
        if not text:
            return None
        return _parse_scores(text, len(titles))
    except Exception as e:
        print(f"[news] LLM scoring failed for {ticker}, using lexicon: {e!r}")
        return None


# ═══════════════════════════════════════════════════════════════════
# Persistent LLM score cache
#
# scan_once.py is a fresh process every run (every ~2 min on GitHub), so
# the in-memory cache below never survives between runs - without this,
# every run re-sent the same headlines to the LLM and the free-tier quota
# ran out within minutes. A headline's score doesn't change, so each
# (provider, model, ticker, headline) is scored ONCE and kept in
# config.LIVE_STATE_DIR (which the workflow carries between runs).
# ═══════════════════════════════════════════════════════════════════
_SCORE_CACHE_FILE = "news_llm_scores.json"
_SCORE_CACHE_DAYS = 3           # headlines older than the lookbacks are never asked again
_score_cache = None             # key -> [score, scored_at_epoch]
_llm_blocked = False            # set on 429/503: rest of this process uses the lexicon


def _score_cache_path() -> str:
    os.makedirs(config.LIVE_STATE_DIR, exist_ok=True)
    return os.path.join(config.LIVE_STATE_DIR, _SCORE_CACHE_FILE)


def _load_score_cache() -> dict:
    global _score_cache
    if _score_cache is None:
        try:
            with open(_score_cache_path()) as f:
                data = json.load(f)
        except Exception:
            data = {}
        cutoff = time.time() - _SCORE_CACHE_DAYS * 86400
        _score_cache = {k: v for k, v in data.items()
                        if isinstance(v, list) and len(v) == 2 and v[1] >= cutoff}
    return _score_cache


def _save_score_cache():
    try:
        path = _score_cache_path()
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(_score_cache, f)
        os.replace(tmp, path)
    except Exception as e:
        print(f"[news] could not save LLM score cache: {e!r}")


def _score_key(ticker: str, text: str) -> str:
    raw = f"{config.ML_V2_NEWS_LLM_PROVIDER}|{config.ML_V2_NEWS_LLM_MODEL}|{ticker}|{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def _cached_llm_scores(ticker: str, texts: list) -> list:
    """LLM score per text, or None where there is none (never scored and
    the LLM is unavailable). Only headlines never scored before are sent."""
    cache = _load_score_cache()
    keys = [_score_key(ticker, t) for t in texts]
    out = [cache[k][0] if k in cache else None for k in keys]
    todo = [i for i, s in enumerate(out) if s is None]
    if todo and not _llm_blocked:
        fresh = _llm_scores(ticker, [texts[i] for i in todo])
        if fresh is not None:
            now = time.time()
            for i, s in zip(todo, fresh):
                out[i] = s
                cache[keys[i]] = [s, now]
            _save_score_cache()
    return out


# ═══════════════════════════════════════════════════════════════════
# Aggregate
# ═══════════════════════════════════════════════════════════════════
_cache = {}   # (ticker, lookback) -> (fetched_at, CatalystRead)


def _aggregate(ticker: str, items: list, now: datetime, use_llm: bool = True) -> CatalystRead:
    if not items:
        return CatalystRead(method="none")
    titles = [f"{t}. {s}" if s else t for _, t, s in items]
    lexicon = [score_headline(t) for _, t, _ in items]
    scores = lexicon
    method = "lexicon"
    if use_llm and config.ML_V2_NEWS_LLM_ENABLED:
        llm = _cached_llm_scores(ticker, titles)
        n_llm = sum(s is not None for s in llm)
        if n_llm:
            # LLM score where one exists, lexicon for any headline the LLM
            # couldn't score this run (quota) - same veto/boost rules either way.
            scores = [s if s is not None else x for s, x in zip(llm, lexicon)]
            method = "llm" if n_llm == len(llm) else "llm+lexicon"

    # Recency weighting: a 1-hour-old headline counts ~2x a 6-hour-old one.
    num = den = 0.0
    best_i, best_mag = 0, -1.0
    for i, ((ts, title, _), sc) in enumerate(zip(items, scores)):
        age_h = max(0.0, (now - ts).total_seconds() / 3600.0)
        w = math.exp(-age_h / 6.0)
        if sc != 0.0:
            num += w * sc
            den += w
        if abs(sc) * w > best_mag:
            best_i, best_mag = i, abs(sc) * w
    # Averaging only over NON-neutral headlines, then shrinking by how many
    # there were: one strong headline is a signal, but it is not as strong
    # a signal as three agreeing ones.
    avg = num / den if den else 0.0
    n_signal = sum(1 for s in scores if s != 0.0)
    confidence = 1.0 - math.exp(-n_signal / 1.5)
    score = max(-1.0, min(1.0, avg * confidence))
    return CatalystRead(score=round(score, 3), n_items=len(items),
                        top_headline=items[best_i][1], method=method,
                        headlines=[t for _, t, _ in items[:5]])


def read_catalyst(ticker: str, lookback_hours: float = None, market: bool = False,
                  use_llm: bool = True) -> CatalystRead:
    """Cached catalyst read for one ticker. use_llm=False scores with the
    keyword lexicon only - never calls the LLM (L_ML_META_V2_2)."""
    lookback_hours = lookback_hours or config.ML_V2_NEWS_SYMBOL_LOOKBACK_HOURS
    key = (ticker, lookback_hours, market, use_llm)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < config.ML_V2_NEWS_CACHE_SECONDS:
        return hit[1]
    now = datetime.now(timezone.utc)
    items = _headlines_cached(ticker, lookback_hours, now, market)
    read = _aggregate(ticker, items, now, use_llm)
    _cache[key] = (time.time(), read)
    return read


_headline_cache = {}   # (ticker, lookback, market) -> (fetched_at, items)


def _headlines_cached(ticker, lookback_hours, now, market):
    """One headline fetch per ticker per cache window, shared by the LLM and
    lexicon reads so V2 and V2_2 don't fetch the same feed twice."""
    key = (ticker, lookback_hours, market)
    hit = _headline_cache.get(key)
    if hit and time.time() - hit[0] < config.ML_V2_NEWS_CACHE_SECONDS:
        return hit[1]
    items = fetch_headlines(ticker, lookback_hours, now, market)
    _headline_cache[key] = (time.time(), items)
    return items


def read_market_catalyst(use_llm: bool = True) -> CatalystRead:
    """Broad-market read from the index ETFs' own news feeds. The
    strongest reading (by magnitude) wins, so one clear macro shock in
    either feed isn't averaged away by the other feed's noise."""
    reads = [read_catalyst(t, config.ML_V2_NEWS_MARKET_LOOKBACK_HOURS, market=True, use_llm=use_llm)
             for t in config.ML_V2_NEWS_MARKET_TICKERS]
    reads = [r for r in reads if r.n_items]
    if not reads:
        return CatalystRead(method="none")
    return max(reads, key=lambda r: abs(r.score))


def entry_decision(symbol: str, direction: str, use_llm: bool = True):
    """(veto_reason_or_None, prob_adjustment, symbol_read, market_read).

    prob_adjustment is <= 0: subtracted from the probability floor when
    the catalyst is aligned with the trade."""
    sym = read_catalyst(symbol, use_llm=use_llm)
    mkt = read_market_catalyst(use_llm=use_llm)
    sign = 1.0 if direction == "long" else -1.0
    s_al, m_al = sign * sym.score, sign * mkt.score
    if s_al <= -config.ML_V2_NEWS_VETO:
        return f"opposing {symbol} news {sym.summary}", 0.0, sym, mkt
    if m_al <= -config.ML_V2_NEWS_MARKET_VETO:
        return f"opposing market news {mkt.summary}", 0.0, sym, mkt
    adj = -config.ML_V2_NEWS_BOOST_PROB if s_al >= config.ML_V2_NEWS_BOOST else 0.0
    return None, adj, sym, mkt


def exit_check(symbol: str, direction: str):
    """Reason string if fresh news now argues for closing an open trade."""
    sym = read_catalyst(symbol)
    mkt = read_market_catalyst()
    sign = 1.0 if direction == "long" else -1.0
    if sign * sym.score <= -config.ML_V2_NEWS_EXIT:
        return f"bad {symbol} catalyst: {sym.summary}"
    if sign * mkt.score <= -config.ML_V2_NEWS_EXIT:
        return f"bad market catalyst: {mkt.summary}"
    return None


# ═══════════════════════════════════════════════════════════════════
# Decision log: every live news check V2 / V2_2 make (veto, boost, plain
# trade or skip) with the scores and the headline behind them - vetoed
# candidates otherwise leave no trace, so whether news ever protects a
# trade could not be measured. live_state/reports/ is uploaded with the
# day's paper-trading artifact.
# ═══════════════════════════════════════════════════════════════════
DECISION_FIELDS = ["logged_at", "candle", "model", "symbol", "direction", "setup", "prob", "base_bar",
                   "final_bar", "decision", "veto_reason", "news", "news_method", "news_items", "news_headline",
                   "mkt", "mkt_method", "mkt_items", "mkt_headline"]


def _decisions_path() -> str:
    d = os.path.join(config.LIVE_STATE_DIR, "reports")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "news_decisions.csv")


def log_decision(model: str, symbol: str, direction: str, candle, setup: str, prob: float,
                 base_bar: float, veto, adj: float, sym: CatalystRead, mkt: CatalystRead) -> str:
    """Record one news-checked candidate; returns the decision label:
    VETO | SKIP (below the bar even after news) | TRADE_BOOSTED (only
    trades because aligned news lowered the bar) | TRADE."""
    final_bar = base_bar + adj
    if veto:
        decision = "VETO"
    elif prob < final_bar:
        decision = "SKIP"
    elif prob < base_bar:
        decision = "TRADE_BOOSTED"
    else:
        decision = "TRADE"
    print(f"[news-decision] {model} {symbol} {direction} {setup} p={prob:.2f} bar {base_bar:.2f}->{final_bar:.2f} "
          f"news {sym.score:+.2f} ({sym.method}, {sym.n_items}) mkt {mkt.score:+.2f} ({mkt.method}) -> {decision}"
          + (f": {veto}" if veto else ""), flush=True)
    row = {"logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "candle": str(candle),
           "model": model, "symbol": symbol, "direction": direction, "setup": setup, "prob": round(prob, 4),
           "base_bar": round(base_bar, 4), "final_bar": round(final_bar, 4), "decision": decision,
           "veto_reason": veto or "", "news": sym.score, "news_method": sym.method, "news_items": sym.n_items,
           "news_headline": sym.top_headline[:200], "mkt": mkt.score, "mkt_method": mkt.method,
           "mkt_items": mkt.n_items, "mkt_headline": mkt.top_headline[:200]}
    try:
        import csv
        path = _decisions_path()
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=DECISION_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
    except Exception as e:
        print(f"[news] could not log decision: {e!r}")
    return decision


def decisions_summary(day: str) -> str:
    """EOD report block: per model, how news changed today's candidates."""
    path = _decisions_path()
    if not os.path.exists(path):
        return ""
    try:
        import pandas as pd
        d = pd.read_csv(path)
    except Exception:
        return ""
    d = d[d["candle"].astype(str).str[:10] == day]
    if d.empty:
        return ""
    lines = ["📰 NEWS DECISIONS today (candidates near the bar):"]
    for model, g in d.groupby("model"):
        c = g["decision"].value_counts()
        llm = (g["news_method"].astype(str).str.startswith("llm")).mean() * 100 if model == "L_ML_META_V2" else None
        lines.append(f"{model}: {len(g)} checked | TRADE {c.get('TRADE', 0)} | BOOSTED {c.get('TRADE_BOOSTED', 0)} | "
                     f"VETO {c.get('VETO', 0)} | SKIP {c.get('SKIP', 0)}"
                     + (f" | LLM-scored {llm:.0f}%" if llm is not None else ""))
        for _, v in g[g["decision"] == "VETO"].head(5).iterrows():
            lines.append(f"   veto {v['symbol']} p={v['prob']:.2f}: {str(v['veto_reason'])[:120]}")
    return "\n".join(lines)
