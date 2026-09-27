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
  - optional (config.ML_V2_NEWS_LLM_ENABLED): Claude reads the same
    headlines and returns a structured score. Falls back to the lexicon
    on any error, so an API problem can never block the bot.
"""
import json
import math
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


def fetch_headlines(ticker: str, lookback_hours: float, now: datetime = None,
                    market: bool = False) -> list:
    """[(published_utc, title, summary)] newer than lookback_hours and
    relevant: about this company (symbol feeds) or about the market /
    macro backdrop (market feeds)."""
    import yfinance as yf
    now = now or datetime.now(timezone.utc)
    try:
        raw = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out = []
    for item in raw:
        title, summary, ts = _parse_item(item)
        if not title or ts is None:
            continue
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


def _llm_scores(ticker: str, titles: list):
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
        data = json.loads(text)
        scores = [0.0] * len(titles)
        for s in data.get("scores", []):
            i = int(s.get("index", -1))
            if 0 <= i < len(titles):
                scores[i] = max(-1.0, min(1.0, float(s.get("score", 0.0))))
        return scores
    except Exception as e:
        print(f"[news] LLM scoring failed for {ticker}, using lexicon: {e!r}")
        return None


# ═══════════════════════════════════════════════════════════════════
# Aggregate
# ═══════════════════════════════════════════════════════════════════
_cache = {}   # (ticker, lookback) -> (fetched_at, CatalystRead)


def _aggregate(ticker: str, items: list, now: datetime) -> CatalystRead:
    if not items:
        return CatalystRead(method="none")
    titles = [f"{t}. {s}" if s else t for _, t, s in items]
    scores = None
    method = "lexicon"
    if config.ML_V2_NEWS_LLM_ENABLED:
        scores = _llm_scores(ticker, titles)
        if scores is not None:
            method = "llm"
    if scores is None:
        scores = [score_headline(t) for _, t, _ in items]

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


def read_catalyst(ticker: str, lookback_hours: float = None, market: bool = False) -> CatalystRead:
    """Cached catalyst read for one ticker."""
    lookback_hours = lookback_hours or config.ML_V2_NEWS_SYMBOL_LOOKBACK_HOURS
    key = (ticker, lookback_hours, market)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < config.ML_V2_NEWS_CACHE_SECONDS:
        return hit[1]
    now = datetime.now(timezone.utc)
    read = _aggregate(ticker, fetch_headlines(ticker, lookback_hours, now, market), now)
    _cache[key] = (time.time(), read)
    return read


def read_market_catalyst() -> CatalystRead:
    """Broad-market read from the index ETFs' own news feeds. The
    strongest reading (by magnitude) wins, so one clear macro shock in
    either feed isn't averaged away by the other feed's noise."""
    reads = [read_catalyst(t, config.ML_V2_NEWS_MARKET_LOOKBACK_HOURS, market=True)
             for t in config.ML_V2_NEWS_MARKET_TICKERS]
    reads = [r for r in reads if r.n_items]
    if not reads:
        return CatalystRead(method="none")
    return max(reads, key=lambda r: abs(r.score))


def entry_decision(symbol: str, direction: str):
    """(veto_reason_or_None, prob_adjustment, symbol_read, market_read).

    prob_adjustment is <= 0: subtracted from the probability floor when
    the catalyst is aligned with the trade."""
    sym = read_catalyst(symbol)
    mkt = read_market_catalyst()
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
