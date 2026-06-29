"""V2 Roadmap — four-stage autonomous A/B backtest comparison.

Stages
------
1. Price momentum confirmation
2. Earnings surprise confirmation
3. Bayesian (EWM) sentiment smoothing
4. Markowitz mean-variance position sizing

Each stage scores FinBERT once per window (cached to data/cache/),
then runs two (or more) backtests from the same scored data.

Results are written to:
  logs/v2_roadmap.log         — detailed progress log
  logs/v2_roadmap_summary.md  — per-stage summary (one section per stage)
"""
from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE     = ROOT / "logs" / "v2_roadmap.log"
SUMMARY_FILE = ROOT / "logs" / "v2_roadmap_summary.md"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logging.getLogger("src").setLevel(logging.INFO)
_log = logging.getLogger("v2_roadmap")
logging.getLogger("v2_roadmap").setLevel(logging.DEBUG)


def log(msg: str = "") -> None:
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def append_summary(section: str) -> None:
    with open(SUMMARY_FILE, "a") as f:
        f.write(section + "\n\n")


# ── Imports ───────────────────────────────────────────────────────────────────
from src.backtest.runner import BacktestRunner, _load_ticker_csv
from src.ingestion.alpha_vantage import EarningsCache
from src.portfolio import MarkowitzOptimizer, PortfolioConfig
from src.sentiment.scorer import SentimentScorer, aggregate_daily, smooth_daily_scores
from src.strategy.signals import SignalGenerator

# ── Config ────────────────────────────────────────────────────────────────────
TICKERS        = ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "AMD"]
PORTFOLIO_TOML = ROOT / "portfolio.toml"
INITIAL_CASH   = 100_000.0
CACHE_DIR      = ROOT / "data" / "cache"
EARNINGS_DIR   = ROOT / "data" / "earnings"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = [
    {
        "label":     "W1: 2025-12-17→2026-03-17",
        "gdelt_dir": ROOT / "data" / "gdelt_prev",
        "price_dir": ROOT / "data" / "prices_prev",
        "cache":     CACHE_DIR / "prev_scored.parquet",
    },
    {
        "label":     "W2: 2026-03-20→2026-06-22",
        "gdelt_dir": ROOT / "data" / "gdelt",
        "price_dir": ROOT / "data" / "prices",
        "cache":     CACHE_DIR / "current_scored.parquet",
    },
]


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_gdelt(gdelt_dir: Path) -> list[dict]:
    articles: list[dict] = []
    for ticker in TICKERS:
        path = gdelt_dir / f"{ticker}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        articles.extend(data)
    return articles


def get_scored(win: dict) -> pd.DataFrame:
    """Return per-article FinBERT scores, using parquet cache when available."""
    if win["cache"].exists():
        log(f"    [cache] Loading {win['cache'].name}")
        return pd.read_parquet(win["cache"])
    articles = load_gdelt(win["gdelt_dir"])
    log(f"    Scoring {len(articles):,} headlines with FinBERT …")
    scorer = SentimentScorer(device=-1)
    scored = scorer.score(articles)
    scored.to_parquet(win["cache"], index=False)
    log(f"    Cached to {win['cache'].name}")
    return scored


def load_prices(price_dir: Path) -> dict[str, pd.DataFrame]:
    ohlcv: dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        df = _load_ticker_csv(price_dir, ticker)
        if df is not None:
            ohlcv[ticker] = df
    return ohlcv


def load_earnings_cache() -> EarningsCache | None:
    raw: dict[str, list[dict]] = {}
    for ticker in TICKERS:
        path = EARNINGS_DIR / f"{ticker}.json"
        if path.exists():
            raw[ticker] = json.loads(path.read_text(encoding="utf-8"))
    if not raw:
        return None
    return EarningsCache(raw)


def signals_df(signals: list) -> pd.DataFrame:
    if not signals:
        return pd.DataFrame(columns=["timestamp", "ticker", "signal"])
    return pd.DataFrame(
        [{"timestamp": s.timestamp, "ticker": s.ticker, "signal": s.action}
         for s in signals]
    )


def run_bt(
    sig_df: pd.DataFrame,
    price_dir: Path,
    optimizer: MarkowitzOptimizer | None = None,
) -> tuple[dict, BacktestRunner]:
    cfg    = PortfolioConfig.from_toml(PORTFOLIO_TOML)
    runner = BacktestRunner(
        price_data_path=price_dir,
        portfolio_config=cfg,
        initial_cash=INITIAL_CASH,
        optimizer=optimizer,
    )
    return runner.run(sig_df), runner


def count_signals(sig_df: pd.DataFrame) -> tuple[int, int, int]:
    buy   = int((sig_df["signal"] == "buy").sum())
    sell  = int((sig_df["signal"] == "sell").sum())
    hold  = int((sig_df["signal"] == "hold").sum())
    return buy, sell, hold


# ── Formatting ────────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    return f"{v * 100:+.2f}%"


def _pp(v: float) -> str:
    return f"{v * 100:+.2f}pp"


def comparison_table(
    labels: list[str],
    metrics: list[dict],
    sig_counts: list[tuple[int, int, int]],
) -> str:
    COL = 14
    header = f"  {'Metric':<24}" + "".join(f"  {l:>{COL}}" for l in labels)
    sep    = "  " + "-" * 24 + ("  " + "-" * COL) * len(labels)

    rows_data = [
        ("Total return",    [_pct(m["total_return"])  for m in metrics]),
        ("Sharpe ratio",    [f"{m['sharpe_ratio']:+.3f}"    for m in metrics]),
        ("Max drawdown",    [_pct(m["max_drawdown"])  for m in metrics]),
        ("Win rate",        [_pct(m["win_rate"])       for m in metrics]),
        ("Trades executed", [str(m["n_trades"])        for m in metrics]),
        ("Trading days",    [str(m["n_days"])          for m in metrics]),
    ]

    lines = [header, sep]
    for name, values in rows_data:
        row = f"  {name:<24}" + "".join(f"  {v:>{COL}}" for v in values)
        lines.append(row)

    lines.append("")
    lines.append("  Signal counts (BUY / SELL / HOLD):")
    for label, (b, s, h) in zip(labels, sig_counts):
        lines.append(f"    {label}: {b} BUY  {s} SELL  {h} HOLD")

    return "\n".join(lines)


# ── Stage runners ─────────────────────────────────────────────────────────────

def run_stage(
    stage_num: int,
    stage_title: str,
    window_results: list[dict],
) -> None:
    """Append a stage summary section to the markdown file and log."""
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    section_lines = [
        f"## Stage {stage_num}: {stage_title}",
        f"*Completed: {ts}*",
        "",
    ]
    for wr in window_results:
        section_lines += [
            f"### {wr['window']}",
            "",
            wr["table"],
            "",
            wr["notes"],
            "",
        ]
    append_summary("\n".join(section_lines))
    log(f"\n  [Stage {stage_num} summary written to {SUMMARY_FILE.name}]")


# ── STAGE 1 ───────────────────────────────────────────────────────────────────

def stage1() -> None:
    log("\n" + "=" * 64)
    log("STAGE 1 — Price Momentum Confirmation")
    log("=" * 64)

    gen_off = SignalGenerator(momentum_confirm=False)
    gen_on  = SignalGenerator(momentum_confirm=True)

    window_results = []
    for win in WINDOWS:
        log(f"\n  {win['label']}")
        scored  = get_scored(win)
        daily   = aggregate_daily(scored)
        ohlcv   = load_prices(win["price_dir"])

        sig_off  = gen_off.generate(daily)
        sig_on   = gen_on.generate(daily, price_history=ohlcv)
        df_off   = signals_df(sig_off)
        df_on    = signals_df(sig_on)

        log("    Running backtest momentum=OFF …")
        m_off, _ = run_bt(df_off, win["price_dir"])
        log("    Running backtest momentum=ON  …")
        m_on,  _ = run_bt(df_on,  win["price_dir"])

        table = comparison_table(
            ["momentum=OFF", "momentum=ON"],
            [m_off, m_on],
            [count_signals(df_off), count_signals(df_on)],
        )
        log(table)

        filtered = (count_signals(df_off)[0] + count_signals(df_off)[1]) \
                 - (count_signals(df_on)[0]  + count_signals(df_on)[1])
        notes = (
            f"Momentum filter removed {filtered} actionable signals this window. "
            f"Total return delta: {_pp(m_on['total_return'] - m_off['total_return'])}. "
            f"Sharpe delta: {m_on['sharpe_ratio'] - m_off['sharpe_ratio']:+.3f}."
        )
        window_results.append({"window": win["label"], "table": table, "notes": notes})

    run_stage(1, "Price Momentum Confirmation", window_results)


# ── STAGE 2 ───────────────────────────────────────────────────────────────────

def stage2() -> None:
    log("\n" + "=" * 64)
    log("STAGE 2 — Earnings Surprise Confirmation")
    log("=" * 64)

    earnings_cache = load_earnings_cache()
    if earnings_cache is None:
        log("  WARNING: No earnings data found in data/earnings/ — skipping stage 2.")
        append_summary("## Stage 2: Earnings Surprise Confirmation\n\nSKIPPED — no earnings data.\n")
        return

    log(f"  Earnings cache loaded: {earnings_cache}")

    configs = [
        {"label": "base (no filters)",   "momentum": False, "earnings": False},
        {"label": "momentum only",        "momentum": True,  "earnings": False},
        {"label": "earnings only",        "momentum": False, "earnings": True},
        {"label": "momentum + earnings",  "momentum": True,  "earnings": True},
    ]

    window_results = []
    for win in WINDOWS:
        log(f"\n  {win['label']}")
        scored  = get_scored(win)
        daily   = aggregate_daily(scored)
        ohlcv   = load_prices(win["price_dir"])

        labels  = []
        all_m   = []
        all_cnt = []

        for cfg in configs:
            gen = SignalGenerator(
                momentum_confirm=cfg["momentum"],
                earnings_confirm=cfg["earnings"],
            )
            sigs   = gen.generate(daily, price_history=ohlcv, earnings_cache=earnings_cache)
            df     = signals_df(sigs)
            log(f"    Backtest: {cfg['label']} …")
            m, _   = run_bt(df, win["price_dir"])
            labels.append(cfg["label"])
            all_m.append(m)
            all_cnt.append(count_signals(df))

        table = comparison_table(labels, all_m, all_cnt)
        log(table)

        base_ret = all_m[0]["total_return"]
        notes_parts = []
        for i, cfg in enumerate(configs[1:], 1):
            delta_ret    = all_m[i]["total_return"] - base_ret
            delta_sharpe = all_m[i]["sharpe_ratio"]  - all_m[0]["sharpe_ratio"]
            notes_parts.append(
                f"{cfg['label']}: return {_pp(delta_ret)} vs base, Sharpe {delta_sharpe:+.3f}"
            )
        notes = "vs baseline — " + "; ".join(notes_parts) + "."
        window_results.append({"window": win["label"], "table": table, "notes": notes})

    run_stage(2, "Earnings Surprise Confirmation", window_results)


# ── STAGE 3 ───────────────────────────────────────────────────────────────────

def stage3() -> None:
    log("\n" + "=" * 64)
    log("STAGE 3 — Bayesian (EWM) Sentiment Smoothing")
    log("=" * 64)

    # Use alpha=0.3 (≈6-day half-life) vs flat average (alpha=1.0)
    ALPHA = 0.3

    window_results = []
    for win in WINDOWS:
        log(f"\n  {win['label']}")
        scored      = get_scored(win)
        daily_flat  = aggregate_daily(scored)
        daily_smooth = smooth_daily_scores(daily_flat, alpha=ALPHA)

        gen = SignalGenerator()

        sig_flat   = gen.generate(daily_flat)
        sig_smooth = gen.generate(daily_smooth)
        df_flat    = signals_df(sig_flat)
        df_smooth  = signals_df(sig_smooth)

        log("    Backtest flat (alpha=1.0) …")
        m_flat, _   = run_bt(df_flat,   win["price_dir"])
        log(f"    Backtest smoothed (alpha={ALPHA}) …")
        m_smooth, _ = run_bt(df_smooth, win["price_dir"])

        table = comparison_table(
            [f"flat (α=1.0)", f"smoothed (α={ALPHA})"],
            [m_flat, m_smooth],
            [count_signals(df_flat), count_signals(df_smooth)],
        )
        log(table)

        delta_ret    = m_smooth["total_return"] - m_flat["total_return"]
        delta_sharpe = m_smooth["sharpe_ratio"]  - m_flat["sharpe_ratio"]
        notes = (
            f"EWM smoothing (α={ALPHA}) vs flat daily average: "
            f"return {_pp(delta_ret)}, Sharpe {delta_sharpe:+.3f}."
        )
        window_results.append({"window": win["label"], "table": table, "notes": notes})

    run_stage(3, "Bayesian (EWM) Sentiment Smoothing", window_results)


# ── STAGE 4 ───────────────────────────────────────────────────────────────────

def stage4() -> None:
    log("\n" + "=" * 64)
    log("STAGE 4 — Markowitz Mean-Variance Position Sizing")
    log("=" * 64)

    optimizer = MarkowitzOptimizer(lookback=60, max_weight=0.25)

    window_results = []
    for win in WINDOWS:
        log(f"\n  {win['label']}")
        scored = get_scored(win)
        daily  = aggregate_daily(scored)

        gen    = SignalGenerator()
        sigs   = gen.generate(daily)
        df_sig = signals_df(sigs)

        log("    Backtest fixed-pct sizing …")
        m_fixed, _ = run_bt(df_sig, win["price_dir"], optimizer=None)

        log("    Backtest Markowitz sizing …")
        m_mkz, _   = run_bt(df_sig, win["price_dir"], optimizer=optimizer)

        table = comparison_table(
            ["fixed-pct", "markowitz"],
            [m_fixed, m_mkz],
            [count_signals(df_sig), count_signals(df_sig)],
        )
        log(table)

        delta_ret    = m_mkz["total_return"] - m_fixed["total_return"]
        delta_sharpe = m_mkz["sharpe_ratio"]  - m_fixed["sharpe_ratio"]
        notes = (
            f"Markowitz vs fixed max_position_pct=10%: "
            f"return {_pp(delta_ret)}, Sharpe {delta_sharpe:+.3f}. "
            f"Same signal set; only position sizing differs."
        )
        window_results.append({"window": win["label"], "table": table, "notes": notes})

    run_stage(4, "Markowitz Mean-Variance Position Sizing", window_results)


# ── Final conclusion ──────────────────────────────────────────────────────────

def write_conclusion(stage_outcomes: dict[int, str]) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "## Final Conclusion",
        f"*Written: {ts}*",
        "",
        "This section synthesises the A/B results across all four stages and "
        "both backtest windows.",
        "",
        "| Stage | Feature | W1 effect | W2 effect | Verdict |",
        "|-------|---------|-----------|-----------|---------|",
    ]
    for stage_num, outcome in stage_outcomes.items():
        lines.append(outcome)

    lines += [
        "",
        "### Recommended final configuration",
        "",
        "Based on the evidence above, enable the features that showed a "
        "consistent positive (or risk-reducing) effect across **both** windows. "
        "Disable features that were inconsistent or harmful in at least one window.",
        "",
        "See individual stage sections for per-window metric tables.",
        "",
    ]
    append_summary("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    log(f"\n{'#' * 64}")
    log(f"  V2 ROADMAP — started {ts}")
    log(f"{'#' * 64}")

    # Initialise summary file
    if not SUMMARY_FILE.exists():
        SUMMARY_FILE.write_text(
            f"# V2 Roadmap: Feature A/B Comparison\n\n"
            f"*Run started: {ts}*\n\n",
            encoding="utf-8",
        )

    stage_outcomes: dict[int, str] = {}

    for stage_fn, stage_num, stage_name in [
        (stage1, 1, "Momentum"),
        (stage2, 2, "Earnings"),
        (stage3, 3, "Smoothing"),
        (stage4, 4, "Markowitz"),
    ]:
        try:
            stage_fn()
            stage_outcomes[stage_num] = f"| {stage_num} | {stage_name} | see table | see table | completed |"
        except Exception:
            tb = traceback.format_exc()
            log(f"\n  !! STAGE {stage_num} FAILED:\n{tb}")
            append_summary(
                f"## Stage {stage_num}: FAILED\n\n```\n{tb}\n```\n"
            )
            stage_outcomes[stage_num] = f"| {stage_num} | {stage_name} | — | — | FAILED |"

    write_conclusion(stage_outcomes)

    log(f"\n{'#' * 64}")
    log("  ALL STAGES COMPLETE")
    log(f"  Log:     {LOG_FILE}")
    log(f"  Summary: {SUMMARY_FILE}")
    log(f"{'#' * 64}\n")


if __name__ == "__main__":
    main()
