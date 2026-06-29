# News-Sentiment Trading Bot — Project Write-Up

## What it does

This is a paper-trading bot that reads financial news, scores how positive
or negative the coverage is using a finance-specific AI model, and decides
whether to buy, sell, or hold based on whether that sentiment is strong
enough and consistent enough to act on.

**Pipeline:** News ingestion (NewsAPI / GDELT) → relevance filtering (is
this article actually about the company, or just mentioning it in
passing?) → sentiment scoring (FinBERT) → signal generation (buy/sell/hold
with configurable thresholds and confirmation filters) → portfolio risk
management (position sizing, sector caps, drawdown limits) → paper trade
execution (Alpaca).

It trades 8 large-cap tech tickers (AAPL, NVDA, TSLA, MSFT, AMZN, GOOGL,
META, AMD) and is backed by 248+ passing unit tests covering ingestion,
sentiment scoring, signal generation, and portfolio logic.

## The core question

Can publicly available news sentiment — the same headlines anyone can
read for free — produce a tradeable trading signal once you account for
realistic costs and constraints?

## Method

Two independent 90-day backtests were run on historical GDELT news data,
using the same code, same thresholds, and same portfolio rules in both
periods:

- **Window 1:** December 2025 – March 2026
- **Window 2:** March 2026 – June 2026

Running two non-overlapping windows, rather than one, was a deliberate
choice — a single backtest can't tell you whether a result is a real
pattern or a coincidence of that particular period.

## Results — baseline strategy

| Window | Total Return | Sharpe Ratio | Max Drawdown | Win Rate | Trades |
|---|---|---|---|---|---|
| W1 (Dec '25–Mar '26) | **−0.91%** | −1.03 | 2.47% | 0% | 7 |
| W2 (Mar '26–Jun '26) | **+4.28%** | 2.00 | 4.31% | 70% | 23 |

**The central finding:** the exact same strategy, run on the exact same
code with the exact same rules, produced a losing result in one period
and a solidly positive one in the next. Nothing about the strategy
changed between the two windows — only the time period did.

This is the most important result in the whole project, and the rest of
the work exists to investigate it.

## What that result means — and doesn't mean

A result like this could come from two very different underlying causes:

1. The sentiment signal is real but **inconsistent across market
   regimes** — it works when news sentiment and price momentum tend to
   move together, and fails when they decouple.
2. The signal is mostly **noise**, and a 90-day sample is too short to
   distinguish luck from skill (7 trades and 23 trades are both small
   samples — neither window is statistically conclusive on its own).

Both are plausible. The follow-up work below was designed to find out
which one was closer to the truth, not to chase a better number.

## V2: Testing four hypotheses against the divergence

Four independent modifications were each tested as an on/off A/B
comparison, against the same scored sentiment data, across both windows:

| Stage | Feature | What it tests |
|---|---|---|
| 1 | Price momentum confirmation | Does requiring sentiment to agree with the 10-day moving average reduce false signals? |
| 2 | Earnings-surprise confirmation | Does gating signals near earnings reports (requiring the surprise direction to agree with sentiment) help? |
| 3 | EWM sentiment smoothing (α=0.3–0.5) | Does smoothing daily sentiment scores reduce noise from single-day spikes? |
| 4 | Markowitz mean-variance position sizing | Does optimizing position sizes by covariance, instead of fixed percentages, improve risk-adjusted return? |

**Selected results:**

| Stage | Window | Config | Return | Sharpe | Win Rate |
|---|---|---|---|---|---|
| Momentum | W1 | off → on | −2.01% → −4.80% | −1.52 → −3.89 | 25% → 0% |
| Momentum | W2 | off → on | +0.15% → +1.06% | +0.11 → +0.71 | 50% → 38% |
| Earnings | W1 | earnings only | −3.68% | −2.93 | 12.5% |
| Earnings | W2 | earnings only | +0.61% | +0.31 | 50% |
| Smoothing | W1 | α=0.3 | −0.92% | −1.19 | 0% |
| Smoothing | W2 | α=0.3 | −0.46% | −0.41 | 50% |
| Markowitz | W1 | markowitz | −3.46% | −2.99 | 22% |
| **Markowitz** | **W2** | **markowitz** | **+2.92%** | **+1.39** | **60%** |

**Finding: every single filter was regime-dependent.** Momentum
confirmation helped in W2 but made W1 meaningfully worse. Earnings
confirmation added no consistent value. Smoothing reduced volatility but
didn't fix direction. Markowitz position sizing was the single strongest
improvement — but only in W2; it still underperformed in W1.

No filter tested produced a result that was good in *both* windows. That
consistency (or lack of it) is itself the finding: it points toward
explanation #1 above (regime-dependence) being more likely than pure
noise, since the same filters behaved in directionally sensible but
period-specific ways — but it does not prove the signal is reliable
enough to trade with real capital.

## Regime detection — a partial validation

Given that every filter's performance seemed to track *which period* it
was tested in, a regime detector was built: a simple classifier using
20-day realized volatility to label each day as "trending" or
"choppy/volatile."

The detector correctly identified Q3 2025 earnings season as a choppy
period with no manual tuning — a reasonable sanity check, since earnings
season is a known source of volatility spikes. This is a positive but
limited result: one correct classification on one known period is
suggestive, not a validated regime-detection system.

## What didn't work — and was deliberately stopped

A third validation window (W3: September–December 2025) was attempted to
add a second independent test of the regime hypothesis. It was not
completed, due to a stack of real data-infrastructure limits:

- GDELT's BigQuery interface hit quota limits scanning unpartitioned
  tables (~1TB scanned per query)
- The GDELT DOC API rate-limited repeated historical pulls despite retry
  logic
- GDELT's GKG bulk files lacked real article titles (only URL slugs),
  degrading FinBERT's sentiment scoring enough to suppress signal
  generation

Rather than force a result through degraded data, W3 was abandoned and
is documented here as a known limitation rather than presented as a third
clean test. **This is also a result, narrowly:** a third window would
have meaningfully strengthened the regime-dependence conclusion, and its
absence is the main reason that conclusion is stated as "suggestive"
rather than "confirmed" throughout this write-up.

## Conclusion

At retail-accessible capital levels and a 90-day backtest horizon, a
FinBERT-based news sentiment strategy on 8 large-cap tech tickers did
not produce a consistent edge. Performance was highly sensitive to
market regime, and every confirmation filter tested (momentum, earnings,
smoothing, Markowitz sizing) improved results in one window while
hurting them in the other.

This is consistent with — though does not prove — the idea that easily
accessible signals like public news sentiment are either too weak, too
quickly arbitraged away by faster and better-capitalized players, or too
regime-dependent to trade reliably without a much larger validation
sample than two 90-day windows. It does not show that markets cannot be
beaten by any strategy; it shows that this specific, retail-accessible
approach did not clear that bar in this test.

## What I'd do differently with more time

- Run three or more non-overlapping windows before drawing any
  regime-dependence conclusion with confidence — two is the minimum to
  notice a divergence, not enough to characterize it.
- Build the regime detector earlier in the process, so it could gate
  which filters get applied, rather than testing filters in isolation
  after the fact.
- Use a data source with more reliable historical coverage than GDELT's
  free tier, to avoid losing a validation window to infrastructure
  limits rather than to a real negative result.
