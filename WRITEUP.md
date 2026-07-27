# Building a News-Sentiment Trading Bot: What I Built, What I Found, and What It Taught Me

## The Idea

Most algorithmic trading bots look at charts — price patterns, moving averages, volume spikes. I wanted to try something different: what if a bot read the news instead, and used AI to decide whether the coverage around a company sounded good or bad? If positive news reliably predicts price increases, and negative news reliably predicts drops, you'd have a tradeable signal.

That was the hypothesis. I spent the summer building and testing it. The results were more interesting than either a clean win or a clean loss would have been.

## What I Built

The bot is a complete, end-to-end automated pipeline running in Python, split into six functional layers:

### 1. News Ingestion

Every day, the bot pulls recent articles about each of eight companies (Apple, Nvidia, Tesla, Microsoft, Amazon, Google, Meta, AMD) from NewsAPI. For historical backtesting, it queries GDELT, a free global news archive covering years of financial coverage. Getting clean historical data from GDELT turned out to be one of the harder engineering problems of the project — more on that later.

### 2. Relevance Filtering

Raw news searches are noisy. A search for "Apple" returns recipes for apple pie. A search for "Tesla" returns articles about Elon Musk's other companies. A search for "Meta" returns academic papers that use "meta-analysis" as a common English word. I built a multi-layer relevance filter that combines word-boundary regex, spaCy named entity recognition (NER), and a company alias mapping to reject clearly off-topic articles before they can contaminate the sentiment signal. Even this wasn't perfect — the NER model correctly caught most noise, but some edge cases (Elon Musk mentioned without naming Tesla, "Prime Minister" matching "Amazon Prime") required specific fixes, each of which introduced new edge cases to debug.

### 3. Sentiment Scoring

Relevant articles get scored by FinBERT, a free, finance-specific language model from HuggingFace that's been trained specifically to understand the tone of financial text. Unlike generic sentiment tools, it correctly reads phrases like "earnings beat expectations" as positive and "guidance disappoints investors" as negative, even when the surface-level wording seems neutral. Each article gets a score from −1 (very negative) to +1 (very positive).

### 4. Signal Generation

The bot aggregates each company's daily articles into a single score. If that score clears a threshold of +0.3, it generates a BUY signal. Below −0.3, a SELL. Anything in between is HOLD. A minimum headline count (8 per day) prevents the bot from acting on thin, noisy data — this guard was specifically added after the first backtest revealed how badly a single Elon Musk article could skew Tesla's daily score on a quiet news day.

### 5. Portfolio Risk Management

Even with a confirmed signal, the bot runs every proposed trade through a risk layer before acting. Rules include: maximum 15% of the portfolio in any single stock, minimum 20% cash reserve at all times, sector concentration caps (no more than 30% in tech at once), a portfolio-level drawdown circuit breaker that halts new buys if the total value drops more than a configurable threshold from its peak, and per-position trailing stops. Every skipped trade is logged with its reason, so the decision process is fully auditable.

### 6. Execution

Approved trades go to Alpaca's paper trading API — real-time stock prices, real order mechanics, but completely simulated money. This means the backtest and live environments use the same execution code, which matters: a system where backtesting and live trading use different logic can produce backtest results that never replicate in practice.

## The Results

I backtested the strategy on two independent, non-overlapping 90-day windows using GDELT's historical news data.

**Window 1: December 2025 – March 2026**
- Return: −0.91%
- Sharpe ratio: −1.03
- Win rate: 0% (every trade lost)
- Trades executed: 7

**Window 2: March 2026 – June 2026**
- Return: +4.28%
- Sharpe ratio: 2.00
- Win rate: 70%
- Trades executed: 23

Same code. Same rules. Same parameters. Two completely opposite results depending on which 90 days you looked at.

This is the central finding of the project, and it's more valuable than either result on its own would have been: a single backtest proves nothing. The first window, tested alone, would have suggested the strategy doesn't work. The second window, tested alone, would have suggested it was exceptional. Neither conclusion would have been correct. The only honest conclusion from both windows together is that the strategy's performance is highly regime-dependent — it behaves differently depending on market conditions, not just randomly.

## Version 2: Adding Confirmation Signals

The obvious question from the two-window result: what's different between the two periods that explains the gap? That led to version 2, which tests whether requiring a second, independent signal to confirm the sentiment before trading meaningfully improves consistency.

I built and tested four enhancements through an autonomous multi-stage process:

**Price momentum confirmation** — requiring the stock's price to be trending in the same direction as the sentiment before acting. In trending markets, this filtered out false signals and improved results. In choppy markets, it blocked real signals and made results worse. Regime-dependent, same as the base strategy.

**Earnings surprise data** — adding Alpha Vantage earnings data (actual vs. estimated EPS/revenue) as a third confirmation layer. Result: zero additive value beyond what momentum confirmation already captured. Removed from the recommended configuration.

**EWM sentiment smoothing** — instead of resetting to a flat daily average, carrying prior days' sentiment forward with decay. Slightly better in one window, slightly worse in the other. Inconclusive.

**Markowitz portfolio optimization** — replacing fixed position-size caps with mathematically optimal weights based on expected return and volatility. The single strongest individual improvement in one window, but again regime-dependent.

The pattern across all four tests was consistent: every feature helped in one market regime and hurt in another. This pointed toward a more interesting conclusion than "find the right feature" — the right architecture might be one that detects which regime the market is in and adjusts its behavior accordingly, rather than using fixed rules that assume one regime all the time.

## The Regime Gate

I built a regime detector that classifies each trading day as either "trending" (price moving steadily in a direction) or "choppy/volatile" (price swinging around without clear direction), based on 20-day realized volatility relative to its own longer-term average. The detector can then switch the bot's behavior based on which regime it's in — momentum confirmation and Markowitz sizing in trending regimes, simpler fixed-percentage sizing in choppy ones.

Testing the regime detector against a third, completely untouched time window (September–December 2025) produced a clean finding: the regime classification itself was accurate and sensible — it correctly identified October 10 through December 1 as choppy, which aligns precisely with Q3 earnings season, a historically volatile period when analyst forecasts, company guidance, and macro concerns all intersect. The regime labels made intuitive sense with zero tuning toward this specific window.

However, the backtest comparison (baseline vs. uniform-best-config vs. regime-gated) was inconclusive for a specific, identifiable reason: the third window's news data came from GDELT's bulk file archive, which only stores article URLs — not actual headlines. FinBERT was scoring reconstructed URL slugs ("apple-iphone-17-rumors") rather than real headlines ("Apple Smashes Q3 Estimates, iPhone 17 Demand Exceeds Forecast"). Sentiment scores from URL slugs were consistently weaker and more neutral than those from real headlines, and EWM smoothing dampened the one borderline buy signal that existed below the 0.30 threshold. The regime gate never got a fair test because there weren't enough actionable signals to route.

This is itself an important finding: data quality upstream of the sentiment model matters as much as the model's architecture. Garbage in, garbage out — but the garbage can be subtle.

## Engineering Challenges Worth Documenting

**The relevance filtering problem.** There is no perfect keyword filter for company names, because natural language is ambiguous in ways that keyword matching fundamentally cannot resolve. "Apple" matches fruit. "Meta" matches academic methodology. "Amazon" matches a river and a mythology. "Prime" matches a UK Prime Minister. Every fix to one false positive introduces a new edge case elsewhere. The final approach — NER + alias mapping + title-only matching for subsidiary brands — got contamination from 16 headlines/10 ticker-days down to 11/9, a real improvement, but with documented residual cases that the conservative ±0.3 threshold absorbs.

**The GDELT data problem.** Getting historical news data at scale is genuinely hard without a budget. BigQuery's free tier burns its entire monthly 1TB quota on a single query against GDELT's unpartitioned 21.6TB table. The DOC API is rate-limited to roughly one successful call per 35-minute cooldown period from a single IP address. The bulk file archive doesn't have real headlines — it stores URLs only, not article text. Each path was explored, each limitation was real and documented.

**The backtest-vs-live consistency problem.** The single most underappreciated design decision in the codebase: `BacktestRunner` and `AlpacaTrader` both call into the same `PortfolioManager.process_signal()` function, with the same parameters, for the same logic. There's no separate "simulation mode" with different rules. This means the gap between "what the backtest says" and "what the live system does" is as small as possible — not zero (slippage assumptions, fill timing), but contained and documented.

## What I'd Do Differently

The biggest single architectural change I'd make: stop treating sentiment as the primary signal and price as a confirmation. The causality may actually run the other way — price movement often precedes news coverage, especially for retail-accessible sources like NewsAPI and GDELT. A better framing might be: start with price-based signals (momentum, mean-reversion) as the primary trigger, and use sentiment as a filter that rejects trades in clearly adverse news environments. This is structurally different from the current approach, not just a parameter change.

Beyond that: earnings data in structured form (beat/miss percentage, not just headline text) should be a primary signal, not a confirmation layer. The earnings-API test found zero additive value, but that result may be an artifact of how I wired it in — as a binary confirmation gate — rather than evidence that earnings data doesn't predict returns.

## Final Takeaway

The most useful thing this project produced isn't a trading strategy. It's an empirical demonstration of why backtesting is hard, and specifically why a single-window backtest is almost useless as evidence of whether a strategy works. The two-window result proved this concretely, on real data, with a real system — not as an abstract principle from a textbook.

A strategy that made 4% with a 70% win rate in one 90-day window, then lost money on every trade in a different 90-day window, doesn't tell you the strategy is good or bad. It tells you market regimes are real, conditions matter, and the honest answer to "does this work?" is "under some conditions, yes; under others, no; and the current challenge is learning to detect which one you're in."

That's a harder answer than a clean win or loss. It's also a more honest one.
