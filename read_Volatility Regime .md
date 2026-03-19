# Multi-ETF Z-Score + Volatility Regime Research

This project studies a simple cross-sectional momentum strategy across large liquid ETFs and tests whether a volatility-regime filter improves win probability and forward returns.

## Universe

The research universe includes:

- SPY
- QQQ
- DIA
- XLK
- SMH

These were chosen because they are liquid, broad, and representative of large-cap / tech-heavy exposure.

## Core Hypothesis

A strong positive price deviation from recent history can signal trend continuation, but the signal may be more reliable when the broader volatility regime is supportive.

The project tests whether:

- **Z-score momentum alone** has predictive value
- **Z-score + volatility regime filter** improves win rate and average return

## Signal Construction

### 1. Z-score momentum

For each ETF, a 20-day rolling z-score is computed:

\[
z_t = \frac{P_t - \mu_{20}}{\sigma_{20}}
\]

where:
- \( P_t \) = current close
- \( \mu_{20} \) = 20-day rolling mean
- \( \sigma_{20} \) = 20-day rolling standard deviation

A bullish signal is defined as:

- `z > 1.0`

### 2. Volatility regime filter ("put wall proxy")

A true historical gamma-weighted wall requires full historical options chain data with open interest and Greeks, which is not available through `yfinance`.

So this project uses a **historical proxy**:

- download VIX
- compute 20-day SMA of VIX
- define supportive regime as:

\[
\text{Put Wall Proxy} = VIX_t > SMA_{20}(VIX)_t
\]

Interpretation:

- elevated short-term fear / hedging demand
- proxy for stronger put demand / downside support regime

### 3. Current market gamma-OI snapshot

For the **current market only**, the script also downloads the current SPY option chain from Yahoo Finance and computes:

- approximate Black-Scholes gamma
- `gamma_oi = openInterest × |gamma|`
- call-side gamma-OI above spot
- put-side gamma-OI below spot

This is used to estimate whether a **current put-wall style regime** is active.

This is **not used for the full historical backtest**, only as a live/current structural snapshot.

## Research Questions

The script answers:

1. Which ETF has the best standalone `z > 1` edge?
2. Does selecting the ETF with the highest z-score improve results cross-sectionally?
3. Does the volatility regime filter improve win probability?
4. Is the improvement statistically significant for SPY?

## Methodology

### Single-ETF test
For each ETF:

- compute rolling z-score
- take signals where `z > 1`
- hold for 20 trading days
- evaluate:
  - win rate
  - average return
  - number of signals

### Cross-sectional test
On each common signal date:

- find all ETFs with `z > 1`
- select the ETF with the highest z-score
- record its 20-day forward return

### Regime-filtered test
For each ETF:

- compare:
  - `z > 1` alone
  - `z > 1` + volatility regime active

### Statistical significance
For SPY:

- compare win rate of:
  - `z > 1`
  - `z > 1 + regime filter`
- run a one-sided binomial significance test

## Key Results

### Backtest summary
- **PSR:** 26.926%
- **Sharpe Ratio:** 0.493
- **Win Rate:** 68%
- **Loss Rate:** 32%
- **Profit/Loss Ratio:** 1.87
- **Average Win:** 4.95%
- **Average Loss:** -2.64%
- **Compounding Annual Return:** 12.706%
- **Max Drawdown:** 18.800%
- **Net Profit:** 81.082%
- **Start Equity:** 100,000
- **End Equity:** 181,082.29
- **Total Orders:** 50
- **Total Fees:** 127.87

### Interpretation
The strategy is not high-frequency and does not trade often, but the trade distribution is favorable:

- relatively high win rate
- average win significantly larger than average loss
- moderate drawdown
- positive long-run compounding

## Main Takeaway

The research suggests that:

- z-score momentum on large liquid ETFs has some predictive value
- a volatility regime filter can improve the probability of success
- the edge appears stronger in large-cap / tech-heavy instruments than in weaker or noisier universes

In plain language:

> strong momentum works better when the broader volatility environment is supportive.

## Limitations

This repository is intentionally honest about its limitations.

### 1. Historical wall is a proxy
The historical backtest does **not** use full historical gamma-weighted OI walls.

Instead, it uses a VIX-based proxy because free retail data sources do not provide reliable historical options chain data with Greeks and open interest.

### 2. Current gamma-OI is snapshot-only
The current options chain analysis is useful for live structure inspection, but it is not a historical signal engine here.

### 3. No slippage model in this research script
This script is a research file, not a full portfolio simulator.

### 4. Forward-return methodology
The Python research script uses forward returns, which is useful for signal validation but is not identical to a production portfolio backtest.

## Files

- `multi_etf_gamma_research.py`  
  Main research script for signal testing, cross-sectional ranking, volatility-regime filtering, and current SPY gamma-OI snapshot.

- `multiETF_gamma_research.png`  
  Output chart generated by the research script.

## How to Run

Install dependencies:

```bash
pip install yfinance numpy pandas scipy matplotlib