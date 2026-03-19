



# ============================================================
# Multi-ETF Z-Score + Volatility Regime Research
#
# Tests:
#   1. Z-score momentum across SPY, QQQ, DIA, XLK, SMH
#   2. Cross-sectional strongest-Z selection
#   3. Volatility regime filter ("put wall proxy")
#   4. Current SPY gamma-weighted OI snapshot
#
# IMPORTANT:
# Historical "gamma wall" is proxied using VIX > 20d SMA because
# free Yahoo Finance data does not provide full historical options
# chain with Greeks + open interest.
#
# Current SPY option-chain gamma-OI is computed only as a live
# structural snapshot, not as the historical backtest driver.
# ============================================================

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

print("Imports OK")

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------
TICKERS = ["SPY", "QQQ", "DIA", "XLK", "SMH"]
Z_WINDOW = 20
Z_THRESHOLD = 1.0
HOLD_DAYS = 20

print("Downloading ETF data...")
data = {}
for t in TICKERS:
    df = yf.download(t, period="5y", interval="1d", auto_adjust=True, progress=False)
    data[t] = df["Close"].squeeze().dropna()
    print(f"  {t}: {len(data[t])} bars")

# ------------------------------------------------------------
# Compute rolling z-score and forward return
# ------------------------------------------------------------
print("\nComputing rolling z-scores and forward returns...")
results = {}

for t in TICKERS:
    closes = data[t].values
    dates = data[t].index

    zs = []
    fwds = []
    ds = []

    for i in range(Z_WINDOW, len(closes) - HOLD_DAYS):
        w = closes[i - Z_WINDOW:i]
        sd = w.std(ddof=0)
        if sd == 0:
            continue

        z = (closes[i] - w.mean()) / sd
        fwd = (closes[i + HOLD_DAYS] - closes[i]) / closes[i]

        zs.append(z)
        fwds.append(fwd)
        ds.append(dates[i])

    results[t] = {
        "z": np.array(zs),
        "fwd": np.array(fwds),
        "dates": ds
    }

print("Done")

# ------------------------------------------------------------
# Standalone z-score edge by ETF
# ------------------------------------------------------------
print("\n--- Z-SCORE EDGE BY ETF (z > 1.0, 20-day hold) ---")
print(f"{'ETF':>5} | {'Win Rate':>9} | {'Avg Ret':>8} | {'N signals':>10} | {'Verdict':>8}")
print("-" * 55)

etf_stats = {}

for t in TICKERS:
    z = results[t]["z"]
    fwd = results[t]["fwd"]

    mask = z > Z_THRESHOLD
    if mask.sum() < 10:
        continue

    wr = (fwd[mask] > 0).mean()
    avg = fwd[mask].mean() * 100

    verdict = "GOOD" if wr > 0.57 else ("OK" if wr > 0.53 else "WEAK")
    etf_stats[t] = {"wr": wr, "avg": avg, "n": int(mask.sum())}

    print(f"{t:>5} | {wr:>8.1%} | {avg:>7.2f}% | {mask.sum():>10} | {verdict:>8}")

# ------------------------------------------------------------
# Cross-sectional strongest z-score each day
# ------------------------------------------------------------
print("\n--- CROSS-SECTIONAL TEST: PICK HIGHEST Z EACH DAY ---")

all_dates = sorted(set.intersection(*[set(results[t]["dates"]) for t in TICKERS]))
print(f"Common dates across ETFs: {len(all_dates)}")

cross_fwds = []

for d in all_dates[:-HOLD_DAYS]:
    best_z = -999
    best_fwd = np.nan

    for t in TICKERS:
        idxs = [j for j, x in enumerate(results[t]["dates"]) if x == d]
        if not idxs:
            continue

        j = idxs[0]
        z_val = results[t]["z"][j]
        fwd_val = results[t]["fwd"][j]

        if z_val > Z_THRESHOLD and z_val > best_z:
            best_z = z_val
            best_fwd = fwd_val

    if best_z > Z_THRESHOLD:
        cross_fwds.append(best_fwd)

cross_fwds = np.array([x for x in cross_fwds if not np.isnan(x)])

if len(cross_fwds) > 10:
    wr = (cross_fwds > 0).mean()
    avg = cross_fwds.mean() * 100
    print(f"Pick highest Z each day -> win rate: {wr:.1%} | avg return: {avg:.2f}% | n={len(cross_fwds)}")
else:
    print("Not enough cross-sectional signals.")

# ------------------------------------------------------------
# Current SPY gamma-weighted OI snapshot
# ------------------------------------------------------------
print("\n--- CURRENT SPY GAMMA-WEIGHTED OI SNAPSHOT ---")
spy_ticker = yf.Ticker("SPY")
spot = float(spy_ticker.history(period="1d")["Close"].iloc[-1])
opt_dates = spy_ticker.options[:6]

print(f"SPY spot: {spot:.2f}")

all_opts = []

for exp in opt_dates:
    try:
        chain = spy_ticker.option_chain(exp)
        dte = max((pd.Timestamp(exp) - pd.Timestamp.now()).days, 1)
        T = dte / 365.0

        def bs_gamma(S, K, T, iv):
            if T <= 0 or iv <= 0 or K <= 0 or S <= 0:
                return 0.0
            try:
                d1 = (np.log(S / K) + 0.5 * iv**2 * T) / (iv * np.sqrt(T))
                return np.exp(-0.5 * d1**2) / (S * iv * np.sqrt(2 * np.pi * T))
            except:
                return 0.0

        for df_o, right in [(chain.calls, "call"), (chain.puts, "put")]:
            tmp = df_o[["strike", "openInterest", "impliedVolatility"]].copy()
            tmp["right"] = right
            tmp["dte"] = dte
            tmp["gamma"] = tmp.apply(
                lambda r: bs_gamma(spot, r["strike"], T, r["impliedVolatility"]),
                axis=1
            )
            tmp["gamma_oi"] = tmp["openInterest"] * tmp["gamma"].abs()
            all_opts.append(tmp)

    except Exception as e:
        print(f"  {exp}: {e}")

opts = pd.concat(all_opts).dropna() if all_opts else pd.DataFrame()
print(f"Contracts downloaded: {len(opts)}")

put_wall_now = None

if len(opts) > 0:
    filt = opts[(opts["dte"] >= 7) & (opts["dte"] <= 30)]

    call_wt = filt[(filt["right"] == "call") & (filt["strike"] > spot)]["gamma_oi"].sum()
    put_wt = filt[(filt["right"] == "put") & (filt["strike"] < spot)]["gamma_oi"].sum()

    put_wall_now = put_wt > call_wt

    print(f"Call-side gamma × OI: {call_wt:.0f}")
    print(f"Put-side  gamma × OI: {put_wt:.0f}")
    print(f"Put wall active now : {put_wall_now} ({'supportive / bullish' if put_wall_now else 'resistance / bearish'})")

# ------------------------------------------------------------
# Historical wall proxy using VIX > 20d SMA
# ------------------------------------------------------------
print("\n--- HISTORICAL VOLATILITY REGIME PROXY ---")
print("Historical gamma-wall data is not available from free Yahoo Finance history.")
print("Using VIX > 20d SMA as a proxy for elevated short-term hedging / put-wall-like regime.\n")

vix_data = yf.download("^VIX", period="5y", interval="1d", auto_adjust=True, progress=False).dropna()
vix_close = vix_data["Close"].squeeze()
vix_sma20 = vix_close.rolling(20).mean()

put_wall_hist = vix_close > vix_sma20

print(f"Proxy active on {put_wall_hist.mean() * 100:.1f}% of days")
print(f"Proxy inactive on {(~put_wall_hist).mean() * 100:.1f}% of days")

# ------------------------------------------------------------
# Z-score alone vs z-score + wall proxy
# ------------------------------------------------------------
print("\n--- Z-SCORE ALONE vs Z-SCORE + WALL PROXY ---")
print(f"{'ETF':>5} | {'Z alone':>8} | {'Z+Wall':>8} | {'Delta':>8} | {'N_wall':>7}")
print("-" * 45)

vix_aligned = put_wall_hist.reindex(data["SPY"].index, method="ffill")

for t in TICKERS:
    z = results[t]["z"]
    fwd = results[t]["fwd"]
    ds = results[t]["dates"]

    wall_flags = []
    for d in ds:
        try:
            wall_flags.append(bool(vix_aligned.loc[d]))
        except:
            wall_flags.append(False)

    wall_arr = np.array(wall_flags)
    m_z = z > Z_THRESHOLD
    m_both = m_z & wall_arr

    if m_z.sum() < 5:
        continue

    wr_z = (fwd[m_z] > 0).mean()
    wr_both = (fwd[m_both] > 0).mean() if m_both.sum() > 5 else np.nan
    delta = (wr_both - wr_z) * 100 if not np.isnan(wr_both) else np.nan

    wr_both_fmt = f"{wr_both:>7.1%}" if not np.isnan(wr_both) else "    nan"
    delta_fmt = f"{delta:>+7.1f}pp" if not np.isnan(delta) else "    nan"

    print(f"{t:>5} | {wr_z:>7.1%} | {wr_both_fmt} | {delta_fmt} | {m_both.sum():>7}")

# ------------------------------------------------------------
# Statistical significance test on SPY
# ------------------------------------------------------------
print("\n--- STATISTICAL SIGNIFICANCE TEST (SPY) ---")

t = "SPY"
z = results[t]["z"]
fwd = results[t]["fwd"]
ds = results[t]["dates"]

wall_arr = np.array([bool(vix_aligned.loc[d]) if d in vix_aligned.index else False for d in ds])

m_z = z > Z_THRESHOLD
m_both = m_z & wall_arr

bwr = (fwd[m_z] > 0).mean() if m_z.sum() > 5 else np.nan
cwr = (fwd[m_both] > 0).mean() if m_both.sum() > 5 else np.nan

if not np.isnan(bwr) and not np.isnan(cwr):
    n = m_both.sum()
    wins = int(round(cwr * n))
    try:
        pval = stats.binomtest(wins, n, p=bwr, alternative="greater").pvalue
    except:
        pval = 1.0

    print(f"SPY z > 1.0 only    : {bwr:.1%} (n={m_z.sum()})")
    print(f"SPY z + wall proxy  : {cwr:.1%} (n={n})")
    print(f"Improvement         : {(cwr - bwr) * 100:+.1f}pp")
    print(f"p-value             : {pval:.4f} {'*** significant' if pval < 0.05 else '(borderline)' if pval < 0.10 else '(not significant)'}")

# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1) Win rate by ETF
ax = axes[0, 0]
tickers_plot = list(etf_stats.keys())
wrs_plot = [etf_stats[t]["wr"] for t in tickers_plot]
colors = ["green" if w > 0.57 else "orange" if w > 0.53 else "red" for w in wrs_plot]

ax.bar(tickers_plot, wrs_plot, color=colors, alpha=0.8)
ax.axhline(0.57, color="gray", linestyle="--", alpha=0.6, label="57% target")
ax.axhline(0.50, color="black", linestyle="--", alpha=0.3)
for i, (t, w) in enumerate(zip(tickers_plot, wrs_plot)):
    ax.text(i, w + 0.005, f"{w:.0%}", ha="center", fontsize=9)
ax.set_title(f"Win rate by ETF (z > {Z_THRESHOLD}, {HOLD_DAYS}d hold)")
ax.legend()
ax.set_ylim(0.35, 0.80)

# 2) Z-score distributions
ax = axes[0, 1]
for t in TICKERS:
    z = results[t]["z"]
    ax.hist(z, bins=40, alpha=0.35, label=t)
ax.axvline(Z_THRESHOLD, color="red", linestyle="--", linewidth=2, label=f"z = {Z_THRESHOLD}")
ax.set_title("Z-score distributions")
ax.legend(fontsize=8)

# 3) SPY win rate vs hold period
ax = axes[1, 0]
holds = [5, 10, 15, 20, 30]
wr_z_h = []
wr_bh_h = []

for h in holds:
    closes = data["SPY"].values
    zs_h, fwd_h, ds_h = [], [], []

    for i in range(Z_WINDOW, len(closes) - h):
        w = closes[i - Z_WINDOW:i]
        sd = w.std(ddof=0)
        if sd == 0:
            continue
        zs_h.append((closes[i] - w.mean()) / sd)
        fwd_h.append((closes[i + h] - closes[i]) / closes[i])
        ds_h.append(data["SPY"].index[i])

    za = np.array(zs_h)
    fa = np.array(fwd_h)
    wa = np.array([bool(vix_aligned.loc[d]) if d in vix_aligned.index else False for d in ds_h])

    mz = za > Z_THRESHOLD
    mb = mz & wa

    wr_z_h.append((fa[mz] > 0).mean() if mz.sum() > 5 else np.nan)
    wr_bh_h.append((fa[mb] > 0).mean() if mb.sum() > 5 else np.nan)

ax.plot(holds, wr_z_h, "b-o", label="z only")
ax.plot(holds, wr_bh_h, "g-o", label="z + wall proxy")
ax.axhline(0.57, color="gray", linestyle="--", alpha=0.5)
ax.set_title("SPY win rate vs hold period")
ax.set_xlabel("Hold days")
ax.legend()
ax.set_ylim(0.35, 0.80)

# 4) Current gamma-OI by strike
ax = axes[1, 1]
if len(opts) > 0:
    filt = opts[(opts["dte"] >= 7) & (opts["dte"] <= 30)]

    calls_near = filt[
        (filt["right"] == "call") &
        (filt["strike"] >= spot * 0.97) &
        (filt["strike"] <= spot * 1.04)
    ]
    puts_near = filt[
        (filt["right"] == "put") &
        (filt["strike"] >= spot * 0.96) &
        (filt["strike"] <= spot * 1.03)
    ]

    cb = calls_near.groupby("strike")["gamma_oi"].sum()
    pb = puts_near.groupby("strike")["gamma_oi"].sum()

    ax.bar(cb.index, cb.values / 1e3, width=0.4, color="red", alpha=0.7, label="Call gamma × OI")
    ax.bar(pb.index - 0.4, pb.values / 1e3, width=0.4, color="green", alpha=0.7, label="Put gamma × OI")
    ax.axvline(spot, color="blue", linestyle="--", label=f"Spot {spot:.0f}")

ax.set_title("Current SPY gamma-weighted OI by strike")
ax.set_xlabel("Strike")
ax.set_ylabel("Gamma × OI (thousands)")
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("multiETF_gamma_research.png", dpi=150, bbox_inches="tight")
plt.show()

print("Saved multiETF_gamma_research.png")

# ------------------------------------------------------------
# Summary block
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

best_etf = max(etf_stats, key=lambda t: etf_stats[t]["wr"]) if etf_stats else "N/A"
best_wr = etf_stats[best_etf]["wr"] if etf_stats else np.nan
cross_wr = (cross_fwds > 0).mean() if len(cross_fwds) else np.nan

print(f"""
BEST SINGLE ETF:
  {best_etf}: {best_wr:.1%} win rate (z > {Z_THRESHOLD}, {HOLD_DAYS}d hold)

CROSS-SECTIONAL (pick highest z each day):
  Win rate: {cross_wr:.1%} (n={len(cross_fwds)})

VOLATILITY REGIME EFFECT:
  See the 'Delta' column above.
  Positive delta means the wall proxy improved the z-score signal.
  Negative delta means the wall proxy removed good trades.

CURRENT MARKET SNAPSHOT:
  SPY spot: {spot:.2f}
  Put wall active now: {put_wall_now if put_wall_now is not None else 'N/A'}

RESEARCH SETTINGS:
  z_threshold = {Z_THRESHOLD}
  z_window    = {Z_WINDOW}
  hold_days   = {HOLD_DAYS}
""")