import os
import time
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.client import RemoteDisconnected

import numpy as np
import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv
from pathlib import Path
import os

# Force load .env from same directory as this script
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

print("ENV PATH:", env_path)
print("KEY LOADED:", os.getenv("ALPACA_API_KEY") is not None)


# ---------------------------
# CONFIG
# ---------------------------

@dataclass
class Config:
    # Strategy parameters
    TOP_N: int = 20
    MAX_POSITIONS: int = 1           # trade top N by last 15m volume
    BAR_MINUTES: int = 15
    LOOKBACK_AVG_VOL_BARS: int = 20 # for avg volume baseline
    TREND_BARS: int = 3             # "one direction": last 3 bars same color
    VOL_MULT: float = 2.0           # high volume threshold: last_vol > VOL_MULT * avg_vol

    # Exit condition
    EXIT_VOL_DROP_BARS: int = 3     # exit if volumes drop consecutively over last 3 completed bars

    # Risk / sizing
    MAX_POSITIONS: int = 1
    USE_EQUITY_FRACTION: float = 0.50
    MAX_GROSS_EXPOSURE: float = 0.95   # use up to 95% of equity notionally
    PER_POSITION_FRACTION: float = 1/20
    MIN_PRICE: float = 3.0             # avoid penny-ish names
    MAX_SPREAD_PCT: float = 1.0        # (optional) skip if spread too wide, if you add quotes later

    # Operational
    DRY_RUN: bool = False              # if True, prints signals but does not trade
    LOG_EVERYTHING: bool = True
    SLEEP_SECONDS: int = 10            # heartbeat sleep while waiting for next cycle


CFG = Config()


# A starter universe. Replace with your own bigger list (200-500 liquid tickers).
# The bot will choose TOP_N by volume from this universe each cycle.
DEFAULT_UNIVERSE = [
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","AMD","INTC","NFLX",
    "JPM","BAC","WFC","GS","MS","XOM","CVX","BRK.B","UNH","LLY",
    "AVGO","ORCL","ADBE","CRM","QCOM","CSCO","INTU","AMAT","MU","TXN",
    "KO","PEP","COST","WMT","TGT","HD","LOW","NKE","MCD","SBUX",
    "DIS","ABNB","UBER","LYFT","CAT","DE","BA","GE","MMM","HON",
    "PFE","MRK","ABBV","TMO","DHR","ISRG","GILD","VRTX","AMGN","BMY",
    "SOFI","PLTR","SHOP","SNOW","PANW","CRWD","NOW","DDOG","MDB","NET",
]


# ---------------------------
# HELPERS
# ---------------------------

def now_utc():
    return datetime.now(timezone.utc)

def floor_to_last_completed_15m(ts: datetime, bar_minutes: int = 15) -> datetime:
    """Return the end time of the last completed bar."""
    # Example: if 10:07 -> last completed 15m ends at 10:00
    minute = (ts.minute // bar_minutes) * bar_minutes
    floored = ts.replace(minute=minute, second=0, microsecond=0)
    if floored == ts.replace(second=0, microsecond=0) and (ts.minute % bar_minutes == 0):
        # if exactly on boundary, previous bar is completed (we want the bar that just ended)
        return floored
    return floored

def next_cycle_time(bar_minutes: int = 15) -> datetime:
    """Next time we should run (shortly after a bar completes)."""
    ts = now_utc()
    floored = ts.replace(second=0, microsecond=0)
    # move to next multiple of bar_minutes
    add = bar_minutes - (floored.minute % bar_minutes)
    if add == 0:
        add = bar_minutes
    nxt = floored + timedelta(minutes=add)
    # give a small buffer so bar data is available
    return nxt + timedelta(seconds=5)

def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def safe_symbol_list(universe):
    # Alpaca uses BRK.B as BRK.B in many contexts, but data sometimes wants BRK.B or BRK/B.
    # We'll keep as-is and let Alpaca respond; if it fails, remove that ticker.
    return list(dict.fromkeys(universe))


# ---------------------------
# STRATEGY LOGIC
# ---------------------------


def compute_signal_from_bars(df: pd.DataFrame, cfg: Config):
    """
    Returns (signal, score) where signal in {"long","short"} or (None, None)
    score = volume_ratio (higher is stronger)
    """
    if df is None or len(df) < max(cfg.LOOKBACK_AVG_VOL_BARS + 1, cfg.TREND_BARS + 1):
        return None, None

    last = df.iloc[-1]
    if last["close"] <= 0 or last["open"] <= 0:
        return None, None

    recent = df.iloc[-cfg.TREND_BARS:]
    greens = (recent["close"] > recent["open"]).all()
    reds   = (recent["close"] < recent["open"]).all()
    if not (greens or reds):
        return None, None

    prior = df.iloc[-(cfg.LOOKBACK_AVG_VOL_BARS + 1):-1]
    avg_vol = prior["volume"].mean()
    if avg_vol <= 0:
        return None, None

    vol_ratio = float(last["volume"] / avg_vol)
    if vol_ratio <= cfg.VOL_MULT:
        return None, None

    signal = "long" if greens else "short"
    return signal, vol_ratio


def exit_condition_volume_drop(df: pd.DataFrame, cfg: Config) -> bool:
    """
    Exit if volume drops consecutively for EXIT_VOL_DROP_BARS bars.
    Meaning: vol[-3] > vol[-2] > vol[-1]
    """
    n = cfg.EXIT_VOL_DROP_BARS
    if df is None or len(df) < n:
        return False
    vols = df["volume"].iloc[-n:].values
    return all(vols[i] > vols[i+1] for i in range(n-1))


# ---------------------------
# ALPACA INTEGRATION
# ---------------------------

def make_clients():
    import os
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical.stock import StockHistoricalDataClient

    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"

    # Debug (remove later)
    print("make_clients sees key?", bool(key), "secret?", bool(secret), "paper?", paper)

    if not key or not secret:
        raise RuntimeError(
            f"Missing keys. ALPACA_API_KEY present={bool(key)}, ALPACA_API_SECRET present={bool(secret)}"
        )

    trading_client = TradingClient(key, secret, paper=paper)
    data_client = StockHistoricalDataClient(key, secret)
    return trading_client, data_client

from requests.exceptions import ConnectionError
from urllib3.exceptions import ProtocolError

def fetch_15m_bars(data_client, symbols, end_utc: datetime, bars_needed: int, cfg: Config):
    """
    Fetch enough 15m bars ending at end_utc (last completed bar end time).
    Robust version: batches symbols + retries on transient disconnects.
    Returns dict: sym -> dataframe[open,high,low,close,volume]
    """
    minutes = cfg.BAR_MINUTES * (bars_needed + 5)
    start_utc = end_utc - timedelta(minutes=minutes)

    def _one_batch(batch_syms):
        # IMPORTANT: keep your TimeFrame construction that works for your version
        req = StockBarsRequest(
            symbol_or_symbols=batch_syms,
            timeframe=TimeFrame(cfg.BAR_MINUTES, TimeFrameUnit.Minute),
            start=start_utc,
            end=end_utc,
            adjustment="raw",
            feed="iex",  # keep this if you added it
        )
        return data_client.get_stock_bars(req).df

    out = {}
    BATCH_SIZE = 20       # <= keeps URLs smaller, fewer disconnects
    RETRIES = 4

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i+BATCH_SIZE]

        last_err = None
        for attempt in range(1, RETRIES + 1):
            try:
                bars = _one_batch(batch)
                last_err = None
                break
            except (ConnectionError, ProtocolError, RemoteDisconnected) as e:
                last_err = e
                log(f"Data fetch transient error (batch {i//BATCH_SIZE+1}, attempt {attempt}/{RETRIES}): {e}")
                time.sleep(1.5 * attempt)  # simple backoff
            except Exception as e:
                # Non-transient error: re-raise so you see it
                raise

        if last_err is not None:
            log(f"Skipping batch {batch} after {RETRIES} failed attempts.")
            continue

        if bars is None or len(bars) == 0:
            continue

        if isinstance(bars.index, pd.MultiIndex):
            for sym in batch:
                try:
                    sdf = bars.xs(sym).copy()
                    sdf = sdf[["open", "high", "low", "close", "volume"]].dropna()
                    if len(sdf) > 0:
                        out[sym] = sdf
                except Exception:
                    continue
        else:
            # single symbol fallback
            sym = batch[0]
            sdf = bars[["open", "high", "low", "close", "volume"]].dropna()
            if len(sdf) > 0:
                out[sym] = sdf

    return out

def get_positions_map(trading_client):
    pos = trading_client.get_all_positions()
    # map symbol -> position object
    return {p.symbol: p for p in pos}


def submit_market_order(trading_client, symbol: str, side: OrderSide, notional: float = None, qty: float = None):
    if notional is None and qty is None:
        raise ValueError("Provide notional or qty")

    order = MarketOrderRequest(
        symbol=symbol,
        side=side,
        time_in_force=TimeInForce.DAY,
        notional=notional,
        qty=qty
    )
    return trading_client.submit_order(order)


def close_position(trading_client, symbol: str):
    # Alpaca has close_position endpoint in trading client
    return trading_client.close_position(symbol)


# ---------------------------
# MAIN LOOP
# ---------------------------

def run_bot(universe=None, cfg: Config = CFG):
    trading_client, data_client = make_clients()
    universe = safe_symbol_list(universe or DEFAULT_UNIVERSE)

    log(f"Starting bot | paper=True | universe_size={len(universe)} | TOP_N={cfg.TOP_N}")
    if cfg.DRY_RUN:
        log("DRY_RUN=True (will not place orders)")

    while True:
        # Check market open
        clock = trading_client.get_clock()
        if not clock.is_open:
            # Sleep until near next open
            log("Market closed. Waiting 60s...")
            time.sleep(60)
            continue

        # Align to 15-min cycle
        nxt = next_cycle_time(cfg.BAR_MINUTES)
        while now_utc() < nxt:
            if cfg.LOG_EVERYTHING:
                remaining = int((nxt - now_utc()).total_seconds())
                log(f"Waiting for next {cfg.BAR_MINUTES}m cycle... ({remaining}s)")
            time.sleep(cfg.SLEEP_SECONDS)

        cycle_end = floor_to_last_completed_15m(now_utc(), cfg.BAR_MINUTES)
        log(f"Cycle start. Using bars ending at {cycle_end.isoformat()}")

        # Fetch bars for the whole universe
        bars_needed = max(cfg.LOOKBACK_AVG_VOL_BARS + 2, cfg.TREND_BARS + 2, cfg.EXIT_VOL_DROP_BARS + 2)
        bars_map = fetch_15m_bars(data_client, universe, cycle_end, bars_needed, cfg)

        # Compute last 15m volume per symbol & filter by price
        vol_list = []
        for sym, df in bars_map.items():
            if len(df) < 2:
                continue
            last = df.iloc[-1]
            if last["close"] < cfg.MIN_PRICE:
                continue
            vol_list.append((sym, float(last["volume"]), float(last["close"])))

        if not vol_list:
            log("No bars returned / no candidates. Skipping cycle.")
            continue

        # Select top N by volume
       # Use entire universe (no top 20 filtering)
        top_syms = [sym for sym, _, _ in vol_list]
        log(f"Scanning full universe: {len(top_syms)} symbols")

        # Current positions
        positions = get_positions_map(trading_client)

        # 1) Exit checks for existing positions (even if not in top list)
        for sym, pos in list(positions.items()):
            df = bars_map.get(sym)
            if df is None:
                # If sym not in current universe bars, skip exit check this cycle
                continue
            if exit_condition_volume_drop(df, cfg):
                log(f"EXIT signal on {sym}: volume dropped consecutively ({cfg.EXIT_VOL_DROP_BARS} bars). Closing.")
                if not cfg.DRY_RUN:
                    try:
                        close_position(trading_client, sym)
                    except Exception as e:
                        log(f"Close failed for {sym}: {e}")

        # Refresh positions after exits
        positions = get_positions_map(trading_client)

        # 2) Entry checks for top symbols
        # Only enter new positions up to MAX_POSITIONS
        open_count = len(positions)
        capacity = max(0, cfg.MAX_POSITIONS - open_count)
        if capacity == 0:
            log("Max positions reached. No new entries this cycle.")
            continue

        # Decide entries
        candidates = []
        for sym in top_syms:
            if sym in positions:
                continue
            df = bars_map.get(sym)
            sig, score = compute_signal_from_bars(df, cfg)
            if sig is None:
                continue
            candidates.append((sym, sig, score))

        if not candidates:
            log("No entry signals this cycle.")
            continue

        # pick the single best candidate by score (highest volume ratio)
        candidates.sort(key=lambda x: x[2], reverse=True)
        entries = [candidates[0]]  # one trade only

        # Position sizing: equal notional per position based on equity
        acct = trading_client.get_account()
        equity = float(acct.equity)
        target_notional = equity * cfg.USE_EQUITY_FRACTION  # 50% of equity
        log(f"Placing up to {len(entries)} entries | equity={equity:.2f} | target_notional/pos={target_notional:.2f}")

        for sym, sig, score in entries:
            side = OrderSide.BUY if sig == "long" else OrderSide.SELL  # SELL opens short in margin accounts
            log(f"ENTRY {sig.upper()} {sym} (market) score={score:.2f} notional={target_notional:.2f}")
            if cfg.DRY_RUN:
                continue

            try:
                # Notional orders work for long; for short some accounts may require qty.
                # We'll attempt notional; if it fails, fallback to qty using last close.
                try:
                    submit_market_order(trading_client, sym, side=side, notional=target_notional)
                except Exception as e_notional:
                    last_px = float(bars_map[sym].iloc[-1]["close"])
                    qty = max(1, math.floor(target_notional / last_px))
                    log(f"Notional failed for {sym} ({e_notional}). Falling back to qty={qty}.")
                    submit_market_order(trading_client, sym, side=side, qty=qty)

            except Exception as e:
                log(f"Order failed for {sym}: {e}")

        log("Cycle complete.\n")


if __name__ == "__main__":
    run_bot()
