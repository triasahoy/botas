# file: app_dca_tokocrypto.py
# Streamlit dashboard + real-data backtester (Tokocrypto BTC/IDR) with TradingView-style DCA options

import os
import time
import math
import json
import logging
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pytz

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ccxt
import streamlit as st

# ---------------------------
# Logging (console + Streamlit friendliness)
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("DCA-Backtester")

# ---------------------------
# Config dataclasses
# ---------------------------
@dataclass
class ExchangeConfig:
    exchange_id: str = "tokocrypto"
    enable_rate_limit: bool = True
    timeout: int = 20000
    max_retries: int = 5
    retry_backoff_sec: float = 2.0
    api_key: Optional[str] = None
    secret: Optional[str] = None

@dataclass
class StrategyConfig:
    # TradingView-style knobs
    capital_per_trade: float = 5_000_000.0   # fixed IDR per buy
    fee_rate: float = 0.0015                 # 0.15%
    fee_in_cost_basis: bool = True           # include fees in avg price
    buy_mode: str = "monthly"                # "monthly", "weekday", "interval"
    buy_weekday: int = 0                     # 0=Mon .. 6=Sun (if buy_mode="weekday")
    interval_candles: int = 30               # every N candles (if buy_mode="interval")
    price_source: str = "open"               # TV default execution price
    close_all_on_last_candle: bool = False   # TV "Close All on last candle"

    # Monthly specifics
    buy_day_of_month: int = 1                # used in monthly mode

    # Execution realism
    slippage_bps: float = 0.0                # +bps price impact on buy

    # Timezone
    tz_name: str = "Asia/Jakarta"

@dataclass
class BacktestWindow:
    years: int

@dataclass
class BacktestReport:
    window_years: int
    symbol: str
    start_date: str
    end_date: str
    total_invested_idr: float
    total_btc: float
    avg_price_idr: float
    current_value_idr: float
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    volatility_monthly_pct: float
    sharpe_ratio: Optional[float]
    fees_paid_idr: float
    monthly_breakdown: pd.DataFrame
    equity_curve: pd.DataFrame

# ---------------------------
# Utilities
# ---------------------------
def now_jakarta() -> datetime:
    return datetime.now(pytz.timezone("Asia/Jakarta")).replace(microsecond=0)

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def from_ms(ms: int, tz: str = "Asia/Jakarta") -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=pytz.timezone(tz))

def retryable(func, max_retries=5, backoff=2.0, *args, **kwargs):
    last_err = None
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            time.sleep(backoff ** attempt)
    raise last_err

# ---------------------------
# Exchange client
# ---------------------------
class ExchangeClient:
    def __init__(self, cfg: ExchangeConfig):
        ex_class = getattr(ccxt, cfg.exchange_id)
        params = {
            "enableRateLimit": cfg.enable_rate_limit,
            "timeout": cfg.timeout,
        }
        self.exchange = ex_class(params)
        if cfg.api_key and cfg.secret:
            self.exchange.apiKey = cfg.api_key
            self.exchange.secret = cfg.secret
        self.max_retries = cfg.max_retries
        self.retry_backoff_sec = cfg.retry_backoff_sec

    def load_markets(self) -> Dict[str, Any]:
        return retryable(self.exchange.load_markets, self.max_retries, self.retry_backoff_sec)

    def resolve_symbol(self, preferred: List[str]) -> str:
        markets = self.load_markets()
        for s in preferred:
            if s in markets:
                return s
        candidates = [m for m in markets.keys() if m.startswith("BTC/") and ("IDR" in m or "IDRT" in m or "BIDR" in m)]
        if candidates:
            return candidates[0]
        raise ValueError("Could not resolve BTC/IDR-like symbol on Tokocrypto.")

    def fetch_ohlcv(self, symbol: str, timeframe: str, since_ms: int, until_ms: int, limit: int = 1000) -> List[List[float]]:
        all_rows = []
        cursor = since_ms
        while True:
            chunk = retryable(
                self.exchange.fetch_ohlcv,
                self.max_retries,
                self.retry_backoff_sec,
                symbol,
                timeframe=timeframe,
                since=cursor,
                limit=limit,
            )
            if not chunk:
                break
            all_rows.extend(chunk)
            last_ts = chunk[-1][0]
            if last_ts >= until_ms or len(chunk) < limit:
                break
            cursor = last_ts + 1
        return [row for row in all_rows if since_ms <= row[0] <= until_ms]

    def fetch_ticker_safe(self, symbols: List[str]) -> Optional[dict]:
        for sym in symbols:
            try:
                return retryable(self.exchange.fetch_ticker, self.max_retries, self.retry_backoff_sec, sym)
            except Exception:
                continue
        return None

# ---------------------------
# Data preparation
# ---------------------------
def ohlcv_to_df(rows: List[List[float]], tz: str = "Asia/Jakarta") -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["time","open","high","low","close","volume"])
    df = pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"])
    df["time"] = df["ts"].apply(lambda x: from_ms(x, tz))
    df = df.drop(columns=["ts"])
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    df = df[(df["close"] > 0) & (df["high"] > 0) & (df["low"] > 0)]
    return df

def first_close_on_or_after(df: pd.DataFrame, target_dt: datetime, price_col: str = "close") -> Optional[Tuple[datetime, float]]:
    rows = df[df["time"] >= target_dt]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return row["time"], float(row[price_col])

# ---------------------------
# Schedule generation
# ---------------------------
def generate_monthly_schedule(start_dt: datetime, end_dt: datetime, day_of_month: int) -> List[datetime]:
    schedule = []
    candidate = start_dt.replace(day=1)
    dm = min(day_of_month, (candidate + relativedelta(months=1) - timedelta(days=1)).day)
    first_buy = candidate.replace(day=dm, hour=0, minute=0, second=0, microsecond=0)
    if first_buy < start_dt:
        nm = candidate + relativedelta(months=1)
        dm = min(day_of_month, (nm + relativedelta(months=1) - timedelta(days=1)).day)
        first_buy = nm.replace(day=dm, hour=0, minute=0, second=0, microsecond=0)
    cur_buy = first_buy
    while cur_buy <= end_dt:
        schedule.append(cur_buy)
        nm = cur_buy + relativedelta(months=1)
        dm = min(day_of_month, (nm + relativedelta(months=1) - timedelta(days=1)).day)
        cur_buy = nm.replace(day=dm, hour=0, minute=0, second=0, microsecond=0)
    return schedule

def generate_buy_schedule(df: pd.DataFrame, cfg: StrategyConfig) -> List[datetime]:
    if df.empty:
        return []
    if cfg.buy_mode == "monthly":
        return generate_monthly_schedule(df["time"].min(), df["time"].max(), cfg.buy_day_of_month)
    elif cfg.buy_mode == "weekday":
        return [t.to_pydatetime() if hasattr(t, "to_pydatetime") else t
                for t in df["time"].tolist() if t.weekday() == cfg.buy_weekday]
    elif cfg.buy_mode == "interval":
        return [t.to_pydatetime() if hasattr(t, "to_pydatetime") else t
                for t in df["time"].iloc[::cfg.interval_candles].tolist()]
    else:
        raise ValueError(f"Unknown buy_mode: {cfg.buy_mode}")

# ---------------------------
# Risk helpers
# ---------------------------
def compute_max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min()) if len(dd) else 0.0

def annualized_return_from_monthly(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    prod = (1.0 + returns).prod()
    years = len(returns) / 12.0
    if years <= 0:
        return 0.0
    return float(prod ** (1.0 / years) - 1.0)

def sharpe_from_monthly(returns: pd.Series, rf: float = 0.0) -> Optional[float]:
    if returns.empty:
        return None
    excess = returns - rf / 12.0
    mu = excess.mean()
    sigma = excess.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return None
    return float((mu / sigma) * math.sqrt(12.0))

# ---------------------------
# Core backtest
# ---------------------------
def run_dca_backtest(
    df_daily: pd.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
    strat_cfg: StrategyConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float, float]:
    df_win = df_daily[(df_daily["time"] >= start_dt) & (df_daily["time"] <= end_dt)].copy()
    if df_win.empty:
        return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0, 0.0

    schedule = generate_buy_schedule(df_win, strat_cfg)

    records = []
    cum_btc = 0.0
    total_cost = 0.0
    total_fees = 0.0

    for target_dt in schedule:
        got = first_close_on_or_after(df_win, target_dt, strat_cfg.price_source)
        if not got:
            continue
        px_time, px = got
        px_exec = px * (1.0 + strat_cfg.slippage_bps / 1e4)
        fee = strat_cfg.capital_per_trade * strat_cfg.fee_rate
        idr_after_fee = strat_cfg.capital_per_trade - fee
        qty_btc = idr_after_fee / px_exec

        cum_btc += qty_btc
        total_cost += strat_cfg.capital_per_trade
        total_fees += fee

        records.append({
            "buy_time": px_time,
            "scheduled_time": target_dt,
            "price_idr": px,
            "price_exec_idr": px_exec,
            "idr_invested": strat_cfg.capital_per_trade,
            "fee_idr": fee,
            "btc_bought": qty_btc,
            "cum_btc": cum_btc,
            "cum_invested": total_cost,
            "cum_fees": total_fees,
        })

    monthly_df = pd.DataFrame(records).sort_values("buy_time").reset_index(drop=True)

    df_ec = df_win[["time", "open", "close"]].copy()
    df_ec["cum_btc"] = 0.0
    if not monthly_df.empty:
        buys = monthly_df[["buy_time","cum_btc"]].values.tolist()
        idx = 0
        cur_cum = 0.0
        times = df_ec["time"].tolist()
        cum_vals = []
        for t in times:
            while idx < len(buys) and buys[idx][0] <= t:
                cur_cum = buys[idx][1]
                idx += 1
            cum_vals.append(cur_cum)
        df_ec["cum_btc"] = cum_vals

    if strat_cfg.close_all_on_last_candle and not df_ec.empty:
        final_price = df_ec[strat_cfg.price_source].iloc[-1]
        final_value = df_ec["cum_btc"].iloc[-1] * final_price
        df_ec["equity_idr"] = df_ec["cum_btc"] * df_ec["close"]
        df_ec.loc[df_ec.index[-1], "cum_btc"] = 0.0
        df_ec.loc[df_ec.index[-1], "equity_idr"] = final_value
    else:
        df_ec["equity_idr"] = df_ec["cum_btc"] * df_ec["close"]

    return monthly_df, df_ec[["time","close","cum_btc","equity_idr"]].copy(), cum_btc, total_cost, total_fees

# ---------------------------
# Reporting
# ---------------------------
def build_report(
    window_years: int,
    symbol: str,
    monthly_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    total_btc: float,
    total_cost: float,
    total_fees: float,
    strat_cfg: StrategyConfig
) -> BacktestReport:
    if equity_df.empty:
        raise ValueError("Equity curve is empty; cannot build report.")

    start_dt = equity_df["time"].min()
    end_dt = equity_df["time"].max()
    latest_close = equity_df["close"].iloc[-1]
    current_value = total_btc * latest_close
    total_return = (current_value - total_cost) / total_cost if total_cost > 0 else 0.0

    if total_btc > 0:
        if strat_cfg.fee_in_cost_basis:
            avg_price = total_cost / total_btc
        else:
            avg_price = (total_cost - total_fees) / total_btc
    else:
        avg_price = 0.0

    max_dd = compute_max_drawdown(equity_df["equity_idr"])

    mdf = equity_df.copy().set_index("time")
    monthly_equity = mdf["equity_idr"].resample("M").last().dropna()
    monthly_returns = monthly_equity.pct_change().dropna()

    ann_return = annualized_return_from_monthly(monthly_returns)
    monthly_vol = monthly_returns.std(ddof=1) if len(monthly_returns) > 1 else 0.0
    sharpe = sharpe_from_monthly(monthly_returns, rf=0.0)

    return BacktestReport(
        window_years=window_years,
        symbol=symbol,
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
        total_invested_idr=float(total_cost),
        total_btc=float(total_btc),
        avg_price_idr=float(avg_price),
        current_value_idr=float(current_value),
        total_return_pct=float(total_return * 100.0),
        annualized_return_pct=float(ann_return * 100.0),
        max_drawdown_pct=float(max_dd * 100.0),
        volatility_monthly_pct=float(monthly_vol * 100.0 if not np.isnan(monthly_vol) else 0.0),
        sharpe_ratio=sharpe,
        fees_paid_idr=float(total_fees),
        monthly_breakdown=monthly_df,
        equity_curve=equity_df,
    )

def summary_table(reports: List[BacktestReport]) -> pd.DataFrame:
    rows = []
    for r in reports:
        rows.append({
            "WindowYears": r.window_years,
            "Symbol": r.symbol,
            "Start": r.start_date,
            "End": r.end_date,
            "TotalInvested_IDR": round(r.total_invested_idr, 2),
            "TotalBTC": round(r.total_btc, 8),
            "AvgPrice_IDR": round(r.avg_price_idr, 2),
            "CurrentValue_IDR": round(r.current_value_idr, 2),
            "TotalReturn_%": round(r.total_return_pct, 2),
            "AnnualizedReturn_%": round(r.annualized_return_pct, 2),
            "MaxDrawdown_%": round(r.max_drawdown_pct, 2),
            "VolatilityMonthly_%": round(r.volatility_monthly_pct, 2),
            "Sharpe": None if r.sharpe_ratio is None else round(r.sharpe_ratio, 2),
            "FeesPaid_IDR": round(r.fees_paid_idr, 2),
        })
    return pd.DataFrame(rows).sort_values("WindowYears")

# ---------------------------
# Data orchestration for Streamlit
# ---------------------------
@st.cache_data(show_spinner=False)
def fetch_daily_data(symbols_preferred: List[str], years: int, exch_cfg: ExchangeConfig, tz: str) -> Tuple[str, pd.DataFrame]:
    client = ExchangeClient(exch_cfg)
    symbol = client.resolve_symbol(symbols_preferred)
    end_dt = now_jakarta().replace(hour=23, minute=59, second=59)
    start_dt = end_dt - relativedelta(years=years) - timedelta(days=7)
    rows = client.fetch_ohlcv(symbol, "1d", to_ms(start_dt), to_ms(end_dt), limit=1000)
    df_daily = ohlcv_to_df(rows, tz=tz)
    return symbol, df_daily

@st.cache_data(ttl=20, show_spinner=False)
def fetch_live_ticker(symbols_preferred: List[str], exch_cfg: ExchangeConfig) -> Optional[dict]:
    client = ExchangeClient(exch_cfg)
    return client.fetch_ticker_safe(symbols_preferred)

def run_windows_backtests(df_daily: pd.DataFrame, symbol: str, strat_cfg: StrategyConfig, end_dt: datetime, windows: List[int]) -> List[BacktestReport]:
    reports = []
    for yrs in windows:
        start_dt = end_dt - relativedelta(years=yrs)
        df_win = df_daily[(df_daily["time"] >= start_dt - timedelta(days=7)) & (df_daily["time"] <= end_dt)].copy()
        monthly_df, equity_df, total_btc, total_cost, total_fees = run_dca_backtest(
            df_daily=df_win,
            start_dt=start_dt,
            end_dt=end_dt,
            strat_cfg=strat_cfg,
        )
        report = build_report(
            window_years=yrs,
            symbol=symbol,
            monthly_df=monthly_df,
            equity_df=equity_df,
            total_btc=total_btc,
            total_cost=total_cost,
            total_fees=total_fees,
            strat_cfg=strat_cfg
        )
        reports.append(report)
    return reports

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="BTC/IDR DCA Backtester (Tokocrypto)", layout="wide")
st.title("📈 BTC/IDR DCA Backtester — Tokocrypto")

# Live price header
col_lp1, col_lp2, col_lp3 = st.columns(3)
with col_lp1:
    st.caption("Live market")
with col_lp2:
    st.caption("Updated every ~20s")
with col_lp3:
    st.caption(now_jakarta().strftime("%Y-%m-%d %H:%M:%S %Z"))

exch_cfg = ExchangeConfig(
    api_key=os.getenv("TOKO_API_KEY"),
    secret=os.getenv("TOKO_SECRET"),
)

preferred_symbols = ["BTC/IDR", "BTC/IDRT", "BTC/BIDR"]

ticker = fetch_live_ticker(preferred_symbols, exch_cfg)
if ticker:
    st.metric(
        label="BTC/IDR Last",
        value=f"{ticker.get('last', float('nan')):,.0f} IDR",
        delta=(f"{ticker.get('percentage'):.2f}%" if ticker.get('percentage') is not None else None),
        help=f"Symbol: {ticker.get('symbol','N/A')}"
    )
else:
    st.info("Unable to fetch live ticker right now.")

st.divider()

# Sidebar controls
st.sidebar.header("Strategy parameters")

buy_mode = st.sidebar.selectbox("Buy mode", ["monthly", "weekday", "interval"], index=0)
price_source = st.sidebar.selectbox("Execution price", ["open", "close"], index=0)
fee_in_cost_basis = st.sidebar.checkbox("Include fees in cost basis", True)
close_all_on_last_candle = st.sidebar.checkbox("Close all on last candle", False)

col1, col2 = st.sidebar.columns(2)
with col1:
    buy_day_of_month = st.number_input("Day of month", min_value=1, max_value=28, value=1, step=1)
with col2:
    buy_weekday = st.number_input("Weekday (0=Mon)", min_value=0, max_value=6, value=0, step=1)

interval_candles = st.sidebar.number_input("Interval candles", min_value=1, max_value=365, value=30, step=1)
capital_per_trade = st.sidebar.number_input("Capital per trade (IDR)", min_value=10000, max_value=100_000_000, value=5_000_000, step=10000)
fee_rate = st.sidebar.number_input("Fee rate", min_value=0.0, max_value=0.01, value=0.0015, step=0.0001, format="%.4f")
slippage_bps = st.sidebar.number_input("Slippage (bps)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)

windows_sel = st.sidebar.multiselect("Backtest windows (years)", [1, 3, 5], default=[1, 3, 5])

st.sidebar.header("Data controls")
hist_years_fetch = st.sidebar.selectbox("Historical fetch range", [1, 3, 5], index=2)
refresh_data = st.sidebar.button("Refresh historical data")

# Build strategy config
strat_cfg = StrategyConfig(
    capital_per_trade=float(capital_per_trade),
    fee_rate=float(fee_rate),
    fee_in_cost_basis=fee_in_cost_basis,
    buy_mode=buy_mode,
    buy_day_of_month=int(buy_day_of_month),
    buy_weekday=int(buy_weekday),
    interval_candles=int(interval_candles),
    price_source=price_source,
    close_all_on_last_candle=close_all_on_last_candle,
    slippage_bps=float(slippage_bps),
    tz_name="Asia/Jakarta",
)

# Historical data fetch (cached)
if refresh_data or "hist_data_key" not in st.session_state or st.session_state.get("hist_data_years") != hist_years_fetch:
    try:
        symbol, df_daily = fetch_daily_data(preferred_symbols, hist_years_fetch, exch_cfg, strat_cfg.tz_name)
        st.session_state["hist_data_key"] = f"{symbol}_{hist_years_fetch}"
        st.session_state["hist_data_years"] = hist_years_fetch
        st.session_state["symbol"] = symbol
        st.session_state["df_daily"] = df_daily
        st.success(f"Historical data loaded for {symbol} ({hist_years_fetch}y).")
    except Exception as e:
        st.error(f"Failed to load historical data: {e}")

symbol = st.session_state.get("symbol")
df_daily = st.session_state.get("df_daily")

run_bt = st.button("Run backtests")
if run_bt:
    if df_daily is None or df_daily.empty or symbol is None:
        st.error("Historical data not available. Load data first.")
    else:
        with st.spinner("Running backtests..."):
            end_dt = now_jakarta().replace(hour=23, minute=59, second=59)
            try:
                reports = run_windows_backtests(df_daily, symbol, strat_cfg, end_dt, windows_sel)
                st.session_state["reports"] = reports
                st.success("Backtests completed.")
            except Exception as e:
                st.error(f"Backtest error: {e}")

reports: List[BacktestReport] = st.session_state.get("reports", [])

# Render results
if reports:
    st.subheader("Summary metrics")
    df_summary = summary_table(reports)
    st.dataframe(df_summary, use_container_width=True)

    # Download button for summary
    st.download_button(
        label="Download summary CSV",
        data=df_summary.to_csv(index=False).encode("utf-8"),
        file_name="summary_all_windows.csv",
        mime="text/csv"
    )

    # Tabs per window
    tabs = st.tabs([f"{r.window_years}y" for r in reports])
    for tab, rep in zip(tabs, reports):
        with tab:
            colA, colB, colC = st.columns(3)
            colA.metric("Total invested (IDR)", f"{rep.total_invested_idr:,.0f}")
            colB.metric("Total BTC", f"{rep.total_btc:.8f}")
            colC.metric("Avg price (IDR/BTC)", f"{rep.avg_price_idr:,.0f}")

            colD, colE, colF = st.columns(3)
            colD.metric("Current value (IDR)", f"{rep.current_value_idr:,.0f}")
            colE.metric("Total return (%)", f"{rep.total_return_pct:.2f}%")
            colF.metric("Annualized (%)", f"{rep.annualized_return_pct:.2f}%")

            colG, colH, colI = st.columns(3)
            colG.metric("Max drawdown (%)", f"{rep.max_drawdown_pct:.2f}%")
            colH.metric("Volatility monthly (%)", f"{rep.volatility_monthly_pct:.2f}%")
            colI.metric("Sharpe", f"{0.0 if rep.sharpe_ratio is None else rep.sharpe_ratio:.2f}")

            # Charts
            st.markdown("**Equity curve (IDR)**")
            eq_df = rep.equity_curve.copy().set_index("time")[["equity_idr"]]
            st.line_chart(eq_df)

            st.markdown("**Cumulative BTC**")
            btc_df = rep.equity_curve.copy().set_index("time")[["cum_btc"]]
            st.line_chart(btc_df)

            # Monthly breakdown table with download
            st.markdown("**Monthly breakdown**")
            mb = rep.monthly_breakdown.copy()
            mb["buy_time"] = pd.to_datetime(mb["buy_time"]).dt.strftime("%Y-%m-%d")
            mb["scheduled_time"] = pd.to_datetime(mb["scheduled_time"]).dt.strftime("%Y-%m-%d")
            st.dataframe(mb, use_container_width=True, height=300)
            st.download_button(
                label=f"Download monthly breakdown ({rep.window_years}y)",
                data=mb.to_csv(index=False).encode("utf-8"),
                file_name=f"monthly_breakdown_{rep.window_years}y.csv",
                mime="text/csv"
            )

# Gentle auto-refresh for live price
st.caption("Auto-refreshing live price...")
st.experimental_rerun  # no-op reference to avoid lint complaints
st_autorefresh = st.sidebar.checkbox("Auto-refresh live price (20s)", value=True)
if st_autorefresh:
    st.experimental_set_query_params(_=int(time.time()))  # bust cache occasionally
    st.experimental_singleton.clear()  # safe-ish in this small app
    st.cache_data.clear()              # refresh cached ticker
    st.experimental_memo.clear()       # legacy safety
    st.toast("Refreshing live price...", icon="🔄")
    time.sleep(0.1)