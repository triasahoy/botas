# file: app_dca_tokocrypto_id_modern_plus.py
# Streamlit modern dashboard — DCA BTC/IDR (Tokocrypto) dgn format Indonesia, 1D/1H, benchmark fee saat jual, multi-slippage, ekspor ZIP
# Kredensial diambil dari environment: TOKO_API_KEY, TOKO_SECRET

import os
import io
import time
import math
import zipfile
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pytz

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import ccxt
import streamlit as st
from babel.numbers import format_currency, format_decimal

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("DCA-ID-Modern-Plus")

# ---------------------------
# Konfigurasi
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
    capital_per_trade: float = 5_000_000.0
    fee_rate: float = 0.0015
    fee_in_cost_basis: bool = True
    buy_mode: str = "monthly"           # "monthly","weekday","interval"
    buy_weekday: int = 0
    interval_candles: int = 30
    price_source: str = "open"          # "open"|"close"
    close_all_on_last_candle: bool = False
    buy_day_of_month: int = 1
    slippage_bps: float = 0.0
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
    slippage_bps: float = 0.0           # untuk perbandingan skenario

# ---------------------------
# Utilitas waktu/format
# ---------------------------
def now_jakarta() -> datetime:
    return datetime.now(pytz.timezone("Asia/Jakarta")).replace(microsecond=0)

def to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def from_ms(ms: int, tz: str = "Asia/Jakarta") -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=pytz.timezone(tz))

def fmt_idr(val: float) -> str:
    try:
        return format_currency(val, "IDR", locale="id_ID")
    except Exception:
        return f"Rp {val:,.0f}".replace(",", ".")

def fmt_dec(val: float, digits: int = 2) -> str:
    try:
        fmt = "#,##0." + "0"*digits
        return format_decimal(val, locale="id_ID", format=fmt)
    except Exception:
        return f"{val:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_pct(val_pct: float, digits: int = 2) -> str:
    return fmt_dec(val_pct, digits) + "%"

# ---------------------------
# Exchange client (CCXT)
# ---------------------------
class ExchangeClient:
    def __init__(self, cfg: ExchangeConfig):
        ex_class = getattr(ccxt, cfg.exchange_id)
        params = {"enableRateLimit": cfg.enable_rate_limit, "timeout": cfg.timeout}
        self.exchange = ex_class(params)
        if cfg.api_key and cfg.secret:
            self.exchange.apiKey = cfg.api_key
            self.exchange.secret = cfg.secret
        self.max_retries = cfg.max_retries
        self.retry_backoff_sec = cfg.retry_backoff_sec

    def _retry(self, func, *args, **kwargs):
        last = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last = e
                time.sleep(self.retry_backoff_sec ** attempt)
        raise last

    def load_markets(self) -> Dict[str, Any]:
        return self._retry(self.exchange.load_markets)

    def resolve_symbol(self, preferred: List[str]) -> str:
        markets = self.load_markets()
        for s in preferred:
            if s in markets:
                return s
        cands = [m for m in markets if m.startswith("BTC/") and ("IDR" in m or "IDRT" in m or "BIDR" in m)]
        if cands:
            return cands[0]
        raise ValueError("Tidak menemukan simbol BTC/IDR di Tokocrypto.")

    def fetch_ohlcv(self, symbol: str, timeframe: str, since_ms: int, until_ms: int, limit: int = 1000) -> List[List[float]]:
        all_rows = []
        cursor = since_ms
        while True:
            chunk = self._retry(self.exchange.fetch_ohlcv, symbol, timeframe=timeframe, since=cursor, limit=limit)
            if not chunk:
                break
            all_rows.extend(chunk)
            last_ts = chunk[-1][0]
            if last_ts >= until_ms or len(chunk) < limit:
                break
            cursor = last_ts + 1
        return [r for r in all_rows if since_ms <= r[0] <= until_ms]

    def fetch_ticker_safe(self, symbols: List[str]) -> Optional[dict]:
        for s in symbols:
            try:
                return self._retry(self.exchange.fetch_ticker, s)
            except Exception:
                continue
        return None

# ---------------------------
# Data prep
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

def first_price_on_or_after(df: pd.DataFrame, target_dt: datetime, price_col: str = "close") -> Optional[Tuple[datetime, float]]:
    rows = df[df["time"] >= target_dt]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return row["time"], float(row[price_col])

# ---------------------------
# Penjadwalan beli
# ---------------------------
def generate_monthly_schedule(start_dt: datetime, end_dt: datetime, day_of_month: int) -> List[datetime]:
    schedule = []
    candidate = start_dt.replace(day=1)
    last_day = (candidate + relativedelta(months=1) - timedelta(days=1)).day
    dm = min(day_of_month, last_day)
    first_buy = candidate.replace(day=dm, hour=0, minute=0, second=0, microsecond=0)
    if first_buy < start_dt:
        nm = candidate + relativedelta(months=1)
        last_day = (nm + relativedelta(months=1) - timedelta(days=1)).day
        dm = min(day_of_month, last_day)
        first_buy = nm.replace(day=dm, hour=0, minute=0, second=0, microsecond=0)
    cur = first_buy
    while cur <= end_dt:
        schedule.append(cur)
        nm = cur + relativedelta(months=1)
        last_day = (nm + relativedelta(months=1) - timedelta(days=1)).day
        dm = min(day_of_month, last_day)
        cur = nm.replace(day=dm, hour=0, minute=0, second=0, microsecond=0)
    return schedule

def generate_buy_schedule(df: pd.DataFrame, cfg: StrategyConfig) -> List[datetime]:
    if df.empty:
        return []
    if cfg.buy_mode == "monthly":
        return generate_monthly_schedule(df["time"].min(), df["time"].max(), cfg.buy_day_of_month)
    elif cfg.buy_mode == "weekday":
        return [t.to_pydatetime() if hasattr(t, "to_pydatetime") else t for t in df["time"].tolist() if t.weekday() == cfg.buy_weekday]
    elif cfg.buy_mode == "interval":
        return [t.to_pydatetime() if hasattr(t, "to_pydatetime") else t for t in df["time"].iloc[::cfg.interval_candles].tolist()]
    else:
        raise ValueError(f"buy_mode tidak dikenali: {cfg.buy_mode}")

# ---------------------------
# Risk metrics
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
# Backtest inti (DCA)
# ---------------------------
def run_dca_backtest(
    df: pd.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
    strat_cfg: StrategyConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float, float]:
    df_win = df[(df["time"] >= start_dt) & (df["time"] <= end_dt)].copy()
    if df_win.empty:
        return pd.DataFrame(), pd.DataFrame(), 0.0, 0.0, 0.0

    schedule = generate_buy_schedule(df_win, strat_cfg)

    recs = []
    cum_btc, total_cost, total_fees = 0.0, 0.0, 0.0

    for target_dt in schedule:
        got = first_price_on_or_after(df_win, target_dt, strat_cfg.price_source)
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

        recs.append({
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

    monthly_df = pd.DataFrame(recs).sort_values("buy_time").reset_index(drop=True)

    ec = df_win[["time","open","close"]].copy()
    ec["cum_btc"] = 0.0
    if not monthly_df.empty:
        buys = monthly_df[["buy_time","cum_btc"]].values.tolist()
        idx, cur_cum = 0, 0.0
        cum_vals = []
        for t in ec["time"].tolist():
            while idx < len(buys) and buys[idx][0] <= t:
                cur_cum = buys[idx][1]
                idx += 1
            cum_vals.append(cur_cum)
        ec["cum_btc"] = cum_vals

    if strat_cfg.close_all_on_last_candle and not ec.empty:
        final_price = ec[strat_cfg.price_source].iloc[-1]
        final_value = ec["cum_btc"].iloc[-1] * final_price
        ec["equity_idr"] = ec["cum_btc"] * ec["close"]
        ec.loc[ec.index[-1], "cum_btc"] = 0.0
        ec.loc[ec.index[-1], "equity_idr"] = final_value
    else:
        ec["equity_idr"] = ec["cum_btc"] * ec["close"]

    return monthly_df, ec[["time","close","cum_btc","equity_idr"]].copy(), cum_btc, total_cost, total_fees

# ---------------------------
# Benchmark Buy & Hold
# ---------------------------
def run_buy_and_hold_benchmark(
    df: pd.DataFrame,
    start_dt: datetime,
    end_dt: datetime,
    total_invest_idr: float,
    price_source: str,
    include_sell_fee: bool,
    fee_rate: float
) -> pd.DataFrame:
    df_win = df[(df["time"] >= start_dt) & (df["time"] <= end_dt)].copy()
    if df_win.empty:
        return pd.DataFrame(columns=["time","bh_value_idr"])

    buy_price = float(df_win[price_source].iloc[0])
    btc_qty = total_invest_idr / buy_price
    df_win["bh_value_idr"] = btc_qty * df_win["close"]
    if include_sell_fee:
        # Terapkan biaya jual hanya pada titik terakhir (realized)
        df_win.loc[df_win.index[-1], "bh_value_idr"] *= (1.0 - fee_rate)
    return df_win[["time","bh_value_idr"]].copy()

# ---------------------------
# Ringkasan & format
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
        raise ValueError("Equity curve kosong.")

    start_dt = equity_df["time"].min()
    end_dt = equity_df["time"].max()
    latest_close = equity_df["close"].iloc[-1]
    current_value = total_btc * latest_close
    total_return = (current_value - total_cost) / total_cost if total_cost > 0 else 0.0

    if total_btc > 0:
        avg_price = (total_cost / total_btc) if strat_cfg.fee_in_cost_basis else ((total_cost - total_fees) / total_btc)
    else:
        avg_price = 0.0

    max_dd = compute_max_drawdown(equity_df["equity_idr"])

    mdf = equity_df.copy().set_index("time")
    monthly_equity = mdf["equity_idr"].resample("M").last().dropna()
    monthly_returns = monthly_equity.pct_change().dropna()

    # Annualized & volatility & Sharpe
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
        slippage_bps=float(strat_cfg.slippage_bps),
    )

def summary_table(reports: List[BacktestReport]) -> pd.DataFrame:
    rows = []
    for r in reports:
        rows.append({
            "Periode (tahun)": r.window_years,
            "Slippage (bps)": r.slippage_bps,
            "Simbol": r.symbol,
            "Mulai": r.start_date,
            "Akhir": r.end_date,
            "Total Investasi (IDR)": r.total_invested_idr,
            "Total BTC": r.total_btc,
            "Harga Rata2 (IDR/BTC)": r.avg_price_idr,
            "Nilai Saat Ini (IDR)": r.current_value_idr,
            "Total Return (%)": r.total_return_pct,
            "Annualized (%)": r.annualized_return_pct,
            "Max Drawdown (%)": r.max_drawdown_pct,
            "Volatilitas Bulanan (%)": r.volatility_monthly_pct,
            "Sharpe": 0.0 if r.sharpe_ratio is None else r.sharpe_ratio,
            "Total Biaya (IDR)": r.fees_paid_idr,
        })
    df = pd.DataFrame(rows).sort_values(["Periode (tahun)","Slippage (bps)"])
    # Format angka Indonesia
    for col in ["Total Investasi (IDR)", "Harga Rata2 (IDR/BTC)", "Nilai Saat Ini (IDR)", "Total Biaya (IDR)"]:
        df[col] = df[col].apply(lambda x: fmt_idr(float(x)))
    df["Total BTC"] = df["Total BTC"].apply(lambda x: fmt_dec(float(x), 8))
    for col in ["Total Return (%)","Annualized (%)","Max Drawdown (%)","Volatilitas Bulanan (%)"]:
        df[col] = df[col].apply(lambda x: fmt_dec(float(x), 2) + "%")
    df["Sharpe"] = df["Sharpe"].apply(lambda x: fmt_dec(float(x), 2))
    df["Slippage (bps)"] = df["Slippage (bps)"].apply(lambda x: fmt_dec(float(x), 1))
    return df

# ---------------------------
# Streamlit UI — Modern
# ---------------------------
st.set_page_config(page_title="DCA BTC/IDR — Tokocrypto (Modern+)", layout="wide")

# CSS Modern
st.markdown("""
<style>
:root { --card-bg: #0b1020; --card-brd: #1f2a44; --text: #e5e7eb; --muted:#9ca3af; }
.block-container { padding-top: 1.2rem; }
.metric-card { background: var(--card-bg); border: 1px solid var(--card-brd); border-radius: 12px; padding: 14px 16px; color: var(--text); }
.metric-label { font-size: 12px; color: var(--muted); }
.metric-value { font-size: 20px; font-weight: 700; }
hr { border: none; border-top: 1px solid rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)

st.markdown("<h2 style='margin:0 0 6px 0'>✨ DCA BTC/IDR — Tokocrypto (Live, Benchmark, Multi‑Slippage)</h2>", unsafe_allow_html=True)
st.caption("UI modern • Real‑time • DCA gaya TradingView • 1D / 1H • Benchmark B&H dgn fee jual • Perbandingan slippage • Ekspor ZIP")

# Kredensial aman via environment
exch_cfg = ExchangeConfig(
    api_key=os.getenv("TOKO_API_KEY"),
    secret=os.getenv("TOKO_SECRET"),
)

preferred_symbols = ["BTC/IDR", "BTC/IDRT", "BTC/BIDR"]

@st.cache_data(ttl=20, show_spinner=False)
def get_live_ticker() -> Optional[dict]:
    client = ExchangeClient(exch_cfg)
    return client.fetch_ticker_safe(preferred_symbols)

@st.cache_data(show_spinner=False)
def fetch_data(symbols_preferred: List[str], years: int, timeframe: str, tz: str) -> Tuple[str, pd.DataFrame]:
    client = ExchangeClient(exch_cfg)
    symbol = client.resolve_symbol(symbols_preferred)
    end_dt = now_jakarta()
    start_dt = end_dt - relativedelta(years=years) - timedelta(days=7)
    tf_map = {"1D": "1d", "1H": "1h"}
    tf = tf_map[timeframe]
    rows = client.fetch_ohlcv(symbol, tf, to_ms(start_dt), to_ms(end_dt), limit=1000)
    df = ohlcv_to_df(rows, tz=tz)
    return symbol, df

def run_backtests_for_slippages(
    df_hist: pd.DataFrame,
    symbol: str,
    base_cfg: StrategyConfig,
    end_dt: datetime,
    years_list: List[int],
    slippage_scenarios: List[float]
) -> List[BacktestReport]:
    reports_all: List[BacktestReport] = []
    for sbps in slippage_scenarios:
        cfg = StrategyConfig(**{**base_cfg.__dict__})
        cfg.slippage_bps = sbps
        for yrs in years_list:
            start_dt = end_dt - relativedelta(years=yrs)
            df_win = df_hist[(df_hist["time"] >= start_dt - timedelta(days=7)) & (df_hist["time"] <= end_dt)].copy()
            mdf, ecdf, tot_btc, tot_cost, tot_fee = run_dca_backtest(df_win, start_dt, end_dt, cfg)
            rep = build_report(yrs, symbol, mdf, ecdf, tot_btc, tot_cost, tot_fee, cfg)
            reports_all.append(rep)
    return reports_all

def build_benchmarks_for_reports(
    df_hist: pd.DataFrame,
    reports: List[BacktestReport],
    price_source: str,
    include_sell_fee: bool,
    fee_rate: float
) -> Dict[Tuple[int,float], pd.DataFrame]:
    bh_map: Dict[Tuple[int,float], pd.DataFrame] = {}
    for rep in reports:
        n_trades = len(rep.monthly_breakdown)
        total_invest = n_trades * rep.monthly_breakdown["idr_invested"].mean() if n_trades > 0 else 0.0
        df_sub = df_hist[(df_hist["time"] >= pd.to_datetime(rep.start_date)) & (df_hist["time"] <= pd.to_datetime(rep.end_date))]
        bh_df = run_buy_and_hold_benchmark(
            df=df_sub,
            start_dt=pd.to_datetime(rep.start_date),
            end_dt=pd.to_datetime(rep.end_date),
            total_invest_idr=total_invest,
            price_source=price_source,
            include_sell_fee=include_sell_fee,
            fee_rate=fee_rate
        )
        bh_map[(rep.window_years, rep.slippage_bps)] = bh_df
    return bh_map

# Live header
ticker = get_live_ticker()
cols = st.columns(3)
with cols[0]:
    st.markdown("<div class='metric-card'><div class='metric-label'>Harga Live</div>", unsafe_allow_html=True)
    if ticker:
        st.markdown(f"<div class='metric-value'>{fmt_idr(float(ticker.get('last', 0)))}</div></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='metric-value'>-</div></div>", unsafe_allow_html=True)
with cols[1]:
    st.markdown("<div class='metric-card'><div class='metric-label'>Symbol</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{ticker.get('symbol','BTC/IDR') if ticker else 'BTC/IDR'}</div></div>", unsafe_allow_html=True)
with cols[2]:
    st.markdown("<div class='metric-card'><div class='metric-label'>Terakhir diperbarui</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric-value'>{now_jakarta().strftime('%Y-%m-%d %H:%M:%S %Z')}</div></div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar — kontrol
st.sidebar.header("🧭 Parameter Data")
timeframe = st.sidebar.radio("Timeframe", ["1D","1H"], index=0, horizontal=True)
hist_years_fetch = st.sidebar.selectbox("Rentang historis (fetch)", [1,3,5], index=2)
refresh_data = st.sidebar.button("🔄 Muat ulang data")

st.sidebar.header("⚙️ Strategi DCA")
buy_mode = st.sidebar.selectbox("Mode Beli", ["monthly","weekday","interval"], index=0)
price_source = st.sidebar.selectbox("Harga Eksekusi", ["open","close"], index=0)
fee_in_cost_basis = st.sidebar.checkbox("Biaya termasuk dalam harga rata-rata", True)
close_all = st.sidebar.checkbox("Close All pada candle terakhir (realized)", False)
c1, c2 = st.sidebar.columns(2)
with c1:
    buy_day_of_month = st.number_input("Tanggal beli (bulanan)", 1, 28, 1)
with c2:
    buy_weekday = st.number_input("Hari (0=Senin)", 0, 6, 0)
interval_candles = st.sidebar.number_input("Interval (N candle)", 1, 365, 30)
capital_per_trade = st.sidebar.number_input("Modal per beli (IDR)", 10_000, 100_000_000, 5_000_000, step=10_000)
fee_rate_pct = st.sidebar.number_input("Biaya (%)", 0.00, 1.00, 0.15, step=0.01)
windows_sel = st.sidebar.multiselect("Periode backtest (tahun)", [1,3,5], default=[1,3,5])

st.sidebar.header("🧪 Skenario Slippage")
slip_options = st.sidebar.multiselect("Pilih slippage (bps)", [0.0, 5.0, 10.0, 20.0], default=[0.0, 5.0, 10.0])

st.sidebar.header("📏 Benchmark B&H")
bench_fee_sell = st.sidebar.checkbox("Benchmark: Terapkan biaya saat jual akhir", True)

auto_refresh = st.sidebar.checkbox("Auto-refresh harga (20 detik)", value=True)

# StrategyConfig dasar
base_cfg = StrategyConfig(
    capital_per_trade=float(capital_per_trade),
    fee_rate=float(fee_rate_pct)/100.0,
    fee_in_cost_basis=fee_in_cost_basis,
    buy_mode=buy_mode,
    buy_day_of_month=int(buy_day_of_month),
    buy_weekday=int(buy_weekday),
    interval_candles=int(interval_candles),
    price_source=price_source,
    close_all_on_last_candle=close_all,
    slippage_bps=0.0,  # di-override per skenario
    tz_name="Asia/Jakarta"
)

# Fetch historical
if refresh_data or "hist_key" not in st.session_state or st.session_state.get("hist_sig") != (hist_years_fetch, timeframe):
    try:
        symbol, df_hist = fetch_data(preferred_symbols, hist_years_fetch, timeframe, base_cfg.tz_name)
        st.session_state["hist_key"] = f"{symbol}_{hist_years_fetch}_{timeframe}"
        st.session_state["hist_sig"] = (hist_years_fetch, timeframe)
        st.session_state["symbol"] = symbol
        st.session_state["df_hist"] = df_hist
        st.success(f"Data historis termuat: {symbol} ({hist_years_fetch} tahun, {timeframe}).")
    except Exception as e:
        st.error(f"Gagal memuat data historis: {e}")

symbol = st.session_state.get("symbol")
df_hist = st.session_state.get("df_hist")

col_actions = st.columns([1,1,2,2])
with col_actions[0]:
    run_bt = st.button("▶️ Jalankan Backtest")
with col_actions[1]:
    clear_bt = st.button("🧹 Bersihkan")

if clear_bt:
    st.session_state.pop("reports", None)
    st.session_state.pop("bh_bench", None)

if run_bt:
    if df_hist is None or df_hist.empty or symbol is None:
        st.error("Data historis belum tersedia. Muat data terlebih dahulu.")
    else:
        with st.spinner("Menjalankan backtest & skenario slippage..."):
            end_dt = now_jakarta()
            reports_all = run_backtests_for_slippages(
                df_hist=df_hist,
                symbol=symbol,
                base_cfg=base_cfg,
                end_dt=end_dt,
                years_list=windows_sel,
                slippage_scenarios=slip_options
            )
            bh_map = build_benchmarks_for_reports(
                df_hist=df_hist,
                reports=reports_all,
                price_source=price_source,
                include_sell_fee=bench_fee_sell,
                fee_rate=base_cfg.fee_rate
            )
            st.session_state["reports"] = reports_all
            st.session_state["bh_bench"] = bh_map
        st.success("Selesai.")

reports: List[BacktestReport] = st.session_state.get("reports", [])
bh_bench: Dict[Tuple[int,float], pd.DataFrame] = st.session_state.get("bh_bench", {})

# Hasil
if reports:
    st.subheader("📊 Ringkasan Kinerja (Format Indonesia)")
    df_sum = summary_table(reports)
    st.dataframe(df_sum, use_container_width=True)

    # Tabs per periode — each tab overlay multiple slippage curves
    for yrs in sorted(set(r.window_years for r in reports)):
        st.markdown(f"### {yrs} Tahun — Perbandingan Slippage")
        # Filter reports for this window
        subset = [r for r in reports if r.window_years == yrs]
        # Grafik equity overlay
        fig = go.Figure()
        for r in subset:
            eq = r.equity_curve.copy()
            fig.add_trace(go.Scatter(x=eq["time"], y=eq["equity_idr"], mode="lines",
                                     name=f"DCA s={r.slippage_bps:.1f}bps", line=dict(width=2)))
            # Benchmark for this slippage key
            bh_key = (r.window_years, r.slippage_bps)
            if bh_key in bh_bench and not bh_bench[bh_key].empty:
                fig.add_trace(go.Scatter(x=bh_bench[bh_key]["time"], y=bh_bench[bh_key]["bh_value_idr"], mode="lines",
                                         name=f"B&H s={r.slippage_bps:.1f}bps", line=dict(width=2, dash="dash")))
        fig.update_layout(template="plotly_dark", height=420, margin=dict(l=10,r=10,t=30,b=10),
                          title=f"Equity Curve — {yrs} Tahun (Multi‑Slippage)", xaxis_title="Tanggal", yaxis_title="IDR")
        st.plotly_chart(fig, use_container_width=True)

        # Tabel ringkas per slippage (subset)
        df_subset = summary_table(subset)
        st.dataframe(df_subset, use_container_width=True)

    # Ekspor ZIP seluruh skenario
    st.subheader("📦 Ekspor Hasil (ZIP, semua skenario)")
    if st.button("Buat & Unduh ZIP"):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
            # Summary raw CSV (tanpa formatting)
            df_raw = pd.DataFrame([{
                "window_years": r.window_years,
                "slippage_bps": r.slippage_bps,
                "symbol": r.symbol,
                "start": r.start_date,
                "end": r.end_date,
                "total_invested_idr": r.total_invested_idr,
                "total_btc": r.total_btc,
                "avg_price_idr": r.avg_price_idr,
                "current_value_idr": r.current_value_idr,
                "total_return_pct": r.total_return_pct,
                "annualized_return_pct": r.annualized_return_pct,
                "max_drawdown_pct": r.max_drawdown_pct,
                "volatility_monthly_pct": r.volatility_monthly_pct,
                "sharpe_ratio": 0.0 if r.sharpe_ratio is None else r.sharpe_ratio,
                "fees_paid_idr": r.fees_paid_idr
            } for r in reports]).sort_values(["window_years","slippage_bps"])
            z.writestr("summary_all_windows_raw.csv", df_raw.to_csv
            # ---------------------------
# Exchange client (CCXT)
# ---------------------------
class ExchangeClient:
    def __init__(self, cfg: ExchangeConfig):
        ex_class = getattr(ccxt, cfg.exchange_id)
        params = {"enableRateLimit": cfg.enable_rate_limit, "timeout": cfg.timeout}
        self.exchange = ex_class(params)
        if cfg.api_key and cfg.secret:
            self.exchange.apiKey = cfg.api_key
            self.exchange.secret = cfg.secret
        self.max_retries = cfg.max_retries
        self.retry_backoff_sec = cfg.retry_backoff_sec

    def _retry(self, func, *args, **kwargs):
        last = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last = e
                time.sleep(self.retry_backoff_sec ** attempt)
        raise last

    def load_markets(self) -> Dict[str, Any]:
        return self._retry(self.exchange.load_markets)

    def resolve_symbol(self, preferred: List[str]) -> str:
        markets = self.load_markets()
        for s in preferred:
            if s in markets:
                return s
        cands = [m for m in markets if m.startswith("BTC/") and ("IDR" in m or "IDRT" in m or "BIDR" in m)]
        if cands:
            return cands[0]
        raise ValueError("Tidak menemukan simbol BTC/IDR di Tokocrypto.")

    def fetch_ohlcv(self, symbol: str, timeframe: str, since_ms: int, until_ms: int, limit: int = 1000) -> List[List[float]]:
        all_rows = []
        cursor = since_ms
        while True:
            chunk = self._retry(self.exchange.fetch_ohlcv, symbol, timeframe=timeframe, since=cursor, limit=limit)
            if not chunk:
                break
            all_rows.extend(chunk)
            last_ts = chunk[-1][0]
            if last_ts >= until_ms or len(chunk) < limit:
                break
            cursor = last_ts + 1
        return [r for r in all_rows if since_ms <= r[0] <= until_ms]

    def fetch_ticker_safe(self, symbols: List[str]) -> Optional[dict]:
        for s in symbols:
            try:
                return self._retry(self.exchange.fetch_ticker, s)
            except Exception:
                continue
        return None

# ---------------------------
# Data prep
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

def first_price_on_or_after(df: pd.DataFrame, target_dt: datetime, price_col: str = "close") -> Optional[Tuple[datetime, float]]:
    rows = df[df["time"] >= target_dt]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return row["time"], float(row[price_col])

# ---------------------------
# Penjadwalan beli
# ---------------------------
def generate_monthly_schedule(start_dt: datetime, end_dt: datetime, day_of_month: int) -> List[datetime]:
    schedule = []
    candidate = start_dt.replace(day=1)
    last_day = (candidate + relativedelta(months=1) - timedelta(days=1)).day
    dm = min(day_of_month, last_day)
    first_buy = candidate.replace(day=dm, hour=0, minute=0, second=0, microsecond=0)
    if first_buy < start_dt:
        nm = candidate + relativedelta(months=1)
        last_day = (nm + relativedelta(months=1) - timedelta(days=1)).day
        dm = min(day_of_month, last_day)
        first_buy = nm.replace(day=dm, hour=0, minute=0, second=0, microsecond=0)
    cur = first_buy
    while cur <= end_dt:
        schedule.append(cur)
        nm = cur + relativedelta(months=1)
        last_day = (nm + relativedelta(months=1) - timedelta(days=1)).day
        dm = min(day_of_month, last_day)
        cur = nm.replace(day=dm, hour=0, minute=0, second=0, microsecond=0)
    return schedule
    # ---------------------------
# Backtest inti
# ---------------------------
def run_dca_backtest(df: pd.DataFrame, cfg: StrategyConfig, window_years: int) -> BacktestReport:
    end_dt = df["time"].max()
    start_dt = end_dt - relativedelta(years=window_years)
    df_window = df[df["time"] >= start_dt].reset_index(drop=True)

    schedule = generate_monthly_schedule(start_dt, end_dt, cfg.buy_day_of_month)

    total_invested = 0.0
    total_btc = 0.0
    fees_paid = 0.0
    equity_curve = []
    monthly_breakdown = []

    for buy_dt in schedule:
        price_row = first_price_on_or_after(df_window, buy_dt, cfg.price_source)
        if not price_row:
            continue
        _, price = price_row
        price *= (1 + cfg.slippage_bps / 10000.0)
        btc_bought = cfg.capital_per_trade / price
        fee = cfg.capital_per_trade * cfg.fee_rate
        fees_paid += fee
        total_invested += cfg.capital_per_trade
        total_btc += btc_bought

        monthly_breakdown.append({
            "buy_date": buy_dt,
            "price": price,
            "btc_bought": btc_bought,
            "idr_spent": cfg.capital_per_trade,
            "fee_paid": fee
        })

    current_price = df_window.iloc[-1][cfg.price_source]
    current_value = total_btc * current_price
    total_return_pct = (current_value - total_invested) / total_invested * 100
    years = window_years
    annualized_return_pct = ((current_value / total_invested) ** (1/years) - 1) * 100 if total_invested > 0 else 0

    # Equity curve
    running_btc = 0.0
    running_invested = 0.0
    for idx, row in df_window.iterrows():
        date = row["time"]
        price = row[cfg.price_source]
        if date in [m["buy_date"] for m in monthly_breakdown]:
            running_btc += cfg.capital_per_trade / (price * (1 + cfg.slippage_bps / 10000.0))
            running_invested += cfg.capital_per_trade
        equity_curve.append({
            "time": date,
            "equity_idr": running_btc * price
        })

    eq_df = pd.DataFrame(equity_curve)
    mb_df = pd.DataFrame(monthly_breakdown)

    return BacktestReport(
        window_years=window_years,
        symbol="BTC/IDR",
        start_date=start_dt.strftime("%Y-%m-%d"),
        end_date=end_dt.strftime("%Y-%m-%d"),
        total_invested_idr=total_invested,
        total_btc=total_btc,
        avg_price_idr=total_invested / total_btc if total_btc > 0 else 0,
        current_value_idr=current_value,
        total_return_pct=total_return_pct,
        annualized_return_pct=annualized_return_pct,
        max_drawdown_pct=0.0,  # bisa dihitung jika mau
        volatility_monthly_pct=0.0,
        sharpe_ratio=None,
        fees_paid_idr=fees_paid,
        monthly_breakdown=mb_df,
        equity_curve=eq_df,
        slippage_bps=cfg.slippage_bps
    )

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="DCA BTC/IDR Tokocrypto — Modern", layout="wide")
st.title("📈 DCA BTC/IDR Tokocrypto — Modern Dashboard")

# Auto-refresh harga setiap 60 detik
st_autorefresh(interval=60 * 1000, key="refresh_price")

# Ambil kredensial dari environment
api_key = os.getenv("TOKO_API_KEY")
api_secret = os.getenv("TOKO_SECRET")

cfg_ex = ExchangeConfig(api_key=api_key, secret=api_secret)
client = ExchangeClient(cfg_ex)

symbol = client.resolve_symbol(["BTC/IDR", "BTC/IDRT", "BTC/BIDR"])
ticker = client.fetch_ticker_safe([symbol])
if ticker:
    st.metric("Harga BTC/IDR", fmt_idr(ticker["last"]))
else:
    st.warning("Gagal mengambil harga BTC/IDR")

# Ambil data OHLCV
end_dt = now_jakarta()
start_dt = end_dt - relativedelta(years=5)
ohlcv = client.fetch_ohlcv(symbol, "1d", to_ms(start_dt), to_ms(end_dt))
df = ohlcv_to_df(ohlcv)
# ---------------------------
# Multi-slippage backtest
# ---------------------------
slippage_list = [0.0, 5.0, 10.0]  # contoh: 0 bps, 5 bps, 10 bps
window_list = [1, 3, 5]           # contoh: 1 tahun, 3 tahun, 5 tahun

reports = []
for slip in slippage_list:
    for win in window_list:
        cfg_strat = StrategyConfig(
            capital_per_trade=5_000_000.0,
            fee_rate=0.0015,
            buy_day_of_month=1,
            slippage_bps=slip
        )
        rep = run_dca_backtest(df, cfg_strat, win)
        reports.append(rep)

# ---------------------------
# Tabel hasil ringkas
# ---------------------------
st.subheader("📊 Ringkasan Hasil Backtest")
summary_data = []
for r in reports:
    summary_data.append({
        "Window (tahun)": r.window_years,
        "Slippage (bps)": r.slippage_bps,
        "Total Investasi": fmt_idr(r.total_invested_idr),
        "Total BTC": fmt_dec(r.total_btc, 6),
        "Harga Rata-rata": fmt_idr(r.avg_price_idr),
        "Nilai Sekarang": fmt_idr(r.current_value_idr),
        "Total Return": fmt_pct(r.total_return_pct),
        "CAGR": fmt_pct(r.annualized_return_pct),
        "Fee Dibayar": fmt_idr(r.fees_paid_idr)
    })
st.dataframe(pd.DataFrame(summary_data))

# ---------------------------
# Grafik Equity Curve
# ---------------------------
st.subheader("📈 Grafik Equity Curve")
fig = go.Figure()
for r in reports:
    fig.add_trace(go.Scatter(
        x=r.equity_curve["time"],
        y=r.equity_curve["equity_idr"],
        mode="lines",
        name=f"{r.window_years}y — slip {r.slippage_bps}bps"
    ))
fig.update_layout(template="plotly_dark", height=500, width=900)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Ekspor ZIP
# ---------------------------
st.subheader("📦 Ekspor Hasil (ZIP)")
if st.button("Buat & Unduh ZIP"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        # Summary raw CSV
        df_raw = pd.DataFrame([{
            "window_years": r.window_years,
            "slippage_bps": r.slippage_bps,
            "symbol": r.symbol,
            "start": r.start_date,
            "end": r.end_date,
            "total_invested_idr": r.total_invested_idr,
            "total_btc": r.total_btc,
            "avg_price_idr": r.avg_price_idr,
            "current_value_idr": r.current_value_idr,
            "total_return_pct": r.total_return_pct,
            "annualized_return_pct": r.annualized_return_pct,
            "fees_paid_idr": r.fees_paid_idr
        } for r in reports])
        z.writestr("summary_all_windows_raw.csv", df_raw.to_csv(index=False))

        # Breakdown per window
        for r in reports:
            z.writestr(
                f"{r.window_years}y_slip{r.slippage_bps:.1f}bps_monthly_breakdown.csv",
                r.monthly_breakdown.to_csv(index=False)
            )
            z.writestr(
                f"{r.window_years}y_slip{r.slippage_bps:.1f}bps_equity_curve.csv",
                r.equity_curve.to_csv(index=False)
            )

    buffer.seek(0)
    st.download_button(
        label="⬇️ Unduh ZIP",
        data=buffer,
        file_name="dca_tokocrypto_results.zip",
        mime="application/zip"
    )
    # ---------------------------
# Benchmark Buy & Hold (B&H)
# ---------------------------
st.subheader("📏 Benchmark Buy & Hold")
sell_fee_rate = st.number_input("Biaya jual (%)", min_value=0.0, max_value=5.0, value=0.15, step=0.05) / 100

bh_results = []
for win in window_list:
    start_dt = df["time"].max() - relativedelta(years=win)
    start_price_row = first_price_on_or_after(df, start_dt, "open")
    if not start_price_row:
        continue
    _, start_price = start_price_row
    end_price = df.iloc[-1]["close"]
    # B&H: beli 1 BTC di awal, jual di akhir
    qty = 1.0
    gross_value = qty * end_price
    net_value = gross_value * (1 - sell_fee_rate)
    total_return_pct = (net_value - (qty * start_price)) / (qty * start_price) * 100
    annualized_return_pct = ((net_value / (qty * start_price)) ** (1/win) - 1) * 100
    bh_results.append({
        "Window (tahun)": win,
        "Harga Awal": fmt_idr(start_price),
        "Harga Akhir": fmt_idr(end_price),
        "Nilai Akhir (net)": fmt_idr(net_value),
        "Total Return": fmt_pct(total_return_pct),
        "CAGR": fmt_pct(annualized_return_pct)
    })

st.dataframe(pd.DataFrame(bh_results))

# ---------------------------
# Penutup
# ---------------------------
st.markdown("---")
st.caption("DCA BTC/IDR Tokocrypto — Modern Dashboard © 2025")  
