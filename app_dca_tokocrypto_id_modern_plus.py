# app_dca_tokocrypto_id_modern_plus.py
# Streamlit modern dashboard — DCA BTC/IDR (Tokocrypto) dengan format Indonesia, multi-slippage, benchmark fee jual, ekspor ZIP
# Kredensial diambil dari environment variables: TOKO_API_KEY, TOKO_SECRET

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
from streamlit_autorefresh import st_autorefresh  # ✅ Auto-refresh resmi

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
    slippage_bps: float = 0.0

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
        first_buy = nm.replace(day=dm, hour=0, minute=0, second=0, microsecond=
