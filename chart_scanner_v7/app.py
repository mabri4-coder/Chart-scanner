"""
Chart Pattern Scanner
Original implementation for educational / research use.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""
from __future__ import annotations

import contextlib
import io
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

APP_NAME = "Chart Pattern Scanner V9.5"

DUPLICATE_SHARE_CLASS_REMOVE = {"GOOG", "FOXA", "NWS"}  # keep GOOGL, FOX, NWSA by default
YF_CHUNK_SIZE = 75

# Keep Streamlit logs readable. yfinance can print noisy messages for stale/delisted symbols
# or temporary Yahoo crumb/session failures; the app skips failed symbols.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


# Broad built-in fallback universe. Used when Wikipedia/NasdaqTrader live symbol downloads fail.
# This prevents the scanner from silently scanning only a tiny 10-15 symbol fallback list.
STATIC_LIQUID_UNIVERSE = """
A AA AAL AAP AAPL ABBV ABNB ABT ACGL ACHR ACM ACN ADBE ADI ADM ADP ADSK AEE AEP AES AFL AFRM AGG AIG AIZ AJG AKAM ALAB ALB ALGN ALL ALLE AMAT AMCR AMD AME AMGN AMP AMT AMZN ANET ANF ANSS AON AOS APA APO APP APD APH APTV ARE ASML AVB AVGO AVY AWK AXON AXP AZO
BA BAC BALL BAX BBY BDX BEN BG BIIB BK BKNG BKR BLK BMY BR BRK-B BRO BSX BURL BX BXP C CAG CAH CARR CAT CB CBOE CBRE CCI CCL CDNS CDW CE CELH CEG CF CFG CHD CHRW CHTR CI CINF CL CLX CMA CMCSA CME CMG CMI CMS CNC CNP COF COIN COO COP COR COST COTY CPB CPRT CRL CRM CRWD CSCO CSGP CSX CTAS CTLT CTRA CTSH CTVA CVS CVX CZR
D DAL DD DE DFS DG DGX DHI DHR DIS DLTR DOC DOV DOW DPZ DRI DTE DUK DVA DVN DXCM EA EBAY ECL ED EFX EG EIX EL ELV EMN EMR ENPH EOG EPAM EQIX EQR EQT ES ESS ETN ETSY EVRG EW EXC EXPE EXPD EXR F FANG FAST FCX FDS FDX FE FFIV FI FICO FIS FITB FMC FOX FOXA FRT FSLR FTNT FTV GD GE GEV GILD GIS GL GLW GM GNRC GOOG GOOGL GPC GPN GRMN GS GWW HAL HAS HBAN HCA HD HES HIG HLT HOLX HON HPE HPQ HRL HSIC HST HSY HUM HWM IBM ICE IDXX IEX ILMN INCY INTC INTU INVH IP IPG IQV IR IRM ISRG IT ITW IVZ J JBL JCI JKHY JNJ JNPR JPM K KDP KEY KEYS KHC KIM KLAC KMB KMI KMX KO KR KVUE L LDOS LEN LH LHX LIN LKQ LLY LMT LOW LRCX LULU LUV LVS LW LYB LYV MA MAA MAR MAS MCD MCHP MCK MCO MDLZ MDT META MGM MHK MKC MKTX MLM MMC MMM MNST MO MOH MOS MPC MPWR MRK MRNA MRO MS MSCI MSFT MSI MTB MTCH MU NCLH NDAQ NDSN NEE NEM NFLX NI NKE NOC NOW NRG NSC NTAP NTRS NUE NVDA NVR NWS NWSA NXPI O ODFL OKE OMC ON ORCL ORLY OTIS OXY PANW PARA PAYC PAYX PCAR PCG PEG PEP PFE PFG PG PGR PH PHM PKG PLD PLTR PM PNC PNR PNW PODD POOL PPG PPL PRU PSA PSX PYPL QCOM QRVO RCL REG REGN RF RHI RJF RL RMD ROK ROL ROP ROST RSG RTX RVTY SBAC SBUX SCHW SHW SJM SLB SMCI SNA SNOW SNPS SO SOLV SPG SPGI SRE STE STLD STT STX STZ SWK SWKS SYF SYK SYY T TAP TDG TEL TER TFC TGT TJX TMO TMUS TPR TRGP TRMB TROW TRV TSCO TSLA TSN TT TTWO TXN TXT TYL UAL UBER UDR UHS ULTA UNH UNP UPS URI USB V VICI VLO VLTO VMC VRSK VRSN VRTX VST VTR VTRS VZ WAB WAT WBA WBD WDC WEC WELL WFC WM WMB WMT WRB WST WTW WY WYNN XEL XOM XYL YUM ZBH ZBRA ZTS
AAOI AEO AI ALGM AMBA AMC AMKR APPF AR ARCC ARM ASTS AUR BE BILI BITF BLBD BMRN BNTX BROS BTBT BURL BYND CAVA CELH CLS CMPR CNSP CNQ CNX CRDO CROX CVNA DELL DKNG DOCN DUOL ELF ESTC EXAS FHN FIVE FROG FRSH GCT GDDY GH GLBE GTLB GWRE HIMS HOOD HUBS IOT IOVA JOBY LCID LI LLYVA LMND LPLA LUMN MARA MBLY MDB MNDY MRVL MSTR NET NIO NTRA OKTA ONON OSCR PATH PINS PLUG RBLX RDDT RIVN RKLB ROKU RUM RUN SE SHOP SNAP SOFI STEM TOST TTD TWLO U UPST VFS W WIX WOLF XPEV Z ZM ZS
"""

def static_liquid_universe() -> List[str]:
    return parse_tickers(STATIC_LIQUID_UNIVERSE)

# -----------------------------
# Utility / indicators
# -----------------------------


def _clean_ticker(t: str) -> str:
    t = str(t).strip().upper().lstrip("$")
    if not t:
        return ""
    # Yahoo Finance uses BRK-B style for share classes.
    t = t.replace(".", "-")
    return t


def parse_tickers(text: str) -> List[str]:
    raw = re.split(r"[\s,;]+", text or "")
    out = []
    for x in raw:
        t = _clean_ticker(x)
        if t and re.match(r"^[A-Z0-9\-]{1,12}$", t):
            out.append(t)
    return sorted(set(out))


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=max(2, n // 2)).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    return pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).rolling(n, min_periods=max(2, n // 2)).mean()


def stoch_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    rs = rsi(close, n)
    low = rs.rolling(n, min_periods=max(2, n // 2)).min()
    high = rs.rolling(n, min_periods=max(2, n // 2)).max()
    return 100 * (rs - low) / (high - low).replace(0, np.nan)


def macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, 12) - ema(close, 26)
    signal = ema(line, 9)
    hist = line - signal
    return line, signal, hist


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=max(2, n // 2)).std()
    return mid, mid + k * sd, mid - k * sd


def keltner(df: pd.DataFrame, n: int = 20, k: float = 1.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
    mid = ema(df["Close"], n)
    a = atr(df, n)
    return mid, mid + k * a, mid - k * a


def slope(series: pd.Series, bars: int = 20) -> float:
    s = series.dropna().tail(bars)
    if len(s) < max(5, bars // 2):
        return np.nan
    y = s.values.astype(float)
    x = np.arange(len(y))
    try:
        m, _ = np.polyfit(x, y, 1)
        return float(m / max(abs(np.nanmean(y)), 1e-9))
    except Exception:
        return np.nan


def pct(a: float, b: float) -> float:
    if b is None or b == 0 or np.isnan(b):
        return np.nan
    return (a / b - 1.0) * 100.0


def loc_in_range(close: float, low: float, high: float) -> float:
    if high <= low:
        return 50.0
    return 100.0 * (close - low) / (high - low)


def clamp(x: float, lo: float = 0, hi: float = 100) -> float:
    if x is None or np.isnan(x):
        return lo
    return max(lo, min(hi, float(x)))


def grade(score: float) -> str:
    if score >= 90:
        return "A+"
    if score >= 80:
        return "A"
    if score >= 70:
        return "B+"
    if score >= 60:
        return "B"
    return "Watch"


@dataclass
class ScanHit:
    ticker: str
    scanner: str
    score: float
    grade: str
    direction: str
    entry: Optional[float]
    stop: Optional[float]
    target: Optional[float]
    reasons: str


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Standardize columns.
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    df = df[keep].dropna(subset=["Close"])
    if "Open" not in df: df["Open"] = df["Close"]
    if "High" not in df: df["High"] = df["Close"]
    if "Low" not in df: df["Low"] = df["Close"]
    if "Volume" not in df: df["Volume"] = 0
    close = df["Close"]
    vol = df["Volume"]
    for n in [10, 20, 21, 30, 50, 65, 100, 150, 200]:
        df[f"SMA{n}"] = sma(close, n)
    for n in [10, 21, 50, 65]:
        df[f"EMA{n}"] = ema(close, n)
    df["RSI14"] = rsi(close, 14)
    df["StochRSI14"] = stoch_rsi(close, 14)
    m, sig, hist = macd(close)
    df["MACD"] = m
    df["MACDSignal"] = sig
    df["MACDHist"] = hist
    df["ATR14"] = atr(df, 14)
    mid, upper, lower = bollinger(close)
    df["BBMid"], df["BBUpper"], df["BBLower"] = mid, upper, lower
    kmid, kup, klo = keltner(df)
    df["KCMid"], df["KCUpper"], df["KCLower"] = kmid, kup, klo
    df["Vol50"] = sma(vol.astype(float), 50)
    df["Vol20"] = sma(vol.astype(float), 20)
    df["DollarVol50"] = df["Vol50"] * close
    df["RangePct"] = (df["High"] - df["Low"]) / close.replace(0, np.nan) * 100
    df["ADR20"] = df["RangePct"].rolling(20, min_periods=10).mean()
    return df


def lastv(df: pd.DataFrame, col: str, default: float = np.nan) -> float:
    try:
        v = df[col].dropna().iloc[-1]
        return float(v)
    except Exception:
        return default


def base_metrics(df: pd.DataFrame, spy_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
    c = df["Close"]
    last = float(c.iloc[-1])
    prev = float(c.iloc[-2]) if len(c) > 1 else np.nan
    out = {
        "last": last,
        "pct_chg": pct(last, prev),
        "vol": lastv(df, "Volume", 0),
        "vol50": lastv(df, "Vol50", 0),
        "rvol": lastv(df, "Volume", 0) / max(lastv(df, "Vol50", 1), 1),
        "rsi": lastv(df, "RSI14"),
        "stochrsi": lastv(df, "StochRSI14"),
        "atr": lastv(df, "ATR14"),
        "adr20": lastv(df, "ADR20"),
        "sma10": lastv(df, "SMA10"),
        "sma20": lastv(df, "SMA20"),
        "sma50": lastv(df, "SMA50"),
        "sma100": lastv(df, "SMA100"),
        "sma150": lastv(df, "SMA150"),
        "sma200": lastv(df, "SMA200"),
        "ema10": lastv(df, "EMA10"),
        "ema21": lastv(df, "EMA21"),
        "ema65": lastv(df, "EMA65"),
        "high20": float(c.tail(20).max()),
        "low20": float(c.tail(20).min()),
        "high50": float(c.tail(50).max()) if len(c) >= 50 else float(c.max()),
        "low50": float(c.tail(50).min()) if len(c) >= 50 else float(c.min()),
        "high252": float(c.tail(252).max()) if len(c) >= 30 else float(c.max()),
        "low252": float(c.tail(252).min()) if len(c) >= 30 else float(c.min()),
    }
    if len(c) > 252:
        out["ret252"] = pct(last, float(c.iloc[-252]))
    elif len(c) > 60:
        out["ret252"] = pct(last, float(c.iloc[0]))
    else:
        out["ret252"] = np.nan
    if spy_df is not None and len(spy_df) > 60 and len(df) > 60:
        n = min(126, len(df) - 1, len(spy_df) - 1)
        stock_ret = pct(float(df["Close"].iloc[-1]), float(df["Close"].iloc[-n]))
        spy_ret = pct(float(spy_df["Close"].iloc[-1]), float(spy_df["Close"].iloc[-n]))
        out["rs_vs_spy"] = stock_ret - spy_ret
    else:
        out["rs_vs_spy"] = np.nan
    return out


def ta_rating(df: pd.DataFrame, ars: Optional[float] = None) -> float:
    m = base_metrics(df)
    close = m["last"]
    points = 0
    checks = 0
    tests = [
        close > m["sma50"],
        close > m["sma150"],
        close > m["sma200"],
        m["sma50"] > m["sma150"],
        m["sma150"] > m["sma200"],
        slope(df["SMA200"], 30) > 0,
        m["rsi"] > 50,
        lastv(df, "MACD") > lastv(df, "MACDSignal"),
        m["last"] >= 0.8 * m["high252"],
        m["last"] >= 1.3 * m["low252"],
    ]
    for t in tests:
        if not pd.isna(t):
            checks += 1
            points += 1 if t else 0
    if ars is not None and not np.isnan(ars):
        checks += 1
        points += 1 if ars >= 70 else 0
    return round(10 * points / max(checks, 1), 1)


def fa_rating(info: Dict) -> float:
    if not info:
        return np.nan
    score = 0
    checks = 0
    rules = [
        (info.get("profitMargins"), lambda x: x is not None and x > 0.05),
        (info.get("returnOnEquity"), lambda x: x is not None and x > 0.12),
        (info.get("revenueGrowth"), lambda x: x is not None and x > 0.05),
        (info.get("earningsGrowth"), lambda x: x is not None and x > 0.05),
        (info.get("debtToEquity"), lambda x: x is not None and x < 150),
        (info.get("currentRatio"), lambda x: x is None or x > 1.0),
        (info.get("grossMargins"), lambda x: x is not None and x > 0.25),
        (info.get("pegRatio"), lambda x: x is not None and 0 < x < 2.5),
    ]
    for value, fn in rules:
        try:
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                checks += 1
                score += 1 if fn(value) else 0
        except Exception:
            pass
    if checks == 0:
        return np.nan
    return round(10 * score / checks, 1)


# -----------------------------
# Data loading
# -----------------------------


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_nasdaq_trader_symbols(include_etfs: bool = False) -> pd.DataFrame:
    """Fetch Nasdaq Trader symbol directories. Includes NASDAQ + NYSE/AMEX listed symbols."""
    urls = {
        "nasdaq": "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "other": "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    }
    rows = []
    for name, url in urls.items():
        try:
            txt = requests.get(url, timeout=20).text
            txt = "\n".join([ln for ln in txt.splitlines() if not ln.startswith("File Creation")])
            from io import StringIO
            df = pd.read_csv(StringIO(txt), sep="|")
            if name == "nasdaq":
                df = df.rename(columns={"Symbol": "Ticker", "Security Name": "Name"})
                df["Exchange"] = "NASDAQ"
                df = df[df.get("Test Issue", "N") == "N"]
                if not include_etfs and "ETF" in df:
                    df = df[df["ETF"] != "Y"]
                rows.append(df[["Ticker", "Name", "Exchange"]].dropna())
            else:
                df = df.rename(columns={"ACT Symbol": "Ticker", "Security Name": "Name"})
                df = df[df.get("Test Issue", "N") == "N"]
                if not include_etfs and "ETF" in df:
                    df = df[df["ETF"] != "Y"]
                rows.append(df[["Ticker", "Name", "Exchange"]].dropna())
        except Exception:
            continue
    if not rows:
        fallback = static_liquid_universe()
        return pd.DataFrame({"Ticker": fallback, "Name": "Built-in fallback universe", "Exchange": "STATIC"})
    out = pd.concat(rows, ignore_index=True).drop_duplicates("Ticker")
    out["Ticker"] = out["Ticker"].map(_clean_ticker)
    out = out[out["Ticker"].str.match(r"^[A-Z0-9\-]{1,12}$", na=False)]
    return out.sort_values("Ticker").reset_index(drop=True)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_sp500() -> List[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        return [_clean_ticker(x) for x in tables[0]["Symbol"].tolist()]
    except Exception:
        return static_liquid_universe()[:550]


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_nasdaq100() -> List[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
        for table in tables:
            if "Ticker" in table.columns:
                return [_clean_ticker(x) for x in table["Ticker"].tolist()]
            if "Symbol" in table.columns:
                return [_clean_ticker(x) for x in table["Symbol"].tolist()]
    except Exception:
        pass
    # Reasonable static fallback if Wikipedia cannot be reached.
    return parse_tickers("AAPL MSFT NVDA AMZN META GOOGL GOOG AVGO TSLA COST NFLX ADBE AMD QCOM INTC INTU CSCO AMAT AMGN BKNG ADI ARM ASML AZN CDNS CHTR CMCSA CPRT CRWD DASH DDOG GILD HON ISRG LRCX MAR MELI MSTR MU NXPI PANW PAYX PCAR PDD PYPL ROST SBUX SNPS TEAM TMUS TXN VRTX WBD")


@st.cache_data(ttl=10 * 60, show_spinner=False)
def download_prices(tickers: Tuple[str, ...], period: str, interval: str) -> Dict[str, pd.DataFrame]:
    tickers = tuple([_clean_ticker(t) for t in tickers if t])
    out: Dict[str, pd.DataFrame] = {}
    if not tickers:
        return out
    for i in range(0, len(tickers), YF_CHUNK_SIZE):
        chunk = tickers[i : i + YF_CHUNK_SIZE]
        try:
            # Suppress yfinance console noise for invalid/stale tickers; failed symbols are simply skipped.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                raw = yf.download(
                    list(chunk),
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                    progress=False,
                    prepost=False,
                )
        except Exception:
            continue
        if raw is None or raw.empty:
            continue
        if isinstance(raw.columns, pd.MultiIndex):
            level0 = raw.columns.get_level_values(0)
            level1 = raw.columns.get_level_values(1)
            for t in chunk:
                try:
                    if t in level0:
                        df = raw[t].copy()
                    elif t in level1:
                        df = raw.xs(t, axis=1, level=1).copy()
                    else:
                        continue
                    df = df.dropna(subset=["Close"])
                    if len(df) >= 40:
                        out[t] = enrich(df)
                except Exception:
                    pass
        else:
            if len(chunk) == 1:
                t = chunk[0]
                df = raw.dropna(subset=["Close"]).copy()
                if len(df) >= 40:
                    out[t] = enrich(df)
    return out


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_info(ticker: str) -> Dict:
    try:
        info = yf.Ticker(ticker).get_info()
        return info if isinstance(info, dict) else {}
    except Exception:
        try:
            return yf.Ticker(ticker).info or {}
        except Exception:
            return {}


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_next_earnings_date(ticker: str) -> Optional[datetime]:
    try:
        cal = yf.Ticker(ticker).calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            vals = cal.values.flatten().tolist()
            dates = [pd.to_datetime(x).to_pydatetime() for x in vals if not pd.isna(x) and "date" in str(type(x)).lower()]
            future = [d for d in dates if d.replace(tzinfo=None) >= datetime.now() - timedelta(days=1)]
            return min(future) if future else None
        if isinstance(cal, dict):
            for key in ["Earnings Date", "EarningsDate", "earningsDate"]:
                val = cal.get(key)
                if isinstance(val, (list, tuple)) and val:
                    return pd.to_datetime(val[0]).to_pydatetime()
                if val is not None:
                    return pd.to_datetime(val).to_pydatetime()
    except Exception:
        return None
    return None


# -----------------------------
# Scanner functions
# -----------------------------


def current_score_floor() -> float:
    # Loose = more watchlist candidates; Balanced = useful default; Strict = clean textbook patterns.
    mode = st.session_state.get("match_quality", "Balanced") if hasattr(st, "session_state") else "Balanced"
    return {"Loose / candidate": 45.0, "Balanced": 52.0, "Strict / confirmed": 60.0}.get(mode, 52.0)


def current_mode() -> str:
    return st.session_state.get("match_quality", "Balanced") if hasattr(st, "session_state") else "Balanced"


def current_interval() -> str:
    return st.session_state.get("scan_interval", "1d") if hasattr(st, "session_state") else "1d"


def is_intraday() -> bool:
    return current_interval() in {"1h", "30m", "15m", "5m"}


def timeframe_mode_label() -> str:
    return "Intraday" if is_intraday() else "Daily/Swing"


def hit(ticker: str, scanner: str, score: float, direction: str, reasons: List[str], entry=None, stop=None, target=None) -> Optional[ScanHit]:
    score = clamp(score)
    if score < current_score_floor():
        return None
    mode = current_mode()
    if mode == "Loose / candidate" and score < 60:
        reasons = ["candidate setup - verify chart manually"] + reasons
    return ScanHit(ticker, scanner, score, grade(score), direction, entry, stop, target, "; ".join(reasons[:7]))


# -----------------------------
# Strict pattern helpers
# -----------------------------


def _slope_pct_per_bar(series: pd.Series) -> float:
    """Linear-regression slope normalized as % of average price per bar."""
    s = pd.Series(series).dropna()
    if len(s) < 3:
        return 0.0
    y = s.astype(float).values
    x = np.arange(len(y), dtype=float)
    try:
        m, _ = np.polyfit(x, y, 1)
        avg = max(abs(float(np.nanmean(y))), 1e-9)
        return float(m / avg * 100.0)
    except Exception:
        return 0.0


def _position_of_max(series: pd.Series) -> int:
    vals = pd.Series(series).astype(float).values
    if len(vals) == 0 or np.all(np.isnan(vals)):
        return -1
    return int(np.nanargmax(vals))


def _position_of_min(series: pd.Series) -> int:
    vals = pd.Series(series).astype(float).values
    if len(vals) == 0 or np.all(np.isnan(vals)):
        return -1
    return int(np.nanargmin(vals))


def _safe_mean(s: pd.Series, default: float = 0.0) -> float:
    try:
        v = float(pd.Series(s).dropna().mean())
        return v if not np.isnan(v) else default
    except Exception:
        return default




def _recent_swing_points(df: pd.DataFrame, left: int = 2, right: int = 2) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Return simple swing highs and swing lows using bar index positions."""
    highs: List[Tuple[int, float]] = []
    lows: List[Tuple[int, float]] = []
    if len(df) < left + right + 3:
        return highs, lows
    h = df["High"].astype(float).values
    l = df["Low"].astype(float).values
    for i in range(left, len(df) - right):
        if h[i] >= np.nanmax(h[i-left:i+right+1]):
            highs.append((i, float(h[i])))
        if l[i] <= np.nanmin(l[i-left:i+right+1]):
            lows.append((i, float(l[i])))
    return highs, lows


def _intraday_window(df: pd.DataFrame, bars: int = 120) -> pd.DataFrame:
    return df.tail(min(bars, len(df))).copy().reset_index(drop=True)


def intraday_bull_flag(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    """Intraday bull flag: impulse up, controlled sideways/down pullback, entry at flag resistance."""
    if len(df) < 60:
        return None
    m = base_metrics(df, spy_df)
    mode = current_mode(); loose = mode == "Loose / candidate"; strict = mode == "Strict / confirmed"
    w = _intraday_window(df, 110)
    last = float(w["Close"].iloc[-1])
    if last <= 0:
        return None
    # Find recent impulse high excluding the last 2 bars.
    end = max(10, len(w)-2)
    high_i = int(np.nanargmax(w["High"].iloc[max(0, end-55):end].values)) + max(0, end-55)
    bars_since_high = len(w)-1-high_i
    if bars_since_high < (4 if strict else 3) or bars_since_high > (32 if loose else 26):
        return None
    low_start = max(0, high_i-45)
    pole_low_i = int(np.nanargmin(w["Low"].iloc[low_start:high_i+1].values)) + low_start
    if high_i - pole_low_i < 4:
        return None
    pole_low = float(w["Low"].iloc[pole_low_i]); pole_high = float(w["High"].iloc[high_i])
    pole_gain = pct(pole_high, pole_low)
    min_gain = 1.8 if current_interval()=="5m" else 2.5
    if pole_gain < (min_gain if loose else min_gain*1.25):
        return None
    flag = w.iloc[high_i+1:]
    if len(flag) < 3:
        return None
    flag_low = float(flag["Low"].min())
    pole_height = max(pole_high-pole_low, 1e-9)
    pullback_of_pole = (pole_high-flag_low)/pole_height*100
    if not ((12 if loose else 18) <= pullback_of_pole <= (75 if loose else 65 if not strict else 55)):
        return None
    high_slope = _slope_pct_per_bar(flag["High"])
    close_slope = _slope_pct_per_bar(flag["Close"])
    if high_slope > (0.20 if loose else 0.10) and close_slope > (0.14 if loose else 0.06):
        return None
    try:
        x = np.arange(len(flag), dtype=float)
        hi_m, hi_b = np.polyfit(x, flag["High"].astype(float).values, 1)
        line_entry = float(hi_m*len(flag)+hi_b)
    except Exception:
        line_entry = float(flag["High"].tail(min(4, len(flag))).max())
    recent_res = float(flag["High"].tail(min(5, len(flag))).max())
    entry = max(line_entry, recent_res, last*1.001)
    dist = (entry-last)/max(entry, 1e-9)*100
    if dist > (4.5 if loose else 3.2 if not strict else 1.7) or last > entry*1.025:
        return None
    vol_ok = float(flag["Volume"].tail(min(5, len(flag))).mean()) <= float(w["Volume"].iloc[pole_low_i:high_i+1].mean()) * (1.05 if loose else 0.90)
    if strict and not vol_ok:
        return None
    score = 58
    reasons = ["intraday impulse + flag", f"pole gain {pole_gain:.1f}%", f"pullback {pullback_of_pole:.0f}% of pole", f"{dist:.1f}% from flag entry"]
    if vol_ok: score += 10; reasons.append("flag volume quiet")
    if dist <= 1.5: score += 10; reasons.append("near trigger")
    if last >= float(flag["Close"].tail(min(5,len(flag))).mean()): score += 6; reasons.append("holding flag area")
    stop = flag_low
    target = entry + pole_height*0.75
    return hit(ticker, "Bull Flag", score, "Bullish", reasons, entry, stop, target)


def intraday_bear_flag(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    """Intraday bear flag: sharp drop, weak bounce, entry at breakdown below flag support."""
    if len(df) < 60:
        return None
    m = base_metrics(df, spy_df)
    mode = current_mode(); loose = mode == "Loose / candidate"; strict = mode == "Strict / confirmed"
    w = _intraday_window(df, 110)
    last = float(w["Close"].iloc[-1])
    end = max(10, len(w)-2)
    low_i = int(np.nanargmin(w["Low"].iloc[max(0, end-55):end].values)) + max(0, end-55)
    bars_since_low = len(w)-1-low_i
    if bars_since_low < (4 if strict else 3) or bars_since_low > (32 if loose else 26):
        return None
    high_start = max(0, low_i-45)
    pole_high_i = int(np.nanargmax(w["High"].iloc[high_start:low_i+1].values)) + high_start
    if low_i - pole_high_i < 4:
        return None
    pole_high = float(w["High"].iloc[pole_high_i]); pole_low = float(w["Low"].iloc[low_i])
    drop = pct(pole_low, pole_high)
    if drop > -(1.8 if loose else 2.5):
        return None
    flag = w.iloc[low_i+1:]
    if len(flag) < 3:
        return None
    bounce = (float(flag["High"].max())-pole_low)/max(pole_high-pole_low,1e-9)*100
    if not ((8 if loose else 12) <= bounce <= (75 if loose else 60 if not strict else 52)):
        return None
    low_slope = _slope_pct_per_bar(flag["Low"])
    close_slope = _slope_pct_per_bar(flag["Close"])
    if low_slope < (-0.18 if loose else -0.08) and close_slope < (-0.10 if loose else -0.04):
        return None
    support = float(flag["Low"].tail(min(5, len(flag))).min())
    entry = min(support, last*0.999)
    dist = (last-entry)/max(last, 1e-9)*100
    if dist > (4.5 if loose else 3.2 if not strict else 1.7):
        return None
    score = 58
    reasons = ["intraday drop + bear flag", f"prior drop {abs(drop):.1f}%", f"bounce {bounce:.0f}% of drop", f"{dist:.1f}% above breakdown"]
    if dist <= 1.5: score += 10; reasons.append("near trigger")
    if last < m.get("ema21", last)*1.01: score += 6; reasons.append("under/near 21 EMA")
    stop = float(flag["High"].max())
    target = entry - (pole_high-pole_low)*0.75
    return hit(ticker, "Bear Flag", score, "Bearish", reasons, entry, stop, target)


def intraday_vcp(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    """Intraday Volatility Contraction Pattern: successive smaller swing pullbacks under a pivot."""
    if len(df) < 90:
        return None
    m = base_metrics(df, spy_df)
    mode = current_mode(); loose = mode == "Loose / candidate"; strict = mode == "Strict / confirmed"
    w = _intraday_window(df, 150)
    last = float(w["Close"].iloc[-1])
    if last <= 0:
        return None
    # prior advance before the tightening area
    if pct(last, float(w["Close"].iloc[0])) < (2.0 if loose else 3.0):
        return None
    highs, lows = _recent_swing_points(w, 2, 2)
    swings = sorted(highs + lows, key=lambda x: x[0])
    if len(swings) < 7:
        return None
    # Use rolling range contractions over nested recent windows plus higher lows.
    ranges = []
    for bars in [90, 55, 34, 18]:
        ww = w.tail(min(bars, len(w)))
        ranges.append((float(ww["High"].max())-float(ww["Low"].min()))/max(last,1e-9)*100)
    if not (ranges[0] > ranges[1] > ranges[2] > ranges[3]):
        return None
    if ranges[3] > (4.5 if loose else 3.5 if not strict else 2.5):
        return None
    recent_lows = [lo for i, lo in lows if i > len(w)-70]
    if len(recent_lows) >= 3 and not (recent_lows[-1] >= min(recent_lows[-3:-1])*0.995):
        return None
    pivot = float(w["High"].tail(34).max())
    dist = (pivot-last)/max(pivot, 1e-9)*100
    if dist < -0.8 or dist > (4.0 if loose else 2.8 if not strict else 1.5):
        return None
    vol_dry = float(w["Volume"].tail(10).mean()) < float(w["Volume"].tail(60).mean()) * (0.90 if loose else 0.80)
    if strict and not vol_dry:
        return None
    score = 60
    reasons = [f"intraday volatility contraction {ranges[0]:.1f}% > {ranges[1]:.1f}% > {ranges[2]:.1f}% > {ranges[3]:.1f}%", "near pivot"]
    if vol_dry: score += 12; reasons.append("volume dry-up")
    if dist <= 1.5: score += 8; reasons.append("tight under pivot")
    stop = float(w["Low"].tail(34).min())
    target = pivot + (pivot-stop)*1.5
    return hit(ticker, "Volatility Contraction Pattern", score, "Bullish", reasons, pivot, stop, target)


def intraday_ascending_triangle(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    """Intraday ascending triangle: flat resistance, separated touches, rising lows, compression."""
    if len(df) < 70:
        return None
    m = base_metrics(df, spy_df)
    mode = current_mode(); loose = mode == "Loose / candidate"; strict = mode == "Strict / confirmed"
    w = _intraday_window(df, 120)
    last = float(w["Close"].iloc[-1])
    resistance = float(w["High"].tail(80).quantile(0.985))
    tol = (0.008 if loose else 0.0055 if not strict else 0.0035)
    touches_raw = [i for i, h in enumerate(w["High"].values) if abs(float(h)-resistance)/max(resistance,1e-9) <= tol]
    touches = []
    for i in touches_raw:
        if not touches or i - touches[-1] >= (5 if loose else 7):
            touches.append(i)
    if len(touches) < (2 if loose else 3):
        return None
    _, lows = _recent_swing_points(w, 2, 2)
    lows = [(i, lo) for i, lo in lows if i >= max(0, touches[0]-8) and lo < resistance*0.985]
    if len(lows) < (2 if loose else 3):
        return None
    used = lows[-3:] if len(lows) >= 3 else lows[-2:]
    if len(used) == 3:
        rising = used[1][1] > used[0][1]*1.006 and used[2][1] > used[1][1]*1.002
    else:
        rising = used[1][1] > used[0][1]*1.012
    if not rising:
        return None
    base_low = min(x[1] for x in used)
    depth = (resistance-base_low)/max(resistance,1e-9)*100
    if depth < (1.2 if loose else 1.8) or depth > (12 if loose else 9 if not strict else 7):
        return None
    if not (resistance*(1-(0.045 if loose else 0.03)) <= last <= resistance*(1+(0.006 if loose else 0.003))):
        return None
    early = (float(w["High"].tail(80).head(35).max())-float(w["Low"].tail(80).head(35).min()))/max(resistance,1e-9)
    recent = (float(w["High"].tail(20).max())-float(w["Low"].tail(20).min()))/max(resistance,1e-9)
    if recent > early*(0.88 if loose else 0.78):
        return None
    score = 60
    reasons = ["intraday flat resistance", "rising lows", f"base depth {depth:.1f}%", "near breakout"]
    if len(touches) >= 3: score += 8; reasons.append("3+ resistance touches")
    if recent < early*0.65: score += 8; reasons.append("range compression")
    stop = min(lo for _, lo in used[-2:])
    target = resistance + (resistance-base_low)
    return hit(ticker, "Ascending Triangle", score, "Bullish", reasons, resistance, stop, target)


def intraday_falling_wedge(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 70:
        return None
    m = base_metrics(df, spy_df)
    mode = current_mode(); loose = mode == "Loose / candidate"; strict = mode == "Strict / confirmed"
    w = _intraday_window(df, 100)
    last = float(w["Close"].iloc[-1])
    high_slope = _slope_pct_per_bar(w["High"].rolling(4).max().dropna().tail(60))
    low_slope = _slope_pct_per_bar(w["Low"].rolling(4).min().dropna().tail(60))
    converging = high_slope < (-0.025 if loose else -0.04) and low_slope < 0 and high_slope < low_slope*1.15
    early = (float(w["High"].head(35).max())-float(w["Low"].head(35).min()))/max(last,1e-9)
    recent = (float(w["High"].tail(20).max())-float(w["Low"].tail(20).min()))/max(last,1e-9)
    if not (converging and recent < early*(0.85 if loose else 0.72)):
        return None
    upper = float(w["High"].tail(15).max())
    if last < upper*(0.97 if loose else 0.985):
        return None
    score = 58
    reasons = ["intraday falling wedge", "converging trendlines", "range contraction", "near upper wedge line"]
    if last > m.get("ema21", last)*0.995: score += 8; reasons.append("reclaiming/near 21 EMA")
    return hit(ticker, "Falling Wedge", score, "Bullish Reversal", reasons, upper, float(w["Low"].min()), upper + 2*m.get("atr", 0))


def intraday_double_bottom(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 65:
        return None
    m = base_metrics(df, spy_df)
    mode = current_mode(); loose = mode == "Loose / candidate"; strict = mode == "Strict / confirmed"
    w = _intraday_window(df, 110)
    lows = _recent_swing_points(w, 2, 2)[1]
    if len(lows) < 2:
        return None
    l1_i, l1 = lows[-2]; l2_i, l2 = lows[-1]
    if l2_i-l1_i < (8 if loose else 12):
        return None
    similar = abs(l1-l2)/max((l1+l2)/2, 1e-9)*100 <= (2.5 if loose else 1.7 if not strict else 1.2)
    if not similar or l2 < l1*(0.985 if loose else 0.995):
        return None
    neckline = float(w["High"].iloc[l1_i:l2_i+1].max())
    last = float(w["Close"].iloc[-1])
    dist = (neckline-last)/max(neckline,1e-9)*100
    if dist < -1.5 or dist > (4.0 if loose else 2.5 if not strict else 1.3):
        return None
    score = 60
    reasons = ["intraday double bottom", "two similar lows", "near neckline"]
    if last > m.get("ema21", last)*0.995: score += 8; reasons.append("momentum stabilizing")
    return hit(ticker, "Double Bottom", score, "Bullish Reversal", reasons, neckline, min(l1,l2), neckline+(neckline-min(l1,l2)))


def intraday_tight_consolidation(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    """Intraday tight base / flat base: 8-30 bars of tight range after a move."""
    if len(df) < 60:
        return None
    m = base_metrics(df, spy_df)
    mode = current_mode(); loose = mode == "Loose / candidate"; strict = mode == "Strict / confirmed"
    w = _intraday_window(df, 100)
    last = float(w["Close"].iloc[-1])
    base = w.tail(24 if not strict else 30)
    prior = w.iloc[:-len(base)]
    if len(prior) < 20:
        return None
    prior_move = pct(float(base["Close"].iloc[0]), float(prior["Close"].iloc[-20]))
    if prior_move < (1.2 if loose else 2.0):
        return None
    hi = float(base["High"].max()); lo = float(base["Low"].min())
    rng = (hi-lo)/max(last,1e-9)*100
    if rng > (4.5 if loose else 3.2 if not strict else 2.2):
        return None
    if not (hi*0.96 <= last <= hi*1.01):
        return None
    vol_quiet = float(base["Volume"].tail(8).mean()) < float(w["Volume"].tail(60).mean())*(0.95 if loose else 0.85)
    score = 58
    reasons = ["intraday tight consolidation", f"base range {rng:.1f}%", "near pivot"]
    if vol_quiet: score += 10; reasons.append("volume quiet")
    return hit(ticker, "Flat Base", score, "Bullish Breakout", reasons, hi, lo, hi+(hi-lo))

def bull_flag_core(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    """Bull flag candidate/triggered scan using the flag breakout line, not only the old high.

    Visual bull flags often break the descending flag resistance before they reclaim the
    original pole high. Earlier versions used the old pole high as the main pivot, which
    could miss valid setups such as a strong stock pulling back in a controlled channel.
    """
    if len(df) < 80:
        return None

    m = base_metrics(df, spy_df)
    if any(np.isnan(x) for x in [m["last"], m["sma50"], m["sma200"], m["ema21"], m["atr"]]):
        return None

    mode = current_mode()
    loose = mode == "Loose / candidate"
    strict = mode == "Strict / confirmed"
    close = m["last"]

    # Bull flag is a continuation setup. Keep a bullish context, but allow a stock
    # that is pulling back toward the 50-day after a strong impulse.
    if strict:
        trend_ok = close > m["sma50"] * 0.99 and m["sma50"] > m["sma150"] > m["sma200"] and slope(df["SMA50"], 20) > 0
    else:
        trend_ok = (
            close > m["sma200"] * (0.97 if loose else 0.985) and
            m["sma50"] >= m["sma200"] * (0.96 if loose else 0.98) and
            slope(df["SMA50"], 20) > (-0.03 if loose else -0.01)
        )
    if not trend_ok:
        return None

    lookback = 60 if loose else 55
    exclude_recent = 1 if loose else 2
    start = max(0, len(df) - lookback)
    end = max(start + 1, len(df) - exclude_recent)
    pivot_slice = df.iloc[start:end]
    if pivot_slice.empty:
        return None

    pivot_pos = start + _position_of_max(pivot_slice["High"])
    bars_since_pivot = len(df) - 1 - pivot_pos

    min_bars = 3 if loose else (4 if not strict else 5)
    max_bars = 34 if loose else (30 if not strict else 22)
    if bars_since_pivot < min_bars or bars_since_pivot > max_bars:
        return None

    pivot_high = float(df["High"].iloc[pivot_pos])

    # Find the impulse/pole low before the pivot high.
    pole_window = df.iloc[max(0, pivot_pos - 60): pivot_pos + 1]
    if len(pole_window) < 8:
        return None
    pole_low_pos = pole_window["Low"].idxmin()
    pole_low_i = df.index.get_loc(pole_low_pos)
    pole_low = float(df.loc[pole_low_pos, "Low"])
    pole_bars = pivot_pos - pole_low_i
    if pole_bars < 5 or pole_bars > 55:
        return None

    pole_gain = pct(pivot_high, pole_low)
    min_gain = max(6 if loose else (8 if not strict else 11), m.get("adr20", 3.0) * (1.8 if loose else 2.2))
    if pole_gain < min_gain:
        return None

    flag = df.iloc[pivot_pos + 1:]
    if len(flag) < min_bars:
        return None

    flag_high = float(flag["High"].max())
    flag_low = float(flag["Low"].min())
    pole_height = max(pivot_high - pole_low, 1e-9)
    depth_of_pole = (pivot_high - flag_low) / pole_height * 100
    depth_pct_price = (pivot_high - flag_low) / max(pivot_high, 1e-9) * 100
    flag_range_pct = (flag_high - flag_low) / max(close, 1e-9) * 100
    high_slope = _slope_pct_per_bar(flag["High"])
    low_slope = _slope_pct_per_bar(flag["Low"])
    close_slope = _slope_pct_per_bar(flag["Close"])

    # Accept a controlled flag/retrace. Balanced allows deeper mega-cap style flags,
    # while Strict keeps textbook patterns tighter.
    max_depth = 72 if loose else (64 if not strict else 48)
    min_depth = 1.5 if loose else 3.0
    if not (min_depth <= depth_of_pole <= max_depth):
        return None
    if depth_pct_price > max(17 if loose else (14 if not strict else 10), m.get("adr20", 3.0) * (4.8 if loose else 3.8)):
        return None
    if flag_range_pct > max(15 if loose else (12 if not strict else 8), m.get("adr20", 3.0) * (4.2 if loose else 3.2)):
        return None

    # The flag should be sideways/down or slightly drifting, not a new sharp rally.
    recent_highs_declining = False
    if len(flag) >= 8:
        first_half_hi = float(flag["High"].iloc[: max(3, len(flag)//2)].max())
        second_half_hi = float(flag["High"].iloc[max(3, len(flag)//2):].max())
        recent_highs_declining = second_half_hi <= first_half_hi * 1.01

    channel_ok = (
        high_slope <= (0.28 if loose else 0.18) or
        close_slope <= (0.12 if loose else 0.06) or
        recent_highs_declining
    )
    if not channel_ok:
        return None
    if strict and close_slope > 0.18:
        return None

    # Use the actual flag breakout area as entry. This is usually the recent descending
    # flag resistance, not necessarily the old pole high.
    try:
        x = np.arange(len(flag), dtype=float)
        high_vals = flag["High"].astype(float).values
        hi_m, hi_b = np.polyfit(x, high_vals, 1)
        projected_resistance = float(hi_m * (len(flag)) + hi_b)
    except Exception:
        projected_resistance = float(flag["High"].tail(min(8, len(flag))).max())

    # For a descending bull flag, the practical trigger is the current descending
    # flag-resistance line, not the highest high from the early part of the flag.
    # Using the last 8 bars' maximum can push the entry back near the old pole high
    # and miss AMZN-style flags. Use the projected resistance plus the most recent
    # 3-bar resistance area instead.
    recent_resistance_near = float(flag["High"].tail(min(3, len(flag))).max())
    recent_resistance_wide = float(flag["High"].tail(min(8, len(flag))).max())
    if high_slope < -0.03 or recent_highs_declining:
        flag_breakout = max(projected_resistance, recent_resistance_near)
    else:
        flag_breakout = max(projected_resistance, recent_resistance_wide)
    # Do not let an odd projected line put entry above the old pole high or below support.
    entry = min(pivot_high, max(flag_breakout, close * 1.002))

    # Candidate must be reasonably close to the flag breakout line. This is the key
    # AMZN-style fix: do not require price to be close to the old high.
    dist_to_entry = (entry - close) / max(entry, 1e-9) * 100
    max_dist = 9.0 if loose else (7.0 if not strict else 3.0)
    if dist_to_entry > max_dist:
        return None
    if close > entry * (1.08 if loose else 1.05):
        return None

    # Volume contraction helps separate real flags from noisy pullbacks.
    pole_vol = _safe_mean(df["Volume"].iloc[max(0, pole_low_i): pivot_pos + 1])
    flag_vol = _safe_mean(flag["Volume"])
    recent_flag_vol = _safe_mean(flag["Volume"].tail(min(8, len(flag))))
    vol_contract = flag_vol < pole_vol * (0.98 if loose else 0.90) or recent_flag_vol < pole_vol * (0.85 if loose else 0.78)
    if strict and not vol_contract:
        return None

    score = 58
    triggered = close >= entry
    state_word = "flag breakout triggered" if triggered else "bull flag candidate"
    reasons = [
        f"{state_word}: flag resistance near {entry:.2f}",
        f"pole gain {pole_gain:.1f}%",
        f"pullback {depth_of_pole:.0f}% of pole",
        f"{dist_to_entry:.1f}% from entry"
    ]
    if vol_contract:
        score += 12
        reasons.append("flag volume contracted")
    if high_slope <= 0.10 or recent_highs_declining:
        score += 10
        reasons.append("sideways/down flag")
    if dist_to_entry <= 2.0:
        score += 8
        reasons.append("near trigger")
    if close > m["ema21"] * 0.995:
        score += 5
        reasons.append("holding/reclaiming 21 EMA")
    if m.get("rs_vs_spy", 0) > -2:
        score += 5
        reasons.append("relative strength acceptable")
    if triggered and m["rvol"] > 1.15:
        score += 8
        reasons.append("breakout volume confirmation")

    stop = round(flag_low, 2)
    target = round(entry + pole_height * 0.75, 2)
    return hit(ticker, "Bull Flag", score, "Bullish", reasons, round(entry, 2), stop, target)


def bull_flag_visual_candidate(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    """Fallback visual bull-flag detector.

    This catches valid chart-review candidates where the breakout line is the
    descending flag resistance rather than the original pole high. It is still
    filtered for trend, impulse, controlled pullback, and distance to the flag
    line. It returns Watch/Near candidates, not automatic buy signals.
    """
    if len(df) < 80:
        return None
    m = base_metrics(df, spy_df)
    if any(np.isnan(x) for x in [m["last"], m["sma50"], m["sma200"], m["ema21"], m["atr"]]):
        return None
    mode = current_mode()
    strict = mode == "Strict / confirmed"
    loose = mode == "Loose / candidate"
    if strict:
        return None
    close = m["last"]
    # Continuation context: avoid damaged/downtrend names, but allow pullbacks
    # toward the 50-day after a strong rally.
    if not (close > m["sma200"] * 0.97 and m["sma50"] > m["sma200"] * 0.96):
        return None

    lookback = 65 if loose else 55
    start = max(0, len(df) - lookback)
    end = max(start + 1, len(df) - 2)
    pivot_slice = df.iloc[start:end]
    if pivot_slice.empty:
        return None
    pivot_pos = start + _position_of_max(pivot_slice["High"])
    bars_since_pivot = len(df) - 1 - pivot_pos
    if bars_since_pivot < 3 or bars_since_pivot > (38 if loose else 28):
        return None
    pivot_high = float(df["High"].iloc[pivot_pos])

    pole_window = df.iloc[max(0, pivot_pos - 60): pivot_pos + 1]
    if len(pole_window) < 8:
        return None
    pole_low_pos = pole_window["Low"].idxmin()
    pole_low_i = df.index.get_loc(pole_low_pos)
    pole_low = float(df.loc[pole_low_pos, "Low"])
    if pivot_pos - pole_low_i < 5:
        return None
    pole_gain = pct(pivot_high, pole_low)
    if pole_gain < max(7.0, m.get("adr20", 3.0) * 1.8):
        return None

    flag = df.iloc[pivot_pos + 1:]
    if len(flag) < 3:
        return None
    flag_low = float(flag["Low"].min())
    flag_high = float(flag["High"].max())
    pole_height = max(pivot_high - pole_low, 1e-9)
    depth_of_pole = (pivot_high - flag_low) / pole_height * 100
    depth_pct_price = (pivot_high - flag_low) / max(pivot_high, 1e-9) * 100
    if not (4 <= depth_of_pole <= (78 if loose else 70)):
        return None
    if depth_pct_price > (18 if loose else 15):
        return None

    high_slope = _slope_pct_per_bar(flag["High"])
    close_slope = _slope_pct_per_bar(flag["Close"])
    if len(flag) >= 6:
        first_half_hi = float(flag["High"].iloc[: max(3, len(flag)//2)].max())
        second_half_hi = float(flag["High"].iloc[max(3, len(flag)//2):].max())
        highs_declining = second_half_hi <= first_half_hi * 1.015
    else:
        highs_declining = high_slope <= 0.15
    if not (high_slope <= 0.18 or close_slope <= 0.08 or highs_declining):
        return None

    try:
        x = np.arange(len(flag), dtype=float)
        high_vals = flag["High"].astype(float).values
        hi_m, hi_b = np.polyfit(x, high_vals, 1)
        projected_resistance = float(hi_m * len(flag) + hi_b)
    except Exception:
        projected_resistance = float(flag["High"].tail(min(3, len(flag))).max())
    recent_near = float(flag["High"].tail(min(3, len(flag))).max())
    entry = min(pivot_high, max(projected_resistance, recent_near, close * 1.002))
    dist_to_entry = (entry - close) / max(entry, 1e-9) * 100
    if dist_to_entry > (10.0 if loose else 8.0):
        return None
    if close < flag_low * 0.985:
        return None

    pole_vol = _safe_mean(df["Volume"].iloc[max(0, pole_low_i): pivot_pos + 1])
    flag_vol = _safe_mean(flag["Volume"])
    vol_contract = flag_vol < pole_vol * 1.05 if pole_vol > 0 else False

    score = 60
    reasons = [
        f"visual bull flag candidate: flag resistance near {entry:.2f}",
        f"pole gain {pole_gain:.1f}%",
        f"pullback {depth_of_pole:.0f}% of pole",
        f"{dist_to_entry:.1f}% from entry",
    ]
    if highs_declining or high_slope <= 0.05:
        score += 8; reasons.append("descending/sideways flag")
    if vol_contract:
        score += 8; reasons.append("flag volume normal/contracted")
    if close > m["ema21"] * 0.98:
        score += 5; reasons.append("near/above 21 EMA")
    if dist_to_entry <= 5:
        score += 6; reasons.append("near trigger")
    stop = round(flag_low, 2)
    target = round(entry + pole_height * 0.70, 2)
    return hit(ticker, "Bull Flag", score, "Bullish", reasons, round(entry, 2), stop, target)


def bull_flag(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if is_intraday():
        return intraday_bull_flag(ticker, df, meta, spy_df)
    primary = bull_flag_core(ticker, df, meta, spy_df)
    if primary is not None:
        return primary
    return bull_flag_visual_candidate(ticker, df, meta, spy_df)

def bear_flag(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if is_intraday():
        return intraday_bear_flag(ticker, df, meta, spy_df)
    """Balanced bear-flag logic with explicit bullish-leader rejection."""
    if len(df) < 80:
        return None

    m = base_metrics(df, spy_df)
    if any(np.isnan(x) for x in [m["last"], m["sma50"], m["sma200"], m["ema21"], m["atr"]]):
        return None

    mode = current_mode(); loose = mode == "Loose / candidate"; strict = mode == "Strict / confirmed"

    bearish_context = (
        m["last"] < m["sma50"] * (1.02 if loose else 1.005) or
        (m["last"] < m["ema21"] * (1.01 if loose else 1.0) and slope(df["EMA21"], 12) < 0) or
        m["sma50"] < m["sma200"] * (1.01 if loose else 1.0)
    )
    strong_bull_leader = (
        m["last"] > m["sma50"] * 1.01 and m["sma50"] > m["sma200"] and
        slope(df["SMA50"], 20) > 0 and m.get("rs_vs_spy", 0) > 0
    )
    if strict:
        bearish_context = m["last"] < m["sma50"] and (m["ema21"] < m["sma50"] or slope(df["SMA50"], 20) < 0)
    if not bearish_context or strong_bull_leader:
        return None

    lookback = 45 if loose else 38
    exclude_recent = 1 if loose else 2
    start = max(0, len(df) - lookback)
    end = max(start + 1, len(df) - exclude_recent)
    pivot_slice = df.iloc[start:end]
    if pivot_slice.empty:
        return None
    pivot_pos = start + _position_of_min(pivot_slice["Low"])
    bars_since_pivot = len(df) - 1 - pivot_pos
    min_bars = 2 if loose else 3
    max_bars = 28 if loose else (22 if not strict else 16)
    if bars_since_pivot < min_bars or bars_since_pivot > max_bars:
        return None

    pivot_low = float(df["Low"].iloc[pivot_pos])
    pole_start = max(0, pivot_pos - 50)
    pole_window = df.iloc[pole_start : pivot_pos + 1]
    if len(pole_window) < 6:
        return None
    pole_high_pos = pole_start + _position_of_max(pole_window["High"])
    pole_high = float(df["High"].iloc[pole_high_pos])
    pole_bars = pivot_pos - pole_high_pos
    if pole_bars < 4 or pole_bars > 50:
        return None

    pole_drop = pct(pivot_low, pole_high)
    required_drop = -max(5.0 if loose else 7.0, min(14.0 if strict else 11.0, m.get("adr20", 3.0) * (2.4 if strict else 1.8)))
    if pole_drop > required_drop:
        return None

    flag = df.iloc[pivot_pos + 1 :]
    if len(flag) < min_bars:
        return None

    flag_high = float(flag["High"].max())
    flag_low = float(flag["Low"].min())
    close = m["last"]
    drop_height = max(pole_high - pivot_low, 1e-9)
    bounce_of_drop = (flag_high - pivot_low) / drop_height * 100.0
    flag_range_pct = (flag_high - flag_low) / max(close, 1e-9) * 100.0

    if not ((3.0 if loose else 6.0) <= bounce_of_drop <= (72.0 if loose else (60.0 if not strict else 52.0))):
        return None

    close_slope = _slope_pct_per_bar(flag["Close"])
    if strict and close_slope < -0.35:
        return None

    if close > m["sma50"] * (1.035 if loose else 1.015) and close > m["ema21"] * (1.035 if loose else 1.015):
        return None

    near_breakdown = close <= pivot_low + (flag_high - pivot_low) * (0.78 if loose else 0.65)
    if strict:
        near_breakdown = close <= pivot_low + (flag_high - pivot_low) * 0.55
    if not near_breakdown:
        return None

    tight_limit = max(14.0 if loose else 11.0, m.get("adr20", 3.0) * (5.0 if loose else 4.0))
    if strict:
        tight_limit = max(8.0, m.get("adr20", 3.0) * 3.5)
    if flag_range_pct > tight_limit:
        return None

    pole_vol = _safe_mean(df["Volume"].iloc[max(pole_high_pos, pivot_pos - 12) : pivot_pos + 1])
    flag_vol = _safe_mean(flag["Volume"])
    prior_vol = _safe_mean(df["Volume"].iloc[max(0, pivot_pos - 25) : pivot_pos + 1])
    vol_contract = flag_vol < max(pole_vol, prior_vol) * (1.05 if loose else 0.98)

    score = 52
    reasons = [f"drop + bear flag candidate: pivot low {bars_since_pivot} bars ago", f"prior drop {pole_drop:.1f}%", f"bounce {bounce_of_drop:.0f}% of drop"]
    if vol_contract:
        score += 12; reasons.append("bounce volume drying/normalizing")
    if close < m["ema21"]:
        score += 8; reasons.append("below 21 EMA")
    if close < m["sma50"]:
        score += 8; reasons.append("below 50-day MA")
    if m.get("rs_vs_spy", 0) < 0:
        score += 7; reasons.append("lagging SPY")
    if close <= pivot_low * 1.04:
        score += 6; reasons.append("near breakdown trigger")
    if strict and not (vol_contract and close < m["ema21"]):
        return None

    entry = round(pivot_low, 2)
    stop = round(flag_high, 2)
    target = round(entry - drop_height, 2)
    return hit(ticker, "Bear Flag", score, "Bearish", reasons, entry, stop, target)


def vcp(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if is_intraday():
        return intraday_vcp(ticker, df, meta, spy_df)
    """Volatility Contraction Pattern candidate.

    A true Volatility Contraction Pattern should show repeated, measurable pullback contractions:
    larger pullback -> smaller pullback -> final tight pivot area.
    This version intentionally avoids simple sideways boxes or general bases.
    """
    if len(df) < 220:
        return None
    m = base_metrics(df, spy_df)
    mode = current_mode(); loose = mode == "Loose / candidate"; strict = mode == "Strict / confirmed"

    last = float(m["last"])
    if last <= 0:
        return None

    # A Volatility Contraction Pattern is normally a continuation pattern in a leading/constructive stock,
    # not a bottoming pattern far below major moving averages.
    if strict:
        trend_ok = (
            last > m["sma50"] > m["sma150"] > m["sma200"]
            and slope(df["SMA50"], 20) > 0
            and last >= 0.82 * m["high252"]
            and last >= 1.35 * m["low252"]
        )
    elif loose:
        trend_ok = last > m["sma200"] and last >= 0.72 * m["high252"] and last >= 1.18 * m["low252"]
    else:
        trend_ok = (
            last > m["sma50"]
            and last > m["sma150"]
            and last > m["sma200"]
            and last >= 0.78 * m["high252"]
            and last >= 1.25 * m["low252"]
        )
    if not trend_ok:
        return None

    base = df.tail(110).copy()
    work = base.iloc[:-1].copy() if len(base) > 20 else base.copy()
    pivot = float(work["High"].tail(75).max())
    base_low = float(work["Low"].tail(90).min())
    base_depth = (pivot - base_low) / max(pivot, 1e-9) * 100

    min_depth = 7.0 if loose else (10.0 if not strict else 12.0)
    max_depth = 38.0 if loose else (30.0 if not strict else 26.0)
    if not (min_depth <= base_depth <= max_depth):
        return None

    # Prior advance into the base. Without this, a sideways utility/defensive box
    # can look like a contraction mathematically but is not a classic Volatility Contraction Pattern.
    prior = df.iloc[max(0, len(df)-220): max(0, len(df)-110)]
    if prior.empty:
        return None
    prior_low = float(prior["Low"].min())
    prior_advance = (pivot - prior_low) / max(prior_low, 1e-9) * 100
    if prior_advance < (22.0 if loose else (30.0 if not strict else 40.0)):
        return None

    # Must be close to the pivot. Volatility Contraction Pattern is most useful when the stock is tight under resistance.
    dist_to_pivot = (pivot - last) / max(pivot, 1e-9) * 100
    max_dist = 6.0 if loose else (4.0 if not strict else 2.5)
    if dist_to_pivot < -1.0 or dist_to_pivot > max_dist:
        return None

    # Final tightness: this is the most important filter to avoid broad, choppy bases.
    def range_pct(w: pd.DataFrame) -> float:
        if w.empty:
            return 999.0
        hi = float(w["High"].max()); lo = float(w["Low"].min())
        return (hi - lo) / max(hi, 1e-9) * 100

    r20 = range_pct(df.tail(20))
    r10 = range_pct(df.tail(10))
    final_limit = max(5.5 if loose else (4.5 if not strict else 3.5), m.get("adr20", 3.0) * (1.35 if loose else 1.10))
    if r10 > final_limit or r20 > (final_limit * 1.55):
        return None

    # Find local swing highs/lows and require actual pullback contractions,
    # not just overlapping-window range shrink.
    w = work.tail(100).reset_index(drop=True)
    highs = w["High"].astype(float).to_numpy()
    lows = w["Low"].astype(float).to_numpy()
    k = 3
    pts = []
    for i in range(k, len(w) - k):
        hseg = highs[i-k:i+k+1]
        lseg = lows[i-k:i+k+1]
        if highs[i] >= np.nanmax(hseg):
            pts.append((i, "H", float(highs[i])))
        if lows[i] <= np.nanmin(lseg):
            pts.append((i, "L", float(lows[i])))
    pts = sorted(pts, key=lambda x: (x[0], x[1]))

    # Collapse consecutive same-type points and keep the most extreme one.
    clean = []
    for pt in pts:
        if not clean or clean[-1][1] != pt[1]:
            clean.append(pt)
        else:
            last_pt = clean[-1]
            if pt[1] == "H" and pt[2] > last_pt[2]:
                clean[-1] = pt
            elif pt[1] == "L" and pt[2] < last_pt[2]:
                clean[-1] = pt

    pairs = []
    for idx, pt in enumerate(clean[:-1]):
        if pt[1] != "H":
            continue
        # next low after this high, before the next high
        nxt_low = None
        for nxt in clean[idx+1:]:
            if nxt[1] == "H":
                break
            if nxt[1] == "L":
                nxt_low = nxt
                break
        if nxt_low is None:
            continue
        depth = (pt[2] - nxt_low[2]) / max(pt[2], 1e-9) * 100
        if depth >= (2.5 if loose else 3.0):
            pairs.append((pt[0], nxt_low[0], depth, pt[2], nxt_low[2]))

    # Need three material pullbacks that contract meaningfully.
    good_seq = None
    for i in range(0, max(0, len(pairs) - 2)):
        d1, d2, d3 = pairs[i][2], pairs[i+1][2], pairs[i+2][2]
        if d1 < (8.0 if loose else (10.0 if not strict else 12.0)):
            continue
        if d2 <= d1 * (0.88 if loose else 0.80) and d3 <= d2 * (0.88 if loose else 0.80):
            if d3 <= (7.0 if loose else (5.8 if not strict else 4.8)):
                # Last contraction should be recent enough to matter.
                if pairs[i+2][1] >= len(w) - (32 if loose else 25):
                    good_seq = (d1, d2, d3)
                    break
    if good_seq is None:
        return None

    # Lows should generally rise/hold; a flat lower boundary is more of a box or rectangle.
    lows_seq = [p[4] for p in pairs[-3:]]
    if len(lows_seq) >= 3:
        higher_lows = lows_seq[1] >= lows_seq[0] * (0.985 if loose else 1.00) and lows_seq[2] >= lows_seq[1] * (0.985 if loose else 1.00)
        if not higher_lows:
            return None

    # Volume should dry up near the pivot.
    vol10 = float(df["Volume"].tail(10).mean())
    vol50 = float(df["Volume"].tail(50).mean())
    vol_dry = vol10 < vol50 * (0.86 if loose else (0.78 if not strict else 0.70))
    if not vol_dry:
        return None

    score = 62
    d1, d2, d3 = good_seq
    reasons = [
        f"true contraction sequence {d1:.1f}% → {d2:.1f}% → {d3:.1f}%",
        f"final tight range {r10:.1f}%",
        "near pivot",
        "volume dry-up",
    ]
    if m.get("rs_vs_spy", 0) > 0:
        score += 10; reasons.append("outperforming SPY")
    if last >= pivot * 0.985:
        score += 8; reasons.append("tight under pivot")
    if m["sma50"] > m["sma150"] > m["sma200"]:
        score += 8; reasons.append("MA structure supportive")
    if prior_advance >= 50:
        score += 7; reasons.append(f"strong prior advance {prior_advance:.0f}%")
    if base_depth <= 22:
        score += 5; reasons.append(f"controlled base depth {base_depth:.0f}%")

    stop = float(df["Low"].tail(25).min())
    target = pivot + (pivot - stop) * 1.5
    return hit(ticker, "Volatility Contraction Pattern", score, "Bullish", reasons, pivot, stop, target)

def _cup_handle_candidate(ticker: str, w: pd.DataFrame, intraday_style: bool, loose: bool, strict: bool, m: dict) -> Optional[dict]:
    """Evaluate one recent window as a current cup-and-handle candidate.

    This function assumes the handle is the most recent consolidation. It scans several
    possible handle lengths so intraday examples such as ENPH 1h are not missed just
    because the handle is short. It also rejects daily false positives that are really
    one-bar gap moves or flat boxes rather than rounded cups.
    """
    w = w.copy().reset_index(drop=True)
    n = len(w)
    if n < (45 if intraday_style else 90):
        return None

    if intraday_style:
        handle_min = 3 if loose else (4 if not strict else 6)
        handle_max = min(24 if loose else (20 if not strict else 16), max(4, n // 4))
        min_left_space = 8 if loose else 10
        min_right_space = 8 if loose else 10
        min_cup_bars = 24 if loose else 30
        min_depth = 2.8 if loose else (3.5 if not strict else 5.0)
        max_depth = 24.0 if loose else (18.0 if not strict else 14.0)
        max_rim_mismatch = 18.0 if loose else (14.0 if not strict else 10.0)
        max_dist_to_pivot = 8.0 if loose else (6.0 if not strict else 3.5)
        min_bottom_bars = 2 if loose else (3 if not strict else 4)
        upper_half_frac = 0.46 if loose else (0.50 if not strict else 0.56)
        max_handle_depth_abs = 8.5 if loose else (6.5 if not strict else 4.5)
    else:
        handle_min = 7 if loose else (9 if not strict else 12)
        handle_max = min(35 if loose else (30 if not strict else 24), max(10, n // 4))
        min_left_space = 20 if loose else 28
        min_right_space = 24 if loose else 32
        min_cup_bars = 65 if loose else 80
        min_depth = 10.0 if loose else (12.0 if not strict else 14.0)
        max_depth = 42.0 if loose else (34.0 if not strict else 28.0)
        max_rim_mismatch = 12.0 if loose else (8.0 if not strict else 5.5)
        max_dist_to_pivot = 7.0 if loose else (5.0 if not strict else 3.0)
        min_bottom_bars = 5 if loose else (7 if not strict else 9)
        upper_half_frac = 0.56 if loose else (0.62 if not strict else 0.68)
        max_handle_depth_abs = 14.0 if loose else (11.0 if not strict else 8.0)

    best = None
    # Try different recent handle lengths. The handle should be at the far right edge.
    for hlen in range(handle_min, handle_max + 1):
        cup = w.iloc[:-hlen].copy()
        handle = w.iloc[-hlen:].copy()
        if len(cup) < min_cup_bars:
            continue

        low_i = _position_of_min(cup["Low"])
        if low_i < min_left_space or low_i > len(cup) - min_right_space:
            continue

        left = cup.iloc[:low_i]
        right = cup.iloc[low_i:]
        if len(left) < min_left_space or len(right) < min_right_space:
            continue

        left_rim_i = _position_of_max(left["High"])
        right_rim_rel = _position_of_max(right["High"])
        right_rim_i = low_i + right_rim_rel

        if low_i - left_rim_i < (6 if intraday_style else 14):
            continue
        if right_rim_i - low_i < (6 if intraday_style else 18):
            continue

        left_rim = float(cup["High"].iloc[left_rim_i])
        right_rim = float(cup["High"].iloc[right_rim_i])
        cup_low = float(cup["Low"].iloc[low_i])
        rim = min(left_rim, right_rim)
        if rim <= 0 or cup_low <= 0:
            continue

        depth = (rim - cup_low) / rim * 100.0
        if not (min_depth <= depth <= max_depth):
            continue

        rim_mismatch = abs(left_rim - right_rim) / max(left_rim, 1e-9) * 100.0
        if rim_mismatch > max_rim_mismatch:
            continue

        recovery = (right_rim - cup_low) / max(rim - cup_low, 1e-9)
        if recovery < (0.68 if intraday_style and loose else (0.74 if intraday_style else (0.84 if loose else (0.88 if not strict else 0.93)))):
            continue

        # Bottom should spend time near the low. This prevents a sharp V from being called a cup.
        lower_third = cup_low + (rim - cup_low) * (0.36 if intraday_style else 0.33)
        bw = 8 if intraday_style else 18
        bottom_window = cup.iloc[max(0, low_i - bw): min(len(cup), low_i + bw + 1)]
        near_bottom_bars = int((bottom_window["Low"] <= lower_third).sum())
        if near_bottom_bars < min_bottom_bars:
            continue

        cup_bars = right_rim_i - left_rim_i
        if cup_bars < min_cup_bars:
            continue

        # Daily chart protection: reject DOC-style one-gap launches that are not rounded cups.
        if not intraday_style:
            right_side = cup.iloc[low_i:right_rim_i + 1].copy()
            if len(right_side) < min_right_space:
                continue
            close_pct = right_side["Close"].pct_change().abs().replace([np.inf, -np.inf], np.nan).dropna() * 100.0
            max_one_bar_jump = float(close_pct.max()) if not close_pct.empty else 0.0
            # A daily cup should recover over several bars, not by one vertical gap that represents most of the cup depth.
            if max_one_bar_jump > max(8.0, depth * (0.42 if strict else 0.55)):
                continue
            # Require multiple closes on the right side above the midpoint, not just one spike.
            midpoint = cup_low + (rim - cup_low) * 0.50
            if int((right_side["Close"] > midpoint).sum()) < (10 if loose else 14):
                continue

        handle_high = float(handle["High"].max())
        handle_low = float(handle["Low"].min())
        pivot = float(handle["High"].iloc[:-1].max()) if len(handle) > 2 else handle_high
        pivot = max(pivot, right_rim * (0.985 if intraday_style else 0.995))
        if pivot <= 0:
            continue

        handle_depth = (handle_high - handle_low) / max(handle_high, 1e-9) * 100.0
        max_handle_depth = min(max_handle_depth_abs, depth * (0.55 if intraday_style else (0.40 if loose else 0.32)))
        if handle_depth < (0.35 if intraday_style and loose else (0.60 if intraday_style else 1.5)):
            continue
        if handle_depth > max_handle_depth:
            continue

        upper_half_floor = cup_low + (rim - cup_low) * upper_half_frac
        if handle_low < upper_half_floor:
            continue

        # A handle should be tight/sideways/down. It should not be the right side continuing straight up.
        handle_slope = _slope_pct_per_bar(handle["Close"])
        if handle_slope > (0.55 if intraday_style and loose else (0.35 if intraday_style else (0.08 if loose else 0.03))):
            continue

        last = float(m["last"])
        dist_to_pivot = (pivot - last) / max(pivot, 1e-9) * 100.0
        if dist_to_pivot < -2.5 or dist_to_pivot > max_dist_to_pivot:
            continue

        # Trend/quality context.
        if intraday_style:
            if not (last > m["ema21"] or last > m["sma50"]):
                continue
        else:
            if not (last > m["sma200"] and last >= 0.70 * m["high252"]):
                continue
            if strict and not (last > m["sma50"] and m["sma50"] >= m["sma150"] * 0.97):
                continue

        vol_quiet = float(handle["Volume"].tail(min(6, len(handle))).mean()) < float(w["Volume"].tail(min(60, len(w))).mean()) * (1.05 if intraday_style and loose else (0.92 if intraday_style else (0.84 if loose else 0.76)))

        score = 55
        if intraday_style:
            score += 5
        if near_bottom_bars >= min_bottom_bars + 1:
            score += 8
        if rim_mismatch <= (10 if intraday_style else 6):
            score += 8
        if vol_quiet:
            score += 8
        if m.get("rs_vs_spy", 0) > 0:
            score += 6
        if dist_to_pivot <= 2.0:
            score += 6
        if handle_depth <= max_handle_depth * 0.70:
            score += 4
        if not intraday_style and cup_bars >= 90:
            score += 5

        candidate = {
            "score": score,
            "depth": depth,
            "rim_mismatch": rim_mismatch,
            "handle_depth": handle_depth,
            "near_bottom_bars": near_bottom_bars,
            "pivot": pivot,
            "stop": handle_low,
            "target": pivot + (rim - cup_low),
            "vol_quiet": vol_quiet,
            "dist_to_pivot": dist_to_pivot,
            "cup_bars": cup_bars,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


def cup_handle(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    """Cup & Handle scanner with separate daily and intraday logic.

    Key V9.5 fix: intraday mode is now based on the selected interval, not the number
    of bars downloaded. Before this, 1h charts with many bars could accidentally use
    daily-style cup rules and miss intraday cups like ENPH. Daily scans are also
    protected from DOC-style false positives where a flat base plus one vertical gap
    was being mislabeled as a cup.
    """
    if len(df) < 60:
        return None

    m = base_metrics(df, spy_df)
    mode = current_mode()
    loose = mode == "Loose / candidate"
    strict = mode == "Strict / confirmed"
    intraday_style = is_intraday()

    available = len(df)
    if intraday_style:
        # Search multiple recent windows because an intraday cup may form over a few sessions
        # or over a few weeks of hourly bars. The handle must still be at the right edge.
        candidate_windows = [55, 75, 100, 130, 170, 220]
        windows = [min(x, available) for x in candidate_windows if available >= max(45, int(x * 0.75))]
        if not windows:
            windows = [min(available, 70)]
    else:
        candidate_windows = [110, 150, 190, 240]
        windows = [min(x, available) for x in candidate_windows if available >= max(90, int(x * 0.85))]
        if not windows:
            windows = [min(available, 150)]

    best = None
    for w_len in sorted(set(windows)):
        w = df.tail(w_len).copy()
        c = _cup_handle_candidate(ticker, w, intraday_style, loose, strict, m)
        if c and (best is None or c["score"] > best["score"]):
            best = c

    if not best:
        return None

    reasons = [
        f"{'intraday ' if intraday_style else ''}cup depth {best['depth']:.0f}%",
        f"cup duration {int(best['cup_bars'])} bars",
        f"rim mismatch {best['rim_mismatch']:.1f}%",
        f"handle depth {best['handle_depth']:.1f}%",
        "handle in upper half of cup",
        "near handle pivot",
    ]
    if best.get("near_bottom_bars", 0) >= (4 if intraday_style else 7):
        reasons.append("rounded bottom")
    if best.get("vol_quiet"):
        reasons.append("handle volume quiet")

    return hit(ticker, "Cup & Handle", best["score"], "Bullish", reasons, best["pivot"], best["stop"], best["target"])

def ascending_triangle(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if is_intraday():
        return intraday_ascending_triangle(ticker, df, meta, spy_df)
    """Ascending triangle: horizontal resistance plus clearly rising swing lows."""
    if len(df) < 120:
        return None

    m = base_metrics(df, spy_df)
    mode = current_mode()
    loose = mode == "Loose / candidate"
    strict = mode == "Strict / confirmed"

    # Use a medium window. Too short finds bull flags; too long finds broad rectangles.
    lookback = 70 if not strict else 85
    w = df.tail(lookback).copy()
    if len(w) < 55:
        return None

    highs = w["High"].astype(float)
    lows = w["Low"].astype(float)
    closes = w["Close"].astype(float)
    last = float(closes.iloc[-1])

    # Basic trend context: ascending triangles are continuation/accumulation patterns,
    # not damaged downtrends.
    if not (last > m["sma200"] and (last > m["sma50"] * 0.97 or slope(df["SMA50"], 25) >= 0)):
        return None

    # Resistance is the cluster of repeated highs, not a single spike.
    high_ex_last = highs.iloc[:-2]
    resistance = float(high_ex_last.quantile(0.95))
    if not np.isfinite(resistance) or resistance <= 0:
        return None

    # Resistance must be flat and touched several times, with touches separated in time.
    touch_band = 0.020 if loose else (0.014 if not strict else 0.010)
    touch_idx = [i for i, h in enumerate(highs.iloc[:-2]) if abs(float(h) - resistance) / resistance <= touch_band]

    # Compress adjacent bars into separate touches so one multi-day spike does not count as 3 touches.
    separated = []
    min_sep = 5 if loose else (7 if not strict else 9)
    for i in touch_idx:
        if not separated or i - separated[-1] >= min_sep:
            separated.append(i)
        else:
            # keep the bar with the closer high inside the same touch cluster
            if abs(float(highs.iloc[i]) - resistance) < abs(float(highs.iloc[separated[-1]]) - resistance):
                separated[-1] = i

    min_touches = 2 if loose else 3
    if len(separated) < min_touches:
        return None
    if separated[-1] - separated[0] < (18 if loose else 25):
        return None

    # Reject if the resistance area is actually a rising channel/new-high sequence.
    touch_prices = [float(highs.iloc[i]) for i in separated]
    if max(touch_prices) / max(min(touch_prices), 1e-9) - 1 > (0.045 if loose else 0.030 if not strict else 0.022):
        return None
    if float(highs.max()) > resistance * (1.035 if loose else 1.020 if not strict else 1.014):
        return None

    # Find real swing lows. Need rising lows; equal/flat lows are rectangle, not ascending triangle.
    swing_lows = []
    for i in range(3, len(w) - 3):
        lo = float(lows.iloc[i])
        if lo <= float(lows.iloc[i-3:i].min()) and lo <= float(lows.iloc[i+1:i+4].min()):
            swing_lows.append((i, lo))

    # Keep lows that occur after/around the first resistance touch. Earlier lows often belong to the prior trend.
    swing_lows = [(i, lo) for i, lo in swing_lows if i >= max(0, separated[0] - 6) and lo < resistance * 0.98]
    if len(swing_lows) < (2 if loose else 3):
        return None

    # Use the last 3 swing lows. They should be rising and reasonably spaced.
    lows_used = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows[-2:]
    if len(lows_used) >= 3:
        l1, l2, l3 = [x[1] for x in lows_used]
        rising_lows = (l2 >= l1 * (1.015 if loose else 1.025)) and (l3 >= l2 * (1.005 if loose else 1.015))
    else:
        l1, l2 = [x[1] for x in lows_used]
        rising_lows = l2 >= l1 * (1.035 if loose else 1.055)
    if strict and len(lows_used) < 3:
        return None
    if not rising_lows:
        return None

    # Base must be a triangle-sized consolidation, not a huge broad range and not a tiny micro pause.
    base_low = min(lo for _, lo in lows_used)
    base_depth = (resistance - base_low) / max(resistance, 1e-9) * 100
    if base_depth < (5 if loose else 7) or base_depth > (28 if loose else 22 if not strict else 18):
        return None

    # Price should be near but not already far above the breakout area.
    max_below = 0.085 if loose else (0.065 if not strict else 0.045)
    max_above = 0.012 if loose else (0.006 if not strict else 0.002)
    if not (resistance * (1 - max_below) <= last <= resistance * (1 + max_above)):
        return None

    # Require compression: recent range should be tighter than early/middle range.
    early_range = (float(highs.iloc[:25].max()) - float(lows.iloc[:25].min())) / max(resistance, 1e-9)
    recent_range = (float(highs.iloc[-18:].max()) - float(lows.iloc[-18:].min())) / max(resistance, 1e-9)
    if recent_range > early_range * (0.82 if loose else 0.70 if not strict else 0.62):
        return None

    # Reject one-bar breakout/flag structures: if the last resistance touch is too recent and there is no long base,
    # it is usually a bull flag or high-tight base, not an ascending triangle.
    if separated[-1] > len(w) - 7 and separated[-1] - separated[0] < 32:
        return None

    vol_comp = float(df["Volume"].tail(10).mean()) < float(df["Volume"].tail(50).mean()) * (0.95 if loose else 0.85 if not strict else 0.75)
    if strict and not vol_comp:
        return None

    score = 62
    reasons = [
        "flat resistance with separated touches",
        "rising swing lows",
        "price near breakout area",
        f"base depth {base_depth:.0f}%",
    ]
    if len(separated) >= 3:
        score += 8; reasons.append("3+ resistance touches")
    if len(lows_used) >= 3:
        score += 8; reasons.append("3 rising lows")
    if vol_comp:
        score += 10; reasons.append("volume compression")
    if m["last"] > m["sma50"]:
        score += 6; reasons.append("above 50-day MA")
    if m.get("rs_vs_spy", 0) > 0:
        score += 8; reasons.append("RS positive")

    stop = min(lo for _, lo in lows_used[-2:])
    target = resistance + (resistance - base_low)
    return hit(ticker, "Ascending Triangle", score, "Bullish", reasons, resistance, stop, target)

def pocket_pivot(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 70:
        return None
    m = base_metrics(df, spy_df)
    last = df.iloc[-1]
    prior = df.iloc[-11:-1]
    down_days = prior[prior["Close"] < prior["Open"]]
    down_vol_max = float(down_days["Volume"].max()) if len(down_days) else float(prior["Volume"].max())
    up_day = float(last["Close"]) > float(last["Open"])
    vol_ok = float(last["Volume"]) > down_vol_max and float(last["Volume"]) > m["vol50"] * 1.1
    support_ok = float(last["Close"]) > last.get("SMA10", np.nan) or float(last["Close"]) > last.get("EMA21", np.nan)
    trend_ok = m["last"] > m["sma50"] or (m["last"] > m["sma200"] and slope(df["SMA50"], 20) > 0)
    if not (up_day and vol_ok and support_ok and trend_ok):
        return None
    score = 65
    reasons = ["up day", "volume exceeds highest recent down-volume", "above 10/21-day support"]
    if m["last"] > m["sma50"]:
        score += 15; reasons.append("above 50-day MA")
    if m.get("rs_vs_spy", 0) > 0:
        score += 10; reasons.append("RS positive")
    if loc_in_range(last["Close"], last["Low"], last["High"]) >= 65:
        score += 10; reasons.append("strong close")
    return hit(ticker, "Pocket Pivot", score, "Bullish", reasons, m["last"], float(df["Low"].tail(10).min()), m["last"] + 2 * m["atr"])


def minervini_template_bool(df: pd.DataFrame) -> bool:
    if len(df) < 220: return False
    m = base_metrics(df)
    return bool(
        m["last"] > m["sma50"] > m["sma150"] > m["sma200"] and
        slope(df["SMA200"], 30) > 0 and
        m["last"] >= 1.30 * m["low252"] and
        m["last"] >= 0.75 * m["high252"]
    )



def minervini_template(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    """Minervini Trend Template is a trend/leadership filter, not a chart pattern by itself."""
    if len(df) < 220:
        return None
    m = base_metrics(df, spy_df)
    mode = current_mode(); loose = mode == "Loose / candidate"; strict = mode == "Strict / confirmed"
    checks = [
        (m["last"] > m["sma50"], "price above 50-day"),
        (m["last"] > m["sma150"], "price above 150-day"),
        (m["last"] > m["sma200"], "price above 200-day"),
        (m["sma50"] > m["sma150"] > m["sma200"], "MAs stacked bullish"),
        (slope(df["SMA200"], 30) > 0, "200-day MA rising"),
        (m["last"] >= 1.30 * m["low252"], ">30% above 52-week low"),
        (m["last"] >= (0.78 if loose else 0.82) * m["high252"], "near 52-week high"),
        (m.get("rs_vs_spy", 0) > (0 if not strict else 5), "outperforming SPY"),
    ]
    passed = [reason for ok, reason in checks if ok]
    # Balanced/Strict should be a pass/fail leadership filter; Loose can show near-misses.
    min_pass = 7 if loose else 8
    if len(passed) < min_pass:
        return None
    # Avoid very extended entries; this is a filter but still should be tradable.
    extended = (m["last"] - m["sma50"]) / max(m["sma50"], 1e-9) * 100
    if extended > (35 if loose else 25):
        return None
    score = 65 + (len(passed) - min_pass) * 10
    if m["last"] >= 0.90 * m["high252"]:
        score += 10; passed.append("within 10% of 52-week high")
    if extended <= 12:
        score += 10; passed.append("not extended from 50-day")
    return hit(ticker, "Minervini Trend Template", score, "Bullish Trend Filter", passed, m["last"], m["sma50"], m["high252"] * 1.10)

def canslim(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 220: return None
    info = meta or {}
    m = base_metrics(df, spy_df)
    score=0; reasons=[]
    if info.get("earningsGrowth") is not None and info.get("earningsGrowth") > 0.15:
        score += 20; reasons.append("earnings growth positive")
    if info.get("revenueGrowth") is not None and info.get("revenueGrowth") > 0.10:
        score += 15; reasons.append("revenue growth positive")
    if m["last"] >= 0.85 * m["high252"]:
        score += 20; reasons.append("near 52-week high")
    if m["rvol"] > 1.2:
        score += 10; reasons.append("volume demand above average")
    if m.get("rs_vs_spy", 0) > 0:
        score += 15; reasons.append("relative strength positive")
    if minervini_template_bool(df):
        score += 20; reasons.append("stage-2 trend structure")
    return hit(ticker, "CANSLIM Growth", score, "Bullish", reasons, m["last"], m["sma50"], m["high252"] * 1.2)


def pivotal_point(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 130:
        return None
    m = base_metrics(df, spy_df)
    pivot = float(df["High"].iloc[-51:-1].max())
    base_hi = float(df["High"].iloc[-35:-1].max())
    base_lo = float(df["Low"].iloc[-35:-1].min())
    base_depth = (base_hi - base_lo) / max(m["last"], 1e-9) * 100
    tight = base_depth < max(14, m.get("adr20", 3.0) * 4)
    near_or_breakout = pivot * 0.97 <= m["last"] <= pivot * 1.04
    leader = m["last"] > m["sma50"] > m["sma200"] and slope(df["SMA50"], 20) >= 0
    if not (tight and near_or_breakout and leader):
        return None
    score = 60
    reasons = [f"tight consolidation base {base_depth:.1f}%", "near/through pivot", "market-leader structure"]
    if m["last"] > pivot:
        score += 15; reasons.append("breakout over pivot")
    if m["rvol"] > 1.5:
        score += 15; reasons.append(f"volume {m['rvol']:.1f}x normal")
    if m.get("rs_vs_spy", 0) > 0:
        score += 10; reasons.append("RS positive")
    return hit(ticker, "Pivotal Point", score, "Bullish", reasons, pivot, base_lo, pivot + (base_hi - base_lo))


def stage2(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 220:
        return None
    m = base_metrics(df, spy_df)
    structure = m["last"] > m["sma50"] > m["sma200"] and slope(df["SMA200"], 40) >= -0.0002
    s30w = slope(sma(df["Close"], 150), 30) > 0
    if not (structure and s30w):
        return None
    score = 65
    reasons = ["price above 50/200-day MAs", "30-week trend improving"]
    if slope(df["SMA200"], 40) > 0:
        score += 10; reasons.append("200-day MA rising")
    if m["rvol"] > 1.2:
        score += 10; reasons.append("volume expansion")
    if m.get("rs_vs_spy", 0) > 0:
        score += 15; reasons.append("RS improving")
    return hit(ticker, "Weinstein Stage 2", score, "Bullish", reasons, m["last"], m["sma200"], m["high252"] * 1.15)


def stage1(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 220:
        return None
    m = base_metrics(df, spy_df)
    base_range = (m["high50"] - m["low50"]) / max(m["last"], 1e-9) * 100
    flattened = abs(slope(df["SMA200"], 50)) < 0.0008
    after_decline = m["last"] < 0.78 * m["high252"]
    not_stage2 = not (m["last"] > m["sma50"] > m["sma200"] and slope(df["SMA200"], 40) > 0)
    if not (flattened and after_decline and base_range < 25 and not_stage2):
        return None
    score = 65
    reasons = ["200-day MA flattening", "prior decline/base context", "tight base forming"]
    if float(df["Volume"].tail(20).mean()) < float(df["Volume"].tail(100).mean()):
        score += 20; reasons.append("volume contraction")
    if m["last"] > m["sma50"]:
        score += 10; reasons.append("reclaiming 50-day")
    return hit(ticker, "Weinstein Stage 1", score, "Bullish Watch", reasons, m["high50"], m["low50"], m["high252"])


def stage3(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 220:
        return None
    m = base_metrics(df, spy_df)
    flat = abs(slope(df["SMA50"], 30)) < 0.001 and abs(slope(df["SMA200"], 40)) < 0.0015
    choppy = (m["high50"] - m["low50"]) / max(m["last"], 1e-9) * 100 > 12
    failed_high = m["last"] < m["high252"] * 0.90 and float(df["High"].tail(70).max()) >= m["high252"] * 0.95
    lost_support = m["last"] < m["sma50"]
    not_strong_leader = not (m["last"] > m["sma50"] > m["sma200"] and m.get("rs_vs_spy", 0) > 0)
    if not (flat and choppy and (failed_high or lost_support) and not_strong_leader):
        return None
    score = 60
    reasons = ["major averages flattening", "choppy distribution range"]
    if failed_high:
        score += 15; reasons.append("failed near highs")
    if lost_support:
        score += 15; reasons.append("lost 50-day MA")
    if m.get("rs_vs_spy", 0) < 0:
        score += 10; reasons.append("lagging SPY")
    return hit(ticker, "Weinstein Stage 3 Distribution", score, "Bearish Watch", reasons, m["last"], m["high50"], m["low50"])


def volume_surge(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 60: return None
    m=base_metrics(df, spy_df)
    last=df.iloc[-1]
    pos=loc_in_range(last["Close"], last["Low"], last["High"])
    breaks_high=m["last"]>=float(df["High"].iloc[-21:-1].max())
    breaks_low=m["last"]<=float(df["Low"].iloc[-21:-1].min())
    score=0; reasons=[]; direction="Neutral"
    if m["rvol"]>=3: score+=40; reasons.append(f"volume surge {m['rvol']:.1f}x")
    elif m["rvol"]>=2: score+=25; reasons.append(f"relative volume {m['rvol']:.1f}x")
    if pos>=80 and breaks_high: score+=35; direction="Bullish"; reasons.append("closes top of range and breaks 20-day high")
    elif pos<=20 and breaks_low: score+=35; direction="Bearish"; reasons.append("closes bottom of range and breaks 20-day low")
    if abs(m["pct_chg"])>=3: score+=15; reasons.append(f"price move {m['pct_chg']:.1f}%")
    if m["last"] < m["sma50"]*1.2 if not np.isnan(m["sma50"]) else True: score+=10; reasons.append("not extremely extended")
    entry = m["last"]
    stop = float(last["Low"] if direction != "Bearish" else last["High"])
    target = entry + np.sign(m["pct_chg"] if not np.isnan(m["pct_chg"]) else 1) * (last["High"]-last["Low"])
    return hit(ticker,"Volume Surge",score,direction,reasons,entry,stop,target)


def rvol_scan(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df)<60: return None
    m=base_metrics(df, spy_df)
    score=0; reasons=[]
    if m["rvol"]>=2: score+=50; reasons.append(f"RVOL {m['rvol']:.1f}x")
    if abs(m["pct_chg"])>=2: score+=25; reasons.append(f"move {m['pct_chg']:.1f}%")
    if m["adr20"]>=4: score+=25; reasons.append(f"ADR {m['adr20']:.1f}%")
    direction="Bullish" if m["pct_chg"]>=0 else "Bearish"
    return hit(ticker,"Relative Volume",score,direction,reasons,m["last"],m["last"]-m["atr"],m["last"]+2*m["atr"])


def gap_scan(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df)<60: return None
    today=df.iloc[-1]; prev=df.iloc[-2]; m=base_metrics(df,spy_df)
    gap=pct(float(today["Open"]), float(prev["Close"]))
    score=0; reasons=[]
    if abs(gap)>=5: score+=50; reasons.append(f"gap {gap:.1f}%")
    elif abs(gap)>=3: score+=35; reasons.append(f"gap {gap:.1f}%")
    if m["rvol"]>=1.5: score+=25; reasons.append(f"volume {m['rvol']:.1f}x")
    if abs(m["pct_chg"])>=5: score+=25; reasons.append(f"strong move {m['pct_chg']:.1f}%")
    direction="Bullish" if gap>0 else "Bearish"
    return hit(ticker,"5% Gap / Strong Move",score,direction,reasons,m["last"],float(today["Low"]),m["last"]+np.sign(gap)*2*m["atr"])


def buyable_gap_up(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 80:
        return None
    today = df.iloc[-1]; prev = df.iloc[-2]; m = base_metrics(df, spy_df)
    gap = pct(float(today["Open"]), float(prev["Close"]))
    holds_gap = float(today["Low"]) > float(prev["High"]) * 0.98 and m["last"] > float(prev["High"])
    strong_close = loc_in_range(today["Close"], today["Low"], today["High"]) >= 55
    if not (gap >= 3 and holds_gap and m["rvol"] >= 1.3 and strong_close):
        return None
    score = 65
    reasons = [f"gap up {gap:.1f}%", "holding above prior high/gap zone", f"volume {m['rvol']:.1f}x"]
    if m["last"] > m["sma50"]:
        score += 15; reasons.append("above 50-day MA")
    if gap >= 8:
        score += 10; reasons.append("powerful gap")
    if m.get("rs_vs_spy", 0) > 0:
        score += 10; reasons.append("RS positive")
    return hit(ticker, "Buyable Gap Up", score, "Bullish", reasons, m["last"], float(today["Low"]), m["last"] + 2 * m["atr"])


def recent_doublers(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df)<70: return None
    m=base_metrics(df, spy_df)
    low60=float(df["Close"].tail(60).min())
    move=pct(m["last"], low60)
    score=0; reasons=[]
    if move>=100: score+=55; reasons.append(f"doubled in 60 sessions: {move:.0f}%")
    elif move>=75: score+=40; reasons.append(f"strong 60-session move: {move:.0f}%")
    if m["last"]>m["sma10"]>m["sma20"] if "SMA10" in df else False: score+=20; reasons.append("short-term trend intact")
    if m["rvol"]>=1: score+=10; reasons.append("liquidity active")
    if m["adr20"]>=4: score+=15; reasons.append("high ADR momentum")
    return hit(ticker,"Recent Doublers",score,"Bullish Momentum",reasons,m["last"],m["sma20"],m["last"]*1.25)


def golden_pocket(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 120:
        return None
    m = base_metrics(df, spy_df)
    w = df.tail(90)
    high_pos = _position_of_max(w["High"])
    low_pos = _position_of_min(w["Low"])
    # For bullish golden pocket, the swing high should come after the swing low.
    if not (0 <= low_pos < high_pos):
        return None
    swing_low = float(w["Low"].iloc[low_pos])
    swing_high = float(w["High"].iloc[high_pos])
    retr = (swing_high - m["last"]) / max(swing_high - swing_low, 1e-9)
    trend_ok = m["last"] > m["sma200"] and slope(df["SMA50"], 20) >= -0.001
    if not (0.618 <= retr <= 0.786 and trend_ok):
        return None
    score = 60
    reasons = ["in .618-.786 retracement zone", "above 200-day support"]
    if m["rsi"] < 55:
        score += 10; reasons.append("pullback reset")
    if m.get("rs_vs_spy", 0) > 0:
        score += 15; reasons.append("relative strength positive")
    if abs(m["last"] - m["ema21"]) / max(m["last"], 1e-9) * 100 < 5:
        score += 10; reasons.append("near 21 EMA")
    return hit(ticker, "Golden Pocket", score, "Bullish Pullback", reasons, m["last"], swing_low, swing_high)



def rsi_oversold_reversion(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    """Oversold pullback in a still-viable longer-term uptrend."""
    if len(df) < 220:
        return None
    m = base_metrics(df, spy_df)
    mode = current_mode(); loose = mode == "Loose / candidate"; strict = mode == "Strict / confirmed"
    rsi_limit = 42 if loose else (36 if not strict else 32)
    if m["rsi"] > rsi_limit:
        return None
    if m["last"] < m["sma200"] * (0.97 if loose else 1.0):
        return None
    # Need evidence of stabilization, not just a falling knife.
    last = df.iloc[-1]; prev = df.iloc[-2]
    stabilizing = float(last["Close"]) >= float(prev["Close"]) or loc_in_range(last["Close"], last["Low"], last["High"]) >= 55
    near_support = abs(m["last"] - m["sma50"]) / max(m["last"], 1e-9) * 100 < (10 if loose else 7) or abs(m["last"] - m["sma200"]) / max(m["last"], 1e-9) * 100 < (8 if loose else 5)
    if not (stabilizing and near_support):
        return None
    score = 60
    reasons = [f"RSI pullback {m['rsi']:.0f}", "above/near 200-day trend", "stabilizing near support"]
    if m["last"] < m["sma50"]: score += 10; reasons.append("below 50-day pullback")
    if m.get("rs_vs_spy", 0) > -3: score += 10; reasons.append("not materially lagging SPY")
    if m["rvol"] >= 1.0: score += 5; reasons.append("volume active")
    return hit(ticker, "RSI Oversold Reversion", score, "Bullish Mean Reversion", reasons, m["last"], m["last"] - 1.5 * m["atr"], m["sma50"])

def bottom_finder(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df)<160: return None
    m=base_metrics(df,spy_df)
    drawdown=pct(m["last"],m["high252"])
    score=0; reasons=[]
    if drawdown<=-35: score+=30; reasons.append(f"deep drawdown {drawdown:.0f}%")
    if m["rsi"]<35: score+=20; reasons.append("oversold")
    if lastv(df,"MACDHist")>lastv(df.iloc[:-1],"MACDHist") if len(df)>2 else False: score+=15; reasons.append("momentum improving")
    if m["last"]>m["low50"]*1.08: score+=20; reasons.append("lifting off lows")
    if m["rvol"]>1.2: score+=15; reasons.append("volume interest")
    return hit(ticker,"Bottom Finder",score,"Bullish Reversal",reasons,m["last"],m["low50"],m["sma50"])



def falling_wedge(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if is_intraday():
        return intraday_falling_wedge(ticker, df, meta, spy_df)
    """Falling wedge candidate: lower highs and lower lows with range contraction near upper trendline."""
    if len(df) < 120:
        return None
    m = base_metrics(df, spy_df)
    mode = current_mode(); loose = mode == "Loose / candidate"; strict = mode == "Strict / confirmed"
    w = df.tail(65)
    hi = w["High"].rolling(5).max().dropna()
    lo = w["Low"].rolling(5).min().dropna()
    hi_s = _slope_pct_per_bar(hi.tail(45))
    lo_s = _slope_pct_per_bar(lo.tail(45))
    # Both trendlines fall, but resistance falls faster than support: converging wedge.
    converging = hi_s < (-0.04 if loose else -0.06) and lo_s < 0 and hi_s < lo_s * (1.15 if loose else 1.35)
    range_early = (float(w["High"].head(25).max()) - float(w["Low"].head(25).min())) / max(m["last"], 1e-9) * 100
    range_late = (float(w["High"].tail(20).max()) - float(w["Low"].tail(20).min())) / max(m["last"], 1e-9) * 100
    range_contract = range_late < range_early * (0.80 if loose else 0.70)
    vol_down = float(w["Volume"].tail(12).mean()) < float(w["Volume"].head(25).mean()) * (0.90 if loose else 0.80)
    upper = float(w["High"].tail(15).max())
    near_break = m["last"] > upper * (0.965 if loose else 0.985)
    if not (converging and range_contract and vol_down and near_break):
        return None
    score = 62
    reasons = ["falling converging trendlines", "range contraction", "near upper wedge breakout"]
    if m["rsi"] > 38: score += 8; reasons.append("momentum stabilizing")
    if m["last"] > m["ema21"]: score += 10; reasons.append("reclaiming 21 EMA")
    if m.get("rs_vs_spy", 0) > -3: score += 5; reasons.append("RS not weak")
    if strict and m["last"] < m["ema21"]:
        return None
    return hit(ticker, "Falling Wedge", score, "Bullish Reversal", reasons, upper, float(w["Low"].min()), upper + 2 * m["atr"])

def inverse_hs(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 130:
        return None
    m = base_metrics(df, spy_df); w = df.tail(90); thirds = np.array_split(w, 3)
    l1 = float(thirds[0]["Low"].min()); head = float(thirds[1]["Low"].min()); l2 = float(thirds[2]["Low"].min())
    neckline = max(float(thirds[0]["High"].max()), float(thirds[2]["High"].max()))
    head_lower = head < l1 * 0.95 and head < l2 * 0.95
    shoulders_similar = abs(l1 - l2) / max((l1 + l2) / 2, 1e-9) * 100 < 12
    near_neckline = neckline * 0.94 <= m["last"] <= neckline * 1.03
    if not (head_lower and shoulders_similar and near_neckline):
        return None
    score = 65
    reasons = ["head lower than shoulders", "shoulders similar", "near neckline"]
    if m["rvol"] > 1:
        score += 10; reasons.append("volume confirmation improving")
    if m["last"] > m["sma50"]:
        score += 10; reasons.append("reclaiming 50-day")
    if m["rsi"] > 45:
        score += 10; reasons.append("momentum improving")
    return hit(ticker, "Inverse Head & Shoulders", score, "Bullish Reversal", reasons, neckline, min(l1, l2), neckline + (neckline - head))


def head_shoulders_top(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 130:
        return None
    m = base_metrics(df, spy_df); w = df.tail(90); thirds = np.array_split(w, 3)
    s1 = float(thirds[0]["High"].max()); head = float(thirds[1]["High"].max()); s2 = float(thirds[2]["High"].max())
    neckline = min(float(thirds[0]["Low"].min()), float(thirds[2]["Low"].min()))
    head_higher = head > s1 * 1.05 and head > s2 * 1.05
    shoulders_similar = abs(s1 - s2) / max((s1 + s2) / 2, 1e-9) * 100 < 12
    near_break = m["last"] < neckline * 1.05
    weak = m["last"] < m["sma50"] or slope(df["SMA50"], 20) < 0
    if not (head_higher and shoulders_similar and near_break and weak):
        return None
    score = 65
    reasons = ["head above shoulders", "shoulders similar", "near neckline breakdown"]
    if m["last"] < m["sma50"]:
        score += 15; reasons.append("lost 50-day MA")
    if m.get("rs_vs_spy", 0) < 0:
        score += 10; reasons.append("lagging SPY")
    return hit(ticker, "Head & Shoulders Top", score, "Bearish Reversal", reasons, neckline, max(s1, s2), neckline - (head - neckline))


def pullback_21ema(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 120:
        return None
    m = base_metrics(df, spy_df)
    near = abs(m["last"] - m["ema21"]) / max(m["last"], 1e-9) * 100 < 3.0
    up = m["last"] > m["sma50"] > m["sma200"] and slope(df["EMA21"], 20) > 0
    shallow = (m["high20"] - m["last"]) / max(m["high20"], 1e-9) * 100 <= max(12, m.get("adr20", 3.0) * 3)
    if not (near and up and shallow):
        return None
    score = 65
    reasons = ["uptrend intact", "pulling into 21 EMA", "shallow pullback"]
    if float(df["Volume"].tail(5).mean()) < float(df["Volume"].tail(30).mean()):
        score += 15; reasons.append("declining pullback volume")
    if m["rsi"] > 40:
        score += 10; reasons.append("momentum not broken")
    if m.get("rs_vs_spy", 0) > 0:
        score += 10; reasons.append("RS positive")
    return hit(ticker, "Pullback to 21 EMA", score, "Bullish Pullback", reasons, m["last"], m["ema21"] - m["atr"], m["high20"])


def bounce_200d(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 220:
        return None
    m = base_metrics(df, spy_df); today = df.iloc[-1]
    touched = float(today["Low"]) <= m["sma200"] * 1.02 and float(today["Close"]) > m["sma200"]
    reversal = float(today["Close"]) > float(today["Open"]) or loc_in_range(today["Close"], today["Low"], today["High"]) >= 60
    if not (touched and reversal):
        return None
    score = 60
    reasons = ["touched/held 200-day MA", "reversal/strong close"]
    if m["rvol"] > 1:
        score += 15; reasons.append("volume above average")
    if m.get("rs_vs_spy", 0) > -3:
        score += 15; reasons.append("relative strength acceptable")
    if m["last"] > m["sma50"]:
        score += 10; reasons.append("also above 50-day")
    return hit(ticker, "200-Day Bounce", score, "Bullish Mean Reversion", reasons, m["last"], m["sma200"] - m["atr"], m["sma50"])


def oversold_bounce(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 90:
        return None
    m = base_metrics(df, spy_df)
    oversold = m["rsi"] < 35 and m["stochrsi"] < 30
    bounce = df["Close"].iloc[-1] > df["Close"].iloc[-2] and loc_in_range(df["Close"].iloc[-1], df["Low"].iloc[-1], df["High"].iloc[-1]) >= 55
    if not (oversold and bounce):
        return None
    score = 60
    reasons = ["RSI + StochRSI oversold", "first strong bounce day"]
    if m["last"] > m["sma200"] if not np.isnan(m["sma200"]) else True:
        score += 15; reasons.append("above/near long trend")
    if m["rvol"] > 1:
        score += 15; reasons.append("volume interest")
    if m.get("rs_vs_spy", 0) > -5:
        score += 10; reasons.append("not materially lagging SPY")
    return hit(ticker, "Oversold Bounce", score, "Bullish Mean Reversion", reasons, m["last"], m["low20"], m.get("sma20", m["sma50"]))


def macd_cross(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 80:
        return None
    m = base_metrics(df, spy_df)
    prev_line = float(df["MACD"].iloc[-2]); prev_sig = float(df["MACDSignal"].iloc[-2])
    now_line = lastv(df, "MACD"); now_sig = lastv(df, "MACDSignal")
    crossed = prev_line <= prev_sig and now_line > now_sig
    trend_ok = m["last"] > m["sma50"] or m["last"] > m["sma200"]
    if not (crossed and trend_ok):
        return None
    score = 60
    reasons = ["MACD crossed bullish", "price above key trend support"]
    if m["last"] > m["sma50"]:
        score += 15; reasons.append("above 50-day MA")
    if m["last"] > m["sma200"]:
        score += 15; reasons.append("above 200-day MA")
    if m["rvol"] > 1:
        score += 10; reasons.append("volume active")
    return hit(ticker, "MACD Bullish Cross", score, "Bullish Momentum", reasons, m["last"], m["sma50"], m["high20"])


def golden_cross(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 220:
        return None
    m = base_metrics(df, spy_df)
    # Require an actual recent cross, not just an old golden-cross state.
    lookback = df.tail(12)
    prev_cross_state = (lookback["SMA50"].shift(1) <= lookback["SMA200"].shift(1)) & (lookback["SMA50"] > lookback["SMA200"])
    crossed_recently = bool(prev_cross_state.any())
    if not (crossed_recently and m["last"] > m["sma50"]):
        return None
    score = 60
    reasons = ["recent 50-day cross above 200-day", "price above 50-day"]
    if m["rvol"] > 1:
        score += 10; reasons.append("volume active")
    if m.get("rs_vs_spy", 0) > 0:
        score += 15; reasons.append("RS positive")
    if slope(df["SMA200"], 40) > 0:
        score += 15; reasons.append("200-day rising")
    return hit(ticker, "Golden Cross", score, "Bullish Trend", reasons, m["last"], m["sma200"], m["high252"])


def bollinger_squeeze(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 120:
        return None
    m = base_metrics(df, spy_df)
    bb_width = (lastv(df, "BBUpper") - lastv(df, "BBLower")) / max(m["last"], 1e-9) * 100
    hist_width = ((df["BBUpper"] - df["BBLower"]) / df["Close"] * 100).dropna()
    if len(hist_width) < 40:
        return None
    kc_inside = lastv(df, "BBUpper") < lastv(df, "KCUpper") and lastv(df, "BBLower") > lastv(df, "KCLower")
    low_width = bb_width < hist_width.tail(120).quantile(0.25)
    if not (kc_inside and low_width):
        return None
    score = 60
    reasons = ["BB inside Keltner Channel", f"bandwidth compressed {bb_width:.1f}%"]
    if m["last"] > m["sma50"]:
        score += 15; reasons.append("above 50-day")
    if m["rvol"] > 1:
        score += 10; reasons.append("volume starting to build")
    if m.get("rs_vs_spy", 0) > 0:
        score += 10; reasons.append("RS positive")
    return hit(ticker, "Bollinger Squeeze", score, "Volatility Setup", reasons, m["high20"], m["low20"], m["high20"] + 2 * m["atr"])


def inside_day_breakout(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 50:
        return None
    m = base_metrics(df, spy_df)
    inside_count = 0
    for i in range(-3, 0):
        if df["High"].iloc[i] < df["High"].iloc[i - 1] and df["Low"].iloc[i] > df["Low"].iloc[i - 1]:
            inside_count += 1
    range_compress = (m["high20"] - m["low20"]) / max(m["last"], 1e-9) * 100 < max(10, m.get("adr20", 3) * 3)
    trend_ok = m["last"] > m["sma50"] or slope(df["SMA50"], 20) > 0
    if not (inside_count >= 2 and range_compress and trend_ok):
        return None
    score = 60
    reasons = [f"{inside_count} inside/coiling days", "20-day range compressed"]
    if m["last"] > m["sma50"]:
        score += 15; reasons.append("trend bullish")
    if float(df["Volume"].tail(3).mean()) < float(df["Volume"].tail(20).mean()):
        score += 15; reasons.append("volume drying up")
    if m.get("rs_vs_spy", 0) > 0:
        score += 10; reasons.append("RS positive")
    return hit(ticker, "Inside Day Breakout", score, "Breakout Setup", reasons, m["high20"], m["low20"], m["high20"] + m["atr"] * 2)


def new_50day_high(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df)<80: return None
    m=base_metrics(df, spy_df)
    prev_high=float(df["High"].iloc[-51:-1].max())
    is_new=m["last"]>prev_high
    score=0; reasons=[]
    if is_new: score+=50; reasons.append("new 50-day high")
    if m["last"]<m["high252"]*0.98: score+=10; reasons.append("before 52-week high")
    if m["rvol"]>1.2: score+=20; reasons.append("volume confirms")
    if m["last"]>m["sma50"]>m["sma200"]: score+=20; reasons.append("bullish trend")
    return hit(ticker,"New 50-Day High",score,"Bullish Momentum",reasons,m["last"],prev_high-m["atr"],m["high252"])


def institutional_accumulation(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 90:
        return None
    m = base_metrics(df, spy_df)
    w = df.tail(20)
    upvol = float(w.loc[w["Close"] > w["Open"], "Volume"].sum())
    downvol = float(w.loc[w["Close"] <= w["Open"], "Volume"].sum())
    higher_lows = _slope_pct_per_bar(w["Low"].rolling(3).min().dropna()) > 0.02
    trend_ok = m["last"] > m["sma50"] or slope(df["SMA50"], 20) > 0
    if not (upvol > downvol * 1.3 and higher_lows and trend_ok):
        return None
    score = 60
    reasons = ["up-volume dominates down-volume", "higher lows", "constructive trend"]
    if m["last"] > m["sma50"]:
        score += 15; reasons.append("above 50-day")
    if m["rvol"] > 1.1:
        score += 15; reasons.append("volume expanding")
    if m.get("rs_vs_spy", 0) > 0:
        score += 10; reasons.append("RS positive")
    return hit(ticker, "Institutional Accumulation", score, "Bullish", reasons, m["last"], m["low20"], m["high50"])


def high_growth_momentum(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df)<220: return None
    m=base_metrics(df, spy_df)
    ret63=pct(m["last"],float(df["Close"].iloc[-63])) if len(df)>63 else np.nan
    ret126=pct(m["last"],float(df["Close"].iloc[-126])) if len(df)>126 else np.nan
    score=0; reasons=[]
    if ret63>15: score+=25; reasons.append(f"3M momentum {ret63:.0f}%")
    if ret126>25: score+=25; reasons.append(f"6M momentum {ret126:.0f}%")
    if m["last"]>m["sma50"]>m["sma200"]: score+=25; reasons.append("trend aligned")
    if m.get("rs_vs_spy",0)>0: score+=25; reasons.append("relative strength positive")
    return hit(ticker,"High Growth Momentum",score,"Bullish Momentum",reasons,m["last"],m["sma50"],m["last"]*1.2)


def buy_the_dip(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 160:
        return None
    m = base_metrics(df, spy_df)
    trend = m["last"] > m["sma200"] and m["ema65"] > m["sma200"] and slope(df["EMA65"], 30) > 0
    near65 = abs(m["last"] - m["ema65"]) / max(m["last"], 1e-9) * 100 < 4
    reset = m["stochrsi"] < 35 or m["rsi"] < 48
    shallow = (m["high50"] - m["last"]) / max(m["high50"], 1e-9) * 100 < 18
    if not (trend and near65 and reset and shallow):
        return None
    score = 65
    reasons = ["confirmed uptrend", "pullback near 65 EMA", "StochRSI/RSI reset"]
    if m.get("rs_vs_spy", 0) > -2:
        score += 10; reasons.append("relative strength stable")
    if float(df["Volume"].tail(5).mean()) < float(df["Volume"].tail(30).mean()):
        score += 15; reasons.append("pullback volume quiet")
    if m["last"] > m["ema21"]:
        score += 10; reasons.append("above 21 EMA")
    return hit(ticker, "Buy the Dip", score, "Bullish Pullback", reasons, m["last"], m["ema65"] - m["atr"], m["high20"])


def sell_signal(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 160:
        return None
    m = base_metrics(df, spy_df)
    extended = (m["last"] - m["ema65"]) / max(m["ema65"], 1e-9) * 100 > 18
    overbought = m["stochrsi"] > 85 or m["rsi"] > 75
    weak_close = loc_in_range(df["Close"].iloc[-1], df["Low"].iloc[-1], df["High"].iloc[-1]) < 45
    if not (extended and overbought and weak_close):
        return None
    score = 65
    reasons = ["extended above 65 EMA", "overbought momentum", "weak close in range"]
    if m["rvol"] > 1.2:
        score += 15; reasons.append("volume active")
    if m["pct_chg"] < 0:
        score += 10; reasons.append("red reversal day")
    if m.get("rs_vs_spy", 0) < 0:
        score += 10; reasons.append("relative strength weakening")
    return hit(ticker, "Sell Signal", score, "Bearish / Trim", reasons, m["last"], m["high20"], m["ema21"])


def parabolic_short(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 80:
        return None
    m = base_metrics(df, spy_df)
    ret10 = pct(m["last"], float(df["Close"].iloc[-10])) if len(df) > 10 else np.nan
    ret20 = pct(m["last"], float(df["Close"].iloc[-20])) if len(df) > 20 else np.nan
    parabolic = ret10 > 40 or ret20 > 80
    exhaustion = m["rsi"] > 80 and m["last"] > m["ema21"] * 1.20
    stall = loc_in_range(df["Close"].iloc[-1], df["Low"].iloc[-1], df["High"].iloc[-1]) < 55 or m["pct_chg"] < 0
    if not (parabolic and exhaustion and stall):
        return None
    score = 65
    reasons = [f"parabolic move 10d {ret10:.0f}% / 20d {ret20:.0f}%", "RSI extreme", "stall/weak close"]
    if m["last"] > m["ema21"] * 1.25:
        score += 15; reasons.append("far above 21 EMA")
    if m["rvol"] > 1.5:
        score += 10; reasons.append("high volume exhaustion")
    return hit(ticker, "Parabolic Short", score, "Bearish Reversal", reasons, m["last"], m["high20"], m["ema21"])


def qulla_breakout(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 100:
        return None
    m = base_metrics(df, spy_df)
    ret30 = pct(m["last"], float(df["Close"].iloc[-30])) if len(df) > 30 else np.nan
    tight = (m["high20"] - m["low20"]) / max(m["last"], 1e-9) * 100 < max(12, m["adr20"] * 3)
    low_vol = float(df["Volume"].tail(7).mean()) < float(df["Volume"].tail(30).mean())
    near_ema = abs(m["last"] - lastv(df, "EMA10")) / max(m["last"], 1e-9) * 100 < 5 or abs(m["last"] - m["ema21"]) / max(m["last"], 1e-9) * 100 < 5
    if not (ret30 > 30 and tight and low_vol and near_ema):
        return None
    score = 65
    reasons = [f"episodic move {ret30:.0f}%", "tight consolidation", "low-volume pause", "near 10/21 EMA"]
    if m["adr20"] > 4:
        score += 10; reasons.append("high ADR leader")
    if m.get("rs_vs_spy", 0) > 0:
        score += 10; reasons.append("RS positive")
    if m["last"] > m["sma50"]:
        score += 10; reasons.append("above 50-day")
    return hit(ticker, "Qullamaggie Breakout", score, "Bullish Momentum", reasons, m["high20"], m["low20"], m["high20"] + 3 * m["atr"])


def high_tight_flag(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 120:
        return None
    m = base_metrics(df, spy_df)
    low60 = float(df["Close"].tail(60).min())
    move = pct(m["last"], low60)
    pullback = (m["high20"] - m["last"]) / max(m["high20"], 1e-9) * 100
    tight = (m["high20"] - m["low20"]) / max(m["last"], 1e-9) * 100 < 15
    vol_dry = float(df["Volume"].tail(10).mean()) < float(df["Volume"].tail(30).mean())
    if not (move >= 80 and pullback < 20 and tight and vol_dry):
        return None
    score = 75
    reasons = [f"80%+ surge: {move:.0f}%", "shallow pullback", "tight flag", "volume dries up"]
    if m["last"] > m["sma50"]:
        score += 10; reasons.append("above 50-day")
    if m.get("rs_vs_spy", 0) > 0:
        score += 10; reasons.append("RS positive")
    return hit(ticker, "High Tight Flag", score, "Bullish Momentum", reasons, m["high20"], m["low20"], m["high20"] + 3 * m["atr"])


def power_play(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 140:
        return None
    m = base_metrics(df, spy_df)
    ret40 = pct(m["last"], float(df["Close"].iloc[-40])) if len(df) > 40 else np.nan
    tight = (m["high20"] - m["low20"]) / max(m["last"], 1e-9) * 100 < 12
    above = m["last"] > m["sma50"]
    vol_dry = float(df["Volume"].tail(10).mean()) < float(df["Volume"].tail(30).mean())
    if not (ret40 >= 80 and tight and above and vol_dry):
        return None
    score = 75
    reasons = [f"rapid 40-session advance {ret40:.0f}%", "tight sideways consolidation", "above 50-day MA", "volume dries up"]
    if m.get("rs_vs_spy", 0) > 0:
        score += 10; reasons.append("RS positive")
    if m["adr20"] > 4:
        score += 10; reasons.append("high ADR")
    return hit(ticker, "Power Play", score, "Bullish Momentum", reasons, m["high20"], m["low20"], m["last"] * 1.25)


def earnings_scan_factory(days_min: int, days_max: int, name: str) -> Callable:
    def _scan(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
        ed = get_next_earnings_date(ticker)
        if not ed: return None
        days=(ed.replace(tzinfo=None)-datetime.now()).days
        if not (days_min<=days<=days_max): return None
        m=base_metrics(df, spy_df)
        score=50; reasons=[f"earnings in {days} days ({ed.date()})"]
        if m["last"]>m["sma50"]: score+=20; reasons.append("above 50-day MA")
        if m["rvol"]>1: score+=10; reasons.append("active volume")
        if m.get("rs_vs_spy",0)>0: score+=20; reasons.append("RS positive")
        return hit(ticker,name,score,"Catalyst",reasons,m["last"],m["low20"],m["high20"])
    return _scan


def earnings_watch(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    ed = get_next_earnings_date(ticker)
    if not ed: return None
    days=(ed.replace(tzinfo=None)-datetime.now()).days
    if not (0<=days<=30): return None
    bullish_scores=[]
    for fn in [bull_flag,vcp,cup_handle,ascending_triangle,pivotal_point,pocket_pivot,minervini_template,stage2,buyable_gap_up,new_50day_high,macd_cross,bollinger_squeeze,pullback_21ema,qulla_breakout,high_tight_flag]:
        h=fn(ticker,df,meta,spy_df)
        if h and h.score>=60:
            bullish_scores.append(h)
    if not bullish_scores: return None
    best=max(bullish_scores,key=lambda x:x.score)
    score=min(100,55+len(bullish_scores)*8+best.score*0.25)
    reasons=[f"earnings in {days} days", f"{len(bullish_scores)} bullish setup(s)", f"best: {best.scanner} {best.grade}"]
    return hit(ticker,"Earnings Watch",score,"Catalyst + Bullish",reasons,best.entry,best.stop,best.target)


def fundamental_value_scan(ticker: str, df: pd.DataFrame, meta=None, spy_df=None, mode="PEG Value") -> Optional[ScanHit]:
    info=meta or {}
    if not info: return None
    m=base_metrics(df,spy_df)
    score=0; reasons=[]
    pe=info.get("trailingPE") or info.get("forwardPE")
    peg=info.get("pegRatio")
    roe=info.get("returnOnEquity")
    rg=info.get("revenueGrowth")
    pm=info.get("profitMargins")
    div=info.get("dividendYield")
    debt=info.get("debtToEquity")
    if mode=="PEG Value":
        if peg is not None and 0.2<=peg<=2.0: score+=35; reasons.append(f"PEG {peg:.2f}")
        if rg is not None and rg>0.05: score+=20; reasons.append("revenue growth")
        if pe is not None and pe<35: score+=15; reasons.append("valuation not excessive")
        if m["last"]>m["sma200"]: score+=15; reasons.append("price trend positive")
        if roe is not None and roe>0.10: score+=15; reasons.append("ROE positive")
    elif mode=="Lynch GARP":
        if rg is not None and rg>0.08: score+=25; reasons.append("growth profile")
        if peg is not None and 0.5<=peg<=2.0: score+=25; reasons.append(f"reasonable PEG {peg:.2f}")
        if roe is not None and roe>0.12: score+=20; reasons.append("strong ROE")
        if pm is not None and pm>0.08: score+=15; reasons.append("healthy margins")
        if m["last"]>m["sma50"]: score+=15; reasons.append("trend positive")
    elif mode=="Buffett Value":
        if roe is not None and roe>0.15: score+=25; reasons.append("high ROE")
        if pm is not None and pm>0.12: score+=20; reasons.append("strong margins")
        if debt is not None and debt<100: score+=20; reasons.append("debt reasonable")
        if pe is not None and pe<30: score+=15; reasons.append("fair valuation")
        if m["last"]>m["sma200"]: score+=20; reasons.append("long trend positive")
    elif mode=="Magic Formula":
        if roe is not None and roe>0.15: score+=30; reasons.append("quality / high return")
        if pe is not None and pe<25: score+=30; reasons.append("earnings yield attractive")
        if pm is not None and pm>0.10: score+=20; reasons.append("profitable")
        if debt is None or debt<150: score+=20; reasons.append("debt acceptable")
    elif mode=="Dividend Growth":
        if div is not None and div>0.005: score+=25; reasons.append(f"dividend yield {div*100:.1f}%")
        if info.get("payoutRatio") is not None and info.get("payoutRatio")<0.65: score+=25; reasons.append("payout ratio manageable")
        if rg is not None and rg>0: score+=20; reasons.append("business growth positive")
        if roe is not None and roe>0.10: score+=15; reasons.append("ROE positive")
        if m["last"]>m["sma200"]: score+=15; reasons.append("trend positive")
    return hit(ticker,mode,score,"Fundamental / Bullish",reasons,m["last"],m["sma200"],m["high252"])


def munger_200w(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    # Uses 200-day as practical approximation if weekly history unavailable.
    info=meta or {}; m=base_metrics(df,spy_df)
    near=abs(m["last"]-m["sma200"])/max(m["last"],1e-9)*100<8
    quality=(info.get("returnOnEquity") or 0)>0.12 or (info.get("profitMargins") or 0)>0.10
    score=0; reasons=[]
    if near: score+=45; reasons.append("near 200-day/weekly value zone")
    if quality: score+=30; reasons.append("quality fundamentals")
    if m["rsi"]<55: score+=10; reasons.append("pullback/reset")
    if m["last"]>m["sma200"]*0.95: score+=15; reasons.append("long trend not broken")
    return hit(ticker,"Munger 200W",score,"Value Pullback",reasons,m["last"],m["sma200"]-m["atr"],m["high50"])


def short_squeeze(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    info=meta or {}; m=base_metrics(df,spy_df)
    short_pct = info.get("shortPercentOfFloat") or info.get("sharesShortPriorMonth")
    score=0; reasons=[]
    if isinstance(short_pct,(int,float)) and short_pct<1:
        if short_pct>0.15: score+=35; reasons.append(f"short float {short_pct*100:.1f}%")
        elif short_pct>0.08: score+=20; reasons.append(f"short float {short_pct*100:.1f}%")
    if m["last"]>m["high20"]*0.99: score+=25; reasons.append("near 20-day breakout")
    if m["rvol"]>1.5: score+=25; reasons.append("volume pressure")
    if m["last"]>m["sma50"]: score+=15; reasons.append("above 50-day")
    return hit(ticker,"Short Squeeze",score,"Bullish Squeeze",reasons,m["last"],m["low20"],m["last"]+3*m["atr"])


def insider_buying(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    # Yahoo insider data is inconsistent; this is a best-effort placeholder.
    try:
        tx = yf.Ticker(ticker).insider_transactions
        if tx is None or tx.empty: return None
        recent = tx.head(20).copy()
        text = recent.to_string().lower()
        buy_like = text.count("buy") + text.count("purchase")
        if buy_like == 0: return None
        m=base_metrics(df,spy_df)
        score=min(100,55+buy_like*8)
        reasons=[f"recent insider purchase rows detected: {buy_like}"]
        if m["last"]>m["sma50"]: score+=10; reasons.append("price above 50-day")
        return hit(ticker,"Insider Buying",score,"Bullish Fundamental",reasons,m["last"],m["low20"],m["high50"])
    except Exception:
        return None


def high_adr(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 60: return None
    m = base_metrics(df, spy_df)
    score = 0; reasons=[]
    if m["adr20"] >= 4: score += 45; reasons.append(f"ADR {m['adr20']:.1f}%")
    if m["vol50"] >= 300000: score += 20; reasons.append("tradable liquidity")
    if m["last"] > m["sma20"] if not np.isnan(lastv(df,"SMA20")) else True: score += 15; reasons.append("short-term trend active")
    if m["rvol"] >= 1: score += 20; reasons.append("current volume active")
    return hit(ticker, "High Avg Daily Range", score, "Day Trading", reasons, m["last"], m["last"]-m["atr"], m["last"]+2*m["atr"])


def highest_volume(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 60: return None
    m = base_metrics(df, spy_df)
    score=0; reasons=[]
    if m["vol"] >= 10_000_000: score += 45; reasons.append(f"today volume {m['vol']/1e6:.1f}M")
    elif m["vol"] >= 5_000_000: score += 30; reasons.append(f"today volume {m['vol']/1e6:.1f}M")
    if m["rvol"] >= 2: score += 30; reasons.append(f"RVOL {m['rvol']:.1f}x")
    if abs(m["pct_chg"]) >= 2: score += 25; reasons.append(f"price move {m['pct_chg']:.1f}%")
    direction = "Bullish" if m["pct_chg"] >= 0 else "Bearish"
    return hit(ticker, "Highest Volume", score, direction, reasons, m["last"], m["last"]-m["atr"], m["last"]+2*m["atr"])


def change_in_character(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 80: return None
    m=base_metrics(df, spy_df)
    prior=df.iloc[-8:-1]
    quiet_range = float((prior["High"]-prior["Low"]).mean()/prior["Close"].mean()*100) < max(2.5, m["adr20"]*0.55)
    quiet_vol = float(prior["Volume"].mean()) < float(df["Volume"].iloc[-40:-8].mean())*0.8
    range_expansion = float(df["RangePct"].iloc[-1]) > float(prior["RangePct"].mean())*1.7
    vol_spike = m["rvol"] > 1.8
    score=0; reasons=[]
    if quiet_range and quiet_vol: score += 35; reasons.append("7-day quiet base")
    if range_expansion: score += 30; reasons.append("range expansion")
    if vol_spike: score += 25; reasons.append(f"volume spike {m['rvol']:.1f}x")
    if abs(m["pct_chg"]) > 2: score += 10; reasons.append("directional price move")
    direction="Bullish" if m["pct_chg"] >= 0 else "Bearish"
    return hit(ticker,"Change in Character",score,direction,reasons,m["last"],m["low20"],m["high20"])


def double_bottom(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if is_intraday():
        return intraday_double_bottom(ticker, df, meta, spy_df)
    if len(df) < 130:
        return None
    m = base_metrics(df, spy_df); w = df.tail(90); thirds = np.array_split(w, 3)
    low1 = float(thirds[0]["Low"].min()); mid_high = float(thirds[1]["High"].max()); low2 = float(thirds[2]["Low"].min())
    similar = abs(low1 - low2) / max((low1 + low2) / 2, 1e-9) * 100 < 8
    second_higher = low2 >= low1 * 0.95
    near_pivot = mid_high * 0.94 <= m["last"] <= mid_high * 1.04
    improving = m["last"] > m["sma50"] or m["rvol"] > 1.2 or m["rsi"] > 45
    if not (similar and second_higher and near_pivot and improving):
        return None
    score = 65
    reasons = ["two similar lows / W pattern", "near W pivot", "confirmation improving"]
    if m["rvol"] > 1.2:
        score += 10; reasons.append("volume interest")
    if m["rsi"] > 45:
        score += 10; reasons.append("momentum stabilizing")
    if m.get("rs_vs_spy", 0) > -5:
        score += 10; reasons.append("relative strength acceptable")
    return hit(ticker, "Double Bottom", score, "Bullish Reversal", reasons, mid_high, min(low1, low2), mid_high + (mid_high - min(low1, low2)))


def flat_base(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if is_intraday():
        return intraday_tight_consolidation(ticker, df, meta, spy_df)
    if len(df) < 140:
        return None
    m = base_metrics(df, spy_df); w = df.tail(35)
    correction = (float(w["High"].max()) - float(w["Low"].min())) / max(float(w["High"].max()), 1e-9) * 100
    prior_up = pct(float(w["Close"].iloc[0]), float(df["Close"].iloc[-100])) > 15 if len(df) > 100 else False
    near = float(w["High"].max()) * 0.95 <= m["last"] <= float(w["High"].max()) * 1.02
    trend = m["last"] > m["sma50"] > m["sma200"]
    vol_quiet = float(w["Volume"].tail(10).mean()) < float(df["Volume"].tail(80).mean())
    if not (correction <= 15 and prior_up and near and trend):
        return None
    score = 65
    reasons = [f"flat base correction {correction:.1f}%", "base after prior advance", "near flat-base pivot"]
    if vol_quiet:
        score += 15; reasons.append("volume quiet")
    if m.get("rs_vs_spy", 0) > 0:
        score += 10; reasons.append("RS positive")
    return hit(ticker, "Flat Base", score, "Bullish Breakout", reasons, float(w["High"].max()), float(w["Low"].min()), float(w["High"].max()) + 2 * m["atr"])


def ipo_base(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if len(df) < 80 or len(df) > 520:
        return None
    m = base_metrics(df, spy_df)
    all_high = float(df["High"].max()); all_low = float(df["Low"].min())
    drawdown = pct(m["last"], all_high)
    base_range = (m["high50"] - m["low50"]) / max(m["last"], 1e-9) * 100
    reclaim = m["last"] > m["sma50"]
    near_pivot = m["last"] > m["high50"] * 0.90
    if not (-60 <= drawdown <= -10 and base_range < 25 and reclaim and near_pivot):
        return None
    score = 65
    reasons = [f"post-IPO drawdown/base {drawdown:.0f}%", "base tightening", "reclaiming 50-day"]
    if m["rvol"] > 1:
        score += 10; reasons.append("volume interest")
    if m.get("rs_vs_spy", 0) > 0:
        score += 10; reasons.append("RS positive")
    return hit(ticker, "IPO Base", score, "Bullish Watch", reasons, m["high50"], m["low50"], all_high)


def rs_new_high(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    if spy_df is None or len(df) < 160 or len(spy_df) < 160:
        return None
    aligned = pd.concat([df["Close"], spy_df["Close"]], axis=1, join="inner").dropna()
    if aligned.shape[0] < 100:
        return None
    aligned.columns = ["stock", "spy"]
    ratio = aligned["stock"] / aligned["spy"]
    m = base_metrics(df, spy_df)
    rs_high = ratio.iloc[-1] >= ratio.tail(252).max() * 0.995
    constructive = m["last"] > m["sma50"] and m["last"] >= 0.75 * m["high252"]
    if not (rs_high and constructive):
        return None
    score = 65
    reasons = ["RS line at/new 52-week high", "constructive price trend"]
    if 0.03 <= (m["high252"] - m["last"]) / max(m["high252"], 1e-9) <= 0.25:
        score += 10; reasons.append("price still below own 52-week high")
    if m["rvol"] > 1:
        score += 10; reasons.append("volume active")
    if m["last"] > m["sma50"] > m["sma200"]:
        score += 15; reasons.append("trend aligned")
    return hit(ticker, "RS New High", score, "Bullish Leadership", reasons, m["last"], m["sma50"], m["high252"])


def smart_money_confluence(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    bullish_fns=[bull_flag,vcp,cup_handle,ascending_triangle,pocket_pivot,qulla_breakout,minervini_template,stage2,pivotal_point,volume_surge,buyable_gap_up,new_50day_high,institutional_accumulation,macd_cross]
    bearish_fns=[bear_flag,head_shoulders_top,parabolic_short,stage3,sell_signal]
    bullish=[]; bearish=[]
    for fn in bullish_fns:
        h=fn(ticker,df,meta,spy_df)
        if h and h.score>=60: bullish.append(h)
    for fn in bearish_fns:
        h=fn(ticker,df,meta,spy_df)
        if h and h.score>=60: bearish.append(h)
    if len(bullish)<3 and len(bearish)<2: return None
    m=base_metrics(df, spy_df)
    if len(bullish)>=3:
        score=min(100,55+len(bullish)*10+max(x.score for x in bullish)*0.15)
        reasons=[f"{len(bullish)} bullish signals"]+[x.scanner for x in bullish[:5]]
        best=max(bullish,key=lambda x:x.score)
        return hit(ticker,"Smart Money Confluence",score,"Bullish Confluence",reasons,best.entry,best.stop,best.target)
    score=min(100,60+len(bearish)*12+max(x.score for x in bearish)*0.10)
    reasons=[f"{len(bearish)} bearish signals"]+[x.scanner for x in bearish[:5]]
    best=max(bearish,key=lambda x:x.score)
    return hit(ticker,"Smart Money Confluence",score,"Bearish Confluence",reasons,best.entry,best.stop,best.target)


def earnings_gap(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    # Detects power earnings-gap style price action; external earnings confirmation can be added via a paid calendar API.
    if len(df)<80: return None
    today=df.iloc[-1]; prev=df.iloc[-2]; m=base_metrics(df, spy_df)
    gap=pct(float(today["Open"]), float(prev["Close"]))
    score=0; reasons=[]
    if gap>=5: score+=40; reasons.append(f"power gap up {gap:.1f}%")
    if m["rvol"]>=2: score+=25; reasons.append(f"volume {m['rvol']:.1f}x")
    if float(today["Close"])>float(today["Open"]): score+=15; reasons.append("green gap day")
    if m["last"]>float(prev["High"]): score+=20; reasons.append("holds above prior high")
    return hit(ticker,"Power Earnings Gap",score,"Bullish Catalyst",reasons,m["last"],float(today["Low"]),m["last"]+3*m["atr"])


def peg_flag(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    bf=bull_flag(ticker,df,meta,spy_df)
    fv=fundamental_value_scan(ticker,df,meta,spy_df,"PEG Value") if meta else None
    if not bf or not fv: return None
    score=min(100, (bf.score+fv.score)/2 + 10)
    reasons=["PEG value + bull flag confluence", bf.reasons, fv.reasons]
    return hit(ticker,"PEG + Flag",score,"Bullish Confluence",reasons,bf.entry,bf.stop,bf.target)


def sector_leader(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    # Sector-relative leadership requires sector ETF mapping; here we use market-relative leadership + trend.
    if len(df)<160: return None
    m=base_metrics(df, spy_df)
    ret126=pct(m["last"],float(df["Close"].iloc[-126])) if len(df)>126 else np.nan
    score=0; reasons=[]
    if m.get("rs_vs_spy",0)>10: score+=45; reasons.append(f"outperforming SPY by {m['rs_vs_spy']:.1f}%")
    elif m.get("rs_vs_spy",0)>0: score+=25; reasons.append("outperforming SPY")
    if ret126>25: score+=25; reasons.append(f"6M performance {ret126:.0f}%")
    if m["last"]>m["sma50"]>m["sma200"]: score+=20; reasons.append("trend aligned")
    if m["last"]>=0.85*m["high252"]: score+=10; reasons.append("near highs")
    return hit(ticker,"Sector Leader",score,"Bullish Leadership",reasons,m["last"],m["sma50"],m["high252"])


def analyst_upgrade_proxy(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    # Free Yahoo does not reliably expose upgrade events. This proxy finds stocks analysts tend to upgrade: strong RS + momentum + trend.
    h=high_growth_momentum(ticker,df,meta,spy_df)
    if not h: return None
    info=meta or {}
    rec=info.get("recommendationMean")
    score=h.score; reasons=["upgrade proxy: strong momentum/RS"]
    if rec is not None and rec<=2.5:
        score+=10; reasons.append(f"positive analyst recommendation mean {rec}")
    return hit(ticker,"Analyst Upgrade Proxy",score,"Bullish Momentum",reasons,h.entry,h.stop,h.target)


def revenue_acceleration(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    info=meta or {}
    if not info: return None
    rg=info.get("revenueGrowth"); eg=info.get("earningsGrowth")
    m=base_metrics(df, spy_df)
    score=0; reasons=[]
    if rg is not None and rg>0.15: score+=35; reasons.append(f"revenue growth {rg*100:.0f}%")
    elif rg is not None and rg>0.05: score+=20; reasons.append(f"revenue growth {rg*100:.0f}%")
    if eg is not None and eg>0.10: score+=25; reasons.append("earnings growth positive")
    if m["last"]>m["sma50"]>m["sma200"]: score+=25; reasons.append("trend confirms growth")
    if m.get("rs_vs_spy",0)>0: score+=15; reasons.append("RS positive")
    return hit(ticker,"Revenue Acceleration",score,"Bullish Growth",reasons,m["last"],m["sma50"],m["high252"])



# Group scanners for one-pass watchlist generation. These modes help avoid running one pattern at a time.
BULLISH_SETUP_NAMES = [
    "Bull Flag", "Volatility Contraction Pattern", "Cup & Handle", "Ascending Triangle", "Pocket Pivot",
    "Pivotal Point", "Minervini Trend Template", "Weinstein Stage 2",
    "Flat Base", "Pullback to 21 EMA", "200-Day Bounce", "MACD Bullish Cross",
    "Bollinger Squeeze", "Inside Day Breakout", "New 50-Day High", "RS New High",
    "Buyable Gap Up", "Recent Doublers", "Falling Wedge", "Double Bottom", "Inverse Head & Shoulders",
    "Institutional Accumulation", "Volume Surge", "Relative Volume"
]

BEARISH_SETUP_NAMES = ["Bear Flag", "Head & Shoulders Top", "Weinstein Stage 3 Distribution", "Sell Signal", "Parabolic Short"]


def all_bullish_setups(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    hits = []
    for name in BULLISH_SETUP_NAMES:
        fn = SCANNERS.get(name)
        if not fn:
            continue
        try:
            h = fn(ticker, df, meta, spy_df)
            if h and not str(h.direction).lower().startswith("bear"):
                hits.append(h)
        except Exception:
            continue
    if not hits:
        return None
    hits = sorted(hits, key=lambda x: x.score, reverse=True)
    best = hits[0]
    score = min(100, best.score + min(15, (len(hits)-1)*3))
    reasons = [f"best setup: {best.scanner}", f"{len(hits)} bullish setup(s) found"] + [h.scanner for h in hits[:5]]
    return hit(ticker, best.scanner, score, best.direction, reasons, best.entry, best.stop, best.target)


def all_bearish_setups(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    hits = []
    for name in BEARISH_SETUP_NAMES:
        fn = SCANNERS.get(name)
        if not fn:
            continue
        try:
            h = fn(ticker, df, meta, spy_df)
            if h:
                hits.append(h)
        except Exception:
            continue
    if not hits:
        return None
    hits = sorted(hits, key=lambda x: x.score, reverse=True)
    best = hits[0]
    score = min(100, best.score + min(15, (len(hits)-1)*3))
    reasons = [f"best setup: {best.scanner}", f"{len(hits)} bearish setup(s) found"] + [h.scanner for h in hits[:5]]
    return hit(ticker, best.scanner, score, best.direction, reasons, best.entry, best.stop, best.target)


def all_technical_setups(ticker: str, df: pd.DataFrame, meta=None, spy_df=None) -> Optional[ScanHit]:
    bullish = all_bullish_setups(ticker, df, meta, spy_df)
    bearish = all_bearish_setups(ticker, df, meta, spy_df)
    if bullish and bearish:
        # Conflict resolver: choose the higher-quality setup; require a meaningful edge if directions conflict.
        if bullish.score >= bearish.score + 6:
            return bullish
        if bearish.score >= bullish.score + 6:
            return bearish
        return None
    return bullish or bearish

SCANNERS: Dict[str, Callable] = {
    "All Bullish Setups": all_bullish_setups,
    "All Bearish Setups": all_bearish_setups,
    "All Technical Setups": all_technical_setups,
    "Bull Flag": bull_flag,
    "Bear Flag": bear_flag,
    "Volatility Contraction Pattern": vcp,
    "Cup & Handle": cup_handle,
    "Ascending Triangle": ascending_triangle,
    "Pocket Pivot": pocket_pivot,
    "Qullamaggie Breakout": qulla_breakout,
    "High Tight Flag": high_tight_flag,
    "Power Play": power_play,
    "Minervini Trend Template": minervini_template,
    "CANSLIM Growth": canslim,
    "Pivotal Point": pivotal_point,
    "Weinstein Stage 1": stage1,
    "Weinstein Stage 2": stage2,
    "Weinstein Stage 3 Distribution": stage3,
    "Golden Pocket": golden_pocket,
    "RSI Oversold Reversion": rsi_oversold_reversion,
    "Bottom Finder": bottom_finder,
    "Falling Wedge": falling_wedge,
    "Double Bottom": double_bottom,
    "Flat Base": flat_base,
    "IPO Base": ipo_base,
    "Inverse Head & Shoulders": inverse_hs,
    "Head & Shoulders Top": head_shoulders_top,
    "Volume Surge": volume_surge,
    "High Avg Daily Range": high_adr,
    "Highest Volume": highest_volume,
    "Change in Character": change_in_character,
    "Relative Volume": rvol_scan,
    "5% Gap / Strong Move": gap_scan,
    "Buyable Gap Up": buyable_gap_up,
    "Power Earnings Gap": earnings_gap,
    "Recent Doublers": recent_doublers,
    "Pullback to 21 EMA": pullback_21ema,
    "200-Day Bounce": bounce_200d,
    "Oversold Bounce": oversold_bounce,
    "MACD Bullish Cross": macd_cross,
    "Golden Cross": golden_cross,
    "Bollinger Squeeze": bollinger_squeeze,
    "Inside Day Breakout": inside_day_breakout,
    "New 50-Day High": new_50day_high,
    "RS New High": rs_new_high,
    "Smart Money Confluence": smart_money_confluence,
    "Institutional Accumulation": institutional_accumulation,
    "Sector Leader": sector_leader,
    "High Growth Momentum": high_growth_momentum,
    "Analyst Upgrade Proxy": analyst_upgrade_proxy,
    "Buy the Dip": buy_the_dip,
    "Sell Signal": sell_signal,
    "Parabolic Short": parabolic_short,
    "Earnings Next Week": earnings_scan_factory(0, 7, "Earnings Next Week"),
    "Earnings Next Month": earnings_scan_factory(8, 30, "Earnings Next Month"),
    "Earnings Watch": earnings_watch,
    "PEG Value": lambda t,d,m=None,spy_df=None: fundamental_value_scan(t,d,m,spy_df,"PEG Value"),
    "PEG + Flag": peg_flag,
    "Lynch GARP": lambda t,d,m=None,spy_df=None: fundamental_value_scan(t,d,m,spy_df,"Lynch GARP"),
    "Buffett Value": lambda t,d,m=None,spy_df=None: fundamental_value_scan(t,d,m,spy_df,"Buffett Value"),
    "Magic Formula": lambda t,d,m=None,spy_df=None: fundamental_value_scan(t,d,m,spy_df,"Magic Formula"),
    "Dividend Growth": lambda t,d,m=None,spy_df=None: fundamental_value_scan(t,d,m,spy_df,"Dividend Growth"),
    "Revenue Acceleration": revenue_acceleration,
    "Munger 200W": munger_200w,
    "Short Squeeze": short_squeeze,
    "Insider Buying": insider_buying,
}

FUNDAMENTAL_SCANNERS = {"CANSLIM Growth", "PEG Value", "PEG + Flag", "Lynch GARP", "Buffett Value", "Magic Formula", "Dividend Growth", "Revenue Acceleration", "Munger 200W", "Short Squeeze", "Insider Buying", "Analyst Upgrade Proxy"}
EARNINGS_SCANNERS = {"Earnings Next Week", "Earnings Next Month", "Earnings Watch"}


def compute_ars(data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    vals = []
    for t, df in data.items():
        if len(df) < 65: continue
        c=df["Close"]
        ret = 0.5*pct(float(c.iloc[-1]), float(c.iloc[-63])) if len(c)>63 else 0
        if len(c)>126: ret += 0.3*pct(float(c.iloc[-1]), float(c.iloc[-126]))
        if len(c)>252: ret += 0.2*pct(float(c.iloc[-1]), float(c.iloc[-252]))
        vals.append((t, ret))
    if not vals: return {}
    rets = pd.Series({t:r for t,r in vals}).replace([np.inf,-np.inf],np.nan).dropna()
    ranks = rets.rank(pct=True)*100
    return ranks.to_dict()


def trade_status(direction: str, last: float, entry: float, stop: float, target: float, rvol: float) -> Tuple[str, float, float]:
    """Return setup status, distance-to-entry %, and reward/risk.

    Status is intentionally separate from pattern detection:
    - Watch: valid pattern but not close enough to entry.
    - Near: within ~5% of entry.
    - Ready: within ~2% of entry.
    - Triggered: price has crossed the entry level; volume confirmation still matters.
    """
    try:
        last = float(last); entry = float(entry); stop = float(stop); target = float(target)
    except Exception:
        return "Watch", np.nan, np.nan
    if not np.isfinite(last) or not np.isfinite(entry) or entry == 0:
        return "Watch", np.nan, np.nan
    bearish = str(direction).lower().startswith("bear") or "short" in str(direction).lower()
    if bearish:
        dist = (last - entry) / max(last, 1e-9) * 100.0  # positive = still above short trigger
        triggered = last <= entry
        risk = abs(stop - entry)
        reward = abs(entry - target)
    else:
        dist = (entry - last) / max(last, 1e-9) * 100.0  # positive = still below long trigger
        triggered = last >= entry
        risk = abs(entry - stop)
        reward = abs(target - entry)
    rr = reward / risk if risk > 0 else np.nan
    if triggered:
        status = "Triggered" if rvol >= 1.2 else "Triggered - needs volume"
    elif dist <= 2.0:
        status = "Ready"
    elif dist <= 5.0:
        status = "Near"
    else:
        status = "Watch"
    return status, round(dist, 2), round(rr, 2) if np.isfinite(rr) else np.nan


def run_scan(
    tickers: List[str],
    scanner_name: str,
    period: str,
    interval: str,
    min_price: float,
    min_avg_vol: float,
    fetch_fundamentals: bool,
    progress=True,
) -> pd.DataFrame:
    if hasattr(st, "session_state"):
        st.session_state["scan_interval"] = interval
    tickers = [_clean_ticker(t) for t in tickers]
    tickers = [t for t in tickers if t]
    spy = download_prices(("SPY",), period="1y", interval="1d").get("SPY")
    data = download_prices(tuple(tickers), period=period, interval=interval)
    ars_map = compute_ars(data)
    scanner = SCANNERS[scanner_name]
    rows = []
    bar = st.progress(0, text="Scanning symbols...") if progress else None
    n = max(len(data),1)
    for idx, (t, df) in enumerate(data.items()):
        try:
            m = base_metrics(df, spy)
            if m["last"] < min_price or m["vol50"] < min_avg_vol:
                continue
            info = {}
            if fetch_fundamentals or scanner_name in FUNDAMENTAL_SCANNERS:
                info = get_info(t)
            result = scanner(t, df, info, spy)
            if result:
                ars = float(ars_map.get(t, np.nan))
                info2 = info or {}
                rows.append({
                    "Ticker": t,
                    "Scanner": result.scanner,
                    "TimeframeMode": timeframe_mode_label(),
                    "Grade": result.grade,
                    "Score": round(result.score, 1),
                    "Direction": result.direction,
                    "Last": round(m["last"], 2),
                    "%Chg": round(m["pct_chg"], 2) if not np.isnan(m["pct_chg"]) else np.nan,
                    "RVOL": round(m["rvol"], 2),
                    "AvgVol50": int(m["vol50"]) if not np.isnan(m["vol50"]) else np.nan,
                    "ADR20%": round(m["adr20"], 2) if not np.isnan(m["adr20"]) else np.nan,
                    "TA": ta_rating(df, ars),
                    "FA": fa_rating(info2) if info2 else np.nan,
                    "ARS": round(ars, 1) if not np.isnan(ars) else np.nan,
                    "Entry": round(result.entry, 2) if result.entry else np.nan,
                    "Stop": round(result.stop, 2) if result.stop else np.nan,
                    "Target": round(result.target, 2) if result.target else np.nan,
                    "SetupStatus": trade_status(result.direction, m["last"], result.entry, result.stop, result.target, m["rvol"])[0],
                    "DistToEntry%": trade_status(result.direction, m["last"], result.entry, result.stop, result.target, m["rvol"])[1],
                    "RR": trade_status(result.direction, m["last"], result.entry, result.stop, result.target, m["rvol"])[2],
                    "MarketCap": info2.get("marketCap", np.nan),
                    "Sector": info2.get("sector", ""),
                    "Reasons": result.reasons,
                })
        except Exception as e:
            # Keep scanning even if one symbol fails.
            pass
        finally:
            if bar and (idx % 10 == 0 or idx == n-1):
                bar.progress(min(1.0, (idx+1)/n), text=f"Scanning {idx+1}/{n} symbols...")
    if bar:
        bar.empty()
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    status_order = {"Triggered": 0, "Triggered - needs volume": 1, "Ready": 2, "Near": 3, "Watch": 4}
    if "SetupStatus" in out.columns:
        out["StatusRank"] = out["SetupStatus"].map(status_order).fillna(9)
        out = out.sort_values(["StatusRank", "Score", "TA", "ARS"], ascending=[True, False, False, False]).reset_index(drop=True)
        out = out.drop(columns=["StatusRank"])
    else:
        out = out.sort_values(["Score", "TA", "ARS"], ascending=False).reset_index(drop=True)
    return out


# -----------------------------
# Streamlit UI
# -----------------------------


def plot_chart(ticker: str, period="1y", interval="1d"):
    data = download_prices((ticker,), period=period, interval=interval)
    df = data.get(ticker)
    if df is None or df.empty:
        st.warning("No chart data available.")
        return
    tail = df.tail(180)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=tail.index, open=tail["Open"], high=tail["High"], low=tail["Low"], close=tail["Close"], name=ticker))
    for col in ["SMA50", "SMA100", "SMA200", "EMA21"]:
        if col in tail:
            fig.add_trace(go.Scatter(x=tail.index, y=tail[col], name=col, mode="lines"))
    fig.update_layout(height=560, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=35,b=10), title=f"{ticker} Chart")
    st.plotly_chart(fig, width="stretch")


def main():
    st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide")
    st.title("📈 Chart Pattern Scanner")
    st.caption("Chart Pattern Scanner — V9.5")

    with st.sidebar:
        st.header("Scanner Settings")
        scanner_name = st.selectbox("Scanner", list(SCANNERS.keys()), index=list(SCANNERS.keys()).index("All Bullish Setups"))
        match_quality = st.selectbox("Match quality", ["Loose / candidate", "Balanced", "Strict / confirmed"], index=1,
                                      help="Loose returns more candidates for manual chart review. Strict returns fewer textbook patterns.")
        st.session_state["match_quality"] = match_quality
        universe = st.selectbox("Universe", ["Custom", "S&P 500", "Nasdaq 100", "Full U.S. Nasdaq/NYSE/AMEX"], index=1)
        include_etfs = st.checkbox("Include ETFs", value=False)
        hide_duplicate_share_classes = st.checkbox("Hide duplicate share classes", value=True, help="Removes duplicate share classes such as GOOG when GOOGL is also in the universe, so you do not review the same company twice.")
        custom = st.text_area("Custom tickers", value="AAPL MSFT NVDA AMD META AMZN GOOGL TSLA PLTR SHOP COIN SNOW ALAB", height=90)
        max_symbols = st.number_input("Max symbols to scan", min_value=10, max_value=6000, value=500, step=50)
        period = st.selectbox("History", ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"], index=4)
        interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m"], index=0)
        st.session_state["scan_interval"] = interval
        if interval in {"5m", "15m", "30m"} and period not in {"5d", "1mo", "3mo"}:
            st.caption("Intraday data is automatically limited to 3mo or less for Yahoo/yfinance reliability.")
            period = "3mo"
        if interval == "1h" and period == "5y":
            period = "2y"
        min_price = st.number_input("Min price", value=10.0, min_value=0.0, step=1.0)
        min_avg_vol = st.number_input("Min 50-day avg volume", value=500000, min_value=0, step=100000)
        fetch_fa = st.checkbox("Fetch fundamentals/sector for results (slower)", value=scanner_name in FUNDAMENTAL_SCANNERS)
        run = st.button("Run Scan", type="primary", width="stretch")

    if universe == "Custom":
        tickers = parse_tickers(custom)
    elif universe == "S&P 500":
        tickers = load_sp500()
    elif universe == "Nasdaq 100":
        tickers = load_nasdaq100()
    else:
        tickers = load_nasdaq_trader_symbols(include_etfs=include_etfs)["Ticker"].tolist()

    # Always include the custom/priority tickers first, even when scanning S&P 500 or Full U.S.
    # This lets you test names such as AMZN directly without changing Universe to Custom.
    priority_tickers = parse_tickers(custom)
    tickers = list(dict.fromkeys(priority_tickers + tickers))

    if hide_duplicate_share_classes:
        tickers = [t for t in tickers if t not in DUPLICATE_SHARE_CLASS_REMOVE]
    tickers = tickers[: int(max_symbols)]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Universe symbols", f"{len(tickers):,}")
    c2.metric("Scanner", scanner_name)
    c3.metric("Match", st.session_state.get("match_quality", "Balanced"))
    c4.metric("Interval", interval)
    c5.metric("Min avg volume", f"{int(min_avg_vol):,}")
    if len(tickers) <= 25:
        st.warning("You are scanning a very small universe. For better scanner results, select S&P 500, Nasdaq 100, or Full U.S.; or paste a larger custom list.")

    # Protect against stale result tables. In earlier versions, changing the Scanner dropdown
    # updated the header but could leave the previous scan's results on screen until Run was clicked.
    # This key makes sure the displayed table always matches the active scanner/settings.
    current_settings_key = (
        scanner_name, match_quality, universe, include_etfs, hide_duplicate_share_classes, custom.strip(), int(max_symbols),
        period, interval, float(min_price), int(min_avg_vol), bool(fetch_fa), tuple(tickers[: int(max_symbols)])
    )

    if run:
        start = time.time()
        with st.spinner("Downloading data and running scanner..."):
            results = run_scan(tickers, scanner_name, period, interval, float(min_price), float(min_avg_vol), fetch_fa)
        elapsed = time.time() - start
        st.success(f"Scan complete in {elapsed:.1f}s. Found {len(results):,} hits.")
        if results.empty:
            st.warning("No passing setups found. Try lowering filters, choosing a different scanner, or using a larger universe.")
            return
        st.session_state["last_results"] = results
        st.session_state["last_period"] = period
        st.session_state["last_interval"] = interval
        st.session_state["last_scanner_name"] = scanner_name
        st.session_state["last_match_quality"] = match_quality
        st.session_state["last_settings_key"] = current_settings_key

    results = st.session_state.get("last_results")
    last_key = st.session_state.get("last_settings_key")
    if isinstance(results, pd.DataFrame) and not results.empty:
        if last_key != current_settings_key:
            st.warning("Settings changed since the last scan. Click **Run Scan** again before reviewing the table. This prevents stale results from being shown under the wrong scanner name.")
            return
        st.subheader("Scanner Results")
        st.caption(f"Showing results from: {st.session_state.get('last_scanner_name', scanner_name)} | Match: {st.session_state.get('last_match_quality', match_quality)}")
        cols = ["Scanner","Ticker","SetupStatus","DistToEntry%","RR","Grade","Score","Direction","Last","%Chg","RVOL","AvgVol50","ADR20%","TA","FA","ARS","Entry","Stop","Target","Sector","Reasons"]
        show = results[[c for c in cols if c in results.columns]].copy()
        ticker_filter = st.text_input("Filter results by ticker", value="", help="Example: AMZN")
        if ticker_filter.strip():
            keepers = parse_tickers(ticker_filter)
            if keepers and "Ticker" in show.columns:
                show = show[show["Ticker"].isin(keepers)]
        st.dataframe(show, width="stretch", height=430)

        csv = results.to_csv(index=False).encode("utf-8")
        dl_name = str(st.session_state.get('last_scanner_name', scanner_name)).replace(' ','_').lower()
        st.download_button("Download CSV", csv, file_name=f"{dl_name}_scan.csv", mime="text/csv")

        st.subheader("Chart Review")
        tick = st.selectbox("Select ticker to chart", results["Ticker"].tolist())
        plot_chart(tick, st.session_state.get("last_period", "1y"), st.session_state.get("last_interval", "1d"))

        with st.expander("How to use these results"):
            st.markdown(
                """
                - Treat scanner output as a **watchlist**, not an automatic buy/sell instruction.
                - Use **SetupStatus**: Watch = valid pattern but not near trigger, Near = within ~5%, Ready = within ~2%, Triggered = crossed entry.
                - Confirm the chart manually: trend, pivot, volume, support/resistance, and market regime.
                - For bullish breakout setups, typical entry is over the pivot with volume confirmation; stop is below the base/flag/pullback low.
                - For mean-reversion setups, wait for a reversal candle or reclaim of a short moving average.
                - Avoid thin names, wide spreads, and stocks moving only because of low-float noise unless that is your specific strategy.
                """
            )


if __name__ == "__main__":
    main()
