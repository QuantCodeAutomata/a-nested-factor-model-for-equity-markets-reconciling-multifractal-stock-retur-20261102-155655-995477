"""
Market data loader for empirical experiments (exp5, exp6).

Loads daily OHLCV data for S&P500 stocks and the index via the QuantX Dataloader,
with Yahoo Finance as the backstop.  Implements the Garman-Klass (GK) daily volatility
estimator from Eq. (47) and provides utilities for assembling universe panels.

Market data: QuantX Dataloader — https://quantx.limex.pro/

Price history in this project is loaded through the QuantX Dataloader, a
research-ready market-data service available through a REST API and as an
MCP server for AI agents. It is built on market data from Massive
(https://massive.com/) and prepares every stock history for backtesting:
  - prices adjusted for dividends, ready for total-return research;
  - one continuous history when a company changes its ticker;
  - clean separation when a ticker has belonged to different companies;
  - careful corporate-action handling and fine-grained data polishing.

Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
MASSIVE_API_KEY at https://massive.com/.
"""

from __future__ import annotations

import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# QuantX Dataloader API root (hardcoded per market-data skill)
DATALOADER_URL = "https://quantx-api.limex.pro/apis/dataloader"

# Local disk cache for stored-mode downloads (not committed)
CACHE_DIR = Path(__file__).parent / "cache"

# S&P 500 ticker for Yahoo Finance / Dataloader
SP500_INDEX_TICKER = "SPY"  # SPY ETF used as S&P 500 index proxy; ^GSPC rejected by Massive


# ---------------------------------------------------------------------------
# Curated list of long-term S&P 500 constituents (1994-2026)
# The paper (Section 7.2, page 19) uses 243 US stocks that are constituents
# of the S&P 500 throughout the entire period 01/01/1994 to 01/01/2026.
# The exact list is in Appendix I of the paper; this approximation uses
# well-known long-tenure S&P 500 members.
# ---------------------------------------------------------------------------
SP500_LONG_TERM_TICKERS = [
    # Consumer Staples
    "KO", "PG", "WMT", "MO", "PM", "CL", "GIS", "K", "HRL", "SJM",
    "CPB", "CAG", "MKC", "TSN", "HSY", "SYY", "BF-B", "CLX", "KMB",
    # Healthcare
    "JNJ", "PFE", "MRK", "ABT", "BMY", "LLY", "AMGN", "UNH", "HUM",
    "MDT", "BSX", "SYK", "ZBH", "BAX", "TMO", "DHR", "ISRG",
    # Information Technology
    "IBM", "MSFT", "ORCL", "CSCO", "TXN", "ADI", "AMAT", "LRCX",
    "GLW", "KLAC", "MCHP", "NTAP", "HPQ", "MSI", "JNPR", "CDNS",
    "SNPS", "XLNX",
    # Financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "BK", "STT", "MMC",
    "AON", "AFL", "MET", "PRU", "ALL", "TRV", "PGR", "CB", "AIG",
    "CME", "ICE", "SPG",
    # Energy
    "XOM", "CVX", "COP", "OXY", "SLB", "HAL", "BKR", "VLO", "MPC",
    "PSX", "DVN", "EOG", "PXD",
    # Industrials
    "GE", "MMM", "BA", "CAT", "HON", "UTX", "RTX", "EMR", "ITW",
    "ROK", "DOV", "PH", "GWW", "FDX", "UPS", "CSX", "NSC", "UNP",
    "LMT", "NOC", "GD", "HII", "TDG",
    # Materials
    "DD", "DOW", "PPG", "SHW", "ECL", "NUE", "FCX", "APD", "LIN",
    "IFF", "FMC", "ALB",
    # Utilities
    "AEP", "SO", "D", "EXC", "DUK", "SRE", "ED", "PPL", "XEL",
    "CMS", "NI", "OGE",
    # Communication Services
    "T", "VZ", "CMCSA", "DIS", "NFLX", "CBS", "OMC", "IPG",
    # Real Estate
    "SPG", "O", "AVB", "EQR", "VTR",
    # Consumer Discretionary
    "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "ROST", "AZO",
    "ORLY", "BBY", "HAS",
    # Extra (to reach near 243)
    "HBAN", "KEY", "RF", "FITB", "CFG", "MTB", "ZION", "PBCT",
    "STI", "BBT", "TCF",
    "LHX", "L", "BEN", "TROW", "STZ", "TAP", "RAI",
    "ADM", "BG", "MOS",
    "AIZ", "SFM", "PKI", "VAR", "IVZ", "WLTW",
    "DLR", "PSA", "AIV", "KIM", "CBL",
    "ABC", "CAH", "MCK", "CVS", "WBA", "CI",
    "BIIB", "REGN", "GILD", "VRTX", "MYL", "WRK", "IRM",
]
# Remove duplicates (SPG appears twice in the list above), keep order
_seen: set[str] = set()
_deduped: list[str] = []
for _t in SP500_LONG_TERM_TICKERS:
    if _t not in _seen:
        _seen.add(_t)
        _deduped.append(_t)
SP500_LONG_TERM_TICKERS = _deduped[:243]  # cap at 243 for exp6


def _dataloader_headers() -> dict[str, str]:
    """Build request headers for QuantX Dataloader."""
    api_key = os.environ.get("DATALOADER_API_KEY", "")
    massive_key = os.environ.get("MASSIVE_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"}
    if massive_key:
        headers["X-Massive-Api-Key"] = massive_key
    return headers


def _download_stored_ticker(
    result_id: str,
    ticker: str,
    timeframe: str,
    start_date: str,
    end_date: str,
) -> tuple[str, pd.DataFrame | None]:
    """
    Download one ticker's bars from a stored Dataloader result.

    Uses the three-step workflow:
    POST /save-dataframe → GET /artifacts/{id}/download → GET download URL.
    Results are cached to CACHE_DIR to avoid redundant downloads.

    Returns (ticker, DataFrame) or (ticker, None) on any failure.
    """
    # Sanitize ticker for filesystem use
    safe_ticker = ticker.replace("/", "_").replace("^", "hat").replace("*", "x")
    # Use CSV to avoid pyarrow / fastparquet dependency (numpy 1.26 conflict).
    cache_file = CACHE_DIR / f"{safe_ticker}_{start_date}_{end_date}_{timeframe}.csv"
    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            logger.debug("Cache hit for %s", ticker)
            return ticker, df
        except Exception:
            logger.debug("Cache read failed for %s; re-downloading.", ticker)

    try:
        # Step 1: POST /save-dataframe with CSV format (no pyarrow needed)
        save_resp = requests.post(
            f"{DATALOADER_URL}/save-dataframe",
            json={
                "result_id": result_id,
                "ticker": ticker,
                "dataset_name": f"ohlcv_{safe_ticker}_{start_date}_{end_date}",
                "timeframe": timeframe,
                "start_date": start_date,
                "end_date": end_date,
                "file_format": "csv",
                "data_kind": "ohlcv",
            },
            headers=_dataloader_headers(),
            timeout=120,
        )
        if save_resp.status_code != 200:
            logger.warning(
                "save-dataframe failed for %s: HTTP %d %s",
                ticker, save_resp.status_code, save_resp.text[:300],
            )
            return ticker, None
        save_data = save_resp.json()
        artifact_id = save_data.get("result", {}).get("artifact_id")
        if not artifact_id:
            logger.warning("save-dataframe for %s returned no artifact_id: %s", ticker, save_data)
            return ticker, None

        # Step 2: GET /artifacts/{artifact_id}/download
        dl_resp = requests.get(
            f"{DATALOADER_URL}/artifacts/{artifact_id}/download",
            headers=_dataloader_headers(),
            timeout=120,
        )
        if dl_resp.status_code != 200:
            logger.warning(
                "artifact download link failed for %s: HTTP %d %s",
                ticker, dl_resp.status_code, dl_resp.text[:300],
            )
            return ticker, None
        download_url = dl_resp.json().get("result", {}).get("url")
        if not download_url:
            logger.warning("artifact response for %s has no URL: %s", ticker, dl_resp.text[:300])
            return ticker, None

        # Step 3: GET download URL (no auth per skill spec)
        file_resp = requests.get(download_url, timeout=300)
        if file_resp.status_code != 200:
            logger.warning(
                "CSV file download failed for %s: HTTP %d",
                ticker, file_resp.status_code,
            )
            return ticker, None

        df = pd.read_csv(io.BytesIO(file_resp.content), index_col=0, parse_dates=True)

        # Cache to disk for re-use
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file)
        return ticker, df

    except Exception as exc:
        logger.warning("Stored download error for %s: %s", ticker, exc)
        return ticker, None


def _parse_stored_ticker_df(df_raw: pd.DataFrame) -> pd.DataFrame | None:
    """
    Normalize a per-ticker parquet DataFrame returned by the stored-mode download.

    Expects a date/timestamp column and OHLCV columns; returns a DataFrame
    with DatetimeIndex named 'date' and columns [open, high, low, close, volume].
    """
    if df_raw is None or df_raw.empty:
        return None

    df = df_raw.copy()

    # Normalise the date index
    if "timestamp" in df.columns:
        df.index = pd.to_datetime(df["timestamp"]).dt.normalize()
        df = df.drop(columns=["timestamp"], errors="ignore")
    elif "date" in df.columns:
        df.index = pd.to_datetime(df["date"]).dt.normalize()
        df = df.drop(columns=["date"], errors="ignore")
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index).normalize()
        except Exception:
            return None
    else:
        df.index = df.index.normalize()
    df.index.name = "date"
    df = df.sort_index()

    # Lowercase column names for consistency
    df.columns = [c.lower() for c in df.columns]

    # Strip optional ticker prefix from OHLCV columns (e.g. 'ko_open' → 'open')
    ohlcv_suffixes = ["open", "high", "low", "close", "volume"]
    col_map = {}
    for col in df.columns:
        for sfx in ohlcv_suffixes:
            if col == sfx or col.endswith(f"_{sfx}"):
                col_map[col] = sfx
                break
    if col_map:
        df = df.rename(columns=col_map)

    required = ["open", "high", "low", "close"]
    if not all(c in df.columns for c in required):
        return None
    keep = [c for c in required + ["volume"] if c in df.columns]
    df = df[keep].astype(float)

    if len(df) < 10:
        return None
    return df


def _fetch_stored_mode(
    result_id: str,
    ticker_details: list[dict],
    timeframe: str,
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    """
    Handle a stored-mode Dataloader response by downloading each ticker's parquet
    via save-dataframe + artifact download, using up to 3 parallel threads.

    Parameters
    ----------
    result_id : str
        The `result_id` from the stored-mode response.
    ticker_details : list of dict
        Per-ticker metadata from the Dataloader response (carries 'ticker' key).
    timeframe, start_date, end_date : str
        Forwarded from the original fetch_ohlcv call.

    Returns
    -------
    dict of ticker → parsed DataFrame
    """
    stored_tickers = [
        td["ticker"] for td in ticker_details if td.get("ticker") and not td.get("error")
    ]
    if not stored_tickers:
        logger.warning("Stored mode: no valid tickers in ticker_details.")
        return {}

    logger.info(
        "Stored mode (result_id=%s): downloading %d tickers with 3 threads.",
        result_id[:16],
        len(stored_tickers),
    )

    output: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=3) as exe:
        futures = {
            exe.submit(
                _download_stored_ticker,
                result_id, tk, timeframe, start_date, end_date,
            ): tk
            for tk in stored_tickers
        }
        for fut in as_completed(futures):
            tk, df_raw = fut.result()
            df_parsed = _parse_stored_ticker_df(df_raw)
            if df_parsed is not None:
                output[tk] = df_parsed
            else:
                logger.warning("Stored mode: no usable data for %s.", tk)

    logger.info(
        "Stored mode: successfully loaded %d/%d tickers.",
        len(output), len(stored_tickers),
    )
    return output


def fetch_ohlcv(
    tickers: list[str],
    start_date: str,
    end_date: str,
    timeframe: str = "1d",
    adjust_dividends: bool = True,
    identity_date: Optional[str] = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV bars for a list of tickers via the QuantX Dataloader.

    Returns a dict mapping ticker → DataFrame with columns
    [open, high, low, close, volume] and a DatetimeIndex.
    Failed tickers (no data) are excluded from the output dict.

    Parameters
    ----------
    tickers : list of str
        Ticker symbols.
    start_date : str
        Start date "YYYY-MM-DD" (inclusive).
    end_date : str
        End date "YYYY-MM-DD" (inclusive).
    timeframe : str
        Bar frequency (default "1d" = daily).
    adjust_dividends : bool
        Whether to adjust for dividends (default True).
    identity_date : str or None
        Date at which the ticker identifies the security. Defaults to end_date.

    Returns
    -------
    dict of str → pd.DataFrame
    """
    if identity_date is None:
        identity_date = end_date

    payload = {
        "tickers": tickers,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date,
        "source": "massive",
        "adjust_dividends": adjust_dividends,
        "identity_date": identity_date,
        "yahoo_backstop": True,  # use Yahoo Finance when Massive has no bars
    }

    # Market data: QuantX Dataloader — https://quantx.limex.pro/
    resp = requests.post(
        f"{DATALOADER_URL}/fetch-ohlcvs",
        json=payload,
        headers=_dataloader_headers(),
        timeout=300,
    )

    if resp.status_code == 401:
        raise RuntimeError(
            "DATALOADER_API_KEY is missing or invalid. "
            "Set DATALOADER_API_KEY in your .env file."
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Dataloader error {resp.status_code}: {resp.text[:500]}"
        )

    data = resp.json()

    # Validate response structure
    result = data.get("result", {})
    if not result:
        raise RuntimeError(f"Dataloader returned empty result: {data}")

    # ── Stored-mode path ────────────────────────────────────────────────────
    # When data_stored=True the response carries result_id and ticker_details
    # but no inline rows; bars must be fetched via save-dataframe + artifact.
    if result.get("data_stored"):
        result_id = result.get("result_id", "")
        ticker_details = result.get("ticker_details", [])
        if not result_id:
            raise RuntimeError(
                "Dataloader returned data_stored=True but no result_id. "
                f"Response keys: {list(result.keys())}"
            )
        output = _fetch_stored_mode(result_id, ticker_details, timeframe, start_date, end_date)
        if not output:
            raise RuntimeError(
                f"Stored-mode download produced no usable data for tickers {tickers[:5]}... "
                f"between {start_date} and {end_date}."
            )
        return output
    # ── Inline path ─────────────────────────────────────────────────────────

    ohlcv_envelope = result.get("ohlcv_df", {})
    rows = ohlcv_envelope.get("rows", [])
    cols = ohlcv_envelope.get("columns", [])
    if not rows or not cols:
        raise RuntimeError(
            f"No OHLCV rows returned for tickers {tickers[:5]}... "
            f"between {start_date} and {end_date}."
        )

    df_raw = pd.DataFrame(rows, columns=cols)

    # Normalize the timestamp/date column
    date_col = "timestamp" if "timestamp" in df_raw.columns else "date"
    df_raw["_date"] = pd.to_datetime(df_raw[date_col]).dt.normalize()

    # Log ticker details
    ticker_details = result.get("ticker_details", [])
    td_by_ticker: dict[str, dict] = {}
    for td in ticker_details:
        tk = td.get("ticker", "?")
        td_by_ticker[tk] = td
        src = td.get("data_source", "unknown")
        err = td.get("error")
        warn = td.get("warnings")
        if err:
            logger.warning("Ticker %s error: %s", tk, err)
        if src not in ("massive", "yahoo_finance"):
            logger.info("Ticker %s loaded from %s", tk, src)
        if warn:
            logger.info("Ticker %s warnings: %s", tk, warn)

    # Parse response format:
    # - Narrow (single ticker): columns = [timestamp, open, high, low, close, volume]
    # - Wide (multi-ticker):    columns = [timestamp, T1_open, T1_high, ..., T2_open, ...]
    required_fields = ["open", "high", "low", "close"]
    output: dict[str, pd.DataFrame] = {}

    if "open" in df_raw.columns:
        # Narrow format — single ticker
        original_tk = ticker_details[0].get("ticker", tickers[0]) if ticker_details else tickers[0]
        grp = df_raw.rename(columns={"_date": "date"}).set_index("date").sort_index()
        available = [c for c in required_fields if c in grp.columns]
        if available and len(grp) >= 10:
            output[original_tk] = grp[[c for c in required_fields + ["volume"] if c in grp.columns]]
    else:
        # Wide format — multi-ticker: detect prefixes from TICKER_close columns
        prefixes = [c[:-6] for c in cols if c.endswith("_close") and not c.startswith("_")]
        for prefix in prefixes:
            field_map = {}
            for field in required_fields + ["volume"]:
                col_name = f"{prefix}_{field}"
                if col_name in df_raw.columns:
                    field_map[field] = col_name

            if "close" not in field_map:
                continue

            tk_df = df_raw[["_date"] + list(field_map.values())].copy()
            tk_df.columns = ["date"] + list(field_map.keys())
            tk_df = tk_df.set_index("date").sort_index()

            # Map column prefix back to original ticker symbol
            original_tk = prefix
            for tk, td in td_by_ticker.items():
                if tk.upper() == prefix.upper():
                    original_tk = tk
                    break

            if len(tk_df) < 10:
                logger.warning("Ticker %s: only %d rows, skipping.", original_tk, len(tk_df))
                continue
            if "low" in tk_df.columns and "high" in tk_df.columns and "close" in tk_df.columns:
                valid = (tk_df["low"] <= tk_df["close"]) & (tk_df["close"] <= tk_df["high"])
                if valid.mean() < 0.85:
                    logger.warning("Ticker %s: OHLC consistency check failed, skipping.", original_tk)
                    continue
            output[original_tk] = tk_df

    if not output:
        raise RuntimeError(
            f"No valid data returned for any of {tickers[:5]}..."
        )
    total_rows = sum(len(v) for v in output.values())
    logger.info(
        "Loaded %d/%d tickers, %s to %s, total rows: %d",
        len(output),
        len(tickers),
        start_date,
        end_date,
        total_rows,
    )
    return output


def garman_klass(df: pd.DataFrame) -> pd.Series:
    """
    Compute the Garman-Klass daily variance estimator (Eq. 47):

        sigma_GK^2 = 0.5 * log(High/Low)^2 - (2*log(2)-1) * log(Close/Open)^2

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'open', 'high', 'low', 'close'.

    Returns
    -------
    pd.Series
        GK variance estimates aligned with df index.
    """
    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"])
    gk_var = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    return gk_var.clip(lower=1e-12)


def build_gk_panel(
    ohlcv_dict: dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Build a panel of Garman-Klass volatility estimates aligned on trading dates.

    Parameters
    ----------
    ohlcv_dict : dict of str → pd.DataFrame
        Per-ticker OHLCV DataFrames from fetch_ohlcv.
    start_date : str
        Filter start date.
    end_date : str
        Filter end date.

    Returns
    -------
    pd.DataFrame
        Panel of shape (T, N) where rows are dates and columns are tickers.
        Missing values are NaN.
    """
    gk_series: dict[str, pd.Series] = {}
    for ticker, df in ohlcv_dict.items():
        df = df.loc[start_date:end_date]
        if df.empty:
            continue
        # Validate all required columns present
        if not all(c in df.columns for c in ["open", "high", "low", "close"]):
            logger.warning("Ticker %s missing OHLC columns for GK, skipping.", ticker)
            continue
        gk_series[ticker] = garman_klass(df)

    if not gk_series:
        raise RuntimeError("No GK series could be computed from the OHLCV data.")

    panel = pd.DataFrame(gk_series)
    panel = panel.loc[start_date:end_date]
    return panel


def build_return_panel(
    ohlcv_dict: dict[str, pd.DataFrame],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Build a panel of daily close-to-close log returns aligned on trading dates.

    Parameters
    ----------
    ohlcv_dict : dict of str → pd.DataFrame
        Per-ticker OHLCV DataFrames.
    start_date : str
        Filter start date.
    end_date : str
        Filter end date.

    Returns
    -------
    pd.DataFrame
        Panel of shape (T, N).
    """
    ret_series: dict[str, pd.Series] = {}
    for ticker, df in ohlcv_dict.items():
        df = df.loc[start_date:end_date]
        if df.empty or "close" not in df.columns:
            continue
        ret_series[ticker] = np.log(df["close"]).diff().dropna()

    if not ret_series:
        raise RuntimeError("No return series could be computed.")

    panel = pd.DataFrame(ret_series)
    panel = panel.loc[start_date:end_date]
    return panel


def load_sp500_data_exp5(
    start_date: str,
    end_date: str,
    ticker_pool: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load OHLCV data for S&P 500 stocks and the index for exp5.

    Parameters
    ----------
    start_date : str
        Start date "YYYY-MM-DD".
    end_date : str
        End date "YYYY-MM-DD".
    ticker_pool : list of str
        Pool of S&P 500 tickers to load.

    Returns
    -------
    (stocks_ohlcv, index_ohlcv)
        stocks_ohlcv: dict of ticker → OHLCV DataFrame
        index_ohlcv: DataFrame for S&P 500 index (^GSPC)
    """
    # Load stocks in batches (avoid too-large requests)
    batch_size = 20  # moderate batch to avoid stored-result mode
    stocks_ohlcv: dict[str, pd.DataFrame] = {}
    for i in range(0, len(ticker_pool), batch_size):
        batch = ticker_pool[i : i + batch_size]
        logger.info("Loading tickers batch %d–%d...", i, i + len(batch))
        try:
            batch_data = fetch_ohlcv(batch, start_date, end_date, identity_date=end_date)
            stocks_ohlcv.update(batch_data)
        except Exception as e:
            logger.warning("Batch %d failed: %s", i, e)

    if not stocks_ohlcv:
        raise RuntimeError("No stock data could be loaded for exp5.")

    # Load index
    logger.info("Loading S&P 500 index (%s)...", SP500_INDEX_TICKER)
    index_data = fetch_ohlcv(
        [SP500_INDEX_TICKER], start_date, end_date, identity_date=end_date
    )
    if SP500_INDEX_TICKER not in index_data:
        raise RuntimeError(
            f"S&P 500 index {SP500_INDEX_TICKER} could not be loaded."
        )
    index_ohlcv = index_data[SP500_INDEX_TICKER]
    logger.info(
        "Loaded %d stocks and index. Index rows: %d (%s to %s)",
        len(stocks_ohlcv),
        len(index_ohlcv),
        index_ohlcv.index.min().date(),
        index_ohlcv.index.max().date(),
    )
    return stocks_ohlcv, index_ohlcv


def load_sp500_data_exp6(
    start_date: str,
    end_date: str,
    tickers: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Load OHLCV data for long-tenure S&P 500 stocks and the index for exp6.

    Parameters
    ----------
    start_date : str
        Start date "YYYY-MM-DD".
    end_date : str
        End date "YYYY-MM-DD".
    tickers : list of str
        Long-term S&P 500 constituent tickers.

    Returns
    -------
    (stocks_ohlcv, index_ohlcv)
    """
    batch_size = 10  # keep batches small for 30-year daily series to avoid stored-result mode
    stocks_ohlcv: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        logger.info("Loading tickers batch %d–%d for exp6...", i, i + len(batch))
        try:
            batch_data = fetch_ohlcv(
                batch, start_date, end_date,
                adjust_dividends=True,
                identity_date=end_date,
            )
            stocks_ohlcv.update(batch_data)
        except Exception as e:
            logger.warning("Batch %d failed: %s", i, e)

    if not stocks_ohlcv:
        raise RuntimeError("No stock data could be loaded for exp6.")

    # Load index
    logger.info("Loading S&P 500 index for exp6...")
    index_data = fetch_ohlcv(
        [SP500_INDEX_TICKER], start_date, end_date,
        adjust_dividends=False,  # index prices: use unadjusted for GK
        identity_date=end_date,
    )
    if SP500_INDEX_TICKER not in index_data:
        raise RuntimeError("S&P 500 index not loaded for exp6.")
    index_ohlcv = index_data[SP500_INDEX_TICKER]

    # Filter stocks: keep only those with sufficient coverage (>= 90% of trading days)
    index_dates = index_ohlcv.index
    n_index_days = len(index_dates)
    min_coverage = 0.90 * n_index_days
    filtered: dict[str, pd.DataFrame] = {}
    for ticker, df in stocks_ohlcv.items():
        common = df.index.intersection(index_dates)
        if len(common) >= min_coverage:
            filtered[ticker] = df

    logger.info(
        "exp6: %d stocks after coverage filter (>= %.0f%% of %d index trading days)",
        len(filtered), 90, n_index_days,
    )
    return filtered, index_ohlcv
