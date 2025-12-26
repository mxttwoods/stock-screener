"""
GARP Stock Screener - Two-Stage Implementation
Based on RULES.md and IDEAS.md methodology
"""

import argparse
import json
import logging
import os
import time
import warnings
from datetime import timedelta
from typing import Optional, Union
from functools import wraps

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from yfinance import EquityQuery
from scipy.optimize import minimize
import yaml

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("screener.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Suppress yfinance 404 errors for optional data (insider purchases, sustainability, etc.)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# Load environment variables from .env file
load_dotenv()


# =============================================================================
# CONFIGURATION - Load from config.yaml
# =============================================================================
def load_config():
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config.yaml: {e}")
        raise


CONFIG = load_config()

# Extract thresholds for easy access
MCAP_MIN = CONFIG["market_cap"]["min"]
MCAP_MAX = CONFIG["market_cap"]["max"]

PE_MAX = CONFIG["valuation"]["pe_max"]
PEG_MAX = CONFIG["valuation"]["peg_max"]
PFCF_MAX = CONFIG["valuation"]["pfcf_max"]

# Stage 1 Constants (API expects Percentages)
GROWTH_MIN_PCT = CONFIG["growth"]["revenue_growth_min_pct"]
ROIC_MIN_PCT = CONFIG["profitability"]["roic_min_pct"]
ROE_MIN_PCT = CONFIG["profitability"]["roe_min_pct"]
ROA_MIN_PCT = CONFIG["profitability"]["roa_min_pct"]
GROSS_MARGIN_MIN_PCT = CONFIG["profitability"]["gross_margin_min_pct"]
OPERATING_MARGIN_MIN_PCT = CONFIG["profitability"]["operating_margin_min_pct"]
CURRENT_RATIO_MAX_PCT = CONFIG["balance_sheet"]["current_ratio_max"]
BETA_MAX = CONFIG["beta"]["max"]
BETA_MIN = CONFIG["beta"]["min"]

# Stage 2 Constants (Calculations use Decimals)
GROWTH_MIN = GROWTH_MIN_PCT / 100.0
ROIC_MIN = ROIC_MIN_PCT / 100.0
ROE_MIN = ROE_MIN_PCT / 100.0
ROA_MIN = ROA_MIN_PCT / 100.0
GROSS_MARGIN_MIN = GROSS_MARGIN_MIN_PCT / 100.0
OPERATING_MARGIN_MIN = OPERATING_MARGIN_MIN_PCT / 100.0
CAGR_MIN = CONFIG["growth"]["cagr_min_pct"] / 100.0
FIFTYTWOWK_PERCENTCHANGE_MIN = CONFIG["growth"]["fiftytwowk_percentchange_min_pct"]
FIFTYTWOWK_PERCENTCHANGE_MAX = CONFIG["growth"]["fiftytwowk_percentchange_max_pct"]

DE_MAX = CONFIG["balance_sheet"]["debt_to_equity_max"]
ACCEPTABLE_RATINGS = CONFIG["analyst"]["acceptable_ratings"]
DEFAULT_STAGE1_SIZE = CONFIG.get("stage1_size", 250)

# =============================================================================
# SHARED HELPERS
# =============================================================================
_TICKER_CACHE: dict[str, yf.Ticker] = {}


def safe_divide(
    numerator: Union[float, int], denominator: Union[float, int], default: float = 0.0
) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero or invalid.

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Value to return if division is invalid

    Returns:
        Result of division or default value
    """
    try:
        if denominator is None or pd.isna(denominator) or denominator == 0:
            return default
        if numerator is None or pd.isna(numerator):
            return default
        result = float(numerator) / float(denominator)
        return result if np.isfinite(result) else default
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def safe_get_value(series: pd.Series, key: str, default: float = 0.0) -> float:
    """
    Safely get value from pandas Series with fallback.

    Args:
        series: Pandas Series to search
        key: Key to look up
        default: Default value if key not found or value is invalid

    Returns:
        Value from series or default
    """
    try:
        if key in series.index:
            val = series[key]
            if pd.notna(val) and np.isfinite(val):
                return float(val)
        return default
    except (KeyError, TypeError, ValueError):
        return default


def validate_financial_data(
    value: Optional[Union[float, int]],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> Optional[float]:
    """
    Validate financial data value.

    Args:
        value: Value to validate
        min_val: Minimum allowed value (None = no minimum)
        max_val: Maximum allowed value (None = no maximum)

    Returns:
        Validated value or None if invalid
    """
    if value is None or pd.isna(value):
        return None
    try:
        val = float(value)
        if not np.isfinite(val):
            return None
        if min_val is not None and val < min_val:
            return None
        if max_val is not None and val > max_val:
            return None
        return val
    except (TypeError, ValueError):
        return None


def clean_none_nan(value: Optional[Union[str, float, int]]) -> Optional[str]:
    """
    Clean None/NaN values from strings for display.
    Handles both actual None/NaN and string "nan" from CSV.

    Args:
        value: Value to clean (can be None, float, int, or str)

    Returns:
        Cleaned string or None
    """
    # Handle None
    if value is None:
        return None

    # Handle pandas/NumPy NaN (float('nan'))
    if isinstance(value, (float, int)):
        if pd.isna(value) or not np.isfinite(value):
            return None

    # Check pd.isna for any type (handles float NaN)
    if pd.isna(value):
        return None

    # Convert to string and check
    val_str = str(value).strip()
    val_lower = val_str.lower()

    # Check for various representations of None/NaN
    if val_lower in ["none", "nan", "", "null", "<na>", "n/a", "na", "false", "true"]:
        # "false" and "true" are valid strings, but if it's a boolean field, handle separately
        return None

    # Check if it's a float NaN string representation
    try:
        float_val = float(val_str)
        if pd.isna(float_val) or not np.isfinite(float_val):
            return None
    except (ValueError, TypeError):
        pass

    return val_str


def rate_limit(calls_per_second: float = 2.0):
    """
    Decorator to rate limit function calls.

    Args:
        calls_per_second: Maximum number of calls per second
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret

        return wrapper

    return decorator


def get_ticker(symbol: str) -> yf.Ticker:
    """Return a cached yfinance.Ticker instance for the given symbol."""
    sym = symbol.upper()
    if sym not in _TICKER_CACHE:
        _TICKER_CACHE[sym] = yf.Ticker(sym)
    return _TICKER_CACHE[sym]


# =============================================================================
# INDEX MEMBERSHIP HELPER
# =============================================================================
# Cache for index holdings (fetched once per run)
_INDEX_HOLDINGS_CACHE: dict = {}


def get_index_holdings(index_symbol: str) -> set:
    """
    Fetch top holdings for an index ETF (SPY, QQQ).
    Returns a set of ticker symbols in the index.
    Uses caching to avoid repeated API calls.
    """
    global _INDEX_HOLDINGS_CACHE

    if index_symbol in _INDEX_HOLDINGS_CACHE:
        return _INDEX_HOLDINGS_CACHE[index_symbol]

    try:
        ticker = yf.Ticker(index_symbol)
        funds_data = ticker.funds_data
        if funds_data and hasattr(funds_data, "top_holdings"):
            holdings_df = funds_data.top_holdings
            if holdings_df is not None and not holdings_df.empty:
                # Extract ticker symbols from the index
                symbols = set(holdings_df.index.tolist())
                _INDEX_HOLDINGS_CACHE[index_symbol] = symbols
                return symbols
    except Exception as e:
        print(f"  Warning: Could not fetch {index_symbol} holdings: {e}")

    _INDEX_HOLDINGS_CACHE[index_symbol] = set()
    return set()


def get_index_membership(symbol: str) -> str:
    """
    Check if a stock is in SPY (S&P 500) or QQQ (NASDAQ 100).
    Returns: "SPY", "QQQ", "SPY/QQQ", or "—"
    """
    spy_holdings = get_index_holdings("SPY")
    qqq_holdings = get_index_holdings("QQQ")

    in_spy = symbol in spy_holdings
    in_qqq = symbol in qqq_holdings

    if in_spy and in_qqq:
        return "SPY/QQQ"
    elif in_spy:
        return "SPY"
    elif in_qqq:
        return "QQQ"
    else:
        return "—"


SECTOR_ETF_MAP = {
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
    "Basic Materials": "XLB",
}
_SECTOR_METRICS_CACHE = {}


def get_sector_median_metrics(sector: str) -> dict:
    """
    Calculate median P/E, P/S, PB, and ROE for a given sector.
    Uses a corresponding sector ETF to find peer companies.
    Caches results to avoid redundant calculations.
    """
    global _SECTOR_METRICS_CACHE
    if sector in _SECTOR_METRICS_CACHE:
        return _SECTOR_METRICS_CACHE[sector]

    etf_ticker = SECTOR_ETF_MAP.get(sector)
    if not etf_ticker:
        return {}

    print(f"  Calculating medians for {sector} sector using {etf_ticker} holdings...")
    try:
        etf = yf.Ticker(etf_ticker)
        # Fix: Use funds_data.top_holdings as .holdings is deprecated/unreliable
        peers = []
        if (
            hasattr(etf, "funds_data")
            and etf.funds_data
            and hasattr(etf.funds_data, "top_holdings")
        ):
            # funds_data.top_holdings is a DataFrame with Index as tickers
            if etf.funds_data.top_holdings is not None:
                peers = etf.funds_data.top_holdings.index.tolist()
        elif (
            hasattr(etf, "holdings")
            and etf.holdings is not None
            and not etf.holdings.empty
        ):
            peers = etf.holdings.index.tolist()

        if not peers:
            print(f"  Warning: No peers found for {sector} via {etf_ticker}")
            return {}

        # Get top 20 holdings' tickers
        peer_symbols = peers[:20]
        # print(f"  Found {len(peer_symbols)} peers for {sector}") # Debug

        # Fetch info for all peers at once
        peers = yf.Tickers(peer_symbols)
        metrics = {
            "P/E": [],
            "P/S": [],
            "P/B": [],
            "ROE": [],
        }

        for symbol in peers.tickers:
            try:
                info = symbol.info
                if info.get("trailingPE"):
                    metrics["P/E"].append(info["trailingPE"])
                if info.get("priceToSalesTrailing12Months"):
                    metrics["P/S"].append(info["priceToSalesTrailing12Months"])
                if info.get("priceToBook"):
                    metrics["P/B"].append(info["priceToBook"])
                if info.get("returnOnEquity"):
                    metrics["ROE"].append(info["returnOnEquity"])
            except Exception:
                continue  # Ignore tickers that fail to load

        # Calculate medians, ignoring empty lists
        median_metrics = {}
        for key, values in metrics.items():
            if values:
                median_metrics[f"Sector Median {key}"] = np.nanmedian(
                    [v for v in values if v is not None]
                )

        _SECTOR_METRICS_CACHE[sector] = median_metrics
        return median_metrics

    except Exception as e:
        print(f"  Warning: Could not calculate sector medians for {sector}: {e}")
        return {}


# =============================================================================
# DATA FETCHING HELPER FUNCTIONS (YFINANCE)
# =============================================================================
def fetch_treasury_yield() -> float:
    """
    Fetch the latest 10-year Treasury yield using yfinance (^TNX).
    Returns yield as a decimal (e.g., 0.045 for 4.5%).
    """
    try:
        ticker = yf.Ticker("^TNX")
        # TNX price is the yield multiplied by 10 (e.g. 45.0 = 4.5%)
        # But actually Yahoo quotes it as the rate directly (e.g. 4.5)
        # Let's check history to be safe
        hist = ticker.history(period="5d")
        if not hist.empty:
            latest_yield = hist["Close"].iloc[-1]
            return float(latest_yield) / 100.0
        return 0.045
    except Exception as e:
        print(f"Warning: Could not fetch ^TNX: {e}")
        return 0.045


def fetch_earnings_surprise(ticker: yf.Ticker) -> dict:
    """
    Calculate earnings surprise from yfinance data.
    Tries multiple methods to get earnings surprise data.
    """
    try:
        # Method 1: Try earnings_history (most reliable)
        earnings = ticker.earnings_history
        if earnings is not None and not earnings.empty:
            # Check for different possible column names
            surprise_col = None
            for col in ["Surprise(%)", "Surprise", "Surprise %", "Earnings Surprise"]:
                if col in earnings.columns:
                    surprise_col = col
                    break

            if surprise_col:
                surprises = pd.to_numeric(
                    earnings[surprise_col], errors="coerce"
                ).dropna()

                if not surprises.empty:
                    # Get last 4 quarters (or all available if less than 4)
                    last_4q = surprises.head(4)
                    avg = last_4q.mean()
                    last_surprise = surprises.iloc[0] if len(surprises) > 0 else 0

                    # Determine if values are decimals or percentages
                    # yfinance earnings_history typically returns percentages directly
                    # (e.g., 5.2 means 5.2%, not 0.052)
                    # We use the column name as a hint and check value ranges
                    is_percentage = "%" in surprise_col or surprise_col == "Surprise"

                    # If not explicitly percentage and values look like decimals
                    # (all absolute values < 2), assume decimals and convert
                    max_abs = max(abs(last_4q.max()), abs(last_4q.min()))
                    if not is_percentage and max_abs < 2.0:
                        avg_pct = avg * 100
                        last_pct = last_surprise * 100
                    else:
                        avg_pct = avg
                        last_pct = last_surprise

                    # Sanity check: earnings surprises rarely exceed ±100%
                    if abs(avg_pct) > 500:
                        logger.debug(f"Unusual earnings surprise value: {avg_pct}")

                    return {
                        "Earnings Surprise Avg (%)": round(float(avg_pct), 2),
                        "Last Quarter Surprise (%)": round(float(last_pct), 2),
                    }

        # Method 2: Try quarterly_earnings and calculate manually
        try:
            quarterly_earnings = ticker.quarterly_earnings
            if quarterly_earnings is not None and not quarterly_earnings.empty:
                # If we have quarterly earnings, we can't calculate surprise without estimates
                # But we can at least return that we tried
                pass
        except Exception:
            pass

        # Method 3: Try to get from info dict
        info = ticker.info
        # Some tickers have earningsQuarterlyGrowth which is related
        earnings_growth = info.get("earningsQuarterlyGrowth")
        if earnings_growth is not None:
            # This is growth, not surprise, but better than nothing
            # Convert to percentage if needed
            if abs(earnings_growth) < 1.0:
                earnings_growth = earnings_growth * 100
            return {
                "Earnings Surprise Avg (%)": round(earnings_growth, 2)
                if earnings_growth
                else None,
                "Last Quarter Surprise (%)": round(earnings_growth, 2)
                if earnings_growth
                else None,
            }

        return {"Earnings Surprise Avg (%)": None, "Last Quarter Surprise (%)": None}

    except Exception as e:
        logger.debug(f"Earnings surprise calculation failed: {str(e)}")
        return {"Earnings Surprise Avg (%)": None, "Last Quarter Surprise (%)": None}


def fetch_insider_sentiment(ticker) -> dict:
    """
    Analyze recent insider transactions (last 6 months).
    Returns a sentiment label and net share change.
    """
    try:
        insider = ticker.insider_transactions
        if insider is None or insider.empty:
            return {"Insider Sentiment": "Unknown", "Net Insider Shares": 0}

        # Filter for last 6 months
        six_months_ago = pd.Timestamp.now() - pd.DateOffset(months=6)

        # Ensure 'Date' column is datetime
        if "Start Date" in insider.columns:
            # Clean string dates if necessary
            insider["Start Date"] = pd.to_datetime(
                insider["Start Date"], errors="coerce"
            )
            recent = insider[insider["Start Date"] > six_months_ago]
        elif insider.index.name == "Start Date" or isinstance(
            insider.index, pd.DatetimeIndex
        ):
            # Index might be the date
            pass  # Use recent if index filtering works, but yfinance usually returns a DF with Date column
            # Double check yfinance structure, usually specific cols
            recent = (
                insider  # Fallback if date logic fails for now, or just take top rows
            )
        else:
            # Fallback: just take last 20 rows
            recent = insider.head(20)

        # Calculate net shares purchased/sold
        # yfinance columns often: ['Shares', 'Value', 'Text', 'Transaction', 'Start Date', 'Ownership', 'Indicated']
        # 'Shares' is often just the number involved. Need to check 'Transaction' or 'Text' for Buy/Sell.
        # Common structure: 'Shares' +/- based on type?
        # Actually yfinance data is often messy. Let's look at specific pattern.
        # Simpler proxy: Look at "Net Shares Purchased" if available or infer.

        # Let's try a simpler heuristic for now:
        # Check if there are more "Buy" (or "Purchase") rows than "Sale" rows in top 10.

        buys = 0
        sells = 0
        net_shares = 0

        # 'Text' column usually contains "Sale at price..." or "Purchase at price..."
        # 'Transaction' might be explicit

        for index, row in recent.iterrows():
            text = str(row.get("Text", "")).lower()
            transaction = str(row.get("Transaction", "")).lower()
            shares = row.get("Shares", 0)

            if "purchase" in text or "buy" in text or "purchase" in transaction:
                buys += 1
                net_shares += shares
            elif "sale" in text or "sell" in text or "sale" in transaction:
                sells += 1
                net_shares -= shares

        label = "Neutral"
        if buys > sells and net_shares > 0:
            label = "Positive"
        elif sells > buys and net_shares < 0:
            label = "Negative"

        return {"Insider Sentiment": label, "Net Insider Shares": net_shares}

    except Exception:
        # print(f"DEBUG: Insider error: {e}")
        return {"Insider Sentiment": "Unknown", "Net Insider Shares": 0}


def fetch_institutional_holdings(ticker) -> dict:
    """
    Get top institutional holders and total % held.
    """
    try:
        # ticker.institutional_holders returns a DataFrame with ["Holder", "Shares", "Date Reported", "% Out", "Value"]
        holders = ticker.institutional_holders
        major = ticker.major_holders  # Key stats like "% Held by Institutions"

        top_holder = "N/A"
        pct_held_institutions = 0.0

        if holders is not None and not holders.empty:
            top_holder = holders.iloc[0].get("Holder", "N/A")

        if major is not None and not major.empty:
            # major_holders keys are often indices 0, 1 with values in column 0, 1
            # Row with "% of Float Held by Institutions" or similar
            # It's often a dictionary-like DF: 0: percentage, 1: description
            for index, row in major.iterrows():
                desc = str(row.iloc[1])  # Description column
                val = str(row.iloc[0])  # Value column
                if "Institutions" in desc:
                    # localized string cleans, e.g. "80.50%"
                    try:
                        clean_val = val.replace("%", "")
                        pct_held_institutions = float(clean_val)
                    except ValueError:
                        pass
                    break

        return {
            "Top Institutional Holder": top_holder,
            "Institutional Ownership (%)": pct_held_institutions,
        }

    except Exception:
        return {"Top Institutional Holder": "N/A", "Institutional Ownership (%)": 0.0}


def calculate_quality_of_earnings(ticker) -> dict:
    """
    Quality of Earnings = Operating Cash Flow / Net Income.
    Ratio > 1.0 is healthy (cash confirms profits).
    Ratio < 0.8 is a warning sign.
    """
    try:
        cashflow = ticker.cashflow
        financials = ticker.financials

        if cashflow.empty or financials.empty:
            return {
                "Quality of Earnings Ratio": None,
                "Earnings Quality Concern": False,
            }

        # Get TTM or latest annual
        # cashflow.columns are dates. Column 0 is most recent.

        ocf = None
        net_income = None

        # Try to find Operating Cash Flow
        # Keys vary: "Operating Cash Flow", "Total Cash From Operating Activities"
        possible_ocf_keys = [
            "Operating Cash Flow",
            "Total Cash From Operating Activities",
        ]
        for key in possible_ocf_keys:
            if key in cashflow.index:
                ocf = cashflow.loc[key].iloc[0]  # Most recent
                break

        # Try to find Net Income
        # Keys vary: "Net Income", "Net Income Common Stockholders"
        possible_ni_keys = [
            "Net Income",
            "Net Income Common Stockholders",
            "Net Income From Continuing And Discontinued Operation",
        ]
        for key in possible_ni_keys:
            if key in financials.index:
                net_income = financials.loc[key].iloc[0]
                break

        if ocf is not None and net_income is not None and net_income != 0:
            ratio = ocf / net_income
            concern = ratio < 0.8
            return {
                "Quality of Earnings Ratio": ratio,
                "Earnings Quality Concern": concern,
            }

        return {"Quality of Earnings Ratio": None, "Earnings Quality Concern": False}

    except Exception:
        return {"Quality of Earnings Ratio": None, "Earnings Quality Concern": False}


def fetch_sector_trend(sector: str) -> dict:
    """
    Check if the sector ETF is in an uptrend (Price > SMA200).
    Returns trend status and SMA200 gap.
    """
    # Map sectors to ETFs
    SECTOR_ETFS = {
        "Technology": "XLK",
        "Healthcare": "XLV",
        "Financial Services": "XLF",
        "Consumer Cyclical": "XLY",
        "Communication Services": "XLC",
        "Industrials": "XLI",
        "Consumer Defensive": "XLP",
        "Energy": "XLE",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Basic Materials": "XLB",
    }

    ticker_symbol = SECTOR_ETFS.get(sector)
    if not ticker_symbol:
        return {"Sector Trend": "Unknown", "Sector Ticker": "N/A"}

    try:
        etf = yf.Ticker(ticker_symbol)
        hist = etf.history(period="1y")
        if hist.empty or len(hist) < 200:
            return {"Sector Trend": "Unknown", "Sector Ticker": ticker_symbol}

        sma200 = hist["Close"].rolling(window=200).mean().iloc[-1]
        current_price = hist["Close"].iloc[-1]

        trend = "Uptrend" if current_price > sma200 else "Downtrend"
        gap = ((current_price - sma200) / sma200) * 100

        return {
            "Sector Trend": trend,
            "Sector Ticker": ticker_symbol,
            "Sector SMA200 Gap (%)": round(gap, 2),
        }
    except Exception:
        return {"Sector Trend": "Error", "Sector Ticker": ticker_symbol}


def calculate_peg_ratio(ticker, info: dict) -> dict:
    """
    Calculate PEG Ratio (Price/Earnings-to-Growth).
    Uses Forward P/E and Next 5Y Growth estimates if available.
    """
    try:
        # P/E to use: Forward Preferred, Trailing Fallback
        pe = info.get("forwardPE") or info.get("trailingPE")

        # Growth Rate: Try to get 5Y est, or next year est
        # We need a robust growth number. yfinance often has 'pegRatio' directly.

        # Method 1: Direct from info (most reliable if available)
        if info.get("pegRatio"):
            return {"PEG Ratio": info.get("pegRatio")}

        # Method 2: Manual Calculation
        growth_est = fetch_growth_estimates(ticker).get("Next Year Growth Est (%)")
        if pe and growth_est and growth_est > 0:
            peg = pe / growth_est
            return {"PEG Ratio": round(peg, 2)}

        return {"PEG Ratio": None}
    except Exception:
        return {"PEG Ratio": None}


def fetch_rsi(ticker, period=14, close_series=None) -> dict:
    """
    Calculate 14-day RSI (Relative Strength Index).
    Returns RSI value and signal (Oversold/Overbought).
    """
    try:
        if close_series is not None:
            close = close_series
        else:
            hist = ticker.history(period="3mo")  # Need enough data for 14d + smoothing
            if hist.empty:
                return {"RSI": None, "RSI Signal": "Insufficient Data"}
            close = hist["Close"]

        if len(close) < period + 1:
            return {"RSI": None, "RSI Signal": "Insufficient Data"}

        delta = close.diff()

        # Gain/Loss
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Calculate RS with safety checks
        # When loss is 0, RS is infinite (100% gains) -> RSI = 100
        # When gain is 0, RS is 0 -> RSI = 0
        last_gain = gain.iloc[-1] if len(gain) > 0 else 0
        last_loss = loss.iloc[-1] if len(loss) > 0 else 0

        if pd.isna(last_gain) or pd.isna(last_loss):
            return {"RSI": None, "RSI Signal": "Insufficient Data"}

        if last_loss == 0:
            rsi = 100.0 if last_gain > 0 else 50.0  # All gains or no movement
        elif last_gain == 0:
            rsi = 0.0  # All losses
        else:
            rs = last_gain / last_loss
            rsi = 100 - (100 / (1 + rs))

        signal = "Neutral"
        if rsi < 30:
            signal = "Oversold (Buy Dip)"
        elif rsi > 70:
            signal = "Overbought (Risk)"

        return {"RSI": round(float(rsi), 2), "RSI Signal": signal}
    except Exception:
        return {"RSI": None, "RSI Signal": "Error"}


def fetch_growth_estimates(ticker) -> dict:
    """
    Fetch "Next Year" growth estimates from yfinance.
    Returns dictionary with growth percentage.
    """
    try:
        # ticker.earnings_estimate is a DataFrame with Index=['0q','+1q','0y','+1y']
        # Columns: ['avg', 'low', 'high', 'yearAgoEps', 'numberOfAnalysts', 'growth']
        est = ticker.earnings_estimate
        if est is not None and not est.empty and "+1y" in est.index:
            growth = est.loc["+1y", "growth"]
            return {"Next Year Growth Est (%)": round(growth * 100, 2)}
        return {"Next Year Growth Est (%)": None}
    except Exception:
        return {"Next Year Growth Est (%)": None}


def fetch_eps_revisions(ticker) -> dict:
    """
    Analyze EPS revisions (Up vs Down) for the last 30 days.
    Returns Up/Down ratio and raw counts.
    """
    try:
        # ticker.eps_revisions index=['0q','+1q','0y','+1y']
        # Columns: ['upLast7days', 'upLast30days', 'downLast30days', 'downLast7Days']
        rev = ticker.eps_revisions
        if rev is not None and not rev.empty:
            # Aggregate revisions for current/next quarter/year
            up_30d = rev["upLast30days"].sum()
            down_30d = rev["downLast30days"].sum()

            ratio = up_30d / down_30d if down_30d > 0 else (up_30d if up_30d > 0 else 0)

            return {
                "EPS Revisions Up (30d)": int(up_30d),
                "EPS Revisions Down (30d)": int(down_30d),
                "EPS Up/Down Ratio": round(ratio, 2),
            }
        return {
            "EPS Revisions Up (30d)": 0,
            "EPS Revisions Down (30d)": 0,
            "EPS Up/Down Ratio": 0.0,
        }
    except Exception:
        return {
            "EPS Revisions Up (30d)": 0,
            "EPS Revisions Down (30d)": 0,
            "EPS Up/Down Ratio": 0.0,
        }


def fetch_analyst_recommendations(ticker) -> dict:
    """
    Analyze recent analyst recommendations trend (last 3 months).
    Returns trend signal (Positive/Neutral/Negative).
    """
    try:
        # ticker.recommendations is a DataFrame: period, strongBuy, buy, hold, sell, strongSell
        # Rows usually: 0 (-0m), 1 (-1m), 2 (-2m), 3 (-3m) - backwards in time?
        # Actually in inspection it was 0m, -1m, -2m, -3m
        recs = ticker.recommendations
        if recs is not None and not recs.empty:
            # Current month (row 0 typically, or confirm by 'period' col)
            current = recs.iloc[0]  # 0m
            three_mo_ago = recs.iloc[-1]  # -3m if available, or just verify period

            # Simple summation of Bullish (Strong Buy + Buy) vs Bearish (Sell + Strong Sell)
            # Compare current month vs 3 months ago to see trend

            curr_bullish = current["strongBuy"] + current["buy"]
            prev_bullish = three_mo_ago["strongBuy"] + three_mo_ago["buy"]

            trend = "Neutral"
            if curr_bullish > prev_bullish:
                trend = "Improving"
            elif curr_bullish < prev_bullish:
                trend = "Softening"

            return {"Analyst Trend": trend, "Current Buy Ratings": int(curr_bullish)}

        return {"Analyst Trend": "Unknown", "Current Buy Ratings": 0}
    except Exception:
        return {"Analyst Trend": "Unknown", "Current Buy Ratings": 0}


def fetch_technical_signals(ticker, close_series=None) -> dict:
    """
    Calculate simple technical indicators (SMA50, SMA200) from 1y history.
    """
    try:
        if close_series is not None:
            close = close_series
        else:
            # Get 1 year of history
            hist = ticker.history(period="1y")
            if hist.empty:
                return {
                    "SMA50": None,
                    "SMA200": None,
                    "Technical Signal": "Insufficient Data",
                }
            close = hist["Close"]

        if len(close) < 200:
            return {
                "SMA50": None,
                "SMA200": None,
                "Technical Signal": "Insufficient Data",
            }

        # Calculate SMAs
        sma50 = close.rolling(window=50).mean().iloc[-1]
        sma200 = close.rolling(window=200).mean().iloc[-1]
        current_price = close.iloc[-1]

        signal = "Neutral"
        if sma50 > sma200:
            signal = "Golden Cross (Bullish)"
        elif sma50 < sma200:
            signal = "Death Cross (Bearish)"

        # Price vs SMA trend
        trend = "Uptrend" if current_price > sma200 else "Downtrend"

        return {
            "SMA50": sma50,
            "SMA200": sma200,
            "Technical Signal": signal,
            "Price Trend": trend,
        }
    except Exception:
        return {
            "SMA50": None,
            "SMA200": None,
            "Technical Signal": "Error",
            "Price Trend": "Unknown",
        }


def check_earnings_calendar(ticker) -> dict:
    """
    Check if earnings are upcoming in the next 7 days.
    """
    try:
        cal = ticker.calendar
        # Calendar format varies. Often a dict or DataFrame.
        # usually: {'Earnings Date': [datetime.date(2025, 1, 23)], ...}
        # or DataFrame with columns.

        upcoming = False
        days_to_earnings = None

        if cal is not None:
            # Handle different formats (Dict vs DataFrame)
            dates = []
            if isinstance(cal, dict):
                if "Earnings Date" in cal:
                    dates = cal["Earnings Date"]
                elif "Earnings High" in cal:  # Sometimes structure is different
                    pass
            elif isinstance(cal, pd.DataFrame):
                if "Earnings Date" in cal.index:  # Transposed sometimes
                    dates = cal.loc["Earnings Date"].tolist()
                elif "Earnings Date" in cal.columns:
                    dates = cal["Earnings Date"].tolist()

            # If we found dates, check proximity
            if dates:
                # Flatten if list of lists
                if isinstance(dates[0], list):
                    dates = [item for sublist in dates for item in sublist]

                now = pd.Timestamp.now().date()
                for d in dates:
                    # Convert to date if timestamp
                    if isinstance(d, pd.Timestamp):
                        d = d.date()

                    if d >= now:
                        delta = (d - now).days
                        if delta <= 7:
                            upcoming = True
                            days_to_earnings = delta
                            break

        return {"Upcoming Earnings": upcoming, "Days to Earnings": days_to_earnings}

    except Exception:
        return {"Upcoming Earnings": False, "Days to Earnings": None}


def get_news_data(symbol: str, ticker: yf.Ticker) -> dict:
    """
    Get news data from yfinance (sentiment analysis removed).
    Returns basic news availability info.
    """
    try:
        news = ticker.news
        news_count = len(news) if news else 0
        return {
            "News Sentiment Score": None,
            "News Sentiment Label": None,
            "Recent News Count": news_count,
        }
    except Exception:
        return {
            "News Sentiment Score": None,
            "News Sentiment Label": None,
            "Recent News Count": 0,
        }


# =============================================================================
# INVESTMENT ADVICE FUNCTIONS
# =============================================================================
def calculate_conviction_score(stock_data: dict) -> tuple[int, list[str]]:
    """
    Calculate a conviction score (1-10) based on multiple factors.
    Returns (score, list of reasons).
    """
    score = 5  # Start neutral
    reasons = []

    # Graham Undervalued (+2)
    if stock_data.get("Graham Undervalued"):
        score += 2
        reasons.append("Graham undervalued ✓")

    # Analyst Rating (+2 for strong_buy, +1 for buy)
    rating = stock_data.get("Analyst Rating", "").lower()
    if rating == "strong_buy":
        score += 2
        reasons.append("Strong Buy rating ✓")
    elif rating == "buy":
        score += 1
        reasons.append("Buy rating ✓")

    # High ROIC (+1 if > 20%)
    roic = stock_data.get("ROIC (%)", 0) or 0
    if roic > 20:
        score += 1
        reasons.append(f"High ROIC ({roic:.1f}%) ✓")

    # Earnings Surprise (+1 if > 5%)
    earnings_surprise = stock_data.get("Earnings Surprise Avg (%)", 0) or 0
    if earnings_surprise > 5:
        score += 1
        reasons.append(f"Consistent earnings beats ({earnings_surprise:.1f}%) ✓")

    # Low P/FCF (+1 if < 15)
    pfcf = stock_data.get("P/FCF", 100) or 100
    if pfcf < 15:
        score += 1
        reasons.append(f"Low P/FCF ({pfcf:.1f}) ✓")

    # Strong upside to target (+1 if > 15%)
    upside = stock_data.get("Upside (%)", 0) or 0
    if upside > 15:
        score += 1
        reasons.append(f"High upside ({upside:.1f}%) ✓")

    # Strong revenue growth (+1 if > 15%)
    rev_cagr = stock_data.get("3Y Rev CAGR (%)", 0) or 0
    if rev_cagr > 15:
        score += 1
        reasons.append(f"Strong growth ({rev_cagr:.1f}% CAGR) ✓")

    # NEW: Next Year Growth Estimate (+1 if > 15%)
    next_yr_growth = stock_data.get("Next Year Growth Est (%)", 0) or 0
    if next_yr_growth > 15:
        score += 1
        reasons.append(f"Strong future growth est ({next_yr_growth:.1f}%) ✓")

    # NEW: EPS Revisions (+1 if Ratio > 1.5)
    rev_ratio = stock_data.get("EPS Up/Down Ratio", 0) or 0
    if rev_ratio > 1.5:
        score += 1
        reasons.append(f"Bullish analyst revisions ({rev_ratio}x Up/Down) ✓")

    # NEW: Analyst Trend (+1 if Improving)
    analyst_trend = stock_data.get("Analyst Trend", "Neutral")
    if analyst_trend == "Improving":
        score += 1
        reasons.append("Analyst ratings improving ✓")

    # NEW: Technical Signals (+1 for Golden Cross, +1 for Uptrend)
    tech_signal = stock_data.get("Technical Signal", "Neutral")
    price_trend = stock_data.get("Price Trend", "Neutral")

    if "Golden Cross" in tech_signal:
        score += 1
        reasons.append("Golden Cross (Bullish) ✓")

    if price_trend == "Uptrend":
        score += 1
        reasons.append("Price in Uptrend (> SMA200) ✓")

    # NEW: RSI Oversold (Buy Dip)
    rsi = stock_data.get("RSI", 50) or 50
    if rsi < 30:
        score += 1
        reasons.append(f"Oversold (RSI: {rsi:.0f}) - Buy Dip Opp. ✓")

    # NEW: PEG Ratio (Growth at Reasonable Price)
    peg = stock_data.get("PEG Ratio", 999) or 999
    if peg < 1.5:
        score += 1
        reasons.append(f"Reasonable Growth Val (PEG: {peg}) ✓")

    # NEW: Piotroski F-Score (Quality Metric)
    piotroski_score = stock_data.get("Piotroski Score", 0) or 0
    if piotroski_score >= 8:
        score += 2
        reasons.append(f"Excellent Quality (Piotroski: {piotroski_score}/9) ✓")
    elif piotroski_score >= 6:
        score += 1
        reasons.append(f"Good Quality (Piotroski: {piotroski_score}/9) ✓")
    elif piotroski_score < 4 and piotroski_score > 0:
        score -= 1
        reasons.append(f"Poor Quality (Piotroski: {piotroski_score}/9) ⚠️")

    # NEW: EV/EBITDA (Better valuation for capital-intensive companies)
    ev_ebitda = stock_data.get("EV/EBITDA")
    if ev_ebitda and ev_ebitda < 15:
        score += 1
        reasons.append(f"Attractive EV/EBITDA ({ev_ebitda:.1f}) ✓")
    elif ev_ebitda and ev_ebitda > 25:
        score -= 1
        reasons.append(f"Expensive EV/EBITDA ({ev_ebitda:.1f}) ⚠️")

    # NEW: Due Diligence Factors
    # Financial Distress Risk
    distress_risk = stock_data.get("Financial Distress Risk", "")
    if "Low" in distress_risk or "Safe Zone" in distress_risk:
        score += 1
        z_score = stock_data.get("Altman Z-Score")
        reasons.append(f"Low financial distress risk (Z: {z_score}) ✓")
    elif "High" in distress_risk:
        score -= 2
        reasons.append("High financial distress risk ⚠️")

    # Cash Flow Quality
    cf_quality = stock_data.get("Cash Flow Quality", "")
    if cf_quality == "Good":
        score += 1
        reasons.append("Strong cash flow quality ✓")
    elif cf_quality == "Poor":
        score -= 1
        reasons.append("Poor cash flow quality ⚠️")

    # Accounting Red Flags
    red_flag_count = stock_data.get("Red Flag Count", 0) or 0
    if red_flag_count == 0:
        score += 1
        reasons.append("No accounting red flags ✓")
    elif red_flag_count >= 2:
        score -= 2
        reasons.append(f"Multiple accounting red flags ({red_flag_count}) ⚠️")

    # Valuation Consensus
    val_consensus = stock_data.get("Valuation Consensus", "")
    if val_consensus == "Strong Agreement":
        score += 1
        avg_upside = stock_data.get("Average Upside (%)", 0) or 0
        reasons.append(f"Strong valuation consensus ({avg_upside:.1f}% upside) ✓")
    elif val_consensus == "Divergent Views":
        score -= 1
        reasons.append("Divergent valuation views ⚠️")

    # Penalty for high P/E (-1 if > 30), UNLESS PEG is good
    pe = stock_data.get("P/E", 0) or 0
    if pe > 30 and peg > 2.0:
        score -= 1
        reasons.append(f"High P/E ({pe:.1f}) ⚠️")

    # Penalty for expensive valuation or leverage
    pfcf_penalty = stock_data.get("P/FCF")
    if pfcf_penalty is not None and pfcf_penalty > PFCF_MAX:
        score -= 1
        reasons.append(f"Rich P/FCF ({pfcf_penalty:.1f}) ⚠️")

    if peg and peg > 2.5:
        score -= 1
        reasons.append(f"High PEG ({peg}) ⚠️")

    leverage = stock_data.get("D/E", 0) or 0
    leverage_limit = DE_MAX * 100  # Aligns with Stage 1/2 scaling
    if leverage and leverage > leverage_limit:
        score -= 1
        reasons.append(f"High leverage (D/E: {leverage:.1f}) ⚠️")

    # --- NEW: Enhanced yfinance Data Metrics ---

    # Short Interest Analysis
    short_pct = stock_data.get("Short % of Float")
    short_signal = stock_data.get("Short Interest Signal")
    if short_pct and short_pct > 20:
        score -= 1
        reasons.append(f"High short interest ({short_pct:.1f}%) ⚠️")
    elif short_signal == "Short Covering Rally Potential":
        score += 1
        reasons.append("Short covering potential ✓")

    # Analyst Momentum (Upgrades/Downgrades)
    analyst_momentum = stock_data.get("Analyst Momentum")
    if analyst_momentum == "Strong Positive":
        score += 2
        reasons.append("Strong analyst upgrade momentum ✓")
    elif analyst_momentum == "Positive":
        score += 1
        reasons.append("Positive analyst momentum ✓")
    elif analyst_momentum == "Strong Negative":
        score -= 2
        reasons.append("Strong analyst downgrade trend ⚠️")
    elif analyst_momentum == "Negative":
        score -= 1
        reasons.append("Negative analyst momentum ⚠️")

    # Target Price Analysis
    target_upside = stock_data.get("Target Upside (%)")
    analyst_agreement = stock_data.get("Analyst Agreement")
    if target_upside and target_upside > 25 and analyst_agreement == "High":
        score += 2
        reasons.append(f"Strong target upside w/ consensus ({target_upside:.1f}%) ✓")
    elif target_upside and target_upside > 20:
        score += 1
        reasons.append(f"Good target upside ({target_upside:.1f}%) ✓")
    elif target_upside and target_upside < -10:
        score -= 1
        reasons.append(f"Below analyst targets ({target_upside:.1f}%) ⚠️")

    # Insider Ownership (Skin in Game)
    insider_pct = stock_data.get("Insider Ownership (%)")
    insider_signal = stock_data.get("Insider Signal")
    if insider_signal == "Very High (Founder-led)":
        score += 1
        reasons.append(f"Strong insider alignment ({insider_pct:.1f}%) ✓")
    elif insider_signal == "Low Skin in Game" and insider_pct is not None:
        score -= 1
        reasons.append(f"Low insider ownership ({insider_pct:.1f}%) ⚠️")

    # Recent Insider Purchases
    insider_purchases = stock_data.get("Recent Insider Purchases", 0) or 0
    if insider_purchases > 0:
        score += 1
        reasons.append(f"Recent insider buying ({insider_purchases:,} shares) ✓")

    # Dividend Quality (for dividend payers)
    div_consistency = stock_data.get("Dividend Consistency")
    div_cagr = stock_data.get("Dividend Growth CAGR (%)")
    if div_consistency == "Excellent (5+ Years Increases)":
        score += 1
        reasons.append("Excellent dividend track record ✓")
    if stock_data.get("Dividend Aristocrat"):
        score += 1
        reasons.append("Dividend Aristocrat (25+ years) ✓")
    if div_cagr and div_cagr > 10:
        score += 1
        reasons.append(f"Strong dividend growth ({div_cagr:.1f}% CAGR) ✓")

    # ESG/Governance
    esg_risk = stock_data.get("ESG Risk Level")
    gov_score = stock_data.get("Governance Score")
    if esg_risk in ["Negligible Risk", "Low Risk"]:
        score += 1
        reasons.append(f"Low ESG risk ({esg_risk}) ✓")
    elif esg_risk in ["High Risk", "Severe Risk"]:
        score -= 1
        reasons.append(f"High ESG risk ({esg_risk}) ⚠️")

    # Analyst Sentiment Score
    sentiment_score = stock_data.get("Analyst Sentiment Score")
    sentiment_label = stock_data.get("Analyst Sentiment")
    if sentiment_score and sentiment_score >= 1.5:
        score += 1
        reasons.append(f"Very bullish analyst sentiment ({sentiment_label}) ✓")
    elif sentiment_score and sentiment_score <= -0.5:
        score -= 1
        reasons.append(f"Bearish analyst sentiment ({sentiment_label}) ⚠️")

    # Options Implied Volatility
    iv_signal = stock_data.get("IV Signal")
    put_call_ratio = stock_data.get("Put/Call OI Ratio")
    if iv_signal == "Very High Volatility Expected":
        reasons.append("High implied volatility (elevated risk) ⚠️")
    if put_call_ratio and put_call_ratio > 1.5:
        score -= 1
        reasons.append(f"Bearish options positioning (P/C: {put_call_ratio:.2f}) ⚠️")
    elif put_call_ratio and put_call_ratio < 0.5:
        score += 1
        reasons.append(f"Bullish options positioning (P/C: {put_call_ratio:.2f}) ✓")

    # 52-Week Position
    week_52_signal = stock_data.get("52-Week Position Signal")
    if week_52_signal == "Near 52-Week High (Momentum)":
        score += 1
        reasons.append("Strong momentum (near 52W high) ✓")
    elif week_52_signal == "Near 52-Week Low (Contrarian)":
        reasons.append("Contrarian opportunity (near 52W low)")

    # FCF Yield
    fcf_yield = stock_data.get("FCF Yield (%)")
    yield_spread = stock_data.get("Yield Spread vs T-Bill")
    if fcf_yield and fcf_yield > 8:
        score += 1
        reasons.append(f"High FCF yield ({fcf_yield:.1f}%) ✓")
    if yield_spread and yield_spread > 4:
        score += 1
        reasons.append(f"Attractive yield spread (+{yield_spread:.1f}% vs T-Bill) ✓")

    # Growth Quality
    growth_quality_score = stock_data.get("Growth Quality Score")
    margin_trend = stock_data.get("Margin Trend")
    if growth_quality_score and growth_quality_score >= 3:
        score += 2
        reasons.append("Excellent growth quality (3/3) ✓")
    elif growth_quality_score and growth_quality_score >= 2:
        score += 1
        reasons.append("Good growth quality (2/3) ✓")
    if margin_trend == "Compressing":
        score -= 1
        reasons.append("Margin compression ⚠️")

    # Cap score at 1-10
    score = max(1, min(10, score))

    # 6. ML Alpha Score (Return Prediction)
    # Score is predicted 20-day return %.
    # > 2% (approx 25% annualized) is bullish. > 5% very bullish. < 0% bearish.

    # Need to handle if "ML Alpha Score" key is missing or None
    ml_score = stock_data.get("ML Alpha Score", 0.0)

    if ml_score > 2.0:
        score += 5
        reasons.append(f"ML Forecast Bullish (+{ml_score:.1f}%)")
    if ml_score > 5.0:
        score += 5
        reasons.append("ML Strong Conviction")
    elif ml_score < 0.0:
        score -= 5
        reasons.append(f"ML Forecast Bearish ({ml_score:.1f}%)")

    return min(100, max(0, score)), reasons


def get_risk_warnings(stock_data: dict, ticker_info: dict) -> list[str]:
    """
    Generate risk warnings for a stock.
    """
    warnings = []

    # High Beta (> 1.5)
    beta = ticker_info.get("beta", 0) or 0
    if beta > 1.5:
        warnings.append(f"⚠️ High volatility (Beta: {beta:.2f})")

    # High D/E aligned with config max
    de = stock_data.get("D/E", 0) or 0
    if de > DE_MAX * 100:
        warnings.append(f"⚠️ High leverage (D/E: {de:.1f})")

    # Negative Graham Upside (significantly overvalued)
    graham_undervalued = (
        str(stock_data.get("Graham Undervalued", "False")).lower() == "true"
    )
    peg = stock_data.get("PEG Ratio", 999) or 999

    if graham_undervalued:
        pass  # No warning
    elif peg < 2.0:
        pass  # No warning (Justified by growth)
    else:
        warnings.append("⚠️ Significantly above Graham value")

    # Low analyst coverage
    num_analysts = stock_data.get("# Analysts", 0) or 0
    if num_analysts < 5:
        warnings.append(f"⚠️ Low analyst coverage ({num_analysts} analysts)")

    # Technical Warning (Death Cross)
    tech_signal = stock_data.get("Technical Signal", "Neutral")
    if "Death Cross" in tech_signal:
        warnings.append("⚠️ Death Cross (Bearish Trend)")

    # Upcoming Earnings Warning
    if stock_data.get("Upcoming Earnings", False):
        days = stock_data.get("Days to Earnings")
        warnings.append(f"⚠️ Earnings in {days} days (High Volatility)")

    # Technical Warning (RSI Overbought)
    rsi = stock_data.get("RSI", 50) or 50
    if rsi > 70:
        warnings.append(f"⚠️ Overbought (RSI: {rsi:.0f})")

    # Sector Trend Warning
    if stock_data.get("Sector Trend") == "Downtrend":
        gap = stock_data.get("Sector SMA200 Gap (%)")
        warnings.append(f"⚠️ Sector Downtrend (Gap: {gap}%)")

    # Bearish Sentiment
    sentiment_label = str(stock_data.get("News Sentiment Label", "")).lower()
    if "bearish" in sentiment_label and sentiment_label != "nan":
        warnings.append(f"⚠️ Negative news sentiment ({sentiment_label})")

    # NEW: Financial Distress Risk (Altman Z-Score)
    distress_risk = stock_data.get("Financial Distress Risk", "")
    if "High" in distress_risk:
        z_score = stock_data.get("Altman Z-Score")
        warnings.append(f"⚠️ High financial distress risk (Z-Score: {z_score})")

    # NEW: Cash Flow Quality
    cf_quality = stock_data.get("Cash Flow Quality", "")
    if cf_quality == "Poor":
        warnings.append("⚠️ Poor cash flow quality")
    cf_concerns = clean_none_nan(stock_data.get("Cash Flow Concerns"))
    if cf_concerns:
        warnings.append(f"⚠️ Cash flow concerns: {cf_concerns}")

    # NEW: Accounting Red Flags
    red_flag_count = stock_data.get("Red Flag Count", 0) or 0
    if red_flag_count > 0:
        flags = clean_none_nan(stock_data.get("Accounting Red Flags"))
        if flags:
            warnings.append(
                f"⚠️ Accounting red flags detected ({red_flag_count}): {flags}"
            )

    # NEW: Debt Maturity Risk
    debt_risk = stock_data.get("Debt Maturity Risk", "")
    if debt_risk == "High":
        warnings.append("⚠️ High debt maturity/refinancing risk")
    debt_concerns = clean_none_nan(stock_data.get("Debt Concerns"))
    if debt_concerns:
        warnings.append(f"⚠️ Debt concerns: {debt_concerns}")

    # NEW: Dividend Sustainability (for dividend payers only)
    div_paying = stock_data.get("Dividend Paying", False)
    if div_paying:  # Only check if stock actually pays dividends
        div_sustainability = stock_data.get("Dividend Sustainability", "")
        if div_sustainability == "At Risk":
            warnings.append("⚠️ Dividend sustainability at risk")
        div_concerns = clean_none_nan(stock_data.get("Dividend Concerns"))
        if div_concerns:
            warnings.append(f"⚠️ Dividend concerns: {div_concerns}")

    # NEW: Valuation Consensus
    val_consensus = stock_data.get("Valuation Consensus", "")
    if val_consensus == "Divergent Views":
        warnings.append("⚠️ Divergent valuation views (methods disagree)")

    return warnings


def classify_action(conviction: int, graham_undervalued: bool, upside: float) -> str:
    """
    Classify stock as BUY NOW, WATCHLIST, or HOLD based on metrics.
    """
    if conviction >= 8 and graham_undervalued:
        return "🟢 BUY NOW"
    elif conviction >= 7 or (conviction >= 6 and upside > 10):
        return "🟡 BUY"
    elif conviction >= 5:
        return "🔵 WATCHLIST"
    else:
        return "⚪ HOLD"


def generate_investment_summary(
    stock_data: dict, ticker_info: dict, conviction_score: int, reasons: list
) -> str:
    """
    Generate a structured investment summary based on metrics (no AI).
    Returns a formatted string with key signals and action.
    """
    symbol = stock_data.get("Symbol", "Unknown")

    # Collect bullish and warning signals
    bullish = [r for r in reasons if "✓" in r]
    warnings = [r for r in reasons if "⚠️" in r]

    # Build summary
    summary_parts = []

    # Action based on conviction
    if conviction_score >= 70:
        action = "STRONG BUY"
    elif conviction_score >= 55:
        action = "BUY"
    elif conviction_score >= 40:
        action = "HOLD"
    else:
        action = "WATCH"

    summary_parts.append(f"[{action}] Conviction: {conviction_score}/100")

    # Key valuation info
    dcf_upside = stock_data.get("DCF Upside (%)")
    graham_undervalued = stock_data.get("Graham Undervalued")
    if dcf_upside is not None:
        summary_parts.append(f"DCF Upside: {dcf_upside:.1f}%")
    if graham_undervalued:
        summary_parts.append("Graham Undervalued")

    # Top bullish signals (max 3)
    if bullish:
        top_bullish = bullish[:3]
        summary_parts.append(f"Bullish: {', '.join(s.replace(' ✓', '') for s in top_bullish)}")

    # Top warnings (max 2)
    if warnings:
        top_warnings = warnings[:2]
        summary_parts.append(f"Risks: {', '.join(s.replace(' ⚠️', '') for s in top_warnings)}")

    return " | ".join(summary_parts)


def backtest_portfolio(tickers, weights) -> None:
    """
    Backtest the selected portfolio vs SPY for the last 1 year.
    Prints metrics and saves to CSV.
    """
    print("\n" + "=" * 60)
    print("BACKTESTING: Portfolio vs SPY (1 Year)")
    print("=" * 60)

    try:
        if not tickers:
            print("  No tickers to backtest.")
            return

        # Download data
        # Download data (Force auto_adjust=True for Total Return)
        all_symbols = tickers + ["SPY"]
        # print(f"  Downloading adjusted daily data for {len(all_symbols)} symbols...")
        data = yf.download(
            all_symbols, period="1y", interval="1d", progress=False, auto_adjust=True
        )

        # Handle MultiIndex columns (yfinance structure changes often)
        # If auto_adjust=True, usually we just get "Close" which is adjusted.
        if "Close" in data.columns and isinstance(data.columns, pd.MultiIndex):
            data = data["Close"]
        elif "Adj Close" in data.columns:
            data = data["Adj Close"]
        elif "Close" in data.columns:
            data = data["Close"]  # Fallback

        if data.empty:
            print("  Error: No historical data found.")
            return

        # print(f"  Data Shape: {data.shape} | Date Range: {data.index.min().date()} to {data.index.max().date()}")

        # Calculate Returns
        # Drop initial NaNs but be careful not to drop whole rows if just one stock is missing a day
        # Ideally we forward fill prices first?
        data = data.ffill().dropna()
        if data.empty:
            print(
                "  Error: Data empty after cleaning (check if tickers have overlapping history)."
            )
            return

        returns = data.pct_change().dropna()

        # Portfolio Returns
        # weight_map logic...
        sorted_tickers = [t for t in all_symbols if t != "SPY" and t in data.columns]
        weight_map = dict(zip(tickers, weights))

        # Check for missing tickers
        missing = set(tickers) - set(data.columns)
        if missing:
            print(
                f"  Warning: Missing data for {missing}. Weights will be re-normalized."
            )

        # Calculate weighted daily returns (Daily Rebalancing Assumption)
        # Verify: this assumes rebalancing to target weights every day.
        # For small deviations this tracks Buy & Hold well enough for checking.
        port_daily_ret = pd.Series(0.0, index=returns.index)
        total_weight = 0
        for t in sorted_tickers:
            w = weight_map.get(t, 0)
            port_daily_ret += returns[t] * w
            total_weight += w

        # Normalize if weights don't sum to 1 (due to missing data)
        if total_weight > 0:
            port_daily_ret = port_daily_ret / total_weight

        # Buy & Hold Return (No Rebalancing)
        # Start Value = 1.0. End Value = Sum(Weight * (Price_End / Price_Start))
        buy_hold_ret = 0.0
        for t in sorted_tickers:
            w = weight_map.get(t, 0)
            if t in data.columns:
                # Total return for this asset
                asset_ret = (data[t].iloc[-1] / data[t].iloc[0]) - 1
                buy_hold_ret += w * (1 + asset_ret)

        buy_hold_total_ret = (buy_hold_ret - 1) * 100

        spy_daily_ret = returns["SPY"]

        # Cumulative Returns
        port_cum_ret = (1 + port_daily_ret).cumprod()
        spy_cum_ret = (1 + spy_daily_ret).cumprod()

        total_ret_port = (port_cum_ret.iloc[-1] - 1) * 100
        total_ret_spy = (spy_cum_ret.iloc[-1] - 1) * 100

        # Max Drawdown
        # Roll max
        roll_max_port = port_cum_ret.cummax()
        drawdown_port = (port_cum_ret / roll_max_port) - 1
        max_dd_port = drawdown_port.min() * 100

        roll_max_spy = spy_cum_ret.cummax()
        drawdown_spy = (spy_cum_ret / roll_max_spy) - 1
        max_dd_spy = drawdown_spy.min() * 100

        # Sharpe (Annualized) - assuming 252 days
        sharpe_port = (port_daily_ret.mean() / port_daily_ret.std()) * np.sqrt(252)
        sharpe_spy = (spy_daily_ret.mean() / spy_daily_ret.std()) * np.sqrt(252)

        # Sortino Ratio (Annualized)
        target_return = 0
        downside_returns_port = port_daily_ret[port_daily_ret < target_return]
        downside_std_port = downside_returns_port.std()
        sortino_port = (
            (port_daily_ret.mean() / downside_std_port) * np.sqrt(252)
            if downside_std_port > 0
            else None
        )

        downside_returns_spy = spy_daily_ret[spy_daily_ret < target_return]
        downside_std_spy = downside_returns_spy.std()
        sortino_spy = (
            (spy_daily_ret.mean() / downside_std_spy) * np.sqrt(252)
            if downside_std_spy > 0
            else None
        )

        print(f"  Portfolio Return (1Y): {total_ret_port:.2f}% (Daily Rebalance)")
        print(f"  Buy & Hold Return:     {buy_hold_total_ret:.2f}%")
        print(f"  SPY Return (1Y):       {total_ret_spy:.2f}%")
        print(f"  Alpha (vs SPY):        {total_ret_port - total_ret_spy:.2f}%")
        print("-" * 40)
        print(f"  Portfolio Max DD:      {max_dd_port:.2f}%")
        print(f"  SPY Max DD:            {max_dd_spy:.2f}%")
        print(f"  Portfolio Sharpe:      {sharpe_port:.2f}")
        print(f"  SPY Sharpe:            {sharpe_spy:.2f}")
        if sortino_port is not None:
            print(f"  Portfolio Sortino:     {sortino_port:.2f}")
        if sortino_spy is not None:
            print(f"  SPY Sortino:           {sortino_spy:.2f}")

        # Value at Risk (VaR) and Conditional VaR
        try:
            var_95_port = (
                np.percentile(port_daily_ret, 5) * 100
            )  # 95% VaR (5th percentile)
            cvar_95_port = (
                port_daily_ret[
                    port_daily_ret <= np.percentile(port_daily_ret, 5)
                ].mean()
                * 100
            )
            var_95_spy = np.percentile(spy_daily_ret, 5) * 100
            cvar_95_spy = (
                spy_daily_ret[spy_daily_ret <= np.percentile(spy_daily_ret, 5)].mean()
                * 100
            )

            # Annualize VaR
            var_95_port_annual = var_95_port * np.sqrt(252)
            var_95_spy_annual = var_95_spy * np.sqrt(252)

            print("-" * 40)
            print("  Risk Metrics (VaR):")
            print(
                f"  Portfolio VaR (95%):   {var_95_port:.2f}% daily ({var_95_port_annual:.2f}% annualized)"
            )
            print(f"  Portfolio CVaR (95%):   {cvar_95_port:.2f}% daily")
            print(
                f"  SPY VaR (95%):         {var_95_spy:.2f}% daily ({var_95_spy_annual:.2f}% annualized)"
            )
            print(f"  SPY CVaR (95%):        {cvar_95_spy:.2f}% daily")
        except Exception:
            pass  # Skip VaR if calculation fails

        # Save results
        results_df = pd.DataFrame(
            {
                "Date": port_cum_ret.index,
                "Portfolio Value": port_cum_ret,
                "SPY Value": spy_cum_ret,
            }
        )
        results_df.to_csv("backtest_results.csv")
        print("\n  Backtest data saved to: backtest_results.csv")

    except Exception as e:
        print(f"  Error during backtest: {e}")


def generate_stock_summaries(passed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate structured investment summaries for all passing stocks.
    """
    print("\n" + "=" * 60)
    print("INVESTMENT SUMMARY - Metrics-Based Analysis")
    print("=" * 60)

    if passed_df.empty:
        print("No stocks to analyze.")
        return passed_df

    advice_data = []

    for _, row in passed_df.iterrows():
        stock_data = row.to_dict()

        # Clean NaN values that may have come from CSV (pandas converts None to float('nan'))
        cleaned_data = {}
        for key, value in stock_data.items():
            # Handle pandas NaN (float('nan')) - this is what empty CSV cells become
            if pd.isna(value):
                cleaned_data[key] = None
            elif isinstance(value, (float, int)) and (
                pd.isna(value) or not np.isfinite(value)
            ):
                cleaned_data[key] = None
            elif isinstance(value, str):
                val_lower = value.lower().strip()
                if val_lower in ["nan", "none", "", "null", "n/a", "na"]:
                    cleaned_data[key] = None
                elif val_lower == "false" and (
                    "Dividend Paying" in key
                    or "Graham Undervalued" in key
                    or "Stage 2 Pass" in key
                ):
                    cleaned_data[key] = False
                elif val_lower == "true" and (
                    "Dividend Paying" in key
                    or "Graham Undervalued" in key
                    or "Stage 2 Pass" in key
                ):
                    cleaned_data[key] = True
                else:
                    cleaned_data[key] = value
            else:
                cleaned_data[key] = value
        stock_data = cleaned_data

        symbol = stock_data.get("Symbol", "Unknown")

        print(f"  Generating advice for {symbol}...", end=" ")

        # Get additional ticker info for risk warnings
        try:
            ticker = get_ticker(symbol)
            ticker_info = ticker.info
        except Exception as e:
            print(f"ERROR getting ticker info for {symbol}: {e}")
            ticker_info = {}

        # Use pre-calculated conviction score
        conviction = stock_data.get("Conviction Score", 5)
        conviction_reasons = stock_data.get("Conviction Reasons", "").split("; ")

        # Get risk warnings
        risks = get_risk_warnings(stock_data, ticker_info)

        # Classify action
        action = classify_action(
            conviction,
            stock_data.get("Graham Undervalued", False),
            stock_data.get("Upside (%)", 0) or 0,
        )

        # Generate investment summary
        summary = generate_investment_summary(stock_data, ticker_info, conviction, conviction_reasons)

        # Create advice row preserving all original data
        advice_row = stock_data.copy()
        advice_row.update(
            {
                "Action": action,
                "Conviction": conviction,
                "Conviction Reasons": "; ".join(conviction_reasons),
                "Risk Warnings": "; ".join(
                    [r for r in risks if r and "nan" not in r.lower()]
                )
                if risks
                else "None",
                "Investment Summary": summary,
            }
        )

        advice_data.append(advice_row)

        print(f"{action} (Conviction: {conviction}/100)")

    advice_df = pd.DataFrame(advice_data)

    # Display summary
    print("\n" + "-" * 60)
    print("INVESTMENT RECOMMENDATIONS SUMMARY")
    print("-" * 60)

    for _, row in advice_df.iterrows():
        print(f"\n{row['Symbol']} - {row['Name']}")
        print(f"  Action: {row['Action']} | Conviction: {row['Conviction']}/100")
        print(f"  Summary: {row['Investment Summary']}")
        if row["Risk Warnings"] != "None":
            print(f"  Risks: {row['Risk Warnings']}")

    # Export summary
    advice_df.to_csv("investment_summary.csv", index=False)
    print("\n\nInvestment summary exported to: investment_summary.csv")

    return advice_df


# =============================================================================
# DCF VALUATION MODEL
# =============================================================================
def calculate_dcf_fair_value(
    ticker: yf.Ticker, info: dict, risk_free_rate: float = 0.045
) -> dict:
    """
    Calculate DCF (Discounted Cash Flow) intrinsic value per share.
    This version uses a more robust, company-specific WACC.
    """
    try:
        # Get cash flow statement
        cf = ticker.cash_flow
        if cf.empty:
            return {
                "DCF Fair Value": None,
                "DCF Upside (%)": None,
                "DCF Notes": "No cash flow data",
            }

        # Get Free Cash Flow - try multiple possible key names
        operating_cf = None
        possible_ocf_keys = [
            "Total Cash From Operating Activities",
            "Operating Cash Flow",
            "Cash From Operating Activities",
        ]
        for key in possible_ocf_keys:
            if key in cf.index:
                operating_cf = cf.loc[key].dropna().tolist()
                break

        capex = None
        possible_capex_keys = [
            "Capital Expenditure",
            "Capital Expenditures",
            "Capital Spending",
        ]
        for key in possible_capex_keys:
            if key in cf.index:
                capex = cf.loc[key].dropna().tolist()
                break

        if not operating_cf or not capex or len(operating_cf) < 2:
            return {
                "DCF Fair Value": None,
                "DCF Upside (%)": None,
                "DCF Notes": "Insufficient FCF data",
            }

        current_fcf = operating_cf[0] + capex[0]
        if current_fcf <= 0:
            return {
                "DCF Fair Value": None,
                "DCF Upside (%)": None,
                "DCF Notes": "Negative FCF",
            }

        # FCF growth rate
        if len(operating_cf) >= 3:
            older_fcf = operating_cf[2] + (capex[2] if len(capex) > 2 else 0)
            if older_fcf > 0:
                ratio = safe_divide(current_fcf, older_fcf, default=1.0)
                fcf_growth = (ratio ** (1 / 2)) - 1
                fcf_growth = max(0.03, min(0.15, fcf_growth))  # Cap growth at 3-15%
            else:
                fcf_growth = 0.05  # Default 5%
        else:
            fcf_growth = 0.05  # Default 5%

        # --- Enhanced WACC Calculation ---
        market_cap = info.get("marketCap", 0)
        total_debt = info.get("totalDebt", 0)
        total_capital = market_cap + total_debt
        if total_capital == 0:
            return {
                "DCF Fair Value": None,
                "DCF Upside (%)": None,
                "DCF Notes": "Missing capital structure",
            }

        equity_weight = market_cap / total_capital
        debt_weight = total_debt / total_capital

        beta = info.get("beta", 1.0) or 1.0
        market_premium = 0.055
        cost_of_equity = risk_free_rate + beta * market_premium

        # Estimate cost of debt (e.g., interest expense / total debt)
        income_stmt = ticker.financials
        interest_expense = 0
        if not income_stmt.empty:
            possible_interest_keys = [
                "Interest Expense",
                "Interest And Debt Expense",
                "Total Interest Expense",
            ]
            for key in possible_interest_keys:
                if key in income_stmt.index:
                    try:
                        interest_expense = income_stmt.loc[key].iloc[0] or 0
                        break
                    except (IndexError, KeyError):
                        continue
        cost_of_debt = abs(interest_expense) / total_debt if total_debt > 0 else 0.05
        cost_of_debt = max(0.04, min(cost_of_debt, 0.09))  # Bound cost of debt

        tax_rate = 0.21  # Default tax rate
        if not income_stmt.empty:
            tax_provision = None
            pretax_income = None

            # Try to find tax provision
            for key in ["Tax Provision", "Income Tax Expense", "Taxes"]:
                if key in income_stmt.index:
                    try:
                        tax_provision = income_stmt.loc[key].iloc[0]
                        break
                    except (IndexError, KeyError):
                        continue

            # Try to find pretax income
            for key in ["Pretax Income", "Income Before Tax", "Earnings Before Tax"]:
                if key in income_stmt.index:
                    try:
                        pretax_income = income_stmt.loc[key].iloc[0]
                        break
                    except (IndexError, KeyError):
                        continue

            if (
                tax_provision is not None
                and pretax_income is not None
                and pretax_income > 0
            ):
                tax_rate = safe_divide(tax_provision, pretax_income, default=0.21)
                tax_rate = max(0.0, min(0.35, tax_rate))  # Bound between 0-35%

        wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (
            1 - tax_rate
        )
        wacc = max(0.07, min(0.12, wacc))  # Cap WACC at 7-12% for stability

        # --- DCF Projection ---
        projection_years = 5
        terminal_growth = 0.025
        projected_fcf = []
        fcf = current_fcf
        growth_decay = 0.9
        for _ in range(1, projection_years + 1):
            fcf *= 1 + fcf_growth
            fcf_growth *= growth_decay
            projected_fcf.append(fcf)

        pv_fcf = sum(
            fcf / ((1 + wacc) ** (i + 1)) for i, fcf in enumerate(projected_fcf)
        )

        terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
        # Guard against division by zero when WACC ≈ terminal_growth
        if wacc <= terminal_growth + 0.01:
            return {
                "DCF Fair Value": None,
                "DCF Upside (%)": None,
                "DCF Notes": "WACC too close to terminal growth rate",
            }
        terminal_value = terminal_fcf / (wacc - terminal_growth)
        pv_terminal = terminal_value / ((1 + wacc) ** projection_years)

        enterprise_value = pv_fcf + pv_terminal
        net_debt = total_debt - info.get("totalCash", 0)
        equity_value = enterprise_value - net_debt

        shares_outstanding = info.get("sharesOutstanding")
        if not shares_outstanding:
            return {"DCF Fair Value": None, "DCF Notes": "No shares data"}

        dcf_fair_value = equity_value / shares_outstanding
        if dcf_fair_value <= 0:
            return {"DCF Fair Value": None, "DCF Notes": "Negative equity value"}

        current_price = info.get("currentPrice")
        dcf_upside = (
            ((dcf_fair_value - current_price) / current_price) * 100
            if current_price
            else None
        )

        # --- Scenario Analysis ---
        # Helper function to calculate DCF value for a given growth/WACC
        def calc_dcf_value(base_fcf, growth_rate, discount_rate, term_growth, years, decay):
            proj_fcf = []
            fcf_temp = base_fcf
            gr = growth_rate
            for _ in range(years):
                fcf_temp *= 1 + gr
                gr *= decay
                proj_fcf.append(fcf_temp)

            pv = sum(f / ((1 + discount_rate) ** (i + 1)) for i, f in enumerate(proj_fcf))

            if discount_rate <= term_growth + 0.01:
                return None
            term_fcf = proj_fcf[-1] * (1 + term_growth)
            term_val = term_fcf / (discount_rate - term_growth)
            pv_term = term_val / ((1 + discount_rate) ** years)

            ev = pv + pv_term
            eq_val = ev - net_debt
            return eq_val / shares_outstanding if shares_outstanding else None

        # Base case (current calculation)
        base_value = dcf_fair_value

        # Bull case: Growth +2%, WACC -1%
        bull_growth = min(0.20, fcf_growth + 0.02)  # Recalculate from original
        bull_wacc = max(0.06, wacc - 0.01)
        bull_value = calc_dcf_value(current_fcf, bull_growth, bull_wacc, terminal_growth, projection_years, growth_decay)

        # Bear case: Growth -2%, WACC +1%
        bear_growth = max(0.01, fcf_growth - 0.02)
        bear_wacc = min(0.15, wacc + 0.01)
        bear_value = calc_dcf_value(current_fcf, bear_growth, bear_wacc, terminal_growth, projection_years, growth_decay)

        # Calculate upsides for each scenario
        bull_upside = ((bull_value - current_price) / current_price * 100) if bull_value and current_price else None
        bear_upside = ((bear_value - current_price) / current_price * 100) if bear_value and current_price else None

        return {
            "DCF Fair Value": round(dcf_fair_value, 2),
            "DCF Upside (%)": round(dcf_upside, 1) if dcf_upside is not None else None,
            "DCF Bull Value": round(bull_value, 2) if bull_value else None,
            "DCF Bear Value": round(bear_value, 2) if bear_value else None,
            "DCF Bull Upside (%)": round(bull_upside, 1) if bull_upside is not None else None,
            "DCF Bear Upside (%)": round(bear_upside, 1) if bear_upside is not None else None,
            "DCF Range": f"${bear_value:.0f} - ${bull_value:.0f}" if bear_value and bull_value else None,
            "DCF Notes": f"WACC={wacc * 100:.1f}%, Growth={fcf_growth * 100:.1f}%",
        }

    except KeyError as e:
        logger.debug(f"DCF calculation failed - missing key: {str(e)}")
        return {
            "DCF Fair Value": None,
            "DCF Upside (%)": None,
            "DCF Bull Value": None,
            "DCF Bear Value": None,
            "DCF Range": None,
            "DCF Notes": f"Missing data key: {str(e)[:30]}",
        }
    except Exception as e:
        logger.warning(f"DCF calculation failed: {str(e)}")
        return {
            "DCF Fair Value": None,
            "DCF Upside (%)": None,
            "DCF Bull Value": None,
            "DCF Bear Value": None,
            "DCF Range": None,
            "DCF Notes": f"Error: {str(e)[:30]}",
        }


def calculate_fcf_yield(ticker: yf.Ticker, info: dict, risk_free_rate: float = 0.045) -> dict:
    """
    Calculate Free Cash Flow Yield = FCF / Market Cap.
    This is the inverse of P/FCF and represents the yield an investor
    would receive if all FCF were distributed.
    """
    try:
        cf = ticker.cash_flow
        if cf.empty:
            return {"FCF Yield (%)": None, "Yield Spread": None, "Yield Signal": None}

        # Get operating cash flow
        operating_cf = None
        for key in ["Total Cash From Operating Activities", "Operating Cash Flow", "Cash From Operating Activities"]:
            if key in cf.index:
                operating_cf = cf.loc[key].iloc[0]
                break

        # Get CapEx
        capex = None
        for key in ["Capital Expenditure", "Capital Expenditures", "Capital Spending"]:
            if key in cf.index:
                capex = cf.loc[key].iloc[0]
                break

        if operating_cf is None or capex is None:
            return {"FCF Yield (%)": None, "Yield Spread": None, "Yield Signal": None}

        fcf = operating_cf + capex  # capex is negative
        market_cap = info.get("marketCap")

        if not market_cap or market_cap <= 0:
            return {"FCF Yield (%)": None, "Yield Spread": None, "Yield Signal": None}

        fcf_yield = (fcf / market_cap) * 100

        # Compare to risk-free rate (10Y Treasury) as "equity bond yield"
        risk_free_pct = risk_free_rate * 100
        yield_spread = fcf_yield - risk_free_pct

        # Signal based on yield spread
        if yield_spread > 3:
            signal = "Very Attractive"
        elif yield_spread > 1:
            signal = "Attractive"
        elif yield_spread > -1:
            signal = "Fair"
        else:
            signal = "Unattractive"

        return {
            "FCF Yield (%)": round(fcf_yield, 2),
            "Yield Spread": round(yield_spread, 2),
            "Yield Signal": signal,
        }

    except Exception as e:
        logger.debug(f"FCF Yield calculation failed: {str(e)}")
        return {"FCF Yield (%)": None, "Yield Spread": None, "Yield Signal": None}


def calculate_growth_quality(ticker: yf.Ticker) -> dict:
    """
    Assess if growth is sustainable and profitable.
    Returns a Growth Quality Score (0-3) based on:
    1. Margin trend (stable/expanding vs compressing)
    2. Operating leverage (earnings growing faster than revenue)
    3. FCF conversion (FCF growth backing earnings growth)
    """
    try:
        income = ticker.financials
        cf = ticker.cash_flow

        if income.empty or len(income.columns) < 3:
            return {
                "Growth Quality Score": None,
                "Margin Trend": None,
                "Operating Leverage": None,
                "FCF Backs Growth": None,
            }

        quality_score = 0
        details = {}

        # 1. Margin Trend Analysis
        try:
            revenues = []
            op_income = []
            for i in range(min(3, len(income.columns))):
                for rev_key in ["Total Revenue", "Revenue", "Net Sales"]:
                    if rev_key in income.index:
                        revenues.append(income.loc[rev_key].iloc[i])
                        break
                for op_key in ["Operating Income", "EBIT"]:
                    if op_key in income.index:
                        op_income.append(income.loc[op_key].iloc[i])
                        break

            if len(revenues) >= 2 and len(op_income) >= 2:
                current_margin = op_income[0] / revenues[0] if revenues[0] else 0
                old_margin = op_income[-1] / revenues[-1] if revenues[-1] else 0

                if old_margin > 0:
                    margin_change = (current_margin - old_margin) / abs(old_margin)
                    if margin_change > 0.05:
                        details["Margin Trend"] = "Expanding"
                        quality_score += 1
                    elif margin_change > -0.05:
                        details["Margin Trend"] = "Stable"
                        quality_score += 1
                    else:
                        details["Margin Trend"] = "Compressing"
                else:
                    details["Margin Trend"] = "Unknown"
            else:
                details["Margin Trend"] = "Insufficient Data"
        except Exception:
            details["Margin Trend"] = "Error"

        # 2. Operating Leverage (Earnings growth vs Revenue growth)
        try:
            if len(revenues) >= 2 and revenues[-1] > 0:
                rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
            else:
                rev_growth = None

            if len(op_income) >= 2 and op_income[-1] and op_income[-1] > 0:
                earnings_growth = (op_income[0] - op_income[-1]) / abs(op_income[-1])
            else:
                earnings_growth = None

            if rev_growth and rev_growth > 0 and earnings_growth:
                op_leverage = earnings_growth / rev_growth
                details["Operating Leverage"] = round(op_leverage, 2)
                if op_leverage > 1.0:
                    quality_score += 1
            else:
                details["Operating Leverage"] = None
        except Exception:
            details["Operating Leverage"] = None

        # 3. FCF Conversion (FCF growth relative to earnings growth)
        try:
            if not cf.empty and len(cf.columns) >= 2:
                fcf_current = None
                fcf_old = None

                for ocf_key in ["Operating Cash Flow", "Total Cash From Operating Activities"]:
                    if ocf_key in cf.index:
                        for capex_key in ["Capital Expenditure", "Capital Expenditures"]:
                            if capex_key in cf.index:
                                fcf_current = cf.loc[ocf_key].iloc[0] + cf.loc[capex_key].iloc[0]
                                fcf_old = cf.loc[ocf_key].iloc[-1] + cf.loc[capex_key].iloc[-1]
                                break
                        break

                if fcf_current and fcf_old and fcf_old > 0:
                    fcf_growth = (fcf_current - fcf_old) / abs(fcf_old)
                    if earnings_growth and earnings_growth > 0:
                        fcf_conversion = fcf_growth / earnings_growth
                        details["FCF Backs Growth"] = fcf_conversion > 0.8
                        if fcf_conversion > 0.8:
                            quality_score += 1
                    else:
                        details["FCF Backs Growth"] = fcf_current > 0
                        if fcf_current > 0:
                            quality_score += 1
                else:
                    details["FCF Backs Growth"] = None
            else:
                details["FCF Backs Growth"] = None
        except Exception:
            details["FCF Backs Growth"] = None

        return {
            "Growth Quality Score": quality_score,
            "Margin Trend": details.get("Margin Trend"),
            "Operating Leverage": details.get("Operating Leverage"),
            "FCF Backs Growth": details.get("FCF Backs Growth"),
        }

    except Exception as e:
        logger.debug(f"Growth quality calculation failed: {str(e)}")
        return {
            "Growth Quality Score": None,
            "Margin Trend": None,
            "Operating Leverage": None,
            "FCF Backs Growth": None,
        }


def build_stage1_query() -> EquityQuery:
    """
    Stage 1: Build composite EquityQuery for fast API filtering.
    Filters: Market Cap, Positive EPS, P/E, PEG, D/E, ROE, Revenue Growth, EBITDA Growth
    """
    filters = [
        # Region: US only
        EquityQuery("eq", ["region", "us"]),
        # Exchange: ASE, BTS, CXI, NCM, NGM, NMS, NYQ, OEM, OQB, OQX, PCX, PNK, YHD
        EquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
        # Market Cap: $1B - $500B
        EquityQuery(
            "btwn", ["lastclosemarketcap.lasttwelvemonths", MCAP_MIN, MCAP_MAX]
        ),
        # Positive EPS
        EquityQuery("gte", ["basicepscontinuingoperations.lasttwelvemonths", 0.001]),
        EquityQuery("gte", ["epsgrowth.lasttwelvemonths", GROWTH_MIN]),
        # P/E < 50
        EquityQuery("btwn", ["peratio.lasttwelvemonths", 0, PE_MAX]),
        # PEG < 3
        EquityQuery("lte", ["pegratio_5y", PEG_MAX]),
        # D/E < 100 (1.0)
        EquityQuery("btwn", ["totaldebtequity.lasttwelvemonths", 0, (DE_MAX * 100)]),
        # ROE >= 10%
        EquityQuery("gte", ["returnonequity.lasttwelvemonths", ROE_MIN_PCT]),
        # ROIC Proxy (Return on Total Capital) >= 7%
        EquityQuery("gte", ["returnontotalcapital.lasttwelvemonths", ROIC_MIN_PCT]),
        # Revenue Growth >= 6%
        EquityQuery("gte", ["totalrevenues1yrgrowth.lasttwelvemonths", GROWTH_MIN_PCT]),
        EquityQuery("gte", ["quarterlyrevenuegrowth.quarterly", GROWTH_MIN_PCT]),
        # EBITDA Growth >= 6%
        # EquityQuery("gte", ["ebitda1yrgrowth.lasttwelvemonths", GROWTH_MIN_PCT]),
        # Net Income Growth >= 6%
        # EquityQuery("gte", ["netincome1yrgrowth.lasttwelvemonths", GROWTH_MIN_PCT]),
        # Diluted EPS Growth >= 6%
        # EquityQuery("gte", ["dilutedeps1yrgrowth.lasttwelvemonths", GROWTH_MIN_PCT]),
        # Current Ratio >= 1.0 (Liquidity)
        EquityQuery(
            "btwn", ["currentratio.lasttwelvemonths", 1, CURRENT_RATIO_MAX_PCT]
        ),
        # ROA >= 5% (Efficiency)
        EquityQuery("gte", ["returnonassets.lasttwelvemonths", ROA_MIN_PCT]),
        # Gross Margin >= 35% (Buffett moat indicator)
        EquityQuery(
            "gte", ["grossprofitmargin.lasttwelvemonths", GROSS_MARGIN_MIN_PCT]
        ),
        # Operating Margin >= 15% (efficiency)
        EquityQuery("gte", ["ebitdamargin.lasttwelvemonths", OPERATING_MARGIN_MIN_PCT]),
        # ROE >= 10%
        EquityQuery("gte", ["returnonequity.lasttwelvemonths", ROE_MIN_PCT]),
        # Beta >= 1.0 (Liquidity)
        EquityQuery("btwn", ["beta", BETA_MIN, BETA_MAX]),
        # fiftytwowkpercentchange
        EquityQuery(
            "btwn",
            [
                "fiftytwowkpercentchange",
                FIFTYTWOWK_PERCENTCHANGE_MIN,
                FIFTYTWOWK_PERCENTCHANGE_MAX,
            ],
        ),
    ]
    return EquityQuery("and", filters)


def run_stage1(limit: int = DEFAULT_STAGE1_SIZE) -> pd.DataFrame:
    """
    Execute Stage 1 screening via yfinance API.
    Returns DataFrame of stocks passing initial filters.
    """
    print("=" * 60)
    print("STAGE 1: Running EquityQuery API Filter")
    print("=" * 60)

    query = build_stage1_query()

    try:
        result = yf.screen(
            query, size=limit, sortField="intradaymarketcap", sortAsc=False
        )

        if result is None or "quotes" not in result:
            print("No results from Stage 1 screening")
            return pd.DataFrame()

        df = pd.DataFrame(result["quotes"])
        initial_count = len(df)
        print(f"Stage 1 found {initial_count} stocks passing initial filters")

        if df.empty:  # Added check for empty DataFrame
            return pd.DataFrame()

        # Filter out excluded symbols from config
        excluded_symbols = CONFIG.get("excluded_symbols", [])
        if excluded_symbols:
            pre_filter_count = len(df)
            df = df[~df["symbol"].isin(excluded_symbols)]
            excluded_count = pre_filter_count - len(df)
            if excluded_count > 0:
                print(
                    f"  Excluded {excluded_count} stocks based on your symbol exclusion list."
                )

        return df

    except Exception as e:
        print(f"Error in Stage 1: {e}")
        return pd.DataFrame()


def build_manual_stage1_df(symbols: list[str]) -> pd.DataFrame:
    """Build a Stage 1-like DataFrame from user supplied tickers."""
    cleaned = [s.upper() for s in symbols if s]
    if not cleaned:
        return pd.DataFrame()
    return pd.DataFrame({"symbol": cleaned})


def calculate_pfcf(
    ticker: yf.Ticker, market_cap: float, info: Optional[dict] = None
) -> Optional[float]:
    """Calculate Price-to-Free Cash Flow ratio with robust fallbacks."""
    try:
        info = info or ticker.info
        cashflow = getattr(ticker, "cashflow", pd.DataFrame())
        if cashflow is None or cashflow.empty:
            cashflow = getattr(ticker, "cash_flow", pd.DataFrame()) or pd.DataFrame()

        fcf = None
        if cashflow is not None and not cashflow.empty:
            if "Free Cash Flow" in cashflow.index:
                fcf = cashflow.loc["Free Cash Flow"].iloc[0]
            else:
                ocf = None
                capex = None
                for key in [
                    "Operating Cash Flow",
                    "Total Cash From Operating Activities",
                ]:
                    if key in cashflow.index:
                        ocf = cashflow.loc[key].iloc[0]
                        break
                for key in ["Capital Expenditure"]:
                    if key in cashflow.index:
                        capex = cashflow.loc[key].iloc[0]
                        break
                if ocf is not None and capex is not None:
                    fcf = ocf + capex  # CapEx is negative

        if fcf is None and info:
            fcf = info.get("freeCashflow")

        if fcf is None or fcf <= 0:
            return None

        return market_cap / fcf
    except Exception as e:
        logger.warning(f"P/FCF calculation failed for {ticker.ticker}: {str(e)}")
        return None


def calculate_roic(ticker: yf.Ticker) -> Optional[float]:
    """
    Calculate Return on Invested Capital (ROIC) using enhanced NOPAT method.
    ROIC = NOPAT / Average Invested Capital
    NOPAT = EBIT * (1 - Tax Rate)
    Invested Capital = Total Debt + Total Equity - Excess Cash
    Excess Cash = max(0, Total Cash - 3% of Revenue)
    """
    try:
        income = ticker.financials
        balance = ticker.balance_sheet

        if income.empty or balance.empty:
            return None

        # Get EBIT (prefer most recent, but validate)
        if "EBIT" in income.index:
            ebit = income.loc["EBIT"].iloc[0]
        elif "Operating Income" in income.index:
            ebit = income.loc["Operating Income"].iloc[0]
        else:
            return None

        if pd.isna(ebit) or ebit is None:
            return None

        # Calculate effective tax rate
        if "Tax Provision" in income.index and "Pretax Income" in income.index:
            tax = income.loc["Tax Provision"].iloc[0]
            pretax = income.loc["Pretax Income"].iloc[0]
            if pretax and pretax > 0 and not pd.isna(tax):
                tax_rate = max(0, min(0.40, tax / pretax))  # Cap at 0-40%
            else:
                tax_rate = 0.21
        else:
            tax_rate = 0.21

        nopat = ebit * (1 - tax_rate)

        # Get revenue for excess cash calculation
        revenue = 0
        for key in ["Total Revenue", "Revenue", "Net Sales"]:
            if key in income.index:
                revenue = income.loc[key].iloc[0] or 0
                break

        # Calculate current period invested capital
        def get_invested_capital(bs, period_idx=0):
            if len(bs.columns) <= period_idx:
                return None

            debt = 0
            if "Total Debt" in bs.index:
                debt = bs.loc["Total Debt"].iloc[period_idx] or 0

            equity = 0
            if "Total Equity Gross Minority Interest" in bs.index:
                equity = bs.loc["Total Equity Gross Minority Interest"].iloc[period_idx] or 0
            elif "Stockholders Equity" in bs.index:
                equity = bs.loc["Stockholders Equity"].iloc[period_idx] or 0

            # Get cash for excess cash calculation
            cash = 0
            for key in ["Cash And Cash Equivalents", "Cash", "Cash And Short Term Investments"]:
                if key in bs.index:
                    cash = bs.loc[key].iloc[period_idx] or 0
                    break

            # Excess cash = cash beyond 3% of revenue (operating cash needs)
            operating_cash_need = revenue * 0.03
            excess_cash = max(0, cash - operating_cash_need)

            # Invested Capital = Debt + Equity - Excess Cash
            invested_capital = debt + equity - excess_cash
            return invested_capital

        current_ic = get_invested_capital(balance, 0)
        prior_ic = get_invested_capital(balance, 1) if len(balance.columns) > 1 else current_ic

        if current_ic is None or current_ic <= 0:
            return None

        # Use average invested capital for more accurate ROIC
        if prior_ic and prior_ic > 0:
            avg_invested_capital = (current_ic + prior_ic) / 2
        else:
            avg_invested_capital = current_ic

        roic = safe_divide(nopat, avg_invested_capital)
        return roic if roic is not None and np.isfinite(roic) else None

    except Exception as e:
        logger.warning(f"ROIC calculation failed for {ticker.ticker}: {str(e)}")
        return None


def calculate_cagr(values: list, years: int = 3) -> Optional[float]:
    """Calculate Compound Annual Growth Rate."""
    try:
        if len(values) < years + 1:
            return None

        end_value = values[0]  # Most recent
        start_value = values[years]  # N years ago

        if start_value is None or end_value is None:
            return None
        if start_value <= 0:  # Can't calculate CAGR from negative/zero base
            return None

        ratio = safe_divide(end_value, start_value, default=1.0)
        if ratio <= 0:
            return None
        cagr = (ratio ** (1 / years)) - 1
        return cagr if np.isfinite(cagr) else None
    except Exception as e:
        logger.warning(f"CAGR calculation failed: {str(e)}")
        return None


def calculate_graham_number(eps: float, book_value_per_share: float) -> Optional[float]:
    """
    Calculate Benjamin Graham's intrinsic value (Graham Number).
    Formula: √(22.5 × EPS × Book Value Per Share)

    22.5 = 15 (max P/E) × 1.5 (max P/B) from Graham's defensive investor criteria.
    Stocks trading below Graham Number are considered undervalued.
    """
    try:
        if eps is None or book_value_per_share is None:
            return None
        if eps <= 0 or book_value_per_share <= 0:
            return None  # Graham Number requires positive EPS and BVPS

        graham_number = (22.5 * eps * book_value_per_share) ** 0.5
        return graham_number
    except Exception as e:
        logger.debug(f"Graham number calculation failed: {str(e)}")
        return None


def calculate_piotroski_score(ticker: yf.Ticker) -> dict:
    """
    Calculate Piotroski F-Score (0-9) for quality assessment.

    The F-Score combines 9 fundamental signals:
    1. Positive ROA
    2. Positive Operating Cash Flow
    3. ROA increased YoY
    4. Quality of Earnings (OCF > Net Income)
    5. Decreasing Leverage (Long-term Debt/Assets)
    6. Increasing Current Ratio
    7. No new shares issued
    8. Increasing Gross Margin
    9. Increasing Asset Turnover

    Returns: Dict with score (0-9) and breakdown of signals
    """
    try:
        income = ticker.financials
        balance = ticker.balance_sheet
        cashflow = ticker.cashflow

        if income.empty or balance.empty or cashflow.empty:
            return {"Piotroski Score": None, "Piotroski Quality": "Unknown"}

        # Get current and previous year data (most recent = column 0)
        current_income = income.iloc[:, 0] if len(income.columns) > 0 else pd.Series()
        prev_income = income.iloc[:, 1] if len(income.columns) > 1 else pd.Series()

        current_balance = (
            balance.iloc[:, 0] if len(balance.columns) > 0 else pd.Series()
        )
        prev_balance = balance.iloc[:, 1] if len(balance.columns) > 1 else pd.Series()

        current_cf = cashflow.iloc[:, 0] if len(cashflow.columns) > 0 else pd.Series()

        score = 0

        # Helper to safely get value
        def get_value(series, key, default=0):
            if key in series.index:
                val = series[key]
                return val if pd.notna(val) else default
            return default

        # 1. Positive ROA
        net_income = get_value(current_income, "Net Income")
        total_assets = get_value(current_balance, "Total Assets")
        roa_current = net_income / total_assets if total_assets > 0 else 0
        if roa_current > 0:
            score += 1

        # 2. Positive Operating Cash Flow
        ocf = get_value(
            current_cf,
            "Operating Cash Flow",
            get_value(current_cf, "Total Cash From Operating Activities"),
        )
        if ocf > 0:
            score += 1

        # 3. ROA increased YoY
        prev_net_income = get_value(prev_income, "Net Income")
        prev_total_assets = get_value(prev_balance, "Total Assets")
        roa_prev = prev_net_income / prev_total_assets if prev_total_assets > 0 else 0
        if roa_current > roa_prev:
            score += 1

        # 4. Quality of Earnings (OCF > Net Income)
        if ocf > net_income:
            score += 1

        # 5. Decreasing Leverage
        current_lt_debt = get_value(
            current_balance, "Long Term Debt", get_value(current_balance, "Total Debt")
        )
        prev_lt_debt = get_value(
            prev_balance, "Long Term Debt", get_value(prev_balance, "Total Debt")
        )
        leverage_current = current_lt_debt / total_assets if total_assets > 0 else 0
        leverage_prev = prev_lt_debt / prev_total_assets if prev_total_assets > 0 else 0
        if leverage_current < leverage_prev:
            score += 1

        # 6. Increasing Current Ratio
        current_assets = get_value(current_balance, "Current Assets")
        current_liabilities = get_value(current_balance, "Current Liabilities")
        prev_current_assets = get_value(prev_balance, "Current Assets")
        prev_current_liabilities = get_value(prev_balance, "Current Liabilities")

        cr_current = (
            current_assets / current_liabilities if current_liabilities > 0 else 0
        )
        cr_prev = (
            prev_current_assets / prev_current_liabilities
            if prev_current_liabilities > 0
            else 0
        )
        if cr_current > cr_prev:
            score += 1

        # 7. No new shares issued (simplified: shares outstanding decreased or stayed same)
        current_shares = get_value(
            current_balance, "Share Issued", get_value(current_balance, "Common Stock")
        )
        prev_shares = get_value(
            prev_balance, "Share Issued", get_value(prev_balance, "Common Stock")
        )
        if current_shares > 0 and prev_shares > 0:
            if current_shares <= prev_shares:
                score += 1

        # 8. Increasing Gross Margin
        current_revenue = get_value(current_income, "Total Revenue")
        current_cogs = get_value(
            current_income,
            "Cost Of Revenue",
            get_value(current_income, "Cost Of Goods Sold"),
        )
        prev_revenue = get_value(prev_income, "Total Revenue")
        prev_cogs = get_value(
            prev_income, "Cost Of Revenue", get_value(prev_income, "Cost Of Goods Sold")
        )

        gm_current = (
            (current_revenue - current_cogs) / current_revenue
            if current_revenue > 0
            else 0
        )
        gm_prev = (prev_revenue - prev_cogs) / prev_revenue if prev_revenue > 0 else 0
        if gm_current > gm_prev:
            score += 1

        # 9. Increasing Asset Turnover
        turnover_current = current_revenue / total_assets if total_assets > 0 else 0
        turnover_prev = prev_revenue / prev_total_assets if prev_total_assets > 0 else 0
        if turnover_current > turnover_prev:
            score += 1

        # Interpret score
        if score >= 8:
            quality_label = "Excellent"
        elif score >= 6:
            quality_label = "Good"
        elif score >= 4:
            quality_label = "Average"
        else:
            quality_label = "Poor"

        return {
            "Piotroski Score": score,
            "Piotroski Quality": quality_label,
        }

    except Exception:
        return {"Piotroski Score": None, "Piotroski Quality": "Unknown"}


def calculate_ev_ebitda(ticker: yf.Ticker, info: dict) -> Optional[float]:
    """
    Calculate EV/EBITDA ratio.
    EV = Market Cap + Total Debt - Cash
    EBITDA = Earnings Before Interest, Taxes, Depreciation, Amortization
    """
    try:
        market_cap = info.get("marketCap", 0)
        total_debt = info.get("totalDebt", 0)
        cash = info.get("totalCash", 0) or info.get("cash", 0) or 0

        enterprise_value = market_cap + total_debt - cash

        # Try to get EBITDA from income statement
        income = ticker.financials
        if income.empty:
            # Fallback to info dict
            ebitda = info.get("ebitda")
            if not ebitda:
                return None
        else:
            # Try different EBITDA keys
            ebitda = None
            for key in ["EBITDA", "Ebitda"]:
                if key in income.index:
                    ebitda = income.loc[key].iloc[0]
                    break

            if ebitda is None:
                # Calculate from components
                ebit = None
                for key in ["EBIT", "Operating Income", "Ebit"]:
                    if key in income.index:
                        ebit = income.loc[key].iloc[0]
                        break

                if ebit is not None:
                    # Add back depreciation/amortization
                    da = None
                    for key in [
                        "Depreciation And Amortization",
                        "Depreciation",
                        "Amortization Of Intangibles",
                    ]:
                        if key in income.index:
                            da = income.loc[key].iloc[0]
                            break
                    ebitda = ebit + (da if da else 0)
                else:
                    return None

        if ebitda is None or ebitda <= 0:
            return None

        return enterprise_value / ebitda

    except Exception:
        return None


def calculate_altman_z_score(ticker: yf.Ticker, info: dict) -> dict:
    """
    Calculate Altman Z-Score to assess financial distress probability.
    Z-Score < 1.8 = Distress Zone (High bankruptcy risk)
    Z-Score 1.8-3.0 = Grey Zone (Moderate risk)
    Z-Score > 3.0 = Safe Zone (Low bankruptcy risk)

    Formula: Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
    Where:
    A = Working Capital / Total Assets
    B = Retained Earnings / Total Assets
    C = EBIT / Total Assets
    D = Market Value of Equity / Total Liabilities
    E = Sales / Total Assets
    """
    try:
        balance = ticker.balance_sheet
        income = ticker.financials

        if balance.empty or income.empty:
            return {"Altman Z-Score": None, "Financial Distress Risk": "Unknown"}

        # Get most recent data
        current_balance = (
            balance.iloc[:, 0] if len(balance.columns) > 0 else pd.Series()
        )
        current_income = income.iloc[:, 0] if len(income.columns) > 0 else pd.Series()

        def get_value(series, key, default=0):
            if key in series.index:
                val = series[key]
                return val if pd.notna(val) else default
            return default

        total_assets = get_value(current_balance, "Total Assets")
        if total_assets <= 0:
            return {"Altman Z-Score": None, "Financial Distress Risk": "Unknown"}

        # A = Working Capital / Total Assets
        current_assets = get_value(current_balance, "Current Assets")
        current_liabilities = get_value(current_balance, "Current Liabilities")
        working_capital = current_assets - current_liabilities
        A = safe_divide(working_capital, total_assets)

        # B = Retained Earnings / Total Assets
        retained_earnings = get_value(
            current_balance,
            "Retained Earnings",
            get_value(current_balance, "Retained Earnings Total Equity"),
        )
        B = safe_divide(retained_earnings, total_assets)

        # C = EBIT / Total Assets
        ebit = get_value(
            current_income, "EBIT", get_value(current_income, "Operating Income")
        )
        C = safe_divide(ebit, total_assets)

        # D = Market Value of Equity / Total Liabilities
        market_cap = info.get("marketCap", 0) or 0
        total_liabilities = get_value(current_balance, "Total Liabilities")
        D = safe_divide(market_cap, total_liabilities)

        # E = Sales / Total Assets
        revenue = get_value(current_income, "Total Revenue")
        E = safe_divide(revenue, total_assets)

        # Calculate Z-Score
        z_score = 1.2 * A + 1.4 * B + 3.3 * C + 0.6 * D + 1.0 * E

        # Interpret
        if z_score < 1.8:
            risk = "High (Distress Zone)"
        elif z_score < 3.0:
            risk = "Moderate (Grey Zone)"
        else:
            risk = "Low (Safe Zone)"

        return {"Altman Z-Score": round(z_score, 2), "Financial Distress Risk": risk}

    except Exception as e:
        logger.debug(f"Altman Z-Score calculation failed: {str(e)}")
        return {"Altman Z-Score": None, "Financial Distress Risk": "Unknown"}


def calculate_cash_flow_quality(ticker: yf.Ticker) -> dict:
    """
    Analyze cash flow quality and sustainability.
    Checks:
    1. Free Cash Flow conversion (FCF / Net Income)
    2. Working capital trends (increasing WC = cash tied up)
    3. CapEx vs Depreciation (sustainable growth)
    """
    try:
        cashflow = ticker.cashflow
        income = ticker.financials
        balance = ticker.balance_sheet

        if cashflow.empty or income.empty or balance.empty:
            return {}

        # Get current and previous year
        current_cf = cashflow.iloc[:, 0] if len(cashflow.columns) > 0 else pd.Series()
        current_income = income.iloc[:, 0] if len(income.columns) > 0 else pd.Series()
        current_balance = (
            balance.iloc[:, 0] if len(balance.columns) > 0 else pd.Series()
        )
        prev_balance = balance.iloc[:, 1] if len(balance.columns) > 1 else pd.Series()

        def get_value(series, key, default=0):
            if key in series.index:
                val = series[key]
                return val if pd.notna(val) else default
            return default

        # FCF Conversion
        ocf = get_value(
            current_cf,
            "Operating Cash Flow",
            get_value(current_cf, "Total Cash From Operating Activities"),
        )
        capex = get_value(current_cf, "Capital Expenditure")
        fcf = ocf + capex  # CapEx is negative

        net_income = get_value(current_income, "Net Income")
        fcf_conversion = fcf / net_income if net_income > 0 else None

        # Working Capital Change
        current_wc = get_value(current_balance, "Current Assets") - get_value(
            current_balance, "Current Liabilities"
        )
        prev_wc = get_value(prev_balance, "Current Assets") - get_value(
            prev_balance, "Current Liabilities"
        )
        wc_change = current_wc - prev_wc
        wc_change_pct = (wc_change / abs(prev_wc)) * 100 if prev_wc != 0 else None

        # CapEx vs Depreciation
        depreciation = get_value(
            current_cf,
            "Depreciation",
            get_value(current_income, "Depreciation And Amortization"),
        )
        capex_ratio = abs(capex) / depreciation if depreciation > 0 else None

        quality_score = "Good"
        concerns = []

        if fcf_conversion and fcf_conversion < 0.5:
            quality_score = "Poor"
            concerns.append("Low FCF conversion")
        elif fcf_conversion and fcf_conversion < 0.7:
            quality_score = "Fair"
            concerns.append("Moderate FCF conversion")

        if wc_change_pct and wc_change_pct > 20:
            concerns.append("Rapid working capital growth (cash tied up)")

        if capex_ratio and capex_ratio > 1.5:
            concerns.append("High CapEx vs Depreciation (aggressive growth)")

        return {
            "FCF Conversion Ratio": round(fcf_conversion, 2)
            if fcf_conversion
            else None,
            "Working Capital Change (%)": round(wc_change_pct, 1)
            if wc_change_pct
            else None,
            "CapEx/Depreciation Ratio": round(capex_ratio, 2) if capex_ratio else None,
            "Cash Flow Quality": quality_score,
            "Cash Flow Concerns": "; ".join(concerns) if concerns else None,
        }

    except Exception as e:
        logger.debug(f"Cash flow quality analysis failed: {str(e)}")
        return {}


def detect_accounting_red_flags(ticker: yf.Ticker) -> dict:
    """
    Detect potential accounting red flags:
    1. Inventory growth > Revenue growth (potential obsolescence)
    2. Accounts Receivable growth > Revenue growth (collection issues)
    3. Revenue growth but declining margins (pricing pressure)
    """
    try:
        income = ticker.financials
        balance = ticker.balance_sheet

        if (
            income.empty
            or balance.empty
            or len(income.columns) < 2
            or len(balance.columns) < 2
        ):
            return {}

        current_income = income.iloc[:, 0]
        prev_income = income.iloc[:, 1]
        current_balance = balance.iloc[:, 0]
        prev_balance = balance.iloc[:, 1]

        def get_value(series, key, default=0):
            if key in series.index:
                val = series[key]
                return val if pd.notna(val) else default
            return default

        red_flags = []

        # Revenue growth
        current_revenue = get_value(current_income, "Total Revenue")
        prev_revenue = get_value(prev_income, "Total Revenue")
        revenue_growth = (
            (current_revenue - prev_revenue) / prev_revenue * 100
            if prev_revenue > 0
            else None
        )

        # Inventory growth vs Revenue growth
        current_inventory = get_value(current_balance, "Inventory")
        prev_inventory = get_value(prev_balance, "Inventory")
        if prev_inventory > 0 and current_inventory > 0:
            inventory_growth = (
                (current_inventory - prev_inventory) / prev_inventory * 100
            )
            if revenue_growth and inventory_growth > revenue_growth * 1.5:
                red_flags.append("Inventory growing faster than revenue")

        # AR growth vs Revenue growth
        current_ar = get_value(
            current_balance,
            "Accounts Receivable",
            get_value(current_balance, "Net Receivables"),
        )
        prev_ar = get_value(
            prev_balance,
            "Accounts Receivable",
            get_value(prev_balance, "Net Receivables"),
        )
        if prev_ar > 0 and current_ar > 0:
            ar_growth = (current_ar - prev_ar) / prev_ar * 100
            if revenue_growth and ar_growth > revenue_growth * 1.5:
                red_flags.append("AR growing faster than revenue (collection risk)")

        # Margin compression
        current_gross = get_value(current_income, "Gross Profit")
        prev_gross = get_value(prev_income, "Gross Profit")
        if current_revenue > 0 and prev_revenue > 0:
            current_gm = (current_gross / current_revenue) * 100
            prev_gm = (prev_gross / prev_revenue) * 100
            if prev_gm > 0 and current_gm < prev_gm * 0.9:  # 10%+ decline
                red_flags.append("Gross margin compression")

        return {
            "Accounting Red Flags": "; ".join(red_flags) if red_flags else "None",
            "Red Flag Count": len(red_flags),
        }

    except Exception as e:
        logger.debug(f"Accounting red flags detection failed: {str(e)}")
        return {"Accounting Red Flags": "Unknown", "Red Flag Count": 0}


def analyze_debt_maturity_risk(ticker: yf.Ticker, info: dict) -> dict:
    """
    Analyze debt maturity and refinancing risk.
    High short-term debt relative to cash = liquidity risk.
    """
    try:
        balance = ticker.balance_sheet
        if balance.empty:
            return {}

        current_balance = (
            balance.iloc[:, 0] if len(balance.columns) > 0 else pd.Series()
        )

        def get_value(series, key, default=0):
            if key in series.index:
                val = series[key]
                return val if pd.notna(val) else default
            return default

        short_term_debt = get_value(
            current_balance,
            "Short Term Debt",
            get_value(current_balance, "Current Debt"),
        )
        long_term_debt = get_value(
            current_balance,
            "Long Term Debt",
            get_value(current_balance, "Long Term Debt And Capital Lease Obligation"),
        )
        total_debt = short_term_debt + long_term_debt
        cash = info.get("totalCash", 0) or info.get("cash", 0) or 0

        # Short-term debt coverage
        st_debt_coverage = cash / short_term_debt if short_term_debt > 0 else None

        # Debt maturity ratio
        st_debt_ratio = (short_term_debt / total_debt * 100) if total_debt > 0 else None

        risk_level = "Low"
        concerns = []

        if st_debt_coverage and st_debt_coverage < 1.0:
            risk_level = "High"
            concerns.append("Insufficient cash to cover short-term debt")
        elif st_debt_coverage and st_debt_coverage < 1.5:
            risk_level = "Moderate"
            concerns.append("Tight cash coverage of short-term debt")

        if st_debt_ratio and st_debt_ratio > 40:
            concerns.append("High proportion of short-term debt")

        return {
            "Short-Term Debt Coverage": round(st_debt_coverage, 2)
            if st_debt_coverage
            else None,
            "Short-Term Debt %": round(st_debt_ratio, 1) if st_debt_ratio else None,
            "Debt Maturity Risk": risk_level,
            "Debt Concerns": "; ".join(concerns) if concerns else None,
        }

    except Exception as e:
        logger.debug(f"Debt maturity analysis failed: {str(e)}")
        return {}


def check_dividend_sustainability(ticker: yf.Ticker, info: dict) -> dict:
    """
    Check dividend sustainability for dividend-paying stocks.
    Key metrics: Payout ratio, FCF coverage, dividend growth trend.
    """
    try:
        dividend_yield = info.get("dividendYield", 0) or 0
        if dividend_yield == 0:
            return {"Dividend Paying": False}

        income = ticker.financials
        cashflow = ticker.cashflow

        if income.empty or cashflow.empty:
            return {"Dividend Paying": True, "Dividend Sustainability": "Unknown"}

        current_income = income.iloc[:, 0]
        current_cf = cashflow.iloc[:, 0]

        def get_value(series, key, default=0):
            if key in series.index:
                val = series[key]
                return val if pd.notna(val) else default
            return default

        net_income = get_value(current_income, "Net Income")
        ocf = get_value(
            current_cf,
            "Operating Cash Flow",
            get_value(current_cf, "Total Cash From Operating Activities"),
        )
        capex = get_value(current_cf, "Capital Expenditure")
        fcf = ocf + capex

        # Estimate dividends paid (often in cashflow statement)
        dividends_paid = get_value(
            current_cf,
            "Dividends Paid",
            get_value(current_cf, "Common Stock Dividends Paid"),
        )

        # If not found, estimate from yield
        if dividends_paid == 0:
            market_cap = info.get("marketCap", 0)
            dividends_paid = market_cap * dividend_yield

        # Payout ratio
        payout_ratio = (dividends_paid / net_income * 100) if net_income > 0 else None

        # FCF coverage
        fcf_coverage = (fcf / dividends_paid) if dividends_paid > 0 else None

        sustainability = "Sustainable"
        concerns = []

        if payout_ratio and payout_ratio > 80:
            sustainability = "At Risk"
            concerns.append("High payout ratio (>80%)")
        elif payout_ratio and payout_ratio > 60:
            sustainability = "Moderate"
            concerns.append("Elevated payout ratio")

        if fcf_coverage and fcf_coverage < 1.0:
            sustainability = "At Risk"
            concerns.append("FCF insufficient to cover dividends")
        elif fcf_coverage and fcf_coverage < 1.5:
            concerns.append("Tight FCF coverage")

        return {
            "Dividend Paying": True,
            "Dividend Yield (%)": round(dividend_yield * 100, 2),
            "Payout Ratio (%)": round(payout_ratio, 1) if payout_ratio else None,
            "FCF Coverage": round(fcf_coverage, 2) if fcf_coverage else None,
            "Dividend Sustainability": sustainability,
            "Dividend Concerns": "; ".join(concerns) if concerns else None,
        }

    except Exception as e:
        logger.debug(f"Dividend sustainability check failed: {str(e)}")
        return {"Dividend Paying": False}


def check_valuation_consistency(stock_data: dict) -> dict:
    """
    Check if multiple valuation methods agree.
    If DCF, Graham, and Analyst targets all suggest similar upside/downside,
    that's a stronger signal.
    """
    try:
        dcf_upside = stock_data.get("DCF Upside (%)")
        graham_upside = stock_data.get("Graham Upside (%)")
        analyst_upside = stock_data.get("Upside (%)")

        valuations = []
        if dcf_upside is not None:
            valuations.append(("DCF", dcf_upside))
        if graham_upside is not None:
            valuations.append(("Graham", graham_upside))
        if analyst_upside is not None:
            valuations.append(("Analyst", analyst_upside))

        if len(valuations) < 2:
            return {"Valuation Consensus": "Insufficient Data"}

        # Check if valuations agree (within 20% of each other)
        upsides = [v[1] for v in valuations]
        avg_upside = np.mean(upsides)
        std_upside = np.std(upsides)

        if std_upside < 15:  # Low variance = high consensus
            consensus = "Strong Agreement"
        elif std_upside < 30:
            consensus = "Moderate Agreement"
        else:
            consensus = "Divergent Views"

        # Overall signal
        if avg_upside > 20:
            signal = "Strongly Undervalued"
        elif avg_upside > 10:
            signal = "Moderately Undervalued"
        elif avg_upside > 0:
            signal = "Slightly Undervalued"
        elif avg_upside > -10:
            signal = "Fairly Valued"
        else:
            signal = "Overvalued"

        return {
            "Valuation Consensus": consensus,
            "Average Upside (%)": round(avg_upside, 1),
            "Valuation Signal": signal,
            "Valuation Methods": len(valuations),
        }

    except Exception as e:
        logger.debug(f"Valuation consistency check failed: {str(e)}")
        return {"Valuation Consensus": "Unknown"}


def calculate_ttm_metrics(ticker: yf.Ticker) -> dict:
    """
    Calculate Trailing Twelve Months (TTM) metrics using quarterly data.
    More current than annual financials.
    """
    try:
        quarterly_income = ticker.quarterly_financials
        quarterly_cashflow = ticker.quarterly_cashflow

        if quarterly_income.empty:
            return {}

        # Sum last 4 quarters (most recent = column 0)
        if len(quarterly_income.columns) < 4:
            return {}  # Need at least 4 quarters

        ttm_metrics = {}

        # TTM Revenue
        if "Total Revenue" in quarterly_income.index:
            ttm_revenue = quarterly_income.loc["Total Revenue"].iloc[:4].sum()
            ttm_metrics["TTM Revenue"] = ttm_revenue

        # TTM Net Income
        if "Net Income" in quarterly_income.index:
            ttm_net_income = quarterly_income.loc["Net Income"].iloc[:4].sum()
            ttm_metrics["TTM Net Income"] = ttm_net_income

        # TTM EBITDA (if available)
        if "EBITDA" in quarterly_income.index:
            ttm_ebitda = quarterly_income.loc["EBITDA"].iloc[:4].sum()
            ttm_metrics["TTM EBITDA"] = ttm_ebitda

        # TTM Operating Cash Flow
        if (
            not quarterly_cashflow.empty
            and "Operating Cash Flow" in quarterly_cashflow.index
        ):
            ttm_ocf = quarterly_cashflow.loc["Operating Cash Flow"].iloc[:4].sum()
            ttm_metrics["TTM Operating Cash Flow"] = ttm_ocf

        return ttm_metrics

    except Exception:
        return {}


# =============================================================================
# NEW YFINANCE DATA METRICS
# =============================================================================


def get_short_interest_metrics(info: dict) -> dict:
    """
    Extract short interest metrics from ticker.info.
    High short interest can indicate contrarian opportunities or high risk.
    """
    try:
        short_percent = info.get("shortPercentOfFloat")
        short_ratio = info.get("shortRatio")  # Days to cover
        shares_short = info.get("sharesShort")
        shares_short_prior = info.get("sharesShortPriorMonth")

        result = {
            "Short % of Float": round(short_percent * 100, 2) if short_percent else None,
            "Short Ratio (Days)": round(short_ratio, 2) if short_ratio else None,
            "Shares Short": shares_short,
        }

        # Calculate short interest trend
        if shares_short and shares_short_prior and shares_short_prior > 0:
            short_change = ((shares_short - shares_short_prior) / shares_short_prior) * 100
            result["Short Interest Change (%)"] = round(short_change, 2)

            if short_change > 20:
                result["Short Interest Signal"] = "Increasing Bearish Pressure"
            elif short_change < -20:
                result["Short Interest Signal"] = "Short Covering Rally Potential"
            else:
                result["Short Interest Signal"] = "Stable"
        else:
            result["Short Interest Change (%)"] = None
            result["Short Interest Signal"] = None

        # Short squeeze potential: High short % + positive momentum
        if short_percent and short_percent > 0.15:  # >15% of float shorted
            result["Short Squeeze Risk"] = "High"
        elif short_percent and short_percent > 0.10:
            result["Short Squeeze Risk"] = "Moderate"
        else:
            result["Short Squeeze Risk"] = "Low"

        return result

    except Exception as e:
        logger.debug(f"Short interest metrics failed: {str(e)}")
        return {}


def get_analyst_upgrades_downgrades(ticker: yf.Ticker) -> dict:
    """
    Analyze recent analyst upgrades/downgrades for sentiment momentum.
    Early warning system for deteriorating/improving fundamentals.
    """
    try:
        upgrades = ticker.upgrades_downgrades
        if upgrades is None or upgrades.empty:
            return {
                "Recent Upgrades (3M)": None,
                "Recent Downgrades (3M)": None,
                "Analyst Momentum": None,
            }

        # Filter to last 3 months
        three_months_ago = pd.Timestamp.now() - pd.DateOffset(months=3)
        if isinstance(upgrades.index, pd.DatetimeIndex):
            recent = upgrades[upgrades.index >= three_months_ago]
        else:
            recent = upgrades  # Use all if index isn't datetime

        if recent.empty:
            return {
                "Recent Upgrades (3M)": 0,
                "Recent Downgrades (3M)": 0,
                "Analyst Momentum": "No Recent Activity",
            }

        # Count upgrades vs downgrades
        upgrades_count = 0
        downgrades_count = 0

        for _, row in recent.iterrows():
            action = str(row.get("Action", "")).lower()
            if "upgrade" in action or "initiated" in action:
                upgrades_count += 1
            elif "downgrade" in action or "lowered" in action:
                downgrades_count += 1

        # Determine momentum
        net_momentum = upgrades_count - downgrades_count
        if net_momentum >= 3:
            momentum = "Strong Positive"
        elif net_momentum >= 1:
            momentum = "Positive"
        elif net_momentum <= -3:
            momentum = "Strong Negative"
        elif net_momentum <= -1:
            momentum = "Negative"
        else:
            momentum = "Neutral"

        return {
            "Recent Upgrades (3M)": upgrades_count,
            "Recent Downgrades (3M)": downgrades_count,
            "Net Analyst Actions": net_momentum,
            "Analyst Momentum": momentum,
        }

    except Exception as e:
        logger.debug(f"Analyst upgrades/downgrades failed: {str(e)}")
        return {}


def get_target_price_analysis(info: dict) -> dict:
    """
    Analyze analyst price targets for upside potential.
    """
    try:
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        target_mean = info.get("targetMeanPrice")
        target_high = info.get("targetHighPrice")
        target_low = info.get("targetLowPrice")
        num_analysts = info.get("numberOfAnalystOpinions")

        result = {
            "Target Mean Price": target_mean,
            "Target High Price": target_high,
            "Target Low Price": target_low,
            "Analyst Coverage": num_analysts,
        }

        if current_price and target_mean:
            upside = ((target_mean - current_price) / current_price) * 100
            result["Target Upside (%)"] = round(upside, 2)

            if upside > 30:
                result["Target Signal"] = "Strong Upside"
            elif upside > 15:
                result["Target Signal"] = "Moderate Upside"
            elif upside > 0:
                result["Target Signal"] = "Slight Upside"
            elif upside > -15:
                result["Target Signal"] = "Near Target"
            else:
                result["Target Signal"] = "Overvalued vs Targets"
        else:
            result["Target Upside (%)"] = None
            result["Target Signal"] = None

        # Target range spread (analyst agreement)
        if target_high and target_low and target_mean:
            spread = ((target_high - target_low) / target_mean) * 100
            result["Target Spread (%)"] = round(spread, 2)
            if spread < 30:
                result["Analyst Agreement"] = "High"
            elif spread < 60:
                result["Analyst Agreement"] = "Moderate"
            else:
                result["Analyst Agreement"] = "Low (Wide Range)"

        return result

    except Exception as e:
        logger.debug(f"Target price analysis failed: {str(e)}")
        return {}


def get_insider_ownership_metrics(info: dict, ticker: yf.Ticker) -> dict:
    """
    Analyze insider and institutional ownership for 'skin in game' assessment.
    """
    try:
        insider_pct = info.get("heldPercentInsiders")
        inst_pct = info.get("heldPercentInstitutions")

        result = {
            "Insider Ownership (%)": round(insider_pct * 100, 2) if insider_pct else None,
            "Institutional Ownership (%)": round(inst_pct * 100, 2) if inst_pct else None,
        }

        # Insider ownership interpretation
        if insider_pct:
            if insider_pct > 0.30:
                result["Insider Signal"] = "Very High (Founder-led)"
            elif insider_pct > 0.10:
                result["Insider Signal"] = "Good Alignment"
            elif insider_pct > 0.03:
                result["Insider Signal"] = "Moderate"
            else:
                result["Insider Signal"] = "Low Skin in Game"
        else:
            result["Insider Signal"] = None

        # Institutional ownership interpretation
        if inst_pct:
            if inst_pct > 0.90:
                result["Institutional Signal"] = "Very High (Crowded)"
            elif inst_pct > 0.70:
                result["Institutional Signal"] = "High Institutional Interest"
            elif inst_pct > 0.40:
                result["Institutional Signal"] = "Moderate"
            else:
                result["Institutional Signal"] = "Under-owned by Institutions"

        # Check for recent insider purchases
        try:
            insider_purchases = ticker.insider_purchases
            if insider_purchases is not None and not insider_purchases.empty:
                total_purchases = insider_purchases.get("Shares", pd.Series()).sum()
                result["Recent Insider Purchases"] = int(total_purchases) if total_purchases > 0 else 0
            else:
                result["Recent Insider Purchases"] = 0
        except Exception:
            result["Recent Insider Purchases"] = None

        return result

    except Exception as e:
        logger.debug(f"Insider ownership metrics failed: {str(e)}")
        return {}


def get_dividend_growth_analysis(ticker: yf.Ticker, info: dict) -> dict:
    """
    Analyze dividend history for quality and growth.
    """
    try:
        dividends = ticker.dividends
        if dividends is None or dividends.empty:
            return {
                "Dividend Growth CAGR (%)": None,
                "Dividend Consistency": "No Dividends",
                "Years of Dividends": 0,
            }

        # Get annual dividend totals
        annual_divs = dividends.resample("YE").sum()
        annual_divs = annual_divs[annual_divs > 0]  # Remove years with no dividends

        if len(annual_divs) < 2:
            current_yield = info.get("dividendYield")
            return {
                "Dividend Growth CAGR (%)": None,
                "Dividend Consistency": "New Dividend Payer",
                "Years of Dividends": len(annual_divs),
                "Current Yield (%)": round(current_yield * 100, 2) if current_yield else None,
            }

        years = len(annual_divs)
        first_div = annual_divs.iloc[0]
        last_div = annual_divs.iloc[-1]

        # Calculate CAGR
        if first_div > 0 and years > 1:
            cagr = ((last_div / first_div) ** (1 / (years - 1)) - 1) * 100
        else:
            cagr = None

        # Check consistency (how many years had increases)
        increases = 0
        decreases = 0
        for i in range(1, len(annual_divs)):
            if annual_divs.iloc[i] > annual_divs.iloc[i - 1]:
                increases += 1
            elif annual_divs.iloc[i] < annual_divs.iloc[i - 1]:
                decreases += 1

        # Determine consistency rating
        if decreases == 0 and increases >= 5:
            consistency = "Excellent (5+ Years Increases)"
        elif decreases == 0 and increases >= 3:
            consistency = "Good (Consistent Growth)"
        elif decreases <= 1:
            consistency = "Fair (Mostly Stable)"
        else:
            consistency = "Variable"

        # Dividend aristocrat check (25+ years of increases)
        is_aristocrat = years >= 25 and decreases == 0

        current_yield = info.get("dividendYield")
        payout_ratio = info.get("payoutRatio")

        return {
            "Dividend Growth CAGR (%)": round(cagr, 2) if cagr else None,
            "Dividend Consistency": consistency,
            "Years of Dividends": years,
            "Consecutive Increases": increases,
            "Dividend Aristocrat": is_aristocrat,
            "Current Yield (%)": round(current_yield * 100, 2) if current_yield else None,
            "Payout Ratio (%)": round(payout_ratio * 100, 2) if payout_ratio else None,
        }

    except Exception as e:
        logger.debug(f"Dividend growth analysis failed: {str(e)}")
        return {}


def get_recommendations_sentiment(ticker: yf.Ticker) -> dict:
    """
    Calculate aggregate analyst sentiment score from recommendations summary.
    """
    try:
        summary = ticker.recommendations_summary
        if summary is None or summary.empty:
            return {
                "Analyst Sentiment Score": None,
                "Analyst Sentiment": None,
                "Strong Buy %": None,
            }

        # Extract counts (handle both Series and DataFrame)
        if isinstance(summary, pd.DataFrame):
            # Take the most recent period
            summary = summary.iloc[0] if len(summary) > 0 else summary

        strong_buy = summary.get("strongBuy", 0) or 0
        buy = summary.get("buy", 0) or 0
        hold = summary.get("hold", 0) or 0
        sell = summary.get("sell", 0) or 0
        strong_sell = summary.get("strongSell", 0) or 0

        total = strong_buy + buy + hold + sell + strong_sell

        if total == 0:
            return {
                "Analyst Sentiment Score": None,
                "Analyst Sentiment": "No Coverage",
                "Strong Buy %": None,
            }

        # Weighted sentiment score: -2 (Strong Sell) to +2 (Strong Buy)
        sentiment_score = (
            (strong_buy * 2 + buy * 1 + hold * 0 - sell * 1 - strong_sell * 2) / total
        )

        # Calculate percentages
        strong_buy_pct = (strong_buy / total) * 100
        buy_pct = ((strong_buy + buy) / total) * 100

        # Interpret sentiment
        if sentiment_score >= 1.5:
            sentiment = "Very Bullish"
        elif sentiment_score >= 0.8:
            sentiment = "Bullish"
        elif sentiment_score >= 0.3:
            sentiment = "Moderately Bullish"
        elif sentiment_score >= -0.3:
            sentiment = "Neutral"
        elif sentiment_score >= -0.8:
            sentiment = "Moderately Bearish"
        else:
            sentiment = "Bearish"

        return {
            "Analyst Sentiment Score": round(sentiment_score, 2),
            "Analyst Sentiment": sentiment,
            "Strong Buy %": round(strong_buy_pct, 1),
            "Buy or Better %": round(buy_pct, 1),
            "Total Analyst Count": int(total),
        }

    except Exception as e:
        logger.debug(f"Recommendations sentiment failed: {str(e)}")
        return {}


def get_esg_scores(ticker: yf.Ticker) -> dict:
    """
    Extract ESG (Environmental, Social, Governance) sustainability scores.
    """
    try:
        sustainability = ticker.sustainability
        if sustainability is None or sustainability.empty:
            return {
                "ESG Total Score": None,
                "ESG Risk Level": None,
                "Governance Score": None,
            }

        # Extract scores (sustainability is typically a DataFrame with index as metric names)
        def get_score(key):
            if key in sustainability.index:
                val = sustainability.loc[key].iloc[0] if isinstance(sustainability.loc[key], pd.Series) else sustainability.loc[key]
                return float(val) if pd.notna(val) else None
            return None

        total_esg = get_score("totalEsg")
        env_score = get_score("environmentScore")
        social_score = get_score("socialScore")
        gov_score = get_score("governanceScore")
        esg_perf = get_score("esgPerformance")

        # ESG risk interpretation (lower is better for risk scores)
        risk_level = None
        if total_esg is not None:
            if total_esg < 15:
                risk_level = "Negligible Risk"
            elif total_esg < 25:
                risk_level = "Low Risk"
            elif total_esg < 35:
                risk_level = "Medium Risk"
            elif total_esg < 45:
                risk_level = "High Risk"
            else:
                risk_level = "Severe Risk"

        return {
            "ESG Total Score": round(total_esg, 1) if total_esg else None,
            "ESG Risk Level": risk_level,
            "Environmental Score": round(env_score, 1) if env_score else None,
            "Social Score": round(social_score, 1) if social_score else None,
            "Governance Score": round(gov_score, 1) if gov_score else None,
        }

    except Exception as e:
        logger.debug(f"ESG scores failed: {str(e)}")
        return {}


def get_options_implied_volatility(ticker: yf.Ticker, info: dict) -> dict:
    """
    Extract implied volatility from options chain for forward-looking volatility expectations.
    Uses ATM options near 30 days expiry.
    """
    try:
        # Get available expiration dates
        expirations = ticker.options
        if not expirations:
            return {
                "Implied Volatility (%)": None,
                "IV Signal": None,
            }

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        if not current_price:
            return {"Implied Volatility (%)": None, "IV Signal": None}

        # Find expiration closest to 30 days
        today = pd.Timestamp.now()
        target_date = today + pd.DateOffset(days=30)

        best_expiry = None
        best_diff = float("inf")
        for exp in expirations:
            exp_date = pd.Timestamp(exp)
            diff = abs((exp_date - target_date).days)
            if diff < best_diff:
                best_diff = diff
                best_expiry = exp

        if not best_expiry or best_diff > 45:  # No expiry within 45 days
            return {"Implied Volatility (%)": None, "IV Signal": None}

        # Get option chain
        chain = ticker.option_chain(best_expiry)
        calls = chain.calls
        puts = chain.puts

        if calls.empty and puts.empty:
            return {"Implied Volatility (%)": None, "IV Signal": None}

        # Find ATM options (closest to current price)
        def find_atm_iv(options_df):
            if options_df.empty or "strike" not in options_df.columns:
                return None
            options_df = options_df.copy()
            options_df["distance"] = abs(options_df["strike"] - current_price)
            atm = options_df.nsmallest(3, "distance")  # Take 3 closest
            if "impliedVolatility" in atm.columns:
                ivs = atm["impliedVolatility"].dropna()
                return ivs.mean() if len(ivs) > 0 else None
            return None

        call_iv = find_atm_iv(calls)
        put_iv = find_atm_iv(puts)

        # Average call and put IV
        ivs = [iv for iv in [call_iv, put_iv] if iv is not None]
        if not ivs:
            return {"Implied Volatility (%)": None, "IV Signal": None}

        avg_iv = sum(ivs) / len(ivs) * 100  # Convert to percentage

        # Put/Call ratio for sentiment
        put_call_ratio = None
        if not puts.empty and not calls.empty:
            put_oi = puts["openInterest"].sum() if "openInterest" in puts.columns else 0
            call_oi = calls["openInterest"].sum() if "openInterest" in calls.columns else 0
            if call_oi > 0:
                put_call_ratio = put_oi / call_oi

        # IV interpretation
        if avg_iv > 60:
            iv_signal = "Very High Volatility Expected"
        elif avg_iv > 40:
            iv_signal = "High Volatility Expected"
        elif avg_iv > 25:
            iv_signal = "Moderate Volatility"
        else:
            iv_signal = "Low Volatility Expected"

        return {
            "Implied Volatility (%)": round(avg_iv, 1),
            "IV Signal": iv_signal,
            "Put/Call OI Ratio": round(put_call_ratio, 2) if put_call_ratio else None,
            "Options Expiry Used": best_expiry,
        }

    except Exception as e:
        logger.debug(f"Options IV analysis failed: {str(e)}")
        return {}


def get_52_week_position(info: dict) -> dict:
    """
    Analyze stock's position relative to 52-week high/low.
    """
    try:
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        high_52w = info.get("fiftyTwoWeekHigh")
        low_52w = info.get("fiftyTwoWeekLow")

        if not all([current_price, high_52w, low_52w]):
            return {}

        # Distance from 52-week high (drawdown)
        drawdown = ((high_52w - current_price) / high_52w) * 100

        # Position in 52-week range (0% = at low, 100% = at high)
        range_position = ((current_price - low_52w) / (high_52w - low_52w)) * 100

        # Interpretation
        if range_position > 90:
            position_signal = "Near 52-Week High (Momentum)"
        elif range_position > 70:
            position_signal = "Upper Range (Strong)"
        elif range_position > 30:
            position_signal = "Middle Range"
        elif range_position > 10:
            position_signal = "Lower Range (Potential Value)"
        else:
            position_signal = "Near 52-Week Low (Contrarian)"

        return {
            "52-Week Drawdown (%)": round(drawdown, 2),
            "52-Week Range Position (%)": round(range_position, 1),
            "52-Week Position Signal": position_signal,
            "52-Week High": high_52w,
            "52-Week Low": low_52w,
        }

    except Exception as e:
        logger.debug(f"52-week position analysis failed: {str(e)}")
        return {}


def run_stage2(stage1_df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2: Deep financial analysis on Stage 1 candidates.
    Calculates P/FCF, ROIC, and 3-year CAGR, and applies sector-relative screening.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Running Custom Ticker Analysis")
    print("=" * 60)

    if stage1_df.empty:
        return pd.DataFrame()

    results = []
    tickers = stage1_df["symbol"].tolist() if "symbol" in stage1_df.columns else []
    risk_free_rate = fetch_treasury_yield()
    print(f"  Using Risk-Free Rate (10Y Treasury): {risk_free_rate * 100:.2f}%")

    # Define relative tolerance from config, with a fallback
    SECTOR_TOLERANCE = CONFIG.get("sector_relative_tolerance", 1.2)  # e.g., 20% premium

    # Phase 3: Batch fetch historical data for technicals and ML
    print(f"  Batch fetching 1Y price history for {len(tickers)} tickers...")
    try:
        # We need historical data for ML training (last ~2 years is better but 1y is okay for now)
        # Extending to 2y for better ML training
        batch_history = yf.download(
            tickers + ["SPY"],
            period="2y",
            interval="1d",
            progress=False,
            group_by="ticker",
        )
    except Exception as e:
        print(f"  Batch download failed: {e}. Falling back to individual.")
        batch_history = pd.DataFrame()

    # Initialize ML Engine
    from ml_engine import AlphaPredictor

    ml_predictor = None
    if not batch_history.empty:
        # Convert batch_history MultiIndex to dict of DataFrames {ticker: df}
        print("  Initializing Quant-ML Alpha Predictor...", end=" ")
        data_dict = {}
        # Handle the structure of yf.download result
        # If multiple tickers: columns are (Ticker, PriceType) or (PriceType, Ticker) ?
        # usually group_by='ticker' -> Level 0 is Ticker.

        # Check structure
        if isinstance(batch_history.columns, pd.MultiIndex):
            # Level 0 is Ticker if group_by='ticker'
            for t in tickers + ["SPY"]:
                try:
                    if t in batch_history.columns.levels[0]:
                        data_dict[t] = batch_history[t]["Close"]
                except KeyError:
                    pass
        else:
            # Single ticker case
            pass

        if len(data_dict) > 1:
            try:
                ml_predictor = AlphaPredictor(data_dict, benchmark_symbol="SPY")
                ml_predictor.train_model()
            except Exception as e:
                logger.warning(f"ML Training failed: {str(e)}")
                ml_predictor = None
        else:
            print("Insufficient data for ML.")

    for i, symbol in enumerate(tickers):
        print(f"  Analyzing {symbol} ({i + 1}/{len(tickers)})...", end=" ")
        try:
            ticker = get_ticker(symbol)
            info = ticker.info
            rating = (info.get("recommendationKey") or "").lower()
            num_analysts = info.get("numberOfAnalystOpinions") or 0

            current_sector = info.get("sector")
            if not current_sector or current_sector in CONFIG.get(
                "excluded_sectors", []
            ):
                print(f"SKIP (Sector: {current_sector or 'N/A'})")
                continue

            market_cap = info.get("marketCap", 0)
            if not market_cap:
                print("SKIP (no market cap)")
                continue

            # --- Key Metric Calculations ---
            pfcf = calculate_pfcf(ticker, market_cap, info)
            roic = calculate_roic(ticker)
            income = ticker.financials
            rev_cagr = (
                calculate_cagr(income.loc["Total Revenue"].dropna().tolist(), 3)
                if not income.empty and "Total Revenue" in income.index
                else None
            )
            rev_cagr_pct = rev_cagr * 100 if rev_cagr is not None else None

            # --- NEW: Enhanced Quality & Valuation Metrics ---
            piotroski = calculate_piotroski_score(ticker)
            ev_ebitda = calculate_ev_ebitda(ticker, info)
            ttm_metrics = calculate_ttm_metrics(ticker)

            # --- NEW: Comprehensive Due Diligence Checks ---
            altman_z = calculate_altman_z_score(ticker, info)
            cash_flow_quality = calculate_cash_flow_quality(ticker)
            accounting_flags = detect_accounting_red_flags(ticker)
            debt_maturity = analyze_debt_maturity_risk(ticker, info)
            dividend_analysis = check_dividend_sustainability(ticker, info)

            # --- Sector-Relative Analysis ---
            sector_medians = get_sector_median_metrics(current_sector)

            # --- New Advanced Metrics (Phase 3) ---
            growth_est = fetch_growth_estimates(ticker)
            growth_est_pct = growth_est.get("Next Year Growth Est (%)")
            eps_rev = fetch_eps_revisions(ticker)
            analyst_trend = fetch_analyst_recommendations(ticker)

            # --- Technicals & Calendar (Phase 4), & Phase 5 (Valuation/Timing) ---
            # Extract batch data if available
            ticker_close = None
            if not batch_history.empty:
                try:
                    # yf.download with group_by='ticker' returns MultiIndex (Ticker, PriceType)
                    # or if single ticker, just PriceType cols.
                    if len(tickers) == 1:
                        if "Close" in batch_history.columns:
                            ticker_close = batch_history["Close"]
                    else:
                        if symbol in batch_history.columns.levels[0]:
                            ticker_close = batch_history[symbol]["Close"]
                except Exception:
                    pass

            technicals = fetch_technical_signals(ticker, close_series=ticker_close)
            calendar = check_earnings_calendar(ticker)
            rsi = fetch_rsi(ticker, close_series=ticker_close)
            peg = calculate_peg_ratio(ticker, info)
            peg_ratio = peg.get("PEG Ratio")
            sector_trend = fetch_sector_trend(current_sector)

            # --- ML Alpha Prediction (Return Forecast) ---
            ml_score = 0.0  # Default 0% return
            if ml_predictor:
                try:
                    ml_score = ml_predictor.predict_alpha_probability(symbol)
                except Exception:
                    ml_score = 0.0

            # --- Stage 2 Filtering ---
            passed = True
            fail_reasons = []

            # Enforce Trend Filter (Price > SMA200) from Phase 4 calculations
            sma200 = technicals.get("SMA200")
            price_for_trend = info.get("currentPrice") or info.get("previousClose")

            if price_for_trend and sma200:
                if price_for_trend < sma200:
                    passed = False
                    fail_reasons.append(
                        f"Downtrend (Price {price_for_trend:.2f} < SMA200 {sma200:.2f})"
                    )

            # Absolute checks (hard rules)
            if pfcf is None or pfcf <= 0:
                passed = False
                fail_reasons.append("No/negative FCF")
            elif pfcf > PFCF_MAX:
                passed = False
                fail_reasons.append(f"P/FCF > {PFCF_MAX}")
            if roic is not None and roic < ROIC_MIN:
                passed = False
                fail_reasons.append(f"ROIC < {ROIC_MIN * 100:.0f}%")
            if rev_cagr is not None and rev_cagr < CAGR_MIN:
                passed = False
                fail_reasons.append(f"3Y Rev CAGR < {CAGR_MIN * 100:.0f}%")
            if rating and rating not in ACCEPTABLE_RATINGS:
                passed = False
                fail_reasons.append(f"Rating:{rating}")

            # Sector-relative checks
            pe = info.get("trailingPE")
            median_pe = sector_medians.get("Sector Median P/E")
            if pe and median_pe:
                pe_rich = pe > median_pe * SECTOR_TOLERANCE
                if pe_rich:
                    growth_ok = False
                    if peg_ratio is not None and peg_ratio <= 1.5:
                        growth_ok = True
                    if rev_cagr_pct is not None and rev_cagr_pct >= (
                        CAGR_MIN * 100 * 1.5
                    ):
                        growth_ok = True
                    if growth_est_pct is not None and growth_est_pct >= 15:
                        growth_ok = True
                    if not growth_ok:
                        passed = False
                        fail_reasons.append("P/E rich vs sector without growth")

            roe = info.get("returnOnEquity")
            median_roe = sector_medians.get("Sector Median ROE")
            if (
                roe and median_roe and roe < median_roe * 0.8
            ):  # Must be at least 80% of sector median
                passed = False
                fail_reasons.append("ROE < Sector Median")

            # --- Compile Data ---
            status = "PASS" if passed else f"FAIL ({', '.join(fail_reasons)})"
            print(status)

            current_price = info.get("currentPrice")
            de_ratio = info.get("debtToEquity")
            beta = info.get("beta")
            dividend_yield = info.get("dividendYield")
            revenue_growth = info.get("revenueGrowth") or info.get(
                "revenueQuarterlyGrowth"
            )
            earnings_growth = info.get("earningsQuarterlyGrowth")
            fifty_two_low = info.get("fiftyTwoWeekLow")
            fifty_two_high = info.get("fiftyTwoWeekHigh")
            graham_number = calculate_graham_number(
                info.get("trailingEps"), info.get("bookValue")
            )
            graham_upside = (
                ((graham_number - current_price) / current_price) * 100
                if graham_number and current_price
                else None
            )

            dcf_data = calculate_dcf_fair_value(ticker, info, risk_free_rate)

            # Fetch earnings surprise data
            earnings_surprise_data = fetch_earnings_surprise(ticker)

            # Fetch news data (sentiment analysis removed)
            news_sentiment_data = get_news_data(symbol, ticker)

            # --- NEW: Enhanced yfinance Data Metrics ---
            short_interest = get_short_interest_metrics(info)
            analyst_upgrades = get_analyst_upgrades_downgrades(ticker)
            target_analysis = get_target_price_analysis(info)
            ownership_metrics = get_insider_ownership_metrics(info, ticker)
            dividend_growth = get_dividend_growth_analysis(ticker, info)
            analyst_sentiment = get_recommendations_sentiment(ticker)
            esg_data = get_esg_scores(ticker)
            options_iv = get_options_implied_volatility(ticker, info)
            week_52_position = get_52_week_position(info)
            fcf_yield_data = calculate_fcf_yield(ticker, info, risk_free_rate)
            growth_quality = calculate_growth_quality(ticker)

            # Calculate valuation consistency (needs all valuation data)
            valuation_consistency = check_valuation_consistency(
                {
                    "DCF Upside (%)": dcf_data.get("DCF Upside (%)"),
                    "Graham Upside (%)": graham_upside,
                    "Upside (%)": (info.get("targetMeanPrice", 0) - current_price)
                    / current_price
                    * 100
                    if info.get("targetMeanPrice") and current_price
                    else None,
                }
            )

            result_row = {
                "Symbol": symbol,
                "Name": info.get("shortName", "N/A"),
                "Sector": current_sector,
                "Market Cap ($B)": market_cap / 1e9,
                "P/E": pe,
                "ROE (%)": (roe or 0) * 100,
                "P/FCF": pfcf,
                "D/E": de_ratio,
                "Beta": beta,
                "ROIC (%)": (roic or 0) * 100,
                "Dividend Yield (%)": (dividend_yield or 0) * 100
                if dividend_yield
                else None,
                "Quarterly Revenue Growth (%)": (revenue_growth or 0) * 100
                if revenue_growth
                else None,
                "Quarterly Earnings Growth (%)": (earnings_growth or 0) * 100
                if earnings_growth
                else None,
                "52W Low": fifty_two_low,
                "52W High": fifty_two_high,
                "Current Price": current_price,
                "3Y Rev CAGR (%)": rev_cagr_pct or 0,
                "Analyst Rating": rating,
                "# Analysts": num_analysts,
                "Upside (%)": (
                    (info.get("targetMeanPrice", 0) - current_price) / current_price
                )
                * 100
                if info.get("targetMeanPrice") and current_price
                else None,
                "Graham Undervalued": current_price < graham_number
                if current_price and graham_number
                else False,
                "Graham Upside (%)": graham_upside,
                "Stage 2 Pass": passed,
                "Sector Median P/E": sector_medians.get("Sector Median P/E"),
                "Sector Median ROE": sector_medians.get("Sector Median ROE"),
                **dcf_data,
                **sector_medians,
                **growth_est,
                **eps_rev,
                **analyst_trend,
                **technicals,
                **calendar,
                **rsi,
                **peg,
                **sector_trend,
                "ML Alpha Score": ml_score,
                **piotroski,
                "EV/EBITDA": ev_ebitda,
                **ttm_metrics,
                **altman_z,
                **cash_flow_quality,
                **accounting_flags,
                **debt_maturity,
                **dividend_analysis,
                **valuation_consistency,
                **earnings_surprise_data,
                **news_sentiment_data,
                # NEW: Enhanced yfinance data metrics
                **short_interest,
                **analyst_upgrades,
                **target_analysis,
                **ownership_metrics,
                **dividend_growth,
                **analyst_sentiment,
                **esg_data,
                **options_iv,
                **week_52_position,
                **fcf_yield_data,
                **growth_quality,
            }

            # Calculate Conviction Score (Fix for KeyError)
            # We explicitly pass ml_score to the calculation if needed, or rely on it being in result_row
            score, reasons = calculate_conviction_score(result_row)
            result_row["Conviction Score"] = score
            result_row["Conviction Reasons"] = "; ".join(reasons)

            results.append(result_row)

        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}", exc_info=True)
            print(f"ERROR processing {symbol}: {e}")
            continue

    return pd.DataFrame(results)


# =============================================================================
# PORTFOLIO OPTIMIZATION
# =============================================================================
def build_optimized_portfolio(
    passed_df: pd.DataFrame,
    min_stocks: int = 6,
    max_weight: float = 0.20,
    objective: str = "sharpe",
) -> pd.DataFrame:
    """
    Build an optimized portfolio from screener results.
    Objectives: 'sharpe' (default), 'return', 'sortino'
    """
    print("\n" + "=" * 60)
    print(f"PORTFOLIO OPTIMIZATION - Objective: Maximize {objective.capitalize()}")
    print("=" * 60)

    # Get unique symbols and sectors
    if passed_df.empty or len(passed_df) < min_stocks:
        print(
            f"\nInsufficient stocks ({len(passed_df)}) for optimization. Using fallback strategy."
        )
        return _build_fallback_portfolio(passed_df)

    # Apply sector diversification - max 2 stocks per sector initially
    symbols = []
    sector_counts = {}
    max_sector_weight = max_weight
    max_stocks_per_sector = max_sector_weight / passed_df["Sector"].nunique()
    # Sort by Conviction Score (descending) to prioritize best ideas, then Market Cap as tiebreaker
    passed_df = passed_df.sort_values(
        by=["Conviction Score", "Market Cap ($B)"], ascending=[False, False]
    )
    for _, row in passed_df.iterrows():
        sector = row.get("Sector", "Unknown")
        if sector_counts.get(sector, 0) < max_stocks_per_sector:  # Max 2 per sector
            symbols.append(row["Symbol"])
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

    # Ensure minimum stocks
    if len(symbols) < min_stocks:
        # Add more stocks if needed, relaxing sector constraint
        for _, row in passed_df.iterrows():
            if row["Symbol"] not in symbols:
                symbols.append(row["Symbol"])
            if len(symbols) >= min_stocks:
                break

    print(f"Selected {len(symbols)} stocks for optimization: {', '.join(symbols)}")

    # Fetch historical data for optimization
    try:
        # Download historical returns
        print("Fetching 5-year historical data...")
        data = yf.download(
            symbols, period="5y", progress=False, threads=4, rounding=True
        )["Close"]

        if data.empty:
            print("Could not fetch historical data. Using equal weight.")
            return _build_equal_weight_portfolio(symbols)

        # Calculate daily returns
        returns = data.pct_change().dropna()

        if len(returns) < 30:
            print("Insufficient historical data. Using equal weight.")
            return _build_equal_weight_portfolio(symbols)

        # Calculate expected returns and covariance
        mean_returns = returns.mean() * 252  # Annualize
        cov_matrix = returns.cov() * 252  # Annualize

        # Regularize the covariance matrix for stability
        cov_matrix_reg = cov_matrix + np.identity(len(symbols)) * 1e-6

        # Optimize for Sharpe ratio with constraints
        n_assets = len(symbols)
        risk_free_rate = fetch_treasury_yield()
        print(f"  Using Risk-Free Rate for optimization: {risk_free_rate * 100:.2f}%")

        # Optimize for Sharpe ratio using scipy.optimize
        def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
            p_ret = np.dot(weights, mean_returns)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            # Handle cases where volatility is zero or near-zero
            if p_vol < 1e-9:
                return np.inf
            return -(p_ret - risk_free_rate) / p_vol

        def negative_total_return(weights, mean_returns):
            return -np.dot(weights, mean_returns)

        def negative_sortino(weights, returns_df, risk_free_rate):
            # Sortino needs full return series, not just mean/cov
            p_daily_ret = must_calculate_portfolio_daily_returns(weights, returns_df)
            mean_ret = p_daily_ret.mean() * 252

            # Downside dev
            target = 0  # or risk_free_rate / 252
            downside = p_daily_ret[p_daily_ret < target]
            if len(downside) == 0:
                downside_dev = 1e-9
            else:
                downside_dev = downside.std() * np.sqrt(252)

            if downside_dev < 1e-9:
                return np.inf

            return -(mean_ret - risk_free_rate) / downside_dev

        # Helper for sortino
        def must_calculate_portfolio_daily_returns(weights, returns_df):
            return returns_df.dot(weights)

        # Constraints: sum of weights = 1
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        # Bounds: 0 <= weight <= max_weight
        bounds = tuple((0.02, max_weight) for _ in range(n_assets))

        print(f"Optimizing for {objective} (scipy)...")

        args = ()
        fun = None

        if objective == "return":
            fun = negative_total_return
            args = (mean_returns,)
        elif objective == "sortino":
            # For Sortino we need the raw returns dataframe passed to the cost function
            # optimization function signature match
            def sortino_wrapper(w):
                return negative_sortino(w, returns, risk_free_rate)

            fun = sortino_wrapper
            args = ()
        else:
            # Default to Sharpe
            fun = negative_sharpe
            args = (mean_returns, cov_matrix_reg, risk_free_rate)

        result = minimize(
            fun,
            n_assets * [1.0 / n_assets],  # Initial guess: equal weights
            args=args,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            best_weights = result.x
            best_sharpe = -result.fun
        else:
            print(f"Optimization failed: {result.message}. Using equal weights.")
            best_weights = np.array([1 / n_assets] * n_assets)
            best_sharpe = 0

        print("Optimization complete.")
        # Build result dataframe
        portfolio_df = pd.DataFrame(
            {
                "Symbol": symbols,
                "Weight (%)": best_weights * 100,
                "Sector": [
                    passed_df[passed_df["Symbol"] == s]["Sector"].values[0]
                    if s in passed_df["Symbol"].values
                    else "N/A"
                    for s in symbols
                ],
            }
        )

        # Sort by weight
        portfolio_df = portfolio_df.sort_values("Weight (%)", ascending=False)

        # Calculate portfolio metrics
        port_return = np.dot(best_weights, mean_returns) * 100
        port_volatility = (
            np.sqrt(np.dot(best_weights.T, np.dot(cov_matrix, best_weights))) * 100
        )

        print("\nOptimized Portfolio:")
        print(portfolio_df.to_string(index=False))
        print("\nPortfolio Metrics:")
        print(f"  Expected Annual Return: {port_return:.1f}%")
        print(f"  Expected Volatility: {port_volatility:.1f}%")
        print(f"  Sharpe Ratio: {best_sharpe:.2f}")
        print(f"  Max Position: {portfolio_df['Weight (%)'].max():.1f}%")

        # Correlation Analysis
        try:
            corr_matrix = returns.corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val > 0.7:
                        high_corr_pairs.append(
                            {
                                "Stock 1": corr_matrix.columns[i],
                                "Stock 2": corr_matrix.columns[j],
                                "Correlation": corr_val,
                            }
                        )

            if high_corr_pairs:
                print("\n⚠️  High Correlation Warnings (>0.7):")
                for pair in high_corr_pairs:
                    print(
                        f"  {pair['Stock 1']} <-> {pair['Stock 2']}: {pair['Correlation']:.2f}"
                    )
            else:
                avg_corr = corr_matrix.values[
                    np.triu_indices_from(corr_matrix.values, k=1)
                ].mean()
                print(
                    f"\n✓ Portfolio Diversification: Avg Correlation = {avg_corr:.2f}"
                )
        except Exception:
            pass  # Skip correlation if calculation fails

        # Export portfolio
        portfolio_df.to_csv("portfolio_allocation.csv", index=False)
        print("\nPortfolio exported to: portfolio_allocation.csv")

        return portfolio_df

    except Exception as e:
        print(f"Optimization failed: {e}. Using equal weight.")
        return _build_equal_weight_portfolio(symbols)


def _build_fallback_portfolio(passed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback portfolio: equal weight with stock picks + BIL + TLT.
    Used when fewer than 4 stocks pass screening.
    """
    print("\nBuilding fallback portfolio with BIL (T-bills) and TLT (20-yr bonds)...")

    # Get any passing stocks
    stock_symbols = passed_df["Symbol"].tolist() if not passed_df.empty else []

    # Add BIL and TLT
    all_symbols = stock_symbols + ["BIL", "TLT"]

    # Equal weight
    weight = 100 / len(all_symbols)

    portfolio_df = pd.DataFrame(
        {
            "Symbol": all_symbols,
            "Weight (%)": [weight] * len(all_symbols),
            "Sector": [
                passed_df[passed_df["Symbol"] == s]["Sector"].values[0]
                if s in passed_df["Symbol"].values
                else ("T-Bills" if s == "BIL" else "Bonds")
                for s in all_symbols
            ],
        }
    )

    print("\nFallback Portfolio (Equal Weight):")
    print(portfolio_df.to_string(index=False))

    # Export
    portfolio_df.to_csv("portfolio_allocation.csv", index=False)
    print("\nPortfolio exported to: portfolio_allocation.csv")

    return portfolio_df


def _build_equal_weight_portfolio(symbols: list) -> pd.DataFrame:
    """Simple equal-weight portfolio when optimization fails."""
    weight = 100 / len(symbols)

    portfolio_df = pd.DataFrame(
        {
            "Symbol": symbols,
            "Weight (%)": [weight] * len(symbols),
            "Sector": ["N/A"] * len(symbols),
        }
    )

    print("\nEqual Weight Portfolio:")
    print(portfolio_df.to_string(index=False))

    portfolio_df.to_csv("portfolio_allocation.csv", index=False)
    print("\nPortfolio exported to: portfolio_allocation.csv")

    return portfolio_df


def parse_args():
    parser = argparse.ArgumentParser(description="GARP stock screener")
    parser.add_argument(
        "--tickers",
        nargs="+",
        help="Skip Stage 1 and run Stage 2 on specific tickers.",
    )
    parser.add_argument(
        "--stage1-size",
        type=int,
        default=DEFAULT_STAGE1_SIZE,
        help="Max number of tickers to request from the Stage 1 API filter.",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip investment summary generation.",
    )
    parser.add_argument(
        "--skip-portfolio",
        action="store_true",
        help="Skip portfolio optimization/backtest and only export screener results.",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Skip PDF research report generation.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run a 1-year backtest performance analysis of the final portfolio.",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="return",
        choices=["sharpe", "return", "sortino"],
        help="Portfolio optimization objective (sharpe, return, sortino).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    """Main entry point for the GARP stock screener."""
    print("\n" + "=" * 60)
    print("       GARP STOCK SCREENER - Two-Stage Implementation")
    print("=" * 60)

    # Stage 1: Fast API filtering (or manual tickers)
    if args.tickers:
        print(
            f"Skipping Stage 1 API call. Running Stage 2 on user tickers: {', '.join(args.tickers)}"
        )
        stage1_df = build_manual_stage1_df(args.tickers)
    else:
        stage1_df = run_stage1(limit=args.stage1_size)

    if stage1_df.empty:
        print("\nNo stocks passed Stage 1 filters. Consider relaxing criteria.")
        return

    # Stage 2: Deep analysis
    results_df = run_stage2(stage1_df)

    if results_df.empty:
        print("\nNo stocks passed Stage 2 analysis.")
        return

    # Filter to only passing stocks and sort by Conviction Score (descending)
    passed_df = results_df[results_df["Stage 2 Pass"]].copy()
    passed_df = passed_df.sort_values("Conviction Score", ascending=False)

    # Display results
    print("\n" + "=" * 60)
    print("FINAL RESULTS - Stocks Passing All GARP Criteria")
    print("=" * 60)

    if passed_df.empty:
        print("No stocks passed all criteria.")
        print("\nAll Stage 1 candidates (for review):")
        print(results_df.to_string(index=False))
    else:
        display_cols = [
            "Symbol",
            "Name",
            "Sector",
            "Market Cap ($B)",
            "P/E",
            "Sector Median P/E",
            "ROE (%)",
            "Sector Median ROE",
            "P/FCF",
            "ROIC (%)",
            "3Y Rev CAGR (%)",
        ]
        print(passed_df[display_cols].to_string(index=False))
        print(f"\nTotal: {len(passed_df)} stocks passed all criteria")

    # Export to CSV
    output_file = "screener_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nFull results for all analyzed stocks exported to: {output_file}")

    # Build optimized portfolio from passing stocks
    portfolio_df = pd.DataFrame()
    advice_df = pd.DataFrame()
    if not passed_df.empty and not args.skip_portfolio:
        portfolio_df = build_optimized_portfolio(passed_df, objective=args.objective)

        # Run Backtest if requested
        if args.backtest and not portfolio_df.empty:
            tickers = portfolio_df["Symbol"].tolist()
            weights = (portfolio_df["Weight (%)"] / 100.0).tolist()
            backtest_portfolio(tickers, weights)

    if not passed_df.empty and not args.skip_summary:
        # Generate investment summaries
        advice_df = generate_stock_summaries(passed_df)

        # Generate Professional PDF Report
        if not args.skip_pdf:
            try:
                from report_generator import generate_pdf_report

                rf_rate = fetch_treasury_yield()
                generate_pdf_report(advice_df, portfolio_df, rf_rate)
            except ImportError:
                print("\nSkipping PDF report: reportlab library not found.")
                print("Install it with: pip install reportlab")
            except Exception as e:
                print(f"\nCould not generate PDF report: {e}")

    end_time = time.time()
    elapsed_time = timedelta(seconds=end_time - start_time)
    print(f"\nTotal runtime: {elapsed_time}")


if __name__ == "__main__":
    main()
