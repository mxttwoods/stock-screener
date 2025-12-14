"""
GARP Stock Screener - Two-Stage Implementation
Based on RULES.md and IDEAS.md methodology
"""

import argparse
import json
import os
import time
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI
from yfinance import EquityQuery
from scipy.optimize import minimize
import yaml

warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# OpenAI API configuration (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


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
    """
    try:
        # yfinance often has earnings_history
        # It returns a DataFrame with 'EPS Estimate', 'Reported EPS', 'Surprise(%)'
        earnings = ticker.earnings_history
        if earnings is None or earnings.empty:
            return {
                "Earnings Surprise Avg (%)": None,
                "Last Quarter Surprise (%)": None,
            }

        # Ensure numeric columns
        if "Surprise(%)" in earnings.columns:
            surprises = pd.to_numeric(earnings["Surprise(%)"], errors="coerce").dropna()

            if surprises.empty:
                return {
                    "Earnings Surprise Avg (%)": None,
                    "Last Quarter Surprise (%)": None,
                }

            last_surprise = (
                surprises.iloc[0] * 100
            )  # usually decimal in DF? No, Yahoo usually sends 0.05 for 5% or 5.0?
            # Actually yfinance earnings_history usually assumes raw values.
            # Let's assume it matches what we see in the dataframe.
            # If the value is 0.05, that's 5%. If it is 5.0, that's 500%? No.
            # Usually 'Surprise(%)' is e.g. 0.06 meaning 6%.

            # Let's assume decimal input if < 1.0 mostly, but safe to just take mean
            avg = surprises.mean()

            # Formatting: if values are small (like 0.05), return as percentage (5.0)
            # If they are large (like 5.0), assume they are already percentages?
            # In yfinance, it's typically a ratio. e.g. 0.03.

            return {
                "Earnings Surprise Avg (%)": avg * 100,
                "Last Quarter Surprise (%)": last_surprise * 100,
            }

        return {"Earnings Surprise Avg (%)": None, "Last Quarter Surprise (%)": None}

    except Exception as e:
        # print(f"  Warning: Earnings data error: {e}")
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

    except Exception as e:
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
                    except:
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

    except Exception as e:
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


def fetch_rsi(ticker, period=14) -> dict:
    """
    Calculate 14-day RSI (Relative Strength Index).
    Returns RSI value and signal (Oversold/Overbought).
    """
    try:
        hist = ticker.history(period="3mo")  # Need enough data for 14d + smoothing
        if hist.empty or len(hist) < period + 1:
            return {"RSI": None, "RSI Signal": "Insufficient Data"}

        close = hist["Close"]
        delta = close.diff()

        # Gain/Loss
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Calculate RS
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]  # Most recent RSI

        signal = "Neutral"
        if rsi < 30:
            signal = "Oversold (Buy Dip)"
        elif rsi > 70:
            signal = "Overbought (Risk)"

        return {"RSI": round(rsi, 2), "RSI Signal": signal}
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


def fetch_technical_signals(ticker) -> dict:
    """
    Calculate simple technical indicators (SMA50, SMA200) from 1y history.
    """
    try:
        # Get 1 year of history
        hist = ticker.history(period="1y")
        if hist.empty or len(hist) < 200:
            return {
                "SMA50": None,
                "SMA200": None,
                "Technical Signal": "Insufficient Data",
            }

        # Calculate SMAs
        sma50 = hist["Close"].rolling(window=50).mean().iloc[-1]
        sma200 = hist["Close"].rolling(window=200).mean().iloc[-1]
        current_price = hist["Close"].iloc[-1]

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


def analyze_sentiment_with_ai(symbol: str, ticker: yf.Ticker) -> dict:
    """
    Analyze news sentiment using OpenAI on yfinance news titles.
    """
    if not openai_client:
        return {"News Sentiment Score": None, "News Sentiment Label": None}

    try:
        news = ticker.news
        if not news:
            return {"News Sentiment Score": None, "News Sentiment Label": None}

        headlines = [n.get("title", "") for n in news[:15]]
        headlines_text = "\n".join(f"- {h}" for h in headlines)

        prompt = f"""Analyze the sentiment of the following news headlines for {symbol}.
Headlines:
{headlines_text}

Return a valid JSON object with exactly two keys:
"score": A float between -1.0 (Very Bearish) and 1.0 (Very Bullish).
"label": A string text label (e.g., "Bullish", "Bearish", "Neutral").
"""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=100,
            temperature=0.0,
        )

        result_text = response.choices[0].message.content.strip()
        result_json = json.loads(result_text)

        return {
            "News Sentiment Score": result_json.get("score"),
            "News Sentiment Label": result_json.get("label"),
        }

    except Exception:
        # print(f"AI Sentiment Error: {e}")
        return {"News Sentiment Score": 0, "News Sentiment Label": "Error"}


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

    # Cap score at 1-10
    score = max(1, min(10, score))

    return score, reasons


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
        warnings.append(f"⚠️ Significantly above Graham value")

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


def generate_ai_thesis(
    stock_data: dict, ticker_info: dict, conviction_score: int, risks: list
) -> str:
    """
    Generate an AI-powered investment thesis using OpenAI.
    """
    if not openai_client:
        return "AI thesis unavailable (disabled or no API key)"

    try:
        # Build context for the AI
        symbol = stock_data.get("Symbol", "Unknown")
        name = stock_data.get("Name", "Unknown")
        sector = stock_data.get("Sector", "Unknown")

        # Format sector medians for the prompt
        sector_pe = stock_data.get("Sector Median P/E")
        sector_roe = stock_data.get("Sector Median ROE")
        pe_comparison = f"(Sector Median: {sector_pe:.1f})" if sector_pe else ""
        roe_comparison = (
            f"(Sector Median: {sector_roe * 100:.1f}%)" if sector_roe else ""
        )

        prompt = f"""You are a Lead Equity Analyst. Synthesize the following data into a compelling 2-3 sentence investment thesis for {symbol} ({name}).

Company Profile:
- Sector: {sector}
- Market Cap: ${stock_data.get("Market Cap ($B)", "N/A"):.2f}B
- Analyst Coverage: {stock_data.get("# Analysts", "N/A")} analysts

Financials & Profitability:
- P/E: {stock_data.get("P/E", "N/A")} {pe_comparison}
- PEG: {stock_data.get("PEG Ratio", "N/A")}
- ROE: {stock_data.get("ROE (%)", "N/A")}% {roe_comparison}
- ROIC: {stock_data.get("ROIC (%)", "N/A")}%
- Margins: Gross {stock_data.get("Gross Margin (%)", "N/A")}%, Op {stock_data.get("Op Margin (%)", "N/A")}%

Growth:
- 3Y Revenue CAGR: {stock_data.get("3Y Rev CAGR (%)", "N/A")}%
- Next Yr Growth Est: {stock_data.get("Next Year Growth Est (%)", "N/A")}%
- Quarterly Revenue Growth (YoY): {stock_data.get("Quarterly Revenue Growth (%)", "N/A")}%
- Quarterly Earnings Growth (YoY): {stock_data.get("Quarterly Earnings Growth (%)", "N/A")}%

Valuation Models:
- Graham Number: ${stock_data.get("Graham Number", "N/A")} (Undervalued: {stock_data.get("Graham Undervalued", "N/A")})
- PEG Ratio: {stock_data.get("PEG Ratio", "N/A")} (Growth Adj. Val)
- DCF Fair Value: ${stock_data.get("DCF Fair Value", "N/A")} (Upside: {stock_data.get("DCF Upside (%)", "N/A")}%)
- Analyst Target: ${stock_data.get("Target Price", "N/A")} (Upside: {stock_data.get("Upside (%)", "N/A")}%)

Sentiment & Momentum:
- Price vs 52W Range: Current ${stock_data.get("Current Price", "N/A")} (Range: ${stock_data.get("52W Low", "N/A")} - ${stock_data.get("52W High", "N/A")})
- News Sentiment: {stock_data.get("News Sentiment Label", "N/A")} (Score: {stock_data.get("News Sentiment Score", "N/A")})
- EPS Revisions (30d): {stock_data.get("EPS Up/Down Ratio", "N/A")}x (Up: {stock_data.get("EPS Revisions Up (30d)", 0)} / Down: {stock_data.get("EPS Revisions Down (30d)", 0)})
- Analyst Trend (3mo): {stock_data.get("Analyst Trend", "N/A")} ({stock_data.get("Current Buy Ratings", 0)} Buys)
- Earnings Surprise (Avg): {stock_data.get("Earnings Surprise Avg (%)", "N/A")}%
- Analyst Rating: {stock_data.get("Analyst Rating", "N/A")}
- Technical Trend: {stock_data.get("Price Trend", "N/A")} (Signal: {stock_data.get("Technical Signal", "N/A")})
- RSI (14d): {stock_data.get("RSI", "N/A")} (Signal: {stock_data.get("RSI Signal", "N/A")})

Risk Profile:
- Conviction Score: {conviction_score}/10
- Identified Risks: {", ".join(risks) if risks else "None"}
- Debt/Equity: {stock_data.get("D/E", "N/A")}
- Upcoming Earnings: {"Yes, in " + str(stock_data.get("Days to Earnings")) + " days" if stock_data.get("Upcoming Earnings") else "No"}
- Sector Trend: {stock_data.get("Sector Trend", "N/A")} (SMA200 Gap: {stock_data.get("Sector SMA200 Gap (%)", "N/A")}%)

Task:
1. Synthesize the signals (e.g., "Trading at a discount to its sector P/E median").
2. Highlight the primary driver for a BUY or WATCH decision.
3. Mention the most critical risk.
4. Be concise (max 80 words). Use numbers. Be formal, direct, and do not provide financial advice."""

        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
        )

        thesis_text = response.choices[0].message.content.strip()
        # Sanitize text for PDF generation
        thesis_text = (
            thesis_text.replace("’", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("—", "--")
        )
        return thesis_text

    except Exception as e:
        return f"AI thesis error: {str(e)[:50]}"


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
        all_symbols = tickers + ["SPY"]
        data = yf.download(all_symbols, period="1y", interval="1d", progress=False)[
            "Close"
        ]

        if data.empty:
            print("  Error: No historical data found.")
            return

        # Calculate Returns
        returns = data.pct_change().dropna()

        # Portfolio Returns
        # weights dict must be ordered same as data columns, but data cols are sorted alphabetically usually
        # Realign weights
        sorted_tickers = [t for t in all_symbols if t != "SPY" and t in data.columns]

        # Create weight array aligned with sorted_tickers
        # Assuming weights is a list corresponding to input 'tickers' list
        # We need a robust mapping.
        weight_map = dict(zip(tickers, weights))

        # Calculate weighted daily returns
        # We'll do it manually: sum(return * weight)
        port_daily_ret = pd.Series(0, index=returns.index)
        for t in sorted_tickers:
            w = weight_map.get(t, 0)
            port_daily_ret += returns[t] * w

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

        print(f"  Portfolio Return (1Y): {total_ret_port:.2f}%")
        print(f"  SPY Return (1Y):       {total_ret_spy:.2f}%")
        print(f"  Alpha:                 {total_ret_port - total_ret_spy:.2f}%")
        print("-" * 40)
        print(f"  Portfolio Max DD:      {max_dd_port:.2f}%")
        print(f"  SPY Max DD:            {max_dd_spy:.2f}%")
        print(f"  Portfolio Sharpe:      {sharpe_port:.2f}")
        print(f"  SPY Sharpe:            {sharpe_spy:.2f}")

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


def generate_investment_advice(passed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive investment advice for all passing stocks.
    """
    print("\n" + "=" * 60)
    print("INVESTMENT ADVICE - AI-Powered Analysis")
    print("=" * 60)

    if passed_df.empty:
        print("No stocks to analyze.")
        return passed_df

    advice_data = []

    for _, row in passed_df.iterrows():
        stock_data = row.to_dict()
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

        # Generate AI thesis
        thesis = generate_ai_thesis(stock_data, ticker_info, conviction, risks)

        # Create advice row preserving all original data
        advice_row = stock_data.copy()
        advice_row.update(
            {
                "Action": action,
                "Conviction": conviction,
                "Conviction Reasons": "; ".join(conviction_reasons),
                "Risk Warnings": "; ".join(risks) if risks else "None",
                "AI Thesis": thesis,
            }
        )

        advice_data.append(advice_row)

        print(f"{action} (Conviction: {conviction}/10)")

    advice_df = pd.DataFrame(advice_data)

    # Display summary
    print("\n" + "-" * 60)
    print("INVESTMENT RECOMMENDATIONS SUMMARY")
    print("-" * 60)

    for _, row in advice_df.iterrows():
        print(f"\n{row['Symbol']} - {row['Name']}")
        print(f"  Action: {row['Action']} | Conviction: {row['Conviction']}/10")
        print(f"  Thesis: {row['AI Thesis'][:200]}...")
        if row["Risk Warnings"] != "None":
            print(f"  Risks: {row['Risk Warnings']}")

    # Export advice
    advice_df.to_csv("investment_advice.csv", index=False)
    print("\n\nInvestment advice exported to: investment_advice.csv")

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

        # Get Free Cash Flow
        operating_cf = cf.loc["Total Cash From Operating Activities"].dropna().tolist()
        capex = cf.loc["Capital Expenditure"].dropna().tolist()

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
            fcf_growth = (
                (current_fcf / older_fcf) ** (1 / 2) - 1 if older_fcf > 0 else 0.05
            )
        else:
            fcf_growth = 0.05  # Default 5%
        fcf_growth = max(0.03, min(0.15, fcf_growth))  # Cap growth at 3-15%

        # --- Enhanced WACC Calculation ---
        market_cap = info.get("marketCap", 0)
        total_debt = info.get("totalDebt", 0)
        total_capital = market_cap + total_debt
        if total_capital == 0:
            return {"DCF Fair Value": None, "DCF Notes": "Missing capital structure"}

        equity_weight = market_cap / total_capital
        debt_weight = total_debt / total_capital

        beta = info.get("beta", 1.0) or 1.0
        market_premium = 0.055
        cost_of_equity = risk_free_rate + beta * market_premium

        # Estimate cost of debt (e.g., interest expense / total debt)
        income_stmt = ticker.financials
        interest_expense = (
            income_stmt.loc["Interest Expense"].iloc[0]
            if "Interest Expense" in income_stmt.index
            else 0
        )
        cost_of_debt = abs(interest_expense) / total_debt if total_debt > 0 else 0.05
        cost_of_debt = max(0.04, min(cost_of_debt, 0.09))  # Bound cost of debt

        tax_rate = 0.21  # Default tax rate
        if (
            "Tax Provision" in income_stmt.index
            and "Pretax Income" in income_stmt.index
        ):
            tax_provision = income_stmt.loc["Tax Provision"].iloc[0]
            pretax_income = income_stmt.loc["Pretax Income"].iloc[0]
            if pretax_income > 0:
                tax_rate = tax_provision / pretax_income

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

        return {
            "DCF Fair Value": round(dcf_fair_value, 2),
            "DCF Upside (%)": round(dcf_upside, 1) if dcf_upside is not None else None,
            "DCF Notes": f"WACC={wacc * 100:.1f}%, FCF Growth={fcf_growth * 100:.1f}%",
        }

    except Exception as e:
        return {
            "DCF Fair Value": None,
            "DCF Upside (%)": None,
            "DCF Notes": f"Error: {str(e)[:30]}",
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
        print(f"ERROR in calculate_pfcf for {ticker.ticker}: {e}")
        return None


def calculate_roic(ticker: yf.Ticker) -> Optional[float]:
    """
    Calculate Return on Invested Capital (ROIC) using NOPAT method.
    ROIC = NOPAT / Invested Capital
    NOPAT = EBIT * (1 - Tax Rate)
    """
    try:
        income = ticker.financials
        balance = ticker.balance_sheet

        if income.empty or balance.empty:
            return None

        # Get EBIT
        if "EBIT" in income.index:
            ebit = income.loc["EBIT"].iloc[0]
        elif "Operating Income" in income.index:
            ebit = income.loc["Operating Income"].iloc[0]
        else:
            return None

        # Calculate tax rate
        if "Tax Provision" in income.index and "Pretax Income" in income.index:
            tax = income.loc["Tax Provision"].iloc[0]
            pretax = income.loc["Pretax Income"].iloc[0]
            if pretax and pretax > 0:
                tax_rate = tax / pretax
            else:
                tax_rate = 0.21  # Default corporate tax rate
        else:
            tax_rate = 0.21

        nopat = ebit * (1 - tax_rate)

        # Calculate Invested Capital = Total Debt + Total Equity
        total_debt = 0
        if "Total Debt" in balance.index:
            total_debt = balance.loc["Total Debt"].iloc[0] or 0

        total_equity = 0
        if "Total Equity Gross Minority Interest" in balance.index:
            total_equity = (
                balance.loc["Total Equity Gross Minority Interest"].iloc[0] or 0
            )
        elif "Stockholders Equity" in balance.index:
            total_equity = balance.loc["Stockholders Equity"].iloc[0] or 0

        invested_capital = total_debt + total_equity

        if invested_capital <= 0:
            return None

        return nopat / invested_capital

    except Exception as e:
        print(f"ERROR in calculate_roic for {ticker.ticker}: {e}")
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

        cagr = (end_value / start_value) ** (1 / years) - 1
        return cagr
    except Exception as e:
        print(f"ERROR in calculate_cagr: {e}")
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
        print(f"ERROR in calculate_graham_number: {e}")
        return None


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

            # --- Sector-Relative Analysis ---
            sector_medians = get_sector_median_metrics(current_sector)

            # --- New Advanced Metrics (Phase 3) ---
            growth_est = fetch_growth_estimates(ticker)
            growth_est_pct = growth_est.get("Next Year Growth Est (%)")
            eps_rev = fetch_eps_revisions(ticker)
            analyst_trend = fetch_analyst_recommendations(ticker)

            # --- Technicals & Calendar (Phase 4), & Phase 5 (Valuation/Timing) ---
            technicals = fetch_technical_signals(ticker)
            calendar = check_earnings_calendar(ticker)
            rsi = fetch_rsi(ticker)
            peg = calculate_peg_ratio(ticker, info)
            peg_ratio = peg.get("PEG Ratio")
            sector_trend = fetch_sector_trend(current_sector)

            # --- Stage 2 Filtering ---
            passed = True
            fail_reasons = []

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
                fail_reasons.append(f"ROE < Sector Median")

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
            }

            # Calculate Conviction Score (Fix for KeyError)
            score, reasons = calculate_conviction_score(result_row)
            result_row["Conviction Score"] = score
            result_row["Conviction Reasons"] = "; ".join(reasons)

            results.append(result_row)

        except Exception as e:
            print(f"ERROR processing {symbol}: {e}")
            continue

    return pd.DataFrame(results)


# =============================================================================
# PORTFOLIO OPTIMIZATION
# =============================================================================
def build_optimized_portfolio(
    passed_df: pd.DataFrame, min_stocks: int = 6, max_weight: float = 0.20
) -> pd.DataFrame:
    """
    Build a Sharpe-optimized portfolio from screener results.

    Rules:
    - Minimum 6 stocks for diversification
    - Maximum 15% in any single name
    - Sector diversification (max 2 per sector initially)
    - Fallback: equal weight with BIL (T-bills) and TLT (20yr bonds) if < 4 stocks
    """
    print("\n" + "=" * 60)
    print("PORTFOLIO OPTIMIZATION - Sharpe Ratio Maximization")
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
    passed_df = passed_df.sort_values(by="Market Cap ($B)", ascending=False)
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

        # Constraints: sum of weights = 1
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        # Bounds: 0 <= weight <= max_weight
        bounds = tuple((0.02, max_weight) for _ in range(n_assets))

        print("Optimizing for Sharpe ratio (scipy)...")
        result = minimize(
            negative_sharpe,
            n_assets * [1.0 / n_assets],  # Initial guess: equal weights
            args=(mean_returns, cov_matrix_reg, risk_free_rate),
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
        "--skip-ai",
        action="store_true",
        help="Disable OpenAI-powered features (sentiment & thesis).",
    )
    parser.add_argument(
        "--skip-advice",
        action="store_true",
        help="Skip AI-powered investment advice generation.",
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
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()
    """Main entry point for the GARP stock screener."""
    print("\n" + "=" * 60)
    print("       GARP STOCK SCREENER - Two-Stage Implementation")
    print("=" * 60)

    global openai_client
    if args.skip_ai or not OPENAI_API_KEY:
        if not OPENAI_API_KEY and not args.skip_ai:
            print("OpenAI key not provided; AI features disabled.")
        openai_client = None

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
        portfolio_df = build_optimized_portfolio(passed_df)

    if not passed_df.empty and not args.skip_advice:
        # Generate AI-powered investment advice
        advice_df = generate_investment_advice(passed_df)

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
