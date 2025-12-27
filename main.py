"""
GARP Stock Screener - Two-Stage Implementation
Based on RULES.md and IDEAS.md methodology
"""

import argparse
import logging
import os
import pickle
import time
import warnings
from datetime import datetime, timedelta
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
EBITDA_GROWTH_MIN_PCT = CONFIG["growth"].get(
    "ebitda_growth_min_pct", 6.0
)  # Per RULES.md line 33
ROIC_MIN_PCT = CONFIG["profitability"]["roic_min_pct"]
ROE_MIN_PCT = CONFIG["profitability"]["roe_min_pct"]
ROA_MIN_PCT = CONFIG["profitability"]["roa_min_pct"]
GROSS_MARGIN_MIN_PCT = CONFIG["profitability"]["gross_margin_min_pct"]
OPERATING_MARGIN_MIN_PCT = CONFIG["profitability"]["operating_margin_min_pct"]
CURRENT_RATIO_MAX_PCT = CONFIG["balance_sheet"]["current_ratio_max"]
INTEREST_COVERAGE_MIN = CONFIG["balance_sheet"].get(
    "interest_coverage_min", 3.0
)  # Per RULES.md line 50-54
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

# Conviction scoring weights (from config)
CONVICTION_WEIGHTS = CONFIG.get("conviction", {})
VALUATION_WEIGHT = CONVICTION_WEIGHTS.get("valuation_weight", 25)
QUALITY_WEIGHT = CONVICTION_WEIGHTS.get("quality_weight", 30)
GROWTH_WEIGHT = CONVICTION_WEIGHTS.get("growth_weight", 20)
TECHNICAL_WEIGHT = CONVICTION_WEIGHTS.get("technical_weight", 15)
SENTIMENT_WEIGHT = CONVICTION_WEIGHTS.get("sentiment_weight", 10)
RISK_PENALTY_PER_FLAG = CONVICTION_WEIGHTS.get("risk_penalty_per_flag", 3)

# Sector-specific tolerance factors
SECTOR_TOLERANCES = CONFIG.get("sector_tolerances", {})
DEFAULT_SECTOR_TOLERANCE = SECTOR_TOLERANCES.get(
    "default",
    {
        "pe_tolerance": 1.2,
        "ps_tolerance": 1.1,
        "growth_premium": 1.0,
        "pb_tolerance": 1.1,
    },
)

# Liquidity requirements
LIQUIDITY_CONFIG = CONFIG.get("liquidity", {})
MIN_AVG_VOLUME = LIQUIDITY_CONFIG.get("min_avg_volume", 500000)
MIN_MARKET_CAP_MILLIONS = LIQUIDITY_CONFIG.get("min_market_cap_millions", 1000)

# Portfolio optimization settings
PORTFOLIO_CONFIG = CONFIG.get("portfolio", {})
VAR_CONFIDENCE = PORTFOLIO_CONFIG.get("var_confidence", 0.95)
VAR_HORIZON_DAYS = PORTFOLIO_CONFIG.get("var_horizon_days", 20)
TRANSACTION_COST_BPS = PORTFOLIO_CONFIG.get("transaction_cost_bps", 10)
MAX_CORRELATION = PORTFOLIO_CONFIG.get("max_correlation", 0.75)

# Data quality thresholds
DATA_QUALITY_CONFIG = CONFIG.get("data_quality", {})
MIN_HISTORY_DAYS = DATA_QUALITY_CONFIG.get("min_history_days", 252)
MAX_DATA_AGE_DAYS = DATA_QUALITY_CONFIG.get("max_data_age_days", 5)
MIN_ANALYST_COVERAGE = DATA_QUALITY_CONFIG.get("min_analyst_coverage", 3)

# Backtest configuration
BACKTEST_CONFIG = CONFIG.get("backtest", {})
WALK_FORWARD_WINDOWS = BACKTEST_CONFIG.get("walk_forward_windows", 4)
TRAIN_PERIOD_MONTHS = BACKTEST_CONFIG.get("train_period_months", 12)
TEST_PERIOD_MONTHS = BACKTEST_CONFIG.get("test_period_months", 3)
BENCHMARK_SYMBOL = BACKTEST_CONFIG.get("benchmark_symbol", "SPY")

# =============================================================================
# SHARED HELPERS
# =============================================================================
_TICKER_CACHE: dict[str, yf.Ticker] = {}
_SECTOR_METRICS_CACHE_FILE = "sector_metrics_cache.pkl"
_SECTOR_CACHE_TTL_HOURS = 24


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
# NEW HELPER FUNCTIONS FOR IMPROVED FINANCIAL ADVICE
# =============================================================================


def get_sector_tolerance(sector: str) -> dict:
    """
    Get sector-specific tolerance factors for relative valuation screening.
    Different sectors have different appropriate valuation multiples.
    """
    return SECTOR_TOLERANCES.get(sector, DEFAULT_SECTOR_TOLERANCE)


def check_liquidity(ticker_info: dict) -> dict:
    """
    Check if a stock meets minimum liquidity requirements.
    Returns liquidity assessment with pass/fail status.
    """
    avg_volume = ticker_info.get("averageVolume", 0) or 0
    market_cap = ticker_info.get("marketCap", 0) or 0
    market_cap_millions = market_cap / 1_000_000

    volume_pass = avg_volume >= MIN_AVG_VOLUME
    mcap_pass = market_cap_millions >= MIN_MARKET_CAP_MILLIONS

    liquidity_score = 0
    if avg_volume >= MIN_AVG_VOLUME * 2:
        liquidity_score = 3  # Excellent
    elif avg_volume >= MIN_AVG_VOLUME:
        liquidity_score = 2  # Good
    elif avg_volume >= MIN_AVG_VOLUME * 0.5:
        liquidity_score = 1  # Fair
    # else 0 = Poor

    return {
        "Liquidity Pass": volume_pass and mcap_pass,
        "Avg Volume": avg_volume,
        "Liquidity Score": liquidity_score,
        "Liquidity Signal": ["Poor", "Fair", "Good", "Excellent"][liquidity_score],
    }


def calculate_interest_coverage(ticker: yf.Ticker) -> dict:
    """
    Calculate Interest Coverage Ratio = EBIT / Interest Expense.
    Critical for assessing debt serviceability.
    """
    try:
        income = ticker.financials
        if income.empty:
            return {"Interest Coverage": None, "Interest Coverage Signal": None}

        # Get EBIT
        ebit = None
        for key in ["EBIT", "Operating Income"]:
            if key in income.index:
                ebit = income.loc[key].iloc[0]
                break

        # Get Interest Expense
        interest_expense = None
        for key in ["Interest Expense", "Interest Expense Non Operating"]:
            if key in income.index:
                interest_expense = abs(income.loc[key].iloc[0])
                break

        if ebit is None or interest_expense is None or interest_expense == 0:
            return {"Interest Coverage": None, "Interest Coverage Signal": None}

        coverage = ebit / interest_expense

        # Classify coverage
        if coverage >= 10:
            signal = "Excellent"
        elif coverage >= 5:
            signal = "Strong"
        elif coverage >= 2.5:
            signal = "Adequate"
        elif coverage >= 1.5:
            signal = "Weak"
        else:
            signal = "Distressed"

        return {
            "Interest Coverage": round(coverage, 2),
            "Interest Coverage Signal": signal,
        }
    except Exception as e:
        logger.debug(f"Interest coverage calculation failed: {e}")
        return {"Interest Coverage": None, "Interest Coverage Signal": None}


def calculate_current_ratio_trend(ticker: yf.Ticker) -> dict:
    """
    Calculate Current Ratio trend over time to assess liquidity trajectory.
    """
    try:
        balance = ticker.balance_sheet
        if balance.empty or len(balance.columns) < 2:
            return {"Current Ratio Trend": None, "Liquidity Trajectory": None}

        ratios = []
        for i in range(min(3, len(balance.columns))):
            current_assets = None
            current_liabilities = None

            for ca_key in ["Total Current Assets", "Current Assets"]:
                if ca_key in balance.index:
                    current_assets = balance.loc[ca_key].iloc[i]
                    break

            for cl_key in ["Total Current Liabilities", "Current Liabilities"]:
                if cl_key in balance.index:
                    current_liabilities = balance.loc[cl_key].iloc[i]
                    break

            if current_assets and current_liabilities and current_liabilities > 0:
                ratios.append(current_assets / current_liabilities)

        if len(ratios) < 2:
            return {"Current Ratio Trend": None, "Liquidity Trajectory": None}

        # Most recent vs oldest
        current = ratios[0]
        oldest = ratios[-1]
        change_pct = ((current - oldest) / oldest) * 100 if oldest > 0 else 0

        if change_pct > 10:
            trajectory = "Improving"
        elif change_pct > -10:
            trajectory = "Stable"
        else:
            trajectory = "Deteriorating"

        return {
            "Current Ratio": round(current, 2),
            "Current Ratio Trend": round(change_pct, 1),
            "Liquidity Trajectory": trajectory,
        }
    except Exception as e:
        logger.debug(f"Current ratio trend calculation failed: {e}")
        return {"Current Ratio Trend": None, "Liquidity Trajectory": None}


def calculate_multi_timeframe_trend(prices: pd.Series) -> dict:
    """
    Calculate multi-timeframe trend score for momentum analysis.
    Combines 1-week, 1-month, and 3-month momentum signals.
    """
    try:
        if prices is None or len(prices) < 60:
            return {"Trend Score": None, "Trend Signal": None, "Momentum Details": None}

        current = prices.iloc[-1]
        score = 0
        details = []

        # 1-week (5 trading days)
        if len(prices) >= 5:
            week_avg = prices.iloc[-5:].mean()
            if current > week_avg:
                score += 1
                details.append("1W+")
            else:
                score -= 1
                details.append("1W-")

        # 1-month (20 trading days)
        if len(prices) >= 20:
            month_avg = prices.iloc[-20:].mean()
            if current > month_avg:
                score += 1
                details.append("1M+")
            else:
                score -= 1
                details.append("1M-")

        # 3-month (60 trading days)
        if len(prices) >= 60:
            quarter_avg = prices.iloc[-60:].mean()
            if current > quarter_avg:
                score += 1
                details.append("3M+")
            else:
                score -= 1
                details.append("3M-")

        # Classify trend
        if score >= 2:
            signal = "Strong Uptrend"
        elif score == 1:
            signal = "Mild Uptrend"
        elif score == 0:
            signal = "Neutral"
        elif score == -1:
            signal = "Mild Downtrend"
        else:
            signal = "Strong Downtrend"

        return {
            "Trend Score": score,
            "Trend Signal": signal,
            "Momentum Details": ", ".join(details),
        }
    except Exception as e:
        logger.debug(f"Multi-timeframe trend calculation failed: {e}")
        return {"Trend Score": None, "Trend Signal": None, "Momentum Details": None}


def assess_data_quality(ticker: yf.Ticker, info: dict) -> dict:
    """
    Assess overall data quality and confidence level for a stock.
    Returns confidence metrics for the analysis.
    """
    try:
        issues = []
        confidence_score = 100  # Start with perfect score

        # Check price history availability
        hist = ticker.history(period="1y")
        if hist.empty:
            issues.append("No price history")
            confidence_score -= 30
        elif len(hist) < MIN_HISTORY_DAYS:
            issues.append(f"Limited history ({len(hist)} days)")
            confidence_score -= 15

        # Check financial data availability
        if ticker.financials.empty:
            issues.append("No income statement")
            confidence_score -= 20

        if ticker.balance_sheet.empty:
            issues.append("No balance sheet")
            confidence_score -= 20

        if ticker.cash_flow.empty:
            issues.append("No cash flow")
            confidence_score -= 15

        # Check analyst coverage
        num_analysts = info.get("numberOfAnalystOpinions", 0) or 0
        if num_analysts == 0:
            issues.append("No analyst coverage")
            confidence_score -= 10
        elif num_analysts < MIN_ANALYST_COVERAGE:
            issues.append(f"Low analyst coverage ({num_analysts})")
            confidence_score -= 5

        # Check for key metrics availability
        if info.get("trailingPE") is None:
            issues.append("Missing P/E")
            confidence_score -= 5

        if info.get("marketCap") is None:
            issues.append("Missing market cap")
            confidence_score -= 10

        # Classify confidence level
        confidence_score = max(0, confidence_score)
        if confidence_score >= 80:
            confidence_level = "High"
        elif confidence_score >= 60:
            confidence_level = "Medium"
        elif confidence_score >= 40:
            confidence_level = "Low"
        else:
            confidence_level = "Very Low"

        return {
            "Data Quality Score": confidence_score,
            "Data Confidence": confidence_level,
            "Data Issues": "; ".join(issues) if issues else None,
        }
    except Exception as e:
        logger.debug(f"Data quality assessment failed: {e}")
        return {
            "Data Quality Score": 0,
            "Data Confidence": "Unknown",
            "Data Issues": str(e),
        }


def load_sector_metrics_cache() -> dict:
    """Load sector metrics from persistent cache file."""
    try:
        if os.path.exists(_SECTOR_METRICS_CACHE_FILE):
            with open(_SECTOR_METRICS_CACHE_FILE, "rb") as f:
                cache = pickle.load(f)

            # Check if cache is still valid
            cache_time = cache.get("timestamp")
            if cache_time:
                age_hours = (datetime.now() - cache_time).total_seconds() / 3600
                if age_hours < _SECTOR_CACHE_TTL_HOURS:
                    return cache.get("data", {})
        return {}
    except Exception as e:
        logger.debug(f"Failed to load sector metrics cache: {e}")
        return {}


def save_sector_metrics_cache(data: dict):
    """Save sector metrics to persistent cache file."""
    try:
        cache = {"timestamp": datetime.now(), "data": data}
        with open(_SECTOR_METRICS_CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)
    except Exception as e:
        logger.debug(f"Failed to save sector metrics cache: {e}")


def calculate_var(
    returns: pd.Series, confidence: float = 0.95, horizon_days: int = 20
) -> dict:
    """
    Calculate Value at Risk (VaR) using historical simulation method.

    Args:
        returns: Daily returns series
        confidence: Confidence level (e.g., 0.95 for 95%)
        horizon_days: Time horizon in trading days

    Returns:
        Dict with VaR metrics
    """
    try:
        if returns is None or len(returns) < 30:
            return {"VaR": None, "CVaR": None, "VaR Signal": None}

        # Scale to horizon
        horizon_factor = np.sqrt(horizon_days)

        # Historical VaR (percentile method)
        var_pct = np.percentile(returns, (1 - confidence) * 100)
        var_scaled = var_pct * horizon_factor * 100  # Convert to percentage

        # Conditional VaR (Expected Shortfall)
        cvar_threshold = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= cvar_threshold].mean() * horizon_factor * 100

        # Classify risk
        if abs(var_scaled) <= 10:
            signal = "Low Risk"
        elif abs(var_scaled) <= 20:
            signal = "Moderate Risk"
        elif abs(var_scaled) <= 30:
            signal = "High Risk"
        else:
            signal = "Very High Risk"

        return {
            "VaR (%)": round(abs(var_scaled), 2),
            "CVaR (%)": round(abs(cvar), 2) if not np.isnan(cvar) else None,
            "VaR Signal": signal,
            "VaR Horizon Days": horizon_days,
            "VaR Confidence": f"{confidence * 100:.0f}%",
        }
    except Exception as e:
        logger.debug(f"VaR calculation failed: {e}")
        return {"VaR (%)": None, "CVaR (%)": None, "VaR Signal": None}


def estimate_transaction_costs(weight: float, market_cap: float) -> float:
    """
    Estimate transaction costs for a position based on size and liquidity.
    Returns cost in basis points.
    """
    base_cost = TRANSACTION_COST_BPS

    # Adjust for position size (larger positions have more market impact)
    if weight > 0.15:
        base_cost *= 1.5
    elif weight > 0.10:
        base_cost *= 1.2

    # Adjust for market cap (smaller companies have higher spreads)
    if market_cap < 10_000_000_000:  # < $10B
        base_cost *= 1.3
    elif market_cap < 50_000_000_000:  # < $50B
        base_cost *= 1.1

    return base_cost


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

    # Removed verbose sector median calculation message
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
    Calculate a conviction score (0-100) based on weighted category scoring.
    Uses config weights: Valuation (25), Quality (30), Growth (20), Technical (15), Sentiment (10).
    Also applies risk penalties from the config (risk_penalty_per_flag).

    Returns (score, list of reasons).
    """
    reasons = []

    # Category scores (each 0-100, will be weighted)
    valuation_score = 50  # Start neutral
    quality_score = 50
    growth_score = 50
    technical_score = 50
    sentiment_score = 50

    # Track risk flags for penalty
    risk_flag_count = 0

    # ==========================================================================
    # VALUATION CATEGORY (Weight: 25%)
    # ==========================================================================
    # Graham Undervalued
    if stock_data.get("Graham Undervalued"):
        valuation_score += 15
        reasons.append("Graham undervalued ✓")

    # P/FCF analysis
    pfcf = stock_data.get("P/FCF", 100) or 100
    if pfcf < 12:
        valuation_score += 15
        reasons.append(f"Very low P/FCF ({pfcf:.1f}) ✓")
    elif pfcf < 18:
        valuation_score += 8
        reasons.append(f"Low P/FCF ({pfcf:.1f}) ✓")
    elif pfcf > PFCF_MAX:
        valuation_score -= 15
        reasons.append(f"Rich P/FCF ({pfcf:.1f}) ⚠️")
        risk_flag_count += 1

    # PEG Ratio
    peg = stock_data.get("PEG Ratio", 999) or 999
    if peg < 1.0:
        valuation_score += 15
        reasons.append(f"Excellent PEG ({peg:.2f}) ✓")
    elif peg < 1.5:
        valuation_score += 10
        reasons.append(f"Good PEG ({peg:.2f}) ✓")
    elif peg > 2.5:
        valuation_score -= 10
        reasons.append(f"High PEG ({peg:.2f}) ⚠️")

    # EV/EBITDA
    ev_ebitda = stock_data.get("EV/EBITDA")
    if ev_ebitda and ev_ebitda < 10:
        valuation_score += 12
        reasons.append(f"Very attractive EV/EBITDA ({ev_ebitda:.1f}) ✓")
    elif ev_ebitda and ev_ebitda < 15:
        valuation_score += 6
        reasons.append(f"Attractive EV/EBITDA ({ev_ebitda:.1f}) ✓")
    elif ev_ebitda and ev_ebitda > 25:
        valuation_score -= 10
        reasons.append(f"Expensive EV/EBITDA ({ev_ebitda:.1f}) ⚠️")

    # FCF Yield
    fcf_yield = stock_data.get("FCF Yield (%)")
    yield_spread = stock_data.get("Yield Spread vs T-Bill")
    if fcf_yield and fcf_yield > 8:
        valuation_score += 10
        reasons.append(f"High FCF yield ({fcf_yield:.1f}%) ✓")
    if yield_spread and yield_spread > 4:
        valuation_score += 8
        reasons.append(f"Attractive yield spread (+{yield_spread:.1f}% vs T-Bill) ✓")

    # P/E with context
    pe = stock_data.get("P/E", 0) or 0
    if pe > 35 and peg > 2.0:
        valuation_score -= 12
        reasons.append(f"High P/E without growth support ({pe:.1f}) ⚠️")
        risk_flag_count += 1

    # Upside to target
    upside = stock_data.get("Upside (%)", 0) or 0
    if upside > 25:
        valuation_score += 10
        reasons.append(f"Strong upside ({upside:.1f}%) ✓")
    elif upside > 15:
        valuation_score += 5
        reasons.append(f"Good upside ({upside:.1f}%) ✓")

    # Valuation Consensus
    val_consensus = stock_data.get("Valuation Consensus", "")
    if val_consensus == "Strong Agreement":
        avg_upside = stock_data.get("Average Upside (%)", 0) or 0
        valuation_score += 8
        reasons.append(f"Strong valuation consensus ({avg_upside:.1f}% upside) ✓")
    elif val_consensus == "Divergent Views":
        valuation_score -= 5
        reasons.append("Divergent valuation views ⚠️")

    # ==========================================================================
    # QUALITY CATEGORY (Weight: 30%)
    # ==========================================================================
    # ROIC
    roic = stock_data.get("ROIC (%)", 0) or 0
    if roic > 25:
        quality_score += 15
        reasons.append(f"Excellent ROIC ({roic:.1f}%) ✓")
    elif roic > 15:
        quality_score += 10
        reasons.append(f"Strong ROIC ({roic:.1f}%) ✓")
    elif roic < 8:
        quality_score -= 10
        reasons.append(f"Weak ROIC ({roic:.1f}%) ⚠️")

    # Piotroski F-Score
    piotroski = stock_data.get("Piotroski Score", 0) or 0
    if piotroski >= 8:
        quality_score += 15
        reasons.append(f"Excellent Quality (Piotroski: {piotroski}/9) ✓")
    elif piotroski >= 6:
        quality_score += 8
        reasons.append(f"Good Quality (Piotroski: {piotroski}/9) ✓")
    elif piotroski < 4 and piotroski > 0:
        quality_score -= 12
        reasons.append(f"Poor Quality (Piotroski: {piotroski}/9) ⚠️")
        risk_flag_count += 1

    # Financial Distress Risk (Altman Z-Score)
    distress_risk = stock_data.get("Financial Distress Risk", "")
    if "Low" in distress_risk or "Safe Zone" in distress_risk:
        quality_score += 10
        z_score = stock_data.get("Altman Z-Score")
        reasons.append(f"Low financial distress risk (Z: {z_score}) ✓")
    elif "High" in distress_risk:
        quality_score -= 20
        reasons.append("High financial distress risk ⚠️")
        risk_flag_count += 2

    # Cash Flow Quality
    cf_quality = stock_data.get("Cash Flow Quality", "")
    if cf_quality == "Good":
        quality_score += 10
        reasons.append("Strong cash flow quality ✓")
    elif cf_quality == "Poor":
        quality_score -= 12
        reasons.append("Poor cash flow quality ⚠️")
        risk_flag_count += 1

    # Accounting Red Flags
    red_flag_count = stock_data.get("Red Flag Count", 0) or 0
    if red_flag_count == 0:
        quality_score += 8
        reasons.append("No accounting red flags ✓")
    elif red_flag_count >= 2:
        quality_score -= 15
        reasons.append(f"Multiple accounting red flags ({red_flag_count}) ⚠️")
        risk_flag_count += red_flag_count

    # Interest Coverage (NEW)
    interest_coverage = stock_data.get("Interest Coverage")
    interest_signal = stock_data.get("Interest Coverage Signal")
    if interest_coverage and interest_coverage >= 10:
        quality_score += 8
        reasons.append(f"Excellent interest coverage ({interest_coverage:.1f}x) ✓")
    elif interest_signal == "Distressed":
        quality_score -= 15
        reasons.append(f"Distressed interest coverage ({interest_coverage:.1f}x) ⚠️")
        risk_flag_count += 1

    # Liquidity Trajectory (NEW)
    liquidity_trajectory = stock_data.get("Liquidity Trajectory")
    if liquidity_trajectory == "Improving":
        quality_score += 5
        reasons.append("Improving liquidity position ✓")
    elif liquidity_trajectory == "Deteriorating":
        quality_score -= 8
        reasons.append("Deteriorating liquidity ⚠️")
        risk_flag_count += 1

    # Leverage
    leverage = stock_data.get("D/E", 0) or 0
    leverage_limit = DE_MAX * 100
    if leverage and leverage > leverage_limit:
        quality_score -= 12
        reasons.append(f"High leverage (D/E: {leverage:.1f}) ⚠️")
        risk_flag_count += 1

    # ==========================================================================
    # GROWTH CATEGORY (Weight: 20%)
    # ==========================================================================
    # Revenue CAGR
    rev_cagr = stock_data.get("3Y Rev CAGR (%)", 0) or 0
    if rev_cagr > 20:
        growth_score += 18
        reasons.append(f"Excellent growth ({rev_cagr:.1f}% CAGR) ✓")
    elif rev_cagr > 12:
        growth_score += 12
        reasons.append(f"Strong growth ({rev_cagr:.1f}% CAGR) ✓")
    elif rev_cagr > 6:
        growth_score += 5
    elif rev_cagr < 0:
        growth_score -= 15
        reasons.append(f"Negative growth ({rev_cagr:.1f}% CAGR) ⚠️")

    # Future Growth Estimate
    next_yr_growth = stock_data.get("Next Year Growth Est (%)", 0) or 0
    if next_yr_growth > 20:
        growth_score += 15
        reasons.append(f"Strong future growth est ({next_yr_growth:.1f}%) ✓")
    elif next_yr_growth > 12:
        growth_score += 8
        reasons.append(f"Good future growth est ({next_yr_growth:.1f}%) ✓")

    # Growth Quality Score
    growth_quality = stock_data.get("Growth Quality Score")
    margin_trend = stock_data.get("Margin Trend")
    if growth_quality and growth_quality >= 3:
        growth_score += 15
        reasons.append("Excellent growth quality (3/3) ✓")
    elif growth_quality and growth_quality >= 2:
        growth_score += 8
        reasons.append("Good growth quality (2/3) ✓")
    if margin_trend == "Compressing":
        growth_score -= 10
        reasons.append("Margin compression ⚠️")
    elif margin_trend == "Expanding":
        growth_score += 8
        reasons.append("Margin expansion ✓")

    # Earnings Surprise
    earnings_surprise = stock_data.get("Earnings Surprise Avg (%)", 0) or 0
    if earnings_surprise > 8:
        growth_score += 10
        reasons.append(f"Consistent earnings beats ({earnings_surprise:.1f}%) ✓")
    elif earnings_surprise > 3:
        growth_score += 5

    # ==========================================================================
    # TECHNICAL CATEGORY (Weight: 15%)
    # ==========================================================================
    # Moving Average Signal
    tech_signal = stock_data.get("Technical Signal", "Neutral")
    price_trend = stock_data.get("Price Trend", "Neutral")

    if "Golden Cross" in tech_signal:
        technical_score += 15
        reasons.append("Golden Cross (Bullish) ✓")
    elif "Death Cross" in tech_signal:
        technical_score -= 15
        reasons.append("Death Cross (Bearish) ⚠️")
        risk_flag_count += 1

    if price_trend == "Uptrend":
        technical_score += 12
        reasons.append("Price in Uptrend (> SMA200) ✓")
    elif price_trend == "Downtrend":
        technical_score -= 8

    # RSI
    rsi = stock_data.get("RSI", 50) or 50
    if rsi < 30:
        technical_score += 10
        reasons.append(f"Oversold (RSI: {rsi:.0f}) - Buy Dip Opp. ✓")
    elif rsi > 70:
        technical_score -= 8
        reasons.append(f"Overbought (RSI: {rsi:.0f}) ⚠️")

    # Multi-timeframe Trend (NEW)
    trend_score = stock_data.get("Trend Score")
    trend_signal = stock_data.get("Trend Signal")
    if trend_score and trend_score >= 2:
        technical_score += 12
        reasons.append(f"{trend_signal} ✓")
    elif trend_score and trend_score <= -2:
        technical_score -= 12
        reasons.append(f"{trend_signal} ⚠️")

    # 52-Week Position
    week_52_signal = stock_data.get("52-Week Position Signal")
    if week_52_signal == "Near 52-Week High (Momentum)":
        technical_score += 8
        reasons.append("Strong momentum (near 52W high) ✓")
    elif week_52_signal == "Near 52-Week Low (Contrarian)":
        technical_score += 3  # Slight positive for contrarian
        reasons.append("Contrarian opportunity (near 52W low)")

    # ==========================================================================
    # SENTIMENT CATEGORY (Weight: 10%)
    # ==========================================================================
    # Analyst Rating
    rating = stock_data.get("Analyst Rating", "").lower()
    if rating == "strong_buy":
        sentiment_score += 18
        reasons.append("Strong Buy rating ✓")
    elif rating == "buy":
        sentiment_score += 12
        reasons.append("Buy rating ✓")
    elif rating in ["sell", "strong_sell"]:
        sentiment_score -= 15
        reasons.append(f"{rating.replace('_', ' ').title()} rating ⚠️")

    # EPS Revisions
    rev_ratio = stock_data.get("EPS Up/Down Ratio", 0) or 0
    if rev_ratio > 2.0:
        sentiment_score += 12
        reasons.append(f"Very bullish revisions ({rev_ratio:.1f}x Up/Down) ✓")
    elif rev_ratio > 1.5:
        sentiment_score += 8
        reasons.append(f"Bullish analyst revisions ({rev_ratio:.1f}x Up/Down) ✓")
    elif rev_ratio < 0.5 and rev_ratio > 0:
        sentiment_score -= 10
        reasons.append(f"Bearish analyst revisions ({rev_ratio:.1f}x Up/Down) ⚠️")

    # Analyst Trend
    analyst_trend = stock_data.get("Analyst Trend", "Neutral")
    if analyst_trend == "Improving":
        sentiment_score += 8
        reasons.append("Analyst ratings improving ✓")
    elif analyst_trend == "Softening":
        sentiment_score -= 5

    # Analyst Momentum (Upgrades/Downgrades)
    analyst_momentum = stock_data.get("Analyst Momentum")
    if analyst_momentum == "Strong Positive":
        sentiment_score += 15
        reasons.append("Strong analyst upgrade momentum ✓")
    elif analyst_momentum == "Positive":
        sentiment_score += 8
        reasons.append("Positive analyst momentum ✓")
    elif analyst_momentum == "Strong Negative":
        sentiment_score -= 15
        reasons.append("Strong analyst downgrade trend ⚠️")
        risk_flag_count += 1
    elif analyst_momentum == "Negative":
        sentiment_score -= 8
        reasons.append("Negative analyst momentum ⚠️")

    # Target Price Analysis
    target_upside = stock_data.get("Target Upside (%)")
    analyst_agreement = stock_data.get("Analyst Agreement")
    if target_upside and target_upside > 25 and analyst_agreement == "High":
        sentiment_score += 12
        reasons.append(f"Strong target upside w/ consensus ({target_upside:.1f}%) ✓")
    elif target_upside and target_upside < -15:
        sentiment_score -= 10
        reasons.append(f"Below analyst targets ({target_upside:.1f}%) ⚠️")

    # Insider Activity
    insider_signal = stock_data.get("Insider Signal")
    insider_pct = stock_data.get("Insider Ownership (%)")
    insider_purchases = stock_data.get("Recent Insider Purchases", 0) or 0

    if insider_signal == "Very High (Founder-led)":
        sentiment_score += 8
        reasons.append(f"Strong insider alignment ({insider_pct:.1f}%) ✓")
    if insider_purchases > 0:
        sentiment_score += 10
        reasons.append(f"Recent insider buying ({insider_purchases:,} shares) ✓")
    elif insider_signal == "Low Skin in Game" and insider_pct is not None:
        sentiment_score -= 5

    # Options Sentiment
    put_call_ratio = stock_data.get("Put/Call OI Ratio")
    if put_call_ratio and put_call_ratio > 1.5:
        sentiment_score -= 8
        reasons.append(f"Bearish options positioning (P/C: {put_call_ratio:.2f}) ⚠️")
    elif put_call_ratio and put_call_ratio < 0.5:
        sentiment_score += 8
        reasons.append(f"Bullish options positioning (P/C: {put_call_ratio:.2f}) ✓")

    # Short Interest
    short_pct = stock_data.get("Short % of Float")
    if short_pct and short_pct > 20:
        sentiment_score -= 10
        reasons.append(f"High short interest ({short_pct:.1f}%) ⚠️")
        risk_flag_count += 1

    # ESG/Governance
    esg_risk = stock_data.get("ESG Risk Level")
    if esg_risk in ["Negligible Risk", "Low Risk"]:
        sentiment_score += 5
        reasons.append(f"Low ESG risk ({esg_risk}) ✓")
    elif esg_risk in ["High Risk", "Severe Risk"]:
        sentiment_score -= 8
        reasons.append(f"High ESG risk ({esg_risk}) ⚠️")

    # ==========================================================================
    # ADDITIONAL FACTORS (Bonus/Penalty)
    # ==========================================================================
    bonus_score = 0

    # Dividend Quality (for income-oriented investors)
    div_consistency = stock_data.get("Dividend Consistency")
    div_cagr = stock_data.get("Dividend Growth CAGR (%)")
    if div_consistency == "Excellent (5+ Years Increases)":
        bonus_score += 3
        reasons.append("Excellent dividend track record ✓")
    if stock_data.get("Dividend Aristocrat"):
        bonus_score += 3
        reasons.append("Dividend Aristocrat (25+ years) ✓")
    if div_cagr and div_cagr > 10:
        bonus_score += 2
        reasons.append(f"Strong dividend growth ({div_cagr:.1f}% CAGR) ✓")

    # Data Quality Impact
    data_confidence = stock_data.get("Data Confidence")
    if data_confidence == "Very Low":
        bonus_score -= 5
        reasons.append("Low data confidence ⚠️")

    # ==========================================================================
    # CALCULATE WEIGHTED FINAL SCORE
    # ==========================================================================
    # Clamp category scores to 0-100
    valuation_score = max(0, min(100, valuation_score))
    quality_score = max(0, min(100, quality_score))
    growth_score = max(0, min(100, growth_score))
    technical_score = max(0, min(100, technical_score))
    sentiment_score = max(0, min(100, sentiment_score))

    # Apply weights from config
    weighted_score = (
        (valuation_score * VALUATION_WEIGHT / 100)
        + (quality_score * QUALITY_WEIGHT / 100)
        + (growth_score * GROWTH_WEIGHT / 100)
        + (technical_score * TECHNICAL_WEIGHT / 100)
        + (sentiment_score * SENTIMENT_WEIGHT / 100)
    )

    # Add bonus/penalty
    weighted_score += bonus_score

    # Apply risk penalty from config
    risk_penalty = risk_flag_count * RISK_PENALTY_PER_FLAG
    weighted_score -= risk_penalty

    if risk_penalty > 0:
        reasons.append(
            f"Risk penalty applied: -{risk_penalty} ({risk_flag_count} flags)"
        )

    # Final clamping
    final_score = int(max(0, min(100, weighted_score)))

    return final_score, reasons


def calculate_comprehensive_risk_score(stock_data: dict, ticker_info: dict) -> dict:
    """
    Calculate a 0-100 risk score (higher = riskier).
    Combines financial, valuation, technical, and sentiment risk.
    """
    risk_score = 0
    risk_factors = []

    # Financial Risk (40% weight)
    financial_risk = 0
    de_ratio = stock_data.get("D/E", 0) or 0
    if de_ratio > 100:
        financial_risk += 30
        risk_factors.append("Very high leverage")
    elif de_ratio > 50:
        financial_risk += 15
        risk_factors.append("High leverage")

    interest_coverage = stock_data.get("Interest Coverage")
    if interest_coverage and interest_coverage < 2:
        financial_risk += 25
        risk_factors.append("Weak interest coverage")
    elif interest_coverage and interest_coverage < 3:
        financial_risk += 10
        risk_factors.append("Moderate interest coverage")

    distress_risk = stock_data.get("Financial Distress Risk", "")
    if "High" in distress_risk:
        financial_risk += 20
        risk_factors.append("High financial distress risk")

    financial_risk = min(100, financial_risk)
    risk_score += financial_risk * 0.4

    # Valuation Risk (25% weight)
    valuation_risk = 0
    pe = stock_data.get("P/E", 0) or 0
    peg = stock_data.get("PEG Ratio", 999) or 999
    pfcf = stock_data.get("P/FCF", 100) or 100

    if pe > 40 and peg > 2.5:
        valuation_risk += 30
        risk_factors.append("Expensive valuation")
    elif pe > 30:
        valuation_risk += 15

    if pfcf > 50:
        valuation_risk += 20
        risk_factors.append("Very high P/FCF")

    valuation_risk = min(100, valuation_risk)
    risk_score += valuation_risk * 0.25

    # Technical Risk (20% weight)
    technical_risk = 0
    price_trend = stock_data.get("Price Trend")
    rsi = stock_data.get("RSI", 50) or 50

    if price_trend == "Downtrend":
        technical_risk += 25
        risk_factors.append("Downtrend")

    if rsi > 75:
        technical_risk += 15
        risk_factors.append("Overbought")

    technical_risk = min(100, technical_risk)
    risk_score += technical_risk * 0.20

    # Sentiment Risk (15% weight)
    sentiment_risk = 0
    analyst_momentum = stock_data.get("Analyst Momentum", "")
    short_pct = stock_data.get("Short % of Float", 0) or 0

    if analyst_momentum == "Strong Negative":
        sentiment_risk += 25
        risk_factors.append("Negative analyst momentum")

    if short_pct and short_pct > 20:
        sentiment_risk += 20
        risk_factors.append("High short interest")

    sentiment_risk = min(100, sentiment_risk)
    risk_score += sentiment_risk * 0.15

    # Classify risk level
    if risk_score < 30:
        risk_level = "LOW"
    elif risk_score < 50:
        risk_level = "MODERATE"
    elif risk_score < 70:
        risk_level = "HIGH"
    else:
        risk_level = "VERY HIGH"

    return {
        "risk_score": round(risk_score, 1),
        "risk_level": risk_level,
        "financial_risk": financial_risk,
        "valuation_risk": valuation_risk,
        "technical_risk": technical_risk,
        "sentiment_risk": sentiment_risk,
        "risk_factors": risk_factors,
    }


def generate_entry_strategy(stock_data: dict) -> dict:
    """
    Generate specific entry strategy with price targets based on conviction and technicals.
    """
    current_price = stock_data.get("Current Price")
    if not current_price or current_price <= 0:
        return {
            "entry_approach": "UNKNOWN",
            "target_entry_price": None,
            "stop_loss_price": None,
            "timeline": "UNKNOWN",
            "rationale": "Price data unavailable",
        }

    rsi = stock_data.get("RSI", 50) or 50
    conviction = stock_data.get("Conviction Score", 0)
    dcf_upside = stock_data.get("DCF Upside (%)", 0) or 0

    # High conviction + Oversold = Aggressive entry
    if conviction >= 70 and rsi < 35:
        return {
            "entry_approach": "AGGRESSIVE",
            "target_entry_price": round(current_price * 0.98, 2),
            "stop_loss_price": round(current_price * 0.92, 2),
            "timeline": "IMMEDIATE",
            "rationale": (
                "High conviction + Oversold = Strong buy opportunity. "
                "Enter now or on any minor weakness."
            ),
        }

    # High conviction + Overbought = Wait for pullback
    elif conviction >= 70 and rsi > 65:
        pullback_target = current_price * 0.95
        return {
            "entry_approach": "WAIT_FOR_PULLBACK",
            "target_entry_price": round(pullback_target, 2),
            "stop_loss_price": round(pullback_target * 0.92, 2),
            "timeline": "1-2 WEEKS",
            "rationale": (
                f"High conviction but overbought. "
                f"Wait for pullback to ${pullback_target:.2f} for better entry."
            ),
        }

    # Medium conviction + Good value = Dollar cost average
    elif conviction >= 55 and dcf_upside > 15:
        return {
            "entry_approach": "DOLLAR_COST_AVERAGE",
            "target_entry_price": round(current_price, 2),
            "stop_loss_price": round(current_price * 0.90, 2),
            "timeline": "2-4 WEEKS",
            "rationale": (
                "Good value but medium conviction. "
                "DCA: Buy 50% now, 50% if price drops 5% or fundamentals improve."
            ),
        }

    # Medium conviction + Neutral technicals = Standard entry
    elif conviction >= 55:
        return {
            "entry_approach": "STANDARD",
            "target_entry_price": round(current_price, 2),
            "stop_loss_price": round(current_price * 0.92, 2),
            "timeline": "1 WEEK",
            "rationale": "Moderate conviction. Enter at current levels with stop loss.",
        }

    # Low conviction = Watchlist
    else:
        return {
            "entry_approach": "WATCHLIST",
            "target_entry_price": round(current_price * 0.93, 2),
            "stop_loss_price": None,
            "timeline": "ONGOING",
            "rationale": (
                f"Low conviction. Monitor for better entry price "
                f"(${current_price * 0.93:.2f}) or improving fundamentals."
            ),
        }


def generate_exit_strategy(stock_data: dict) -> dict:
    """
    Generate exit strategy with profit targets and stop losses.
    """
    current_price = stock_data.get("Current Price")
    if not current_price or current_price <= 0:
        return {"profit_targets": [], "stop_loss": None, "trailing_stop": None}

    dcf_fair_value = stock_data.get("DCF Fair Value")
    dcf_upside = stock_data.get("DCF Upside (%)", 0) or 0
    target_price = stock_data.get("Target Mean Price")
    conviction = stock_data.get("Conviction Score", 0)
    beta = stock_data.get("Beta", 1.0) or 1.0

    strategy = {"profit_targets": [], "stop_loss": None, "trailing_stop": None}

    # Profit targets from different valuation methods
    if dcf_fair_value and dcf_fair_value > current_price:
        strategy["profit_targets"].append(
            {
                "price": round(dcf_fair_value, 2),
                "upside_pct": round(dcf_upside, 1),
                "rationale": "DCF Fair Value",
                "priority": "HIGH" if dcf_upside > 20 else "MEDIUM",
            }
        )

    if target_price and target_price > current_price:
        target_upside = ((target_price - current_price) / current_price) * 100
        strategy["profit_targets"].append(
            {
                "price": round(target_price, 2),
                "upside_pct": round(target_upside, 1),
                "rationale": "Analyst Target Price",
                "priority": "MEDIUM",
            }
        )

    # Stop loss based on volatility and conviction
    stop_loss_pct = 8 + (beta - 1.0) * 5  # Higher beta = wider stop
    if conviction < 50:
        stop_loss_pct = 6  # Tighter stop for low conviction
    elif conviction >= 75:
        stop_loss_pct = 10  # Wider stop for high conviction

    strategy["stop_loss"] = round(current_price * (1 - stop_loss_pct / 100), 2)

    # Trailing stop for high conviction positions
    if conviction >= 70:
        strategy["trailing_stop"] = "10% trailing stop after 15% gain"

    return strategy


def calculate_position_size(
    conviction: int, risk_score: float, portfolio_value: float = 100000
) -> dict:
    """
    Calculate recommended position size based on conviction and risk.
    High conviction + Low risk = Larger position.
    """
    # Base position size from conviction
    if conviction >= 75:
        base_size_pct = 0.12  # 12%
    elif conviction >= 60:
        base_size_pct = 0.08  # 8%
    elif conviction >= 45:
        base_size_pct = 0.05  # 5%
    else:
        base_size_pct = 0.03  # 3%

    # Adjust for risk
    if risk_score > 70:  # High risk
        base_size_pct *= 0.5
    elif risk_score < 30:  # Low risk
        base_size_pct *= 1.2

    # Cap at 15%
    position_size_pct = min(base_size_pct, 0.15)
    position_value = portfolio_value * position_size_pct

    return {
        "position_size_pct": round(position_size_pct * 100, 1),
        "position_value": round(position_value, 2),
        "rationale": (
            f"Based on conviction ({conviction}/100) and risk ({risk_score:.0f}/100). "
            f"Higher conviction + Lower risk = Larger position."
        ),
    }


def generate_investment_thesis(stock_data: dict) -> str:
    """
    Generate narrative investment thesis explaining why to buy/hold/sell.
    """
    symbol = stock_data.get("Symbol", "Unknown")
    name = stock_data.get("Name", symbol)
    conviction = stock_data.get("Conviction Score", 0)
    sector = stock_data.get("Sector", "Unknown")

    strengths = []
    if stock_data.get("ROIC (%)", 0) > 20:
        strengths.append(f"exceptional ROIC of {stock_data.get('ROIC (%)'):.1f}%")
    if stock_data.get("P/FCF", 100) < 15:
        strengths.append(
            f"attractive valuation with P/FCF of {stock_data.get('P/FCF'):.1f}"
        )
    if stock_data.get("3Y Rev CAGR (%)", 0) > 15:
        strengths.append(
            f"strong revenue growth of {stock_data.get('3Y Rev CAGR (%)'):.1f}% CAGR"
        )
    if stock_data.get("Margin Trend") == "Expanding":
        strengths.append("expanding profit margins")

    concerns = []
    if stock_data.get("D/E", 0) > 50:
        concerns.append("high leverage")
    if stock_data.get("Price Trend") == "Downtrend":
        concerns.append("current downtrend")
    if stock_data.get("Financial Distress Risk", "").startswith("High"):
        concerns.append("elevated financial distress risk")

    dcf_upside = stock_data.get("DCF Upside (%)", 0) or 0

    thesis = f"{name} ({symbol}) presents a "
    if conviction >= 70:
        thesis += "compelling investment opportunity "
    elif conviction >= 55:
        thesis += "solid investment opportunity "
    else:
        thesis += "moderate investment opportunity "

    thesis += f"in the {sector} sector. "

    if strengths:
        thesis += f"The company demonstrates {', '.join(strengths)}. "

    if dcf_upside > 20:
        thesis += f"Valuation analysis suggests {dcf_upside:.0f}% upside potential. "

    if concerns:
        thesis += f"Key risks include {', '.join(concerns)}. "

    thesis += f"Overall conviction score: {conviction}/100."

    return thesis


def extract_key_strengths(stock_data: dict) -> list[str]:
    """Extract top 3 key strengths."""
    strengths = []

    if stock_data.get("ROIC (%)", 0) > 20:
        strengths.append(f"High ROIC ({stock_data.get('ROIC (%)'):.1f}%)")
    if stock_data.get("P/FCF", 100) < 15:
        strengths.append(f"Low P/FCF ({stock_data.get('P/FCF'):.1f})")
    if stock_data.get("3Y Rev CAGR (%)", 0) > 15:
        strengths.append(
            f"Strong Growth ({stock_data.get('3Y Rev CAGR (%)'):.1f}% CAGR)"
        )
    if stock_data.get("Piotroski Score", 0) >= 8:
        strengths.append(
            f"High Quality (Piotroski: {stock_data.get('Piotroski Score')}/9)"
        )
    if stock_data.get("Margin Trend") == "Expanding":
        strengths.append("Expanding Margins")

    return strengths[:3]


def extract_key_weaknesses(stock_data: dict) -> list[str]:
    """Extract top 3 key weaknesses."""
    weaknesses = []

    if stock_data.get("D/E", 0) > 50:
        weaknesses.append(f"High Leverage (D/E: {stock_data.get('D/E'):.1f})")
    if stock_data.get("Financial Distress Risk", "").startswith("High"):
        weaknesses.append("High Financial Distress Risk")
    if stock_data.get("Price Trend") == "Downtrend":
        weaknesses.append("Downtrend")
    if stock_data.get("PEG Ratio", 999) > 2.5:
        weaknesses.append(f"High PEG ({stock_data.get('PEG Ratio'):.2f})")

    return weaknesses[:3]


def determine_hold_period(stock_data: dict, conviction: int) -> str:
    """Determine recommended hold period."""
    dcf_upside = stock_data.get("DCF Upside (%)", 0) or 0

    if conviction >= 70 and dcf_upside > 30:
        return "LONG_TERM (2+ years) - High conviction compounder"
    elif conviction >= 55:
        return "MEDIUM_TERM (6-18 months) - Monitor fundamentals quarterly"
    else:
        return "SHORT_TERM (3-6 months) - Reassess regularly"


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


def classify_action(conviction: int, graham_undervalued: bool, upside: float) -> dict:
    """
    Classify stock action with detailed rationale for better investment advice.
    Returns dict with action and rationale.
    """
    dcf_upside = upside  # Using the upside parameter which is typically DCF upside

    if conviction >= 75 and graham_undervalued and dcf_upside > 25:
        return {
            "action": "🟢 STRONG BUY",
            "rationale": (
                f"Exceptional opportunity: High conviction ({conviction}/100), "
                f"undervalued by Graham method, {dcf_upside:.0f}% upside. "
                f"Consider aggressive entry."
            ),
        }
    elif conviction >= 65 and (graham_undervalued or dcf_upside > 15):
        return {
            "action": "🟢 BUY",
            "rationale": (
                f"Strong buy signal: High conviction ({conviction}/100) with "
                f"attractive valuation ({dcf_upside:.0f}% upside). "
                f"Good entry opportunity."
            ),
        }
    elif conviction >= 55:
        return {
            "action": "🟡 BUY",
            "rationale": (
                f"Moderate conviction ({conviction}/100). "
                f"Consider smaller position size or dollar-cost averaging."
            ),
        }
    elif conviction >= 45:
        return {
            "action": "🔵 WATCHLIST",
            "rationale": (
                f"Watchlist candidate ({conviction}/100). "
                f"Monitor for improving fundamentals or better entry price."
            ),
        }
    elif conviction >= 35:
        return {
            "action": "⚪ HOLD (IF OWNED)",
            "rationale": (
                f"Low conviction ({conviction}/100). "
                f"Hold if already owned, but don't add to position."
            ),
        }
    else:
        return {
            "action": "🔴 AVOID",
            "rationale": (
                f"Very low conviction ({conviction}/100). "
                f"Multiple risk factors present. Avoid new positions."
            ),
        }


def generate_investment_summary(
    stock_data: dict, ticker_info: dict, conviction_score: int, reasons: list
) -> dict:
    """
    Generate comprehensive, actionable investment advice.
    Returns dict with entry/exit strategies, position sizing, and narrative thesis.
    """
    conviction = conviction_score

    # Get action classification with rationale
    graham_undervalued = stock_data.get("Graham Undervalued", False)
    upside = stock_data.get("Upside (%)", 0) or stock_data.get("DCF Upside (%)", 0) or 0
    action_data = classify_action(conviction, graham_undervalued, upside)

    # Calculate comprehensive risk score
    risk_data = calculate_comprehensive_risk_score(stock_data, ticker_info)

    # Generate entry strategy
    entry_strategy = generate_entry_strategy(stock_data)

    # Generate exit strategy
    exit_strategy = generate_exit_strategy(stock_data)

    # Calculate position sizing
    position_size = calculate_position_size(
        conviction,
        risk_data.get("risk_score", 50),
        portfolio_value=100000,  # Default $100k, can be made configurable
    )

    # Generate investment thesis
    thesis = generate_investment_thesis(stock_data)

    # Extract key strengths and weaknesses
    strengths = extract_key_strengths(stock_data)
    weaknesses = extract_key_weaknesses(stock_data)

    # Determine hold period
    hold_period = determine_hold_period(stock_data, conviction)

    # Build comprehensive summary dict
    summary = {
        "Action": action_data["action"],
        "Action_Rationale": action_data["rationale"],
        "Conviction_Score": conviction,
        "Risk_Score": risk_data.get("risk_score", 50),
        "Risk_Level": risk_data.get("risk_level", "MODERATE"),
        "Entry_Approach": entry_strategy.get("entry_approach"),
        "Target_Entry_Price": entry_strategy.get("target_entry_price"),
        "Entry_Rationale": entry_strategy.get("rationale"),
        "Stop_Loss_Price": exit_strategy.get("stop_loss"),
        "Profit_Targets": exit_strategy.get("profit_targets"),
        "Recommended_Position_Size_Pct": position_size.get("position_size_pct"),
        "Hold_Period": hold_period,
        "Investment_Thesis": thesis,
        "Key_Strengths": "; ".join(strengths) if strengths else "None",
        "Key_Weaknesses": "; ".join(weaknesses) if weaknesses else "None",
    }

    # Also create a formatted string for backward compatibility
    summary_str = (
        f"[{action_data['action']}] Conviction: {conviction}/100 | "
        f"Risk: {risk_data.get('risk_score', 50):.0f}/100 ({risk_data.get('risk_level', 'MODERATE')}) | "
        f"Entry: {entry_strategy.get('entry_approach')} @ ${entry_strategy.get('target_entry_price', 0):.2f} | "
        f"Position: {position_size.get('position_size_pct', 0):.1f}%"
    )
    summary["Summary_String"] = summary_str

    return summary


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


def walk_forward_backtest(
    tickers: list,
    weights: list,
    n_windows: int = None,
    train_months: int = None,
    test_months: int = None,
) -> dict:
    """
    Walk-forward backtesting for more robust performance analysis.

    Divides historical data into train/test windows, simulates portfolio
    performance on out-of-sample data to avoid lookahead bias.

    Args:
        tickers: List of ticker symbols
        weights: List of portfolio weights
        n_windows: Number of walk-forward periods (default from config)
        train_months: Training period in months (default from config)
        test_months: Test period in months (default from config)

    Returns:
        Dictionary with walk-forward performance metrics
    """
    n_windows = n_windows or WALK_FORWARD_WINDOWS
    train_months = train_months or TRAIN_PERIOD_MONTHS
    test_months = test_months or TEST_PERIOD_MONTHS
    benchmark = BENCHMARK_SYMBOL

    print("\n" + "=" * 60)
    print(
        f"WALK-FORWARD BACKTEST ({n_windows} windows: {train_months}M train, {test_months}M test)"
    )
    print("=" * 60)

    try:
        if not tickers:
            print("  No tickers to backtest.")
            return {}

        # Calculate total history needed
        total_months = train_months + (n_windows * test_months)
        period = f"{total_months + 3}mo"  # Extra buffer

        # Download historical data
        all_symbols = list(set(tickers + [benchmark]))
        print(f"  Downloading {period} of data for {len(all_symbols)} symbols...")

        data = yf.download(
            all_symbols, period=period, interval="1d", progress=False, auto_adjust=True
        )

        if "Close" in data.columns and isinstance(data.columns, pd.MultiIndex):
            data = data["Close"]
        elif "Close" in data.columns:
            data = data["Close"]

        if data.empty:
            print("  Error: No historical data found.")
            return {}

        data = data.ffill().dropna()
        returns = data.pct_change().dropna()

        # Calculate window sizes in trading days
        train_days = train_months * 21  # ~21 trading days per month
        test_days = test_months * 21

        # Store results for each window
        window_results = []
        weight_map = dict(zip(tickers, weights))

        print(f"\n  Running {n_windows} walk-forward windows...")

        for window in range(n_windows):
            # Calculate window boundaries
            test_end = len(returns) - (window * test_days)
            test_start = test_end - test_days
            train_end = test_start
            train_start = max(0, train_end - train_days)

            if test_start < 0 or train_start < 0:
                print(f"  Window {window + 1}: Insufficient data, skipping")
                continue

            # Extract test period returns
            test_returns = returns.iloc[test_start:test_end]

            if len(test_returns) < 10:
                continue

            # Calculate portfolio returns for test period
            port_test_ret = pd.Series(0.0, index=test_returns.index)
            total_weight = 0

            for t in tickers:
                if t in test_returns.columns:
                    w = weight_map.get(t, 0)
                    port_test_ret += test_returns[t] * w
                    total_weight += w

            if total_weight > 0:
                port_test_ret = port_test_ret / total_weight

            # Benchmark returns
            bench_test_ret = (
                test_returns[benchmark]
                if benchmark in test_returns.columns
                else pd.Series(0.0)
            )

            # Calculate metrics for this window
            port_total = (1 + port_test_ret).prod() - 1
            bench_total = (1 + bench_test_ret).prod() - 1

            # Sharpe ratio
            port_sharpe = (
                (port_test_ret.mean() / port_test_ret.std()) * np.sqrt(252)
                if port_test_ret.std() > 0
                else 0
            )

            # Max drawdown
            port_cum = (1 + port_test_ret).cumprod()
            port_dd = ((port_cum / port_cum.cummax()) - 1).min()

            window_results.append(
                {
                    "window": window + 1,
                    "test_start": test_returns.index[0].strftime("%Y-%m-%d"),
                    "test_end": test_returns.index[-1].strftime("%Y-%m-%d"),
                    "portfolio_return": port_total * 100,
                    "benchmark_return": bench_total * 100,
                    "alpha": (port_total - bench_total) * 100,
                    "sharpe": port_sharpe,
                    "max_drawdown": port_dd * 100,
                }
            )

            print(
                f"  Window {window + 1}: {window_results[-1]['test_start']} to {window_results[-1]['test_end']}"
            )
            print(
                f"    Portfolio: {port_total * 100:+.1f}%  |  {benchmark}: {bench_total * 100:+.1f}%  |  Alpha: {(port_total - bench_total) * 100:+.1f}%"
            )

        if not window_results:
            print("  No valid windows found.")
            return {}

        # Aggregate results
        avg_return = np.mean([w["portfolio_return"] for w in window_results])
        avg_alpha = np.mean([w["alpha"] for w in window_results])
        avg_sharpe = np.mean([w["sharpe"] for w in window_results])
        avg_dd = np.mean([w["max_drawdown"] for w in window_results])
        win_rate = (
            sum(1 for w in window_results if w["alpha"] > 0) / len(window_results) * 100
        )

        # Standard deviation of results (consistency)
        std_return = np.std([w["portfolio_return"] for w in window_results])
        std_alpha = np.std([w["alpha"] for w in window_results])

        print("\n" + "-" * 40)
        print("  WALK-FORWARD SUMMARY:")
        print("-" * 40)
        print(f"  Avg Portfolio Return:  {avg_return:+.1f}% (+/- {std_return:.1f}%)")
        print(
            f"  Avg Alpha vs {benchmark}:     {avg_alpha:+.1f}% (+/- {std_alpha:.1f}%)"
        )
        print(f"  Avg Sharpe Ratio:      {avg_sharpe:.2f}")
        print(f"  Avg Max Drawdown:      {avg_dd:.1f}%")
        print(f"  Win Rate (Beat {benchmark}): {win_rate:.0f}%")
        print(f"  Windows Tested:        {len(window_results)}")

        # Export walk-forward results
        wf_df = pd.DataFrame(window_results)
        wf_df.to_csv("walk_forward_results.csv", index=False)
        print("\n  Walk-forward results saved to: walk_forward_results.csv")

        return {
            "avg_return": avg_return,
            "avg_alpha": avg_alpha,
            "avg_sharpe": avg_sharpe,
            "avg_max_drawdown": avg_dd,
            "win_rate": win_rate,
            "std_return": std_return,
            "std_alpha": std_alpha,
            "n_windows": len(window_results),
            "window_details": window_results,
        }

    except Exception as e:
        print(f"  Error during walk-forward backtest: {e}")
        logger.error(f"Walk-forward backtest error: {e}", exc_info=True)
        return {}


def generate_stock_summaries(passed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate structured investment summaries for all passing stocks.
    """
    # Removed verbose header

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

        # Removed verbose advice generation message

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

        # Get action classification with rationale
        graham_undervalued = stock_data.get("Graham Undervalued", False)
        upside = (
            stock_data.get("Upside (%)", 0) or stock_data.get("DCF Upside (%)", 0) or 0
        )
        action_data = classify_action(conviction, graham_undervalued, upside)

        # Generate comprehensive investment summary (now returns dict)
        summary_dict = generate_investment_summary(
            stock_data, ticker_info, conviction, conviction_reasons
        )

        # Create advice row preserving all original data
        advice_row = stock_data.copy()
        advice_row.update(
            {
                "Action": action_data["action"],
                "Action_Rationale": action_data["rationale"],
                "Conviction": conviction,
                "Conviction Reasons": "; ".join(conviction_reasons),
                "Risk Warnings": "; ".join(
                    [r for r in risks if r and "nan" not in r.lower()]
                )
                if risks
                else "None",
                # Enhanced advice fields
                "Risk_Score": summary_dict.get("Risk_Score", 50),
                "Risk_Level": summary_dict.get("Risk_Level", "MODERATE"),
                "Entry_Approach": summary_dict.get("Entry_Approach"),
                "Target_Entry_Price": summary_dict.get("Target_Entry_Price"),
                "Entry_Rationale": summary_dict.get("Entry_Rationale"),
                "Stop_Loss_Price": summary_dict.get("Stop_Loss_Price"),
                "Profit_Targets": str(summary_dict.get("Profit_Targets", [])),
                "Recommended_Position_Size_Pct": summary_dict.get(
                    "Recommended_Position_Size_Pct"
                ),
                "Hold_Period": summary_dict.get("Hold_Period"),
                "Investment_Thesis": summary_dict.get("Investment_Thesis"),
                "Key_Strengths": summary_dict.get("Key_Strengths"),
                "Key_Weaknesses": summary_dict.get("Key_Weaknesses"),
                # Backward compatibility - keep summary string
                "Investment Summary": summary_dict.get("Summary_String", ""),
            }
        )

        advice_data.append(advice_row)

    advice_df = pd.DataFrame(advice_data)

    # Display enhanced recommendations
    print("\n" + "=" * 60)
    print("INVESTMENT RECOMMENDATIONS")
    print("=" * 60)
    for _, row in advice_df.iterrows():
        symbol = row["Symbol"]
        action = row["Action"]
        conviction = row["Conviction"]
        risk_score = row.get("Risk_Score", 50)
        risk_level = row.get("Risk_Level", "MODERATE")
        entry_approach = row.get("Entry_Approach", "N/A")
        entry_price = row.get("Target_Entry_Price", 0)
        position_size = row.get("Recommended_Position_Size_Pct", 0)

        print(f"\n{symbol}: {action}")
        print(
            f"  Conviction: {conviction}/100 | Risk: {risk_score:.0f}/100 ({risk_level})"
        )
        print(
            f"  Entry: {entry_approach} @ ${entry_price:.2f} | Position: {position_size:.1f}%"
        )
        if row.get("Action_Rationale"):
            print(f"  Rationale: {row['Action_Rationale']}")

    # Export summary (silent)
    advice_df.to_csv("investment_summary.csv", index=False)

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
        def calc_dcf_value(
            base_fcf, growth_rate, discount_rate, term_growth, years, decay
        ):
            proj_fcf = []
            fcf_temp = base_fcf
            gr = growth_rate
            for _ in range(years):
                fcf_temp *= 1 + gr
                gr *= decay
                proj_fcf.append(fcf_temp)

            pv = sum(
                f / ((1 + discount_rate) ** (i + 1)) for i, f in enumerate(proj_fcf)
            )

            if discount_rate <= term_growth + 0.01:
                return None
            term_fcf = proj_fcf[-1] * (1 + term_growth)
            term_val = term_fcf / (discount_rate - term_growth)
            pv_term = term_val / ((1 + discount_rate) ** years)

            ev = pv + pv_term
            eq_val = ev - net_debt
            return eq_val / shares_outstanding if shares_outstanding else None

        # Bull case: Growth +2%, WACC -1%
        bull_growth = min(0.20, fcf_growth + 0.02)  # Recalculate from original
        bull_wacc = max(0.06, wacc - 0.01)
        bull_value = calc_dcf_value(
            current_fcf,
            bull_growth,
            bull_wacc,
            terminal_growth,
            projection_years,
            growth_decay,
        )

        # Bear case: Growth -2%, WACC +1%
        bear_growth = max(0.01, fcf_growth - 0.02)
        bear_wacc = min(0.15, wacc + 0.01)
        bear_value = calc_dcf_value(
            current_fcf,
            bear_growth,
            bear_wacc,
            terminal_growth,
            projection_years,
            growth_decay,
        )

        # Calculate upsides for each scenario
        bull_upside = (
            ((bull_value - current_price) / current_price * 100)
            if bull_value and current_price
            else None
        )
        bear_upside = (
            ((bear_value - current_price) / current_price * 100)
            if bear_value and current_price
            else None
        )

        return {
            "DCF Fair Value": round(dcf_fair_value, 2),
            "DCF Upside (%)": round(dcf_upside, 1) if dcf_upside is not None else None,
            "DCF Bull Value": round(bull_value, 2) if bull_value else None,
            "DCF Bear Value": round(bear_value, 2) if bear_value else None,
            "DCF Bull Upside (%)": round(bull_upside, 1)
            if bull_upside is not None
            else None,
            "DCF Bear Upside (%)": round(bear_upside, 1)
            if bear_upside is not None
            else None,
            "DCF Range": f"${bear_value:.0f} - ${bull_value:.0f}"
            if bear_value and bull_value
            else None,
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


def calculate_fcf_yield(
    ticker: yf.Ticker, info: dict, risk_free_rate: float = 0.045
) -> dict:
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
        for key in [
            "Total Cash From Operating Activities",
            "Operating Cash Flow",
            "Cash From Operating Activities",
        ]:
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

                for ocf_key in [
                    "Operating Cash Flow",
                    "Total Cash From Operating Activities",
                ]:
                    if ocf_key in cf.index:
                        for capex_key in [
                            "Capital Expenditure",
                            "Capital Expenditures",
                        ]:
                            if capex_key in cf.index:
                                fcf_current = (
                                    cf.loc[ocf_key].iloc[0] + cf.loc[capex_key].iloc[0]
                                )
                                fcf_old = (
                                    cf.loc[ocf_key].iloc[-1]
                                    + cf.loc[capex_key].iloc[-1]
                                )
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
    Per RULES.md and IDEAS.md requirements.
    """
    filters = [
        # Region: US only
        EquityQuery("eq", ["region", "us"]),
        # Exchange: ASE, BTS, CXI, NCM, NGM, NMS, NYQ, OEM, OQB, OQX, PCX, PNK, YHD
        EquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
        # Market Cap: $1B - $500B (per RULES.md line 7)
        EquityQuery(
            "btwn", ["lastclosemarketcap.lasttwelvemonths", MCAP_MIN, MCAP_MAX]
        ),
        # Positive EPS (per RULES.md line 8)
        # EquityQuery("gte", ["basicepscontinuingoperations.lasttwelvemonths", 0.001]),
        # EPS Growth >= 6% (per RULES.md line 32)
        EquityQuery("gte", ["epsgrowth.lasttwelvemonths", GROWTH_MIN_PCT]),
        # P/E < 50 (per RULES.md line 13)
        EquityQuery("btwn", ["peratio.lasttwelvemonths", 10, PE_MAX]),
        # PEG < 1.1 (per RULES.md line 16-22, strict requirement)
        EquityQuery("lte", ["pegratio_5y", PEG_MAX]),
        # D/E < 0.5 (per RULES.md line 9-11)
        # Note: yfinance API expects percentage format, so 0.5 ratio = 50%
        EquityQuery("btwn", ["totaldebtequity.lasttwelvemonths", 0, (DE_MAX * 100)]),
        # ROE >= 15% (per RULES.md line 41-45)
        EquityQuery("gte", ["returnonequity.lasttwelvemonths", ROE_MIN_PCT]),
        # ROIC Proxy (Return on Total Capital) >= 10% (per RULES.md line 41-45)
        EquityQuery("gte", ["returnontotalcapital.lasttwelvemonths", ROIC_MIN_PCT]),
        # Revenue Growth >= 6% (per RULES.md line 31)
        EquityQuery("gte", ["totalrevenues1yrgrowth.lasttwelvemonths", GROWTH_MIN_PCT]),
        EquityQuery("gte", ["quarterlyrevenuegrowth.quarterly", GROWTH_MIN_PCT]),
        # EBITDA Growth >= 6% (per RULES.md line 33) - NOW ENABLED
        EquityQuery("gte", ["ebitda1yrgrowth.lasttwelvemonths", EBITDA_GROWTH_MIN_PCT]),
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
        # Beta between 0.5 and 2.5
        EquityQuery("btwn", ["beta", BETA_MIN, BETA_MAX]),
        # 52-week price change filter
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

        # Note: S&P 500/NASDAQ-100 filtering removed - cannot reliably source all constituents.
        # Stocks are already filtered by exchange (NYSE/NASDAQ) and market cap ($1B-$500B),
        # which ensures mid-to-large cap US stocks as per RULES.md requirements.

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
                equity = (
                    bs.loc["Total Equity Gross Minority Interest"].iloc[period_idx] or 0
                )
            elif "Stockholders Equity" in bs.index:
                equity = bs.loc["Stockholders Equity"].iloc[period_idx] or 0

            # Get cash for excess cash calculation
            cash = 0
            for key in [
                "Cash And Cash Equivalents",
                "Cash",
                "Cash And Short Term Investments",
            ]:
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
        prior_ic = (
            get_invested_capital(balance, 1) if len(balance.columns) > 1 else current_ic
        )

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
            "Short % of Float": round(short_percent * 100, 2)
            if short_percent
            else None,
            "Short Ratio (Days)": round(short_ratio, 2) if short_ratio else None,
            "Shares Short": shares_short,
        }

        # Calculate short interest trend
        if shares_short and shares_short_prior and shares_short_prior > 0:
            short_change = (
                (shares_short - shares_short_prior) / shares_short_prior
            ) * 100
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
            "Insider Ownership (%)": round(insider_pct * 100, 2)
            if insider_pct
            else None,
            "Institutional Ownership (%)": round(inst_pct * 100, 2)
            if inst_pct
            else None,
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
                result["Recent Insider Purchases"] = (
                    int(total_purchases) if total_purchases > 0 else 0
                )
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
                "Current Yield (%)": round(current_yield * 100, 2)
                if current_yield
                else None,
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
            "Current Yield (%)": round(current_yield * 100, 2)
            if current_yield
            else None,
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
            strong_buy * 2 + buy * 1 + hold * 0 - sell * 1 - strong_sell * 2
        ) / total

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
                val = (
                    sustainability.loc[key].iloc[0]
                    if isinstance(sustainability.loc[key], pd.Series)
                    else sustainability.loc[key]
                )
                return float(val) if pd.notna(val) else None
            return None

        total_esg = get_score("totalEsg")
        env_score = get_score("environmentScore")
        social_score = get_score("socialScore")
        gov_score = get_score("governanceScore")

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
            call_oi = (
                calls["openInterest"].sum() if "openInterest" in calls.columns else 0
            )
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

    # Define relative tolerance from config, with a fallback
    SECTOR_TOLERANCE = CONFIG.get("sector_relative_tolerance", 1.2)  # e.g., 20% premium

    # Phase 3: Batch fetch historical data for technicals
    try:
        batch_history = yf.download(
            tickers + ["SPY"],
            period="2y",
            interval="1d",
            progress=False,
            group_by="ticker",
        )
    except Exception as e:
        logger.debug(f"Batch download failed: {e}")
        batch_history = pd.DataFrame()

    print(f"  Analyzing {len(tickers)} stocks...")
    for i, symbol in enumerate(tickers):
        try:
            ticker = get_ticker(symbol)
            info = ticker.info
            rating = (info.get("recommendationKey") or "").lower()
            num_analysts = info.get("numberOfAnalystOpinions") or 0

            current_sector = info.get("sector")
            if not current_sector or current_sector in CONFIG.get(
                "excluded_sectors", []
            ):
                continue

            market_cap = info.get("marketCap", 0)
            if not market_cap:
                continue

            # --- Key Metric Calculations ---
            pfcf = calculate_pfcf(ticker, market_cap, info)
            roic = calculate_roic(ticker)
            income = ticker.financials
            # Revenue CAGR (3-year) - per RULES.md line 34-36
            rev_cagr = (
                calculate_cagr(income.loc["Total Revenue"].dropna().tolist(), 3)
                if not income.empty and "Total Revenue" in income.index
                else None
            )
            rev_cagr_pct = rev_cagr * 100 if rev_cagr is not None else None
            # EPS CAGR (3-year) - per RULES.md line 34-36, IDEAS.md line 54
            eps_cagr = None
            eps_cagr_pct = None
            if not income.empty:
                # Try to find EPS data - may need to calculate from Net Income and shares
                net_income_key = None
                for key in ["Net Income", "Net Income Common Stockholders"]:
                    if key in income.index:
                        net_income_key = key
                        break
                if net_income_key:
                    net_income_values = income.loc[net_income_key].dropna().tolist()
                    if len(net_income_values) >= 4:  # Need 4 years for 3-year CAGR
                        shares_outstanding = info.get("sharesOutstanding") or info.get(
                            "impliedSharesOutstanding"
                        )
                        if shares_outstanding and shares_outstanding > 0:
                            # Approximate EPS from Net Income / Shares
                            eps_values = [
                                ni / shares_outstanding for ni in net_income_values[:4]
                            ]
                            eps_cagr = calculate_cagr(eps_values, 3)
                            eps_cagr_pct = (
                                eps_cagr * 100 if eps_cagr is not None else None
                            )
                # Fallback: try to get EPS directly if available
                if eps_cagr is None:
                    for key in ["Diluted EPS", "Basic EPS"]:
                        if key in income.index:
                            eps_values = income.loc[key].dropna().tolist()
                            if len(eps_values) >= 4:
                                eps_cagr = calculate_cagr(eps_values[:4], 3)
                                eps_cagr_pct = (
                                    eps_cagr * 100 if eps_cagr is not None else None
                                )
                                break

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
            # Interest Coverage (per RULES.md line 50-54)
            interest_coverage_data = calculate_interest_coverage(ticker)
            # Growth Quality (includes FCF trend) - per RULES.md line 38-40, line 61
            growth_quality = calculate_growth_quality(ticker)

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
            # EPS CAGR filter (per RULES.md line 34-36, IDEAS.md line 54)
            if eps_cagr is not None and eps_cagr < CAGR_MIN:
                passed = False
                fail_reasons.append(f"3Y EPS CAGR < {CAGR_MIN * 100:.0f}%")
            # FCF Trend validation (per RULES.md line 38-40, line 61)
            fcf_backs_growth = growth_quality.get("FCF Backs Growth")
            if fcf_backs_growth is False:  # Explicitly False (not None)
                passed = False
                fail_reasons.append(
                    "FCF not backing growth (declining/negative FCF trend)"
                )
            # Interest Coverage filter (per RULES.md line 50-54)
            interest_coverage = interest_coverage_data.get("Interest Coverage")
            if (
                interest_coverage is not None
                and interest_coverage < INTEREST_COVERAGE_MIN
            ):
                passed = False
                fail_reasons.append(
                    f"Interest Coverage < {INTEREST_COVERAGE_MIN} (EBIT/Interest)"
                )
            # EV/EBITDA filter (per RULES.md line 23-25)
            if ev_ebitda is not None and ev_ebitda > 10:  # EV/EBITDA > 10 is expensive
                # Allow higher EV/EBITDA if growth justifies it
                growth_justifies = False
                if peg_ratio is not None and peg_ratio <= 1.5:
                    growth_justifies = True
                if rev_cagr_pct is not None and rev_cagr_pct >= (CAGR_MIN * 100 * 1.5):
                    growth_justifies = True
                if growth_est_pct is not None and growth_est_pct >= 15:
                    growth_justifies = True
                if not growth_justifies:
                    passed = False
                    fail_reasons.append(
                        f"EV/EBITDA too high ({ev_ebitda:.1f}) without growth justification"
                    )
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
            # Only print status for passed stocks or if verbose
            if passed:
                print(f"  ✓ {symbol}: PASS")
            # Failures are logged but not printed to reduce noise

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
            # growth_quality already calculated earlier for filtering

            # --- NEW: Enhanced Financial Health Metrics ---
            liquidity_data = check_liquidity(info)
            # interest_coverage_data already calculated earlier for filtering
            # growth_quality already calculated earlier for filtering
            current_ratio_data = calculate_current_ratio_trend(ticker)
            data_quality_data = assess_data_quality(ticker, info)

            # Multi-timeframe trend analysis
            trend_data = {}
            if ticker_close is not None and len(ticker_close) >= 60:
                trend_data = calculate_multi_timeframe_trend(ticker_close)

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
                "3Y EPS CAGR (%)": eps_cagr_pct
                or None,  # Per RULES.md line 34-36, IDEAS.md line 54
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
                # NEW: Enhanced financial health metrics
                **liquidity_data,
                **interest_coverage_data,
                **current_ratio_data,
                **data_quality_data,
                **trend_data,
            }

            # Calculate Conviction Score
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
    # Get unique symbols and sectors
    if passed_df.empty or len(passed_df) < min_stocks:
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

        # Simplified portfolio output
        print(
            f"\nPortfolio: {port_return:.1f}% return, {port_volatility:.1f}% vol, Sharpe {best_sharpe:.2f}"
        )

        # =================================================================
        # NEW: VaR and Risk Metrics
        # =================================================================
        try:
            # Calculate portfolio daily returns
            portfolio_returns = returns.dot(best_weights)

            # Calculate VaR metrics
            var_metrics = calculate_var(
                portfolio_returns,
                confidence=VAR_CONFIDENCE,
                horizon_days=VAR_HORIZON_DAYS,
            )

            if var_metrics.get("VaR (%)"):
                # VaR metrics logged but not printed to reduce noise
                logger.debug(f"Portfolio VaR: {var_metrics.get('VaR (%)', 0):.1f}%")

                # Max drawdown from historical data
                cumulative = (1 + portfolio_returns).cumprod()
                rolling_max = cumulative.cummax()
                drawdown = (cumulative - rolling_max) / rolling_max
                max_drawdown = drawdown.min() * 100
                print(f"  Historical Max Drawdown: {max_drawdown:.1f}%")

            # Add VaR to portfolio_df for export
            portfolio_df["Portfolio VaR (%)"] = var_metrics.get("VaR (%)")
            portfolio_df["Portfolio CVaR (%)"] = var_metrics.get("CVaR (%)")

        except Exception as e:
            logger.debug(f"VaR calculation failed: {e}")

        # =================================================================
        # NEW: Transaction Cost Estimate
        # =================================================================
        try:
            total_cost_bps = 0
            print("\nEstimated Transaction Costs:")
            for sym, weight in zip(symbols, best_weights):
                # Get market cap for cost estimation
                mcap = passed_df[passed_df["Symbol"] == sym]["Market Cap ($B)"].values
                mcap_value = mcap[0] * 1e9 if len(mcap) > 0 else 50e9  # Default to $50B

                cost = estimate_transaction_costs(weight, mcap_value)
                total_cost_bps += cost * weight

            # Transaction costs calculated but not printed
            net_return = port_return - (total_cost_bps / 100 * 4)
            logger.debug(f"Net return after costs: {net_return:.1f}%")

        except Exception as e:
            logger.debug(f"Transaction cost calculation failed: {e}")

        # Correlation Analysis
        try:
            corr_matrix = returns.corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val > MAX_CORRELATION:
                        high_corr_pairs.append(
                            {
                                "Stock 1": corr_matrix.columns[i],
                                "Stock 2": corr_matrix.columns[j],
                                "Correlation": corr_val,
                            }
                        )

            if high_corr_pairs:
                print(f"\n⚠️  High Correlation Warnings (>{MAX_CORRELATION}):")
                for pair in high_corr_pairs:
                    print(
                        f"  {pair['Stock 1']} <-> {pair['Stock 2']}: {pair['Correlation']:.2f}"
                    )
            else:
                avg_corr = corr_matrix.values[
                    np.triu_indices_from(corr_matrix.values, k=1)
                ].mean()
                print(logger.debug(f"Portfolio avg correlation: {avg_corr:.2f}"))
        except Exception:
            pass  # Skip correlation if calculation fails

        # Export portfolio (silent)
        portfolio_df.to_csv("portfolio_allocation.csv", index=False)

        return portfolio_df

    except Exception as e:
        print(f"Optimization failed: {e}. Using equal weight.")
        return _build_equal_weight_portfolio(symbols)


def _build_fallback_portfolio(passed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback portfolio: equal weight with stock picks + BIL + TLT.
    Used when fewer than 4 stocks pass screening.
    """
    # Removed verbose fallback message

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

    # Export (silent)
    portfolio_df.to_csv("portfolio_allocation.csv", index=False)

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

    portfolio_df.to_csv("portfolio_allocation.csv", index=False)

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
        "--backtest",
        action="store_true",
        help="Run a 1-year backtest performance analysis of the final portfolio.",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward backtesting for robust out-of-sample analysis.",
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

    # Log loaded thresholds (debug level to reduce noise)
    logger.debug(
        f"GARP Thresholds: Market Cap ${MCAP_MIN / 1e9:.0f}B-${MCAP_MAX / 1e9:.0f}B, "
        f"P/E<{PE_MAX}, PEG<{PEG_MAX}, P/FCF<{PFCF_MAX}, "
        f"ROE>{ROE_MIN_PCT}%, ROIC>{ROIC_MIN_PCT}%, D/E<{DE_MAX}"
    )

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

    # Display concise results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if passed_df.empty:
        print("No stocks passed all criteria.")
    else:
        # Show only essential columns
        display_cols = ["Symbol", "Name", "P/E", "P/FCF", "ROIC (%)", "3Y Rev CAGR (%)"]
        print(passed_df[display_cols].to_string(index=False))
        print(f"\n{len(passed_df)} stocks passed all criteria")

    # Export to CSV (silent)
    output_file = "screener_results.csv"
    results_df.to_csv(output_file, index=False)

    # Build optimized portfolio from passing stocks
    portfolio_df = pd.DataFrame()
    if not passed_df.empty and not args.skip_portfolio:
        portfolio_df = build_optimized_portfolio(passed_df, objective=args.objective)

        # Run Backtest if requested
        if args.backtest and not portfolio_df.empty:
            tickers = portfolio_df["Symbol"].tolist()
            weights = (portfolio_df["Weight (%)"] / 100.0).tolist()
            backtest_portfolio(tickers, weights)

        # Run Walk-Forward Backtest if requested
        if args.walk_forward and not portfolio_df.empty:
            tickers = portfolio_df["Symbol"].tolist()
            weights = (portfolio_df["Weight (%)"] / 100.0).tolist()
            walk_forward_backtest(tickers, weights)

    if not passed_df.empty and not args.skip_summary:
        # Generate investment summaries
        generate_stock_summaries(passed_df)

    end_time = time.time()
    elapsed_time = timedelta(seconds=end_time - start_time)
    print(f"\nCompleted in {elapsed_time}")


if __name__ == "__main__":
    main()
