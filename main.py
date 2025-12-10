"""
GARP Stock Screener - Two-Stage Implementation
Based on RULES.md and IDEAS.md methodology
"""

import os
import time
import json
import warnings
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from openai import OpenAI
from yfinance import EquityQuery
from scipy.optimize import minimize
import yaml

warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
assert ALPHA_VANTAGE_API_KEY, "ALPHA_VANTAGE_API_KEY not found in environment variables"
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY not found in environment variables"
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


# =============================================================================
# ALPHA VANTAGE API FUNCTIONS
# =============================================================================
def fetch_alpha_vantage_overview(symbol: str) -> Optional[dict]:
    """
    Fetch company fundamental data from Alpha Vantage.
    Returns key metrics like forward P/E, PEG, analyst targets, etc.
    """
    if not ALPHA_VANTAGE_API_KEY:
        return None

    try:
        params = {
            "function": "OVERVIEW",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
        }
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=10)
        data = response.json()

        # Check for API limit or error
        if "Note" in data or "Error Message" in data or not data:
            return None

        return data
    except Exception:
        return None


def get_alpha_vantage_metrics(av_data: Optional[dict]) -> dict:
    """
    Extract useful metrics from Alpha Vantage company overview.
    Returns dict with forward P/E, PEG, analyst target, etc.
    """
    if not av_data:
        return {}

    def safe_float(value):
        try:
            if value in [None, "None", "-", ""]:
                return None
            return float(value)
        except (ValueError, TypeError):
            return None

    return {
        "AV Forward P/E": safe_float(av_data.get("ForwardPE")),
        "AV PEG": safe_float(av_data.get("PEGRatio")),
        "AV Analyst Target": safe_float(av_data.get("AnalystTargetPrice")),
        "AV 52W High": safe_float(av_data.get("52WeekHigh")),
        "AV 52W Low": safe_float(av_data.get("52WeekLow")),
        "AV Beta": safe_float(av_data.get("Beta")),
        "AV Profit Margin": safe_float(av_data.get("ProfitMargin")),
        "AV Operating Margin": safe_float(av_data.get("OperatingMarginTTM")),
        "AV Quarterly Earnings Growth": safe_float(
            av_data.get("QuarterlyEarningsGrowthYOY")
        ),
        "AV Quarterly Revenue Growth": safe_float(
            av_data.get("QuarterlyRevenueGrowthYOY")
        ),
    }


def fetch_treasury_yield() -> float:
    """
    Fetch the latest 10-year Treasury yield from Alpha Vantage.
    Returns yield as a decimal (e.g., 0.045 for 4.5%).
    Default fallback: 0.045 (4.5%)
    """
    if not ALPHA_VANTAGE_API_KEY:
        return 0.045

    try:
        params = {
            "function": "TREASURY_YIELD",
            "interval": "daily",
            "maturity": "10year",
            "apikey": ALPHA_VANTAGE_API_KEY,
        }
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=10)
        data = response.json()

        if "data" in data and len(data["data"]) > 0:
            latest_yield = float(data["data"][0]["value"])
            return latest_yield / 100
        return 0.045
    except Exception as e:
        print("ERROR", e)
        return 0.045


def fetch_earnings_data(symbol: str) -> dict:
    """
    Fetch earnings data (surprises, estimates) from Alpha Vantage.
    """
    if not ALPHA_VANTAGE_API_KEY:
        return {"Earnings Surprise Avg (%)": None, "Last Quarter Surprise (%)": None}

    try:
        params = {
            "function": "EARNINGS",
            "symbol": symbol,
            "apikey": ALPHA_VANTAGE_API_KEY,
        }
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=10)
        data = response.json()

        quarterly = data.get("quarterlyEarnings", [])
        if not quarterly:
            return {
                "Earnings Surprise Avg (%)": None,
                "Last Quarter Surprise (%)": None,
            }

        # Calculate average surprise over last 4 quarters
        surprises = []
        for q in quarterly[:4]:
            if "surprisePercentage" in q and q["surprisePercentage"] not in [
                None,
                "None",
            ]:
                surprises.append(float(q["surprisePercentage"]))

        avg_surprise = sum(surprises) / len(surprises) if surprises else None

        last_surprise = None
        if quarterly and "surprisePercentage" in quarterly[0]:
            val = quarterly[0]["surprisePercentage"]
            if val not in [None, "None"]:
                last_surprise = float(val)

        return {
            "Earnings Surprise Avg (%)": avg_surprise,
            "Last Quarter Surprise (%)": last_surprise,
        }
    except Exception as e:
        print(f"ERROR in fetch_earnings_data: {e}")
        return {"Earnings Surprise Avg (%)": None, "Last Quarter Surprise (%)": None}


def fetch_sentiment_data(symbol: str) -> dict:
    """
    Fetch news sentiment data from Alpha Vantage.
    """
    if not ALPHA_VANTAGE_API_KEY:
        return {"News Sentiment Score": None, "News Sentiment Label": None}

    try:
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "limit": 1,
            "apikey": ALPHA_VANTAGE_API_KEY,
        }
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=10)
        data = response.json()

        # Sentiment is often in the feed, but we look for overall ticker sentiment if available
        # The endpoint returns a feed of articles. We can check the first few for overall sentiment.
        # For simplicity, we'll look at the first article's sentiment for this ticker.

        if "feed" in data and len(data["feed"]) > 0:
            article = data["feed"][0]
            for ticker_sent in article.get("ticker_sentiment", []):
                if ticker_sent["ticker"] == symbol:
                    return {
                        "News Sentiment Score": float(
                            ticker_sent["ticker_sentiment_score"]
                        ),
                        "News Sentiment Label": ticker_sent["ticker_sentiment_label"],
                    }
        return {"News Sentiment Score": None, "News Sentiment Label": None}
    except Exception as e:
        print(f"ERROR in fetch_sentiment_data: {e}")
        return {"News Sentiment Score": None, "News Sentiment Label": None}


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

    # Penalty for high P/E (-1 if > 30)
    pe = stock_data.get("P/E", 0) or 0
    if pe > 30:
        score -= 1
        reasons.append(f"High P/E ({pe:.1f}) ⚠️")

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

    # High D/E (> 100)
    de = stock_data.get("D/E", 0) or 0
    if de > 100:
        warnings.append(f"⚠️ High leverage (D/E: {de:.1f})")

    # Negative Graham Upside (significantly overvalued)
    graham_upside = stock_data.get("Graham Upside (%)", 0) or 0
    if graham_upside < -50:
        warnings.append(f"⚠️ Significantly above Graham value ({graham_upside:.0f}%)")

    # Low analyst coverage
    num_analysts = stock_data.get("# Analysts", 0) or 0
    if num_analysts < 5:
        warnings.append(f"⚠️ Low analyst coverage ({num_analysts} analysts)")

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
        return "AI thesis unavailable (no API key)"

    try:
        # Build context for the AI
        symbol = stock_data.get("Symbol", "Unknown")
        name = stock_data.get("Name", "Unknown")
        sector = stock_data.get("Sector", "Unknown")

        prompt = f"""You are a Lead Equity Analyst. Synthesize the following data into a compelling 2-3 sentence investment thesis for {symbol} ({name}).

Company Profile:
- Sector: {sector}
- Market Cap: ${stock_data.get("Market Cap ($B)", "N/A"):.2f}B
- Analyst Coverage: {stock_data.get("# Analysts", "N/A")} analysts

Financials & Profitability:
- P/E: {stock_data.get("P/E", "N/A")} (Forward P/E: {stock_data.get("AV Forward P/E", "N/A")})
- PEG: {stock_data.get("PEG", "N/A")}
- ROE: {stock_data.get("ROE (%)", "N/A")}%
- ROIC: {stock_data.get("ROIC (%)", "N/A")}%
- Margins: Gross {stock_data.get("Gross Margin (%)", "N/A")}%, Op {stock_data.get("Op Margin (%)", "N/A")}%

Growth:
- 3Y Revenue CAGR: {stock_data.get("3Y Rev CAGR (%)", "N/A")}%
- Quarterly Revenue Growth (YoY): {stock_data.get("AV Quarterly Revenue Growth", "N/A")}%
- Quarterly Earnings Growth (YoY): {stock_data.get("AV Quarterly Earnings Growth", "N/A")}%

Valuation Models:
- Graham Number: ${stock_data.get("Graham Number", "N/A")} (Undervalued: {stock_data.get("Graham Undervalued", "N/A")})
- DCF Fair Value: ${stock_data.get("DCF Fair Value", "N/A")} (Upside: {stock_data.get("DCF Upside (%)", "N/A")}%)
- Analyst Target: ${stock_data.get("Target Price", "N/A")} (Upside: {stock_data.get("Upside (%)", "N/A")}%)

Sentiment & Momentum:
- Price vs 52W Range: Current ${stock_data.get("Current Price", "N/A")} (Range: ${stock_data.get("AV 52W Low", "N/A")} - ${stock_data.get("AV 52W High", "N/A")})
- News Sentiment: {stock_data.get("News Sentiment Label", "N/A")} (Score: {stock_data.get("News Sentiment Score", "N/A")})
- Earnings Surprise (Avg): {stock_data.get("Earnings Surprise Avg (%)", "N/A")}%
- Analyst Rating: {stock_data.get("Analyst Rating", "N/A")}

Risk Profile:
- Conviction Score: {conviction_score}/10
- Identified Risks: {", ".join(risks) if risks else "None"}
- Debt/Equity: {stock_data.get("D/E", "N/A")}

Task:
1. Synthesize the signals (e.g., "Undervalued by DCF but facing bearish sentiment").
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
            ticker = yf.Ticker(symbol)
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

    DCF Formula:
    - Project FCF for 5 years using historical growth rate
    - Calculate terminal value using perpetuity growth model
    - Discount all cash flows to present value using WACC
    - Divide by shares outstanding to get fair value per share
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

        # Get Free Cash Flow (Operating CF - CapEx)
        operating_cf = None
        capex = None

        if "Operating Cash Flow" in cf.index:
            operating_cf = cf.loc["Operating Cash Flow"].dropna().tolist()
        elif "Total Cash From Operating Activities" in cf.index:
            operating_cf = (
                cf.loc["Total Cash From Operating Activities"].dropna().tolist()
            )

        if "Capital Expenditure" in cf.index:
            capex = cf.loc["Capital Expenditure"].dropna().tolist()
        elif "Capital Expenditures" in cf.index:
            capex = cf.loc["Capital Expenditures"].dropna().tolist()

        if not operating_cf or not capex or len(operating_cf) < 2:
            return {
                "DCF Fair Value": None,
                "DCF Upside (%)": None,
                "DCF Notes": "Insufficient FCF data",
            }

        # Calculate current FCF (most recent year)
        current_fcf = operating_cf[0] + capex[0]  # capex is typically negative

        if current_fcf <= 0:
            return {
                "DCF Fair Value": None,
                "DCF Upside (%)": None,
                "DCF Notes": "Negative FCF",
            }

        # Estimate FCF growth rate from historical data
        if len(operating_cf) >= 3:
            older_fcf = operating_cf[2] + (capex[2] if len(capex) > 2 else 0)
            if older_fcf > 0:
                fcf_growth = (current_fcf / older_fcf) ** (1 / 2) - 1
            else:
                fcf_growth = 0.08  # Default 8%
        else:
            fcf_growth = 0.08  # Default 8%

        # Cap growth rate at reasonable bounds
        fcf_growth = max(0.03, min(0.25, fcf_growth))  # 3% to 25%

        # DCF Parameters
        projection_years = 5
        terminal_growth = 0.025  # 2.5% perpetuity growth

        # Estimate WACC (simplified)
        beta = info.get("beta", 1.0) or 1.0
        # risk_free_rate passed as argument
        market_premium = 0.055  # Historical equity risk premium
        cost_of_equity = risk_free_rate + beta * market_premium

        # Simplified WACC (assume 30% debt at 6% cost)
        debt_weight = 0.30
        equity_weight = 0.70
        cost_of_debt = 0.06
        tax_rate = 0.21
        wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (
            1 - tax_rate
        )
        wacc = max(0.08, min(0.15, wacc))  # Cap WACC at 8-15%

        # Project FCF for 5 years
        projected_fcf = []
        fcf = current_fcf
        for year in range(1, projection_years + 1):
            fcf = fcf * (1 + fcf_growth)
            # Apply growth decay (growth slows over time)
            fcf_growth = fcf_growth * 0.9  # 10% decay per year
            projected_fcf.append(fcf)

        # Calculate present value of projected FCFs
        pv_fcf = 0
        for i, fcf in enumerate(projected_fcf):
            pv_fcf += fcf / ((1 + wacc) ** (i + 1))

        # Calculate terminal value (Gordon Growth Model)
        terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (wacc - terminal_growth)
        pv_terminal = terminal_value / ((1 + wacc) ** projection_years)

        # Enterprise Value
        enterprise_value = pv_fcf + pv_terminal

        # Adjust for net debt to get equity value
        total_cash = info.get("totalCash", 0) or 0
        total_debt = info.get("totalDebt", 0) or 0
        net_debt = total_debt - total_cash
        equity_value = enterprise_value - net_debt

        # Get shares outstanding
        shares_outstanding = info.get("sharesOutstanding", 0)
        if not shares_outstanding or shares_outstanding <= 0:
            return {
                "DCF Fair Value": None,
                "DCF Upside (%)": None,
                "DCF Notes": "No shares data",
            }

        # Calculate fair value per share
        dcf_fair_value = equity_value / shares_outstanding

        if dcf_fair_value <= 0:
            return {
                "DCF Fair Value": None,
                "DCF Upside (%)": None,
                "DCF Notes": "Negative equity value",
            }

        # Calculate upside vs current price
        current_price = info.get("currentPrice", 0) or info.get("regularMarketPrice", 0)
        dcf_upside = None
        if current_price and current_price > 0:
            dcf_upside = ((dcf_fair_value - current_price) / current_price) * 100

        return {
            "DCF Fair Value": round(dcf_fair_value, 2),
            "DCF Upside (%)": round(dcf_upside, 1) if dcf_upside else None,
            "DCF Notes": f"WACC={wacc * 100:.1f}%, Growth={fcf_growth * 100:.1f}%",
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
        EquityQuery("btwn", ["peratio.lasttwelvemonths", 10, PE_MAX]),
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
        EquityQuery("gte", ["ebitda1yrgrowth.lasttwelvemonths", GROWTH_MIN_PCT]),
        # Net Income Growth >= 6%
        EquityQuery("gte", ["netincome1yrgrowth.lasttwelvemonths", GROWTH_MIN_PCT]),
        # Diluted EPS Growth >= 6%
        EquityQuery("gte", ["dilutedeps1yrgrowth.lasttwelvemonths", GROWTH_MIN_PCT]),
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


def run_stage1() -> pd.DataFrame:
    """
    Execute Stage 1 screening via yfinance API.
    Returns DataFrame of stocks passing initial filters.
    """
    print("=" * 60)
    print("STAGE 1: Running EquityQuery API Filter")
    print("=" * 60)

    query = build_stage1_query()

    try:
        result = yf.screen(query, size=50, sortField="intradaymarketcap", sortAsc=False)

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


def calculate_pfcf(ticker: yf.Ticker, market_cap: float) -> Optional[float]:
    """Calculate Price-to-Free Cash Flow ratio."""
    try:
        cf = ticker.cashflow
        if cf.empty:
            return None

        # Try to get Free Cash Flow or calculate from Operating CF - CapEx
        if "Free Cash Flow" in cf.index:
            fcf = cf.loc["Free Cash Flow"].iloc[0]
        elif "Operating Cash Flow" in cf.index and "Capital Expenditure" in cf.index:
            ocf = cf.loc["Operating Cash Flow"].iloc[0]
            capex = cf.loc["Capital Expenditure"].iloc[0]
            fcf = ocf + capex  # CapEx is negative
        else:
            return None

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
    Calculates P/FCF, ROIC, and 3-year CAGR.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: Running Custom Ticker Analysis")
    print("=" * 60)

    if stage1_df.empty:
        return pd.DataFrame()

    results = []
    tickers = stage1_df["symbol"].tolist() if "symbol" in stage1_df.columns else []

    # Fetch dynamic risk-free rate for DCF
    risk_free_rate = fetch_treasury_yield()
    print(f"  Using Risk-Free Rate (10Y Treasury): {risk_free_rate * 100:.2f}%")

    for i, symbol in enumerate(tickers):
        print(f"  Analyzing {symbol} ({i + 1}/{len(tickers)})...", end=" ")

        result_row = {
            "Symbol": symbol,
            "Stage 2 Pass": False,
            "Failure Reason": "Unknown",
        }  # Initialize for error cases

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Apply excluded_sectors filter here
            current_sector = info.get("sector")
            excluded_sectors = CONFIG.get("excluded_sectors", [])
            if current_sector and current_sector in excluded_sectors:
                print(f"SKIP (Excluded Sector: {current_sector})")
                continue

            market_cap = info.get("marketCap", 0)
            if not market_cap:
                print("SKIP (no market cap)")
                result_row["Failure Reason"] = "No market cap"
                results.append(result_row)
                continue

            # Calculate P/FCF
            pfcf = calculate_pfcf(ticker, market_cap)

            # Calculate ROIC
            roic = calculate_roic(ticker)

            # Calculate 3-Year Revenue CAGR
            income = ticker.financials
            rev_cagr = None
            eps_cagr = None

            if not income.empty:
                if "Total Revenue" in income.index:
                    revenues = income.loc["Total Revenue"].dropna().tolist()
                    rev_cagr = calculate_cagr(revenues, 3)

                if "Basic EPS" in income.index:
                    eps_values = income.loc["Basic EPS"].dropna().tolist()
                    eps_cagr = calculate_cagr(eps_values, 3)

            # Get analyst data
            current_price = info.get("currentPrice", 0) or info.get(
                "regularMarketPrice", 0
            )
            target_price = info.get("targetMeanPrice", None)
            upside_pct = None
            if current_price and target_price:
                upside_pct = ((target_price - current_price) / current_price) * 100

            # Analyst recommendations (from info dict)
            analyst_rating = info.get("recommendationKey", "N/A")
            num_analysts = info.get("numberOfAnalystOpinions", 0)

            # Calculate Graham Number (intrinsic value)
            eps = info.get("trailingEps", None)
            book_value_per_share = info.get("bookValue", None)
            graham_number = calculate_graham_number(eps, book_value_per_share)

            # Graham margin of safety: is current price below Graham Number?
            graham_upside = None
            is_graham_undervalued = False
            if graham_number and current_price and current_price > 0:
                graham_upside = ((graham_number - current_price) / current_price) * 100
                is_graham_undervalued = current_price < graham_number

            # Fetch Alpha Vantage data (if API key available)
            if ALPHA_VANTAGE_API_KEY:
                # Rate limit: Free tier is 5 calls/minute (1 call every 12s)
                # We add a small delay to be safe
                time.sleep(2)

            av_data = fetch_alpha_vantage_overview(symbol)
            av_metrics = get_alpha_vantage_metrics(av_data)

            # Fetch additional Alpha Vantage data
            earnings_data = fetch_earnings_data(symbol)
            sentiment_data = fetch_sentiment_data(symbol)

            # Check Stage 2 thresholds
            passed = True
            fail_reasons = []

            if pfcf is not None and pfcf > PFCF_MAX:
                passed = False
                fail_reasons.append(f"P/FCF={pfcf:.1f}")

            if roic is not None and roic < ROIC_MIN:
                passed = False
                fail_reasons.append(f"ROIC={roic * 100:.1f}%")

            if rev_cagr is not None and rev_cagr < CAGR_MIN:
                passed = False
                fail_reasons.append(f"RevCAGR={rev_cagr * 100:.1f}%")

            # D/E Filter (Stage 2 validation)
            de_ratio = info.get("debtToEquity", None)
            if (
                de_ratio is not None and de_ratio > DE_MAX * 100
            ):  # yfinance returns D/E as percentage (e.g. 150 for 1.5)
                passed = False
                fail_reasons.append(f"D/E={de_ratio:.1f}")

            if analyst_rating not in ACCEPTABLE_RATINGS:
                passed = False
                fail_reasons.append(f"Analyst Rating={analyst_rating}")

            status = "PASS" if passed else f"FAIL ({', '.join(fail_reasons)})"
            print(status)

            # Calculate DCF Fair Value
            dcf_data = calculate_dcf_fair_value(ticker, info, risk_free_rate)

            # Build result dict with all metrics
            result_row = {
                "Symbol": symbol,
                "Name": info.get("shortName", "N/A"),
                "Sector": info.get("sector", "N/A"),
                "Index": get_index_membership(symbol),  # SPY, QQQ, SPY/QQQ, or —
                "Market Cap ($B)": market_cap / 1e9,
                "P/E": info.get("trailingPE", None),
                "PEG": av_metrics.get("AV PEG", None),
                "D/E": info.get("debtToEquity", None),
                "ROE (%)": (info.get("returnOnEquity", 0) or 0) * 100,
                "Gross Margin (%)": (info.get("grossMargins", 0) or 0) * 100,
                "Op Margin (%)": (info.get("operatingMargins", 0) or 0) * 100,
                "P/FCF": pfcf,
                "ROIC (%)": roic * 100 if roic else None,
                "3Y Rev CAGR (%)": rev_cagr * 100 if rev_cagr else None,
                "3Y EPS CAGR (%)": eps_cagr * 100 if eps_cagr else None,
                # Analyst data
                "Analyst Rating": analyst_rating,
                "# Analysts": num_analysts,
                "Target Price": target_price,
                "Current Price": current_price,
                "Upside (%)": upside_pct,
                # Graham Number (intrinsic value)
                "Graham Number": graham_number,
                "Graham Upside (%)": graham_upside,
                "Graham Undervalued": is_graham_undervalued,
                # DCF Valuation
                "DCF Fair Value": dcf_data.get("DCF Fair Value"),
                "DCF Upside (%)": dcf_data.get("DCF Upside (%)"),
                "DCF Notes": dcf_data.get("DCF Notes"),
                "Stage 2 Pass": passed,
            }

            # Add Alpha Vantage metrics if available
            result_row.update(av_metrics)
            result_row.update(earnings_data)
            result_row.update(sentiment_data)

            # Calculate Conviction Score immediately
            conviction, conviction_reasons = calculate_conviction_score(result_row)
            result_row["Conviction Score"] = conviction
            result_row["Conviction Reasons"] = "; ".join(conviction_reasons)

            results.append(result_row)

        except Exception as e:
            print(f"ERROR ({e})")
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


def main():
    start_time = time.time()
    """Main entry point for the GARP stock screener."""
    print("\n" + "=" * 60)
    print("       GARP STOCK SCREENER - Two-Stage Implementation")
    print("=" * 60)

    # Stage 1: Fast API filtering
    stage1_df = run_stage1()

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
            "PEG",
            "ROE (%)",
            "P/FCF",
            "ROIC (%)",
            "3Y Rev CAGR (%)",
        ]
        print(passed_df[display_cols].to_string(index=False))
        print(f"\nTotal: {len(passed_df)} stocks passed all criteria")

    # Export to CSV
    output_file = "screener_results.csv"
    passed_df.to_csv(output_file, index=False)
    print(f"\nFull results exported to: {output_file}")

    # Build optimized portfolio from passing stocks
    portfolio_df = pd.DataFrame()
    if not passed_df.empty:
        portfolio_df = build_optimized_portfolio(passed_df)

        # Generate AI-powered investment advice
        advice_df = generate_investment_advice(passed_df)

        # Generate Professional PDF Report
        from report_generator import generate_pdf_report

        # Get risk free rate again or pass it down (it was calculated in run_stage2 but local scope)
        # We'll fetch it again or use default
        rf_rate = fetch_treasury_yield()
        generate_pdf_report(advice_df, portfolio_df, rf_rate)
    end_time = time.time()
    elapsed_time = timedelta(seconds=end_time - start_time)
    print(f"\nTotal runtime: {elapsed_time}")


if __name__ == "__main__":
    main()
