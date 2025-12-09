"""
GARP Stock Screener - Two-Stage Implementation
Based on RULES.md and IDEAS.md methodology
"""

import os
import yfinance as yf
from yfinance import EquityQuery
import pandas as pd
import requests
from typing import Optional
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Alpha Vantage API configuration
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# =============================================================================
# CONFIGURATION - Thresholds from RULES.md / IDEAS.md
# =============================================================================
MCAP_MIN = 15_000_000_000  # $15 billion
MCAP_MAX = 500_000_000_000 * 4  # $1 trillion
PE_MAX = 45.0
PEG_MAX = 1
DE_MAX = 50  # More lenient - validate in Stage 2
ROE_MIN = 0.15  # 15%
GROWTH_MIN = 0.06  # 6%
PFCF_MAX = 30.0
ROIC_MIN = 0.10  # 10%
CAGR_MIN = 0.06  # 6%

# Buffett/Ackman quality thresholds (lenient for Stage 1, strict in Stage 2)
GROSS_MARGIN_MIN = 0.25  # 25% - moat indicator (Buffett prefers 40%+)
OPERATING_MARGIN_MIN = 0.12  # 12% - efficiency indicator


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
        EquityQuery("btwn", ["peratio.lasttwelvemonths", 5, PE_MAX]),
        # PEG < 1.1
        EquityQuery("lte", ["pegratio_5y", PEG_MAX]),
        # D/E < 0.5
        # EquityQuery("btwn", ["totaldebtequity.lasttwelvemonths", 0, DE_MAX]),
        # ROE >= 15%
        EquityQuery("gte", ["returnonequity.lasttwelvemonths", ROE_MIN]),
        # Revenue Growth >= 6%
        EquityQuery("gte", ["totalrevenues1yrgrowth.lasttwelvemonths", GROWTH_MIN]),
        EquityQuery("gte", ["quarterlyrevenuegrowth.quarterly", GROWTH_MIN]),
        # EBITDA Growth >= 6%
        EquityQuery("gte", ["ebitda1yrgrowth.lasttwelvemonths", GROWTH_MIN]),
        # Gross Margin >= 25% (Buffett moat indicator)
        EquityQuery("gte", ["grossprofitmargin.lasttwelvemonths", GROSS_MARGIN_MIN]),
        # Operating Margin >= 12% (efficiency)
        EquityQuery("gte", ["ebitdamargin.lasttwelvemonths", OPERATING_MARGIN_MIN]),
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
        result = yf.screen(
            query, size=250, sortField="intradaymarketcap", sortAsc=False
        )

        if result is None or "quotes" not in result:
            print("No results from Stage 1 screening")
            return pd.DataFrame()

        df = pd.DataFrame(result["quotes"])
        print(f"Stage 1 found {len(df)} stocks passing initial filters")
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
    except Exception:
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

    except Exception:
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
    except Exception:
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
    except Exception:
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

    for i, symbol in enumerate(tickers):
        print(f"  Analyzing {symbol} ({i + 1}/{len(tickers)})...", end=" ")

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            market_cap = info.get("marketCap", 0)
            if not market_cap:
                print("SKIP (no market cap)")
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
            av_data = fetch_alpha_vantage_overview(symbol)
            av_metrics = get_alpha_vantage_metrics(av_data)

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

            if analyst_rating not in [
                "buy",
                "strong_buy",
                "outperform",
                # "none",
                # "hold",
            ]:
                passed = False
                fail_reasons.append(f"Analyst Rating={analyst_rating}")

            status = "PASS" if passed else f"FAIL ({', '.join(fail_reasons)})"
            print(status)

            # Build result dict with all metrics
            result_row = {
                "Symbol": symbol,
                "Name": info.get("shortName", "N/A"),
                "Sector": info.get("sector", "N/A"),
                "Market Cap ($B)": market_cap / 1e9,
                "P/E": info.get("trailingPE", None),
                "PEG": info.get("pegRatio", None),
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
                "Stage 2 Pass": passed,
            }

            # Add Alpha Vantage metrics if available
            result_row.update(av_metrics)

            results.append(result_row)

        except Exception as e:
            print(f"ERROR ({e})")
            continue

    return pd.DataFrame(results)


def main():
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

    # Filter to only passing stocks
    passed_df = results_df[results_df["Stage 2 Pass"] == True].copy()

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
    results_df = results_df[results_df["Stage 2 Pass"]]
    results_df.to_csv(output_file, index=False)
    print(f"\nFull results exported to: {output_file}")


if __name__ == "__main__":
    main()
