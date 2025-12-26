# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GARP (Growth At a Reasonable Price) Stock Screener - A Python-based quantitative stock screening tool that implements a two-stage filtering process combining fundamental analysis, technical indicators, and machine learning predictions.

## Running the Application

```bash
# Basic run - Full screening pipeline
python main.py

# Analyze specific tickers (skip Stage 1)
python main.py --tickers AAPL MSFT GOOGL

# Skip portfolio optimization
python main.py --skip-portfolio

# Skip investment summary generation
python main.py --skip-summary

# Skip PDF report generation
python main.py --skip-pdf

# Run with backtesting
python main.py --backtest

# Change portfolio optimization objective
python main.py --objective sharpe  # or 'return' or 'sortino'

# Limit Stage 1 results
python main.py --stage1-size 100
```

## Development Setup

```bash
# Python 3.10+ required (specified in pyproject.toml)
# Install dependencies using uv or pip
uv pip install -e .
# or
pip install -e .

# No API keys required - uses yfinance which doesn't require authentication
```

## Architecture

### Two-Stage Screening Process

**Stage 1** (`run_stage1()`):
- Fast API-based filtering using yfinance's EquityQuery
- Constructs query from config.yaml thresholds (market cap, P/E, ROE, growth, margins, etc.)
- Returns initial candidate list sorted by market cap
- Can be bypassed with `--tickers` flag

**Stage 2** (`run_stage2()`):
- Deep analysis on Stage 1 candidates
- Batch fetches 2-year price history for all tickers
- Initializes ML predictor (AlphaPredictor from ml_engine.py)
- For each ticker, calculates 60+ metrics including:
  - Valuation: P/FCF, EV/EBITDA, Graham Number, DCF fair value with bull/bear/base scenarios, FCF Yield, target price upside
  - Quality: ROIC (TTM with average invested capital), Piotroski Score, Altman Z-Score (sector-adjusted), cash flow quality, Growth Quality Score, ESG scores
  - Growth: Revenue/Earnings CAGR, sector trend analysis, margin trend analysis, operating leverage
  - Technical: RSI (with edge case handling), moving averages, support/resistance, 52-week position
  - Sentiment: Earnings surprises, analyst recommendations, analyst upgrades/downgrades momentum, recommendations sentiment score
  - Ownership: Insider ownership %, institutional ownership %, recent insider purchases
  - Risk: Short interest metrics, implied volatility from options, put/call ratio
  - Dividends: Dividend growth CAGR, consistency rating, Dividend Aristocrat status
  - Sector-relative comparison: compares each stock to sector medians (cached for performance)
- Applies conviction scoring system
- Generates structured investment summaries

### Key Components

**main.py**:
- Primary application logic (~4500 lines)
- Configuration loading from config.yaml
- 60+ financial calculation functions including:
  - New yfinance data metrics: `get_short_interest_metrics()`, `get_analyst_upgrades_downgrades()`, `get_target_price_analysis()`, `get_insider_ownership_metrics()`, `get_dividend_growth_analysis()`, `get_recommendations_sentiment()`, `get_esg_scores()`, `get_options_implied_volatility()`, `get_52_week_position()`
- Stage 1 & Stage 2 orchestration
- Portfolio optimization (Sharpe ratio, returns, Sortino ratio)
- Backtesting engine
- Investment summary generation

**ml_engine.py**:
- AlphaPredictor class: RandomForest-based return predictor
- Features: RSI, relative strength vs SPY, volatility, SMA distance
- Trains on batch historical data, predicts 20-day forward returns
- Integrated into conviction scoring

**report_generator.py**:
- ResearchReport class (extends FPDF)
- Generates professional PDF research reports
- Includes executive summary, portfolio allocation, stock details

**config.yaml**:
- Central configuration for all screening thresholds
- Market cap bounds, valuation limits, profitability minimums
- Growth requirements, balance sheet ratios
- Excluded symbols and sectors
- Analyst rating filters
- Valuation: margin_of_safety, dcf_scenarios (bull/bear/base)
- Quality: growth_quality_weight, quality_trend_lookback_years
- Conviction: category weights (valuation, quality, growth, technical, sentiment), risk_penalty_per_flag

### Data Flow

1. Load config.yaml → extract thresholds
2. Stage 1: API query → initial candidates DataFrame
3. Stage 2: Batch fetch history → train ML model → analyze each ticker
4. Filter passing stocks → sort by conviction score
5. Portfolio optimization → calculate optimal weights
6. Generate investment summaries → PDF report
7. Export: screener_results.csv, portfolio_allocation.csv, investment_summary.csv

### Important Implementation Details

**Rate Limiting**:
- `@rate_limit` decorator throttles API calls to 2/sec
- Used on ticker fetching and API-heavy functions

**Caching**:
- `_TICKER_CACHE` dictionary caches yf.Ticker objects
- `_SECTOR_METRICS_CACHE` dictionary caches sector median calculations
- Reduces redundant API calls during analysis

**Sector-Relative Screening**:
- `get_sector_median_metrics()` fetches and caches sector benchmarks
- Stage 2 compares stocks to sector medians with tolerance factor
- Example: P/E can be 20% above sector median and still pass

**Conviction Scoring**:
- 100-point scale combining fundamentals, technicals, sentiment
- Weights: Quality (30), Valuation (25), Growth (20), Technical (15), Sentiment (10)
- ML alpha prediction adds bonus points
- Generates specific conviction reasons

**Financial Validations**:
- `validate_financial_data()` checks for data quality issues
- Flags missing earnings, negative book value, stale data
- Ensures minimum data requirements before deep analysis

**Portfolio Optimization**:
- Scipy minimize for objective functions (Sharpe, returns, Sortino)
- Constraints: max 15% per position, min 3% allocation
- Sector diversification limits
- Fallback to equal-weight if optimization fails

## Code Patterns

**Safe Value Extraction**:
```python
# Use helper functions to handle missing/NaN values
safe_divide(numerator, denominator, default=0.0)
safe_get_value(series, key, default=0.0)
clean_none_nan(value)  # Returns None for NaN/None, formatted string otherwise
```

**Metric Calculation Functions**:
- Return dict with `{"value": float, "status": str}` pattern
- Status: "pass"/"fail"/"warning"/"error"
- Include confidence indicators and supporting data

**Error Handling**:
- Broad try-except blocks with logging
- Graceful degradation (skip metrics if data unavailable)
- Return empty DataFrames or default values on failures

## Configuration

All screening parameters are in config.yaml:
- Modify thresholds without code changes
- Add/remove excluded symbols or sectors
- Adjust sector-relative tolerance
- Configure analyst rating requirements

## Output Files

- `screener_results.csv`: All analyzed stocks with full metrics
- `portfolio_allocation.csv`: Optimized portfolio weights
- `investment_summary.csv`: Structured investment recommendations
- `GARP_Research_Report.pdf`: Professional research report
- `backtest_results.csv`: Historical performance analysis (with --backtest)
- `screener.log`: Application logs

## Notes

- No existing test suite - tests would need to be created
- Uses yfinance API (no external API keys required)
- Heavy computational load - Stage 2 can take 10-30 minutes for large batches
- Batch operations preferred over individual ticker loops for performance
