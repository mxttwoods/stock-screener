from datetime import datetime
import re

import numpy as np
import pandas as pd
from fpdf import FPDF


class ResearchReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.report_date = datetime.now().strftime("%B %d, %Y")

    def strip_emojis(self, text):
        """
        Remove all emoji characters from text for PDF compatibility.
        """
        if not text or not isinstance(text, str):
            return str(text) if text is not None else ""

        # Remove common emojis used in the screener
        emojis_to_remove = [
            "🟢",
            "🟡",
            "🔵",
            "⚪",
            "🔴",  # Action emojis
            "✓",
            "⚠️",
            "⭐",
            "❗",  # Status emojis
        ]

        cleaned = text
        for emoji in emojis_to_remove:
            cleaned = cleaned.replace(emoji, "")

        # Also remove any other Unicode emoji characters using regex
        # Remove emoji ranges
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags
            "\U00002702-\U000027b0"  # dingbats
            "\U000024c2-\U0001f251"  # enclosed characters
            "]+",
            flags=re.UNICODE,
        )
        cleaned = emoji_pattern.sub("", cleaned)

        return cleaned.strip()

    def safe_multi_cell(
        self, w, h, txt, border=0, align="J", fill=False, max_length=500
    ):
        """
        Safely render multi-cell text with truncation and error handling.
        """
        if not txt:
            return

        # Convert to string and clean
        txt_str = str(txt) if txt is not None else ""
        txt_clean = self.strip_emojis(txt_str)

        # Truncate if too long
        if len(txt_clean) > max_length:
            txt_clean = txt_clean[:max_length] + "..."

        # Remove any problematic characters that might cause rendering issues
        # Keep only printable ASCII and common punctuation
        try:
            txt_clean = "".join(
                c for c in txt_clean if c.isprintable() or c in ["\n", "\r", "\t"]
            )
        except Exception:
            # Fallback: just use ASCII
            txt_clean = txt_clean.encode("ascii", "ignore").decode("ascii")

        # Ensure we have valid width
        if w <= 0:
            w = self.w - self.l_margin - self.r_margin

        try:
            self.multi_cell(w, h, txt_clean, border, align, fill)
        except Exception:
            # Fallback: try with even shorter text
            try:
                txt_clean = txt_clean[:200] + "..."
                self.multi_cell(w, h, txt_clean, border, align, fill)
            except Exception:
                # Last resort: just skip this text
                try:
                    self.cell(
                        w, h, "[Text too long to display]", border, 1, align, fill
                    )
                except Exception:
                    # If even that fails, just move to next line
                    self.ln(h)

    def header(self):
        # Logo or Brand Name
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, "GARP Capital Research", 0, 0, "L")
        self.cell(0, 10, f"Report Date: {self.report_date}", 0, 0, "R")
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def title_page(self, risk_free_rate):
        self.add_page()
        self.set_font("Helvetica", "B", 24)
        self.set_text_color(0, 0, 0)
        self.ln(60)
        self.cell(0, 10, "GARP Strategy", 0, 1, "C")
        self.cell(0, 10, "Research Report", 0, 1, "C")
        self.ln(10)

        self.set_font("Helvetica", "", 12)
        self.cell(0, 10, "Prepared by: AI Quantitative Analyst", 0, 1, "C")
        self.ln(20)

        # Market Context Box
        self.set_fill_color(240, 240, 240)
        self.rect(50, 140, 110, 40, "F")
        self.set_xy(50, 145)
        self.set_font("Helvetica", "B", 12)
        self.cell(110, 10, "Market Context", 0, 1, "C")
        self.set_font("Helvetica", "", 11)
        self.set_xy(50, 155)
        self.cell(
            110, 8, f"10-Year Treasury Yield: {risk_free_rate * 100:.2f}%", 0, 1, "C"
        )
        self.set_xy(50, 163)
        self.cell(110, 8, "Risk-Free Rate Assumption", 0, 1, "C")

    def executive_summary(self, top_picks):
        self.add_page()
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "Executive Summary", 0, 1, "L")
        self.ln(5)

        self.set_font("Helvetica", "", 11)
        self.safe_multi_cell(
            0,
            6,
            "This report identifies high-quality Growth at a Reasonable Price (GARP) opportunities. "
            "Our screening process combines rigorous quantitative filters with AI-powered fundamental analysis. "
            "The selected stocks demonstrate strong return on capital, reasonable valuations, and positive momentum.",
            max_length=500,
        )
        self.ln(10)

        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Top High-Conviction Picks", 0, 1, "L")
        self.ln(5)

        for _, stock in top_picks.head(3).iterrows():
            self.set_font("Helvetica", "B", 12)
            self.cell(30, 8, stock["Symbol"], 0, 0)
            self.set_font("Helvetica", "", 12)
            name_clean = self.strip_emojis(str(stock.get("Name", "N/A")))
            self.cell(0, 8, f"- {name_clean}", 0, 1)

            self.set_font("Helvetica", "I", 10)
            summary = stock.get("Investment Summary", "No summary available")
            summary_clean = self.strip_emojis(str(summary))
            self.safe_multi_cell(
                0, 6, f"Summary: {summary_clean[:200]}...", max_length=300
            )
            self.ln(5)

    def portfolio_allocation(self, portfolio_df):
        self.add_page()
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "Portfolio Strategy", 0, 1, "L")
        self.ln(5)

        self.set_font("Helvetica", "", 11)
        self.safe_multi_cell(
            0,
            6,
            "The following portfolio is optimized to maximize the Sharpe Ratio, balancing expected returns against volatility. "
            "Allocation is weighted towards our highest conviction ideas.",
            max_length=500,
        )
        self.ln(10)

        # Portfolio Allocation Table
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, "Optimized Portfolio Allocation", 0, 1)
        self.ln(2)

        # Table Header
        self.set_font("Helvetica", "B", 10)
        self.set_fill_color(240, 240, 240)
        self.cell(30, 8, "Symbol", 1, 0, "C", 1)
        self.cell(30, 8, "Weight (%)", 1, 0, "C", 1)
        self.cell(40, 8, "Allocation ($10k)", 1, 0, "C", 1)
        self.cell(60, 8, "Sector", 1, 1, "C", 1)

        # Table Rows
        self.set_font("Helvetica", "", 10)
        for _, row in portfolio_df.iterrows():
            weight = row["Weight (%)"]
            allocation_amt = (weight / 100) * 10000

            self.cell(30, 8, row["Symbol"], 1, 0, "C")
            self.cell(30, 8, f"{weight:.2f}%", 1, 0, "C")
            self.cell(40, 8, f"${allocation_amt:,.2f}", 1, 0, "C")
            sector_clean = self.strip_emojis(str(row.get("Sector", "N/A")))
            self.cell(60, 8, sector_clean, 1, 1, "C")

        self.ln(10)

    def stock_analysis_card(self, stock_data):
        self.add_page()

        # Header Section
        self.set_font("Helvetica", "B", 20)
        symbol = stock_data.get("Symbol", "N/A")
        name = self.strip_emojis(str(stock_data.get("Name", "N/A")))
        self.cell(120, 10, f"{symbol} - {name}", 0, 0)

        # Rating Badge
        action = stock_data.get("Action", "N/A")
        # Strip emojis for PDF compatibility
        action_clean = self.strip_emojis(action)

        if "BUY" in action:
            self.set_text_color(0, 150, 0)  # Green
        elif "HOLD" in action:
            self.set_text_color(200, 150, 0)  # Orange
        else:
            self.set_text_color(100, 100, 100)  # Grey

        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, action_clean, 0, 1, "R")
        self.set_text_color(0, 0, 0)  # Reset color

        self.ln(5)
        self.set_font("Helvetica", "", 10)
        sector = self.strip_emojis(str(stock_data.get("Sector", "N/A")))
        price = stock_data.get("Current Price", 0)
        self.cell(
            0,
            6,
            f"Sector: {sector} | Price: ${price:.2f}",
            0,
            1,
        )
        self.ln(10)

        # Metrics Grid
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, "Key Financial Metrics", 0, 1)
        self.ln(2)

        # Helper to safely format floats
        def safe_fmt(val, fmt="{:.2f}"):
            try:
                if val is None or val == "N/A" or pd.isna(val):
                    return "N/A"
                val_float = float(val)
                if not pd.isna(val_float) and np.isfinite(val_float):
                    return fmt.format(val_float)
                return "N/A"
            except (ValueError, TypeError):
                return "N/A"

        metrics = [
            (
                "Market Cap",
                f"${safe_fmt(stock_data.get('Market Cap ($B)'), '{:.1f}')}B",
            ),
            ("P/E Ratio", safe_fmt(stock_data.get("P/E"))),
            ("PEG Ratio", safe_fmt(stock_data.get("PEG Ratio"))),
            ("P/FCF", safe_fmt(stock_data.get("P/FCF"))),
            ("ROIC", f"{safe_fmt(stock_data.get('ROIC (%)'))}%"),
            ("ROE", f"{safe_fmt(stock_data.get('ROE (%)'))}%"),
            ("3Y Rev CAGR", f"{safe_fmt(stock_data.get('3Y Rev CAGR (%)'))}%"),
            ("Div Yield", f"{safe_fmt(stock_data.get('Dividend Yield (%)', 0))}%"),
        ]

        # Draw 2x4 Grid
        col_width = 45
        row_height = 10
        self.set_font("Helvetica", "", 10)

        for i, (label, value) in enumerate(metrics):
            x_pos = 10 + (i % 4) * col_width
            y_pos = self.get_y()

            self.set_xy(x_pos, y_pos)
            self.set_fill_color(245, 245, 245)
            self.rect(x_pos, y_pos, col_width - 2, row_height * 2, "F")

            self.set_xy(x_pos, y_pos + 2)
            self.set_font("Helvetica", "B", 9)
            self.cell(col_width - 2, 5, label, 0, 2, "C")
            self.set_font("Helvetica", "", 11)
            self.cell(col_width - 2, 8, str(value), 0, 0, "C")

            if (i + 1) % 4 == 0:
                self.ln(row_height * 2 + 5)

        if len(metrics) % 4 != 0:
            self.ln(row_height * 2 + 5)

        self.ln(5)

        # Investment Summary
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, "Investment Summary", 0, 1)
        self.set_font("Helvetica", "", 11)
        summary_text = stock_data.get("Investment Summary", "No summary available.")
        if summary_text:
            self.safe_multi_cell(0, 6, summary_text, max_length=800)
        self.ln(10)

        # Enhanced Advice Fields (if available)
        if stock_data.get("Action_Rationale"):
            self.set_font("Helvetica", "B", 10)
            self.cell(0, 6, "Action Rationale:", 0, 1)
            self.set_font("Helvetica", "", 9)
            self.safe_multi_cell(
                0, 5, stock_data.get("Action_Rationale", ""), max_length=600
            )
            self.ln(5)

        if stock_data.get("Investment_Thesis"):
            self.set_font("Helvetica", "B", 10)
            self.cell(0, 6, "Investment Thesis:", 0, 1)
            self.set_font("Helvetica", "", 9)
            self.safe_multi_cell(
                0, 5, stock_data.get("Investment_Thesis", ""), max_length=600
            )
            self.ln(5)

        if stock_data.get("Entry_Approach"):
            self.set_font("Helvetica", "B", 10)
            self.cell(0, 6, "Entry Strategy:", 0, 1)
            self.set_font("Helvetica", "", 9)
            entry_approach = stock_data.get("Entry_Approach", "N/A")
            target_price = stock_data.get("Target_Entry_Price", 0)
            stop_loss = stock_data.get("Stop_Loss_Price", 0)

            # Format entry info safely
            try:
                entry_info = (
                    f"Approach: {entry_approach} | "
                    f"Target Price: ${target_price:.2f} | "
                    f"Stop Loss: ${stop_loss:.2f}"
                )
            except (ValueError, TypeError):
                entry_info = f"Approach: {entry_approach}"

            self.safe_multi_cell(0, 5, entry_info, max_length=200)

            if stock_data.get("Entry_Rationale"):
                self.safe_multi_cell(
                    0,
                    5,
                    f"Rationale: {stock_data.get('Entry_Rationale', '')}",
                    max_length=500,
                )
            self.ln(5)

        # Risk Factors
        if stock_data.get("Risk Warnings") and stock_data["Risk Warnings"] != "None":
            self.ln(5)
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(200, 50, 50)  # Red
            self.cell(0, 6, "Risk Factors:", 0, 1)
            self.set_font("Helvetica", "", 9)

            risks = stock_data["Risk Warnings"].split("; ")
            for risk in risks:
                # Strip emojis for PDF compatibility
                risk_clean = self.strip_emojis(risk)
                if risk_clean:  # Only print if there's content after stripping
                    self.cell(0, 6, f"- {risk_clean}", 0, 1)

            self.set_text_color(0, 0, 0)  # Reset
        self.ln(10)

        # Sentiment & Momentum
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, "Sentiment & Momentum", 0, 1)
        self.set_font("Helvetica", "", 10)

        # News Sentiment Score
        sentiment_score = stock_data.get("News Sentiment Score")
        sentiment_label = stock_data.get("News Sentiment Label", "")
        if sentiment_score is not None and not pd.isna(sentiment_score):
            sentiment_display = f"{sentiment_score:.2f}"
            if sentiment_label:
                sentiment_display += f" ({sentiment_label})"
        else:
            sentiment_display = "N/A"
        self.cell(0, 6, f"News Sentiment Score: {sentiment_display}", 0, 1)

        # Earnings Surprise
        earnings_surprise = stock_data.get("Earnings Surprise Avg (%)")
        last_surprise = stock_data.get("Last Quarter Surprise (%)")
        if earnings_surprise is not None and not pd.isna(earnings_surprise):
            surprise_display = f"{earnings_surprise:.2f}%"
            if last_surprise is not None and not pd.isna(last_surprise):
                surprise_display += f" (Last Q: {last_surprise:.2f}%)"
        else:
            surprise_display = "N/A"
        self.cell(0, 6, f"Avg Earnings Surprise (4Q): {surprise_display}", 0, 1)


def generate_pdf_report(advice_df, portfolio_df, risk_free_rate=0.045):
    """
    Main function to generate the PDF report.
    """
    pdf = ResearchReport()

    # 1. Title Page
    pdf.title_page(risk_free_rate)

    # 2. Executive Summary
    # Filter for top picks (Buy/Strong Buy with high conviction)
    top_picks = advice_df.sort_values("Conviction", ascending=False)
    pdf.executive_summary(top_picks)

    # 3. Portfolio Allocation
    if not portfolio_df.empty:
        pdf.portfolio_allocation(portfolio_df)

    # 4. Stock Analysis Cards
    # Generate cards for top 10 holdings or top 10 picks
    # Prioritize portfolio holdings

    # Get symbols in portfolio
    portfolio_symbols = (
        portfolio_df["Symbol"].tolist() if not portfolio_df.empty else []
    )

    # Create a list of stocks to analyze (Portfolio + Top Picks up to 15 total)
    stocks_to_analyze = []

    # First, portfolio stocks
    for symbol in portfolio_symbols:
        stock_data = advice_df[advice_df["Symbol"] == symbol]
        if not stock_data.empty:
            stocks_to_analyze.append(stock_data.iloc[0])

    # Then fill with other top picks if needed
    for _, row in top_picks.iterrows():
        if row["Symbol"] not in portfolio_symbols and len(stocks_to_analyze) < 15:
            stocks_to_analyze.append(row)

    # Generate pages
    for stock in stocks_to_analyze:
        pdf.stock_analysis_card(stock)

    output_filename = "GARP_Research_Report.pdf"
    pdf.output(output_filename)
    print(f"\nPDF Report generated: {output_filename}")
    return output_filename
