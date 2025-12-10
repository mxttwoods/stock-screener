from datetime import datetime

from fpdf import FPDF


class ResearchReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.report_date = datetime.now().strftime("%B %d, %Y")

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
        self.multi_cell(
            0,
            6,
            "This report identifies high-quality Growth at a Reasonable Price (GARP) opportunities. "
            "Our screening process combines rigorous quantitative filters with AI-powered fundamental analysis. "
            "The selected stocks demonstrate strong return on capital, reasonable valuations, and positive momentum.",
        )
        self.ln(10)

        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Top High-Conviction Picks", 0, 1, "L")
        self.ln(5)

        for _, stock in top_picks.head(3).iterrows():
            self.set_font("Helvetica", "B", 12)
            self.cell(30, 8, stock["Symbol"], 0, 0)
            self.set_font("Helvetica", "", 12)
            self.cell(0, 8, f"- {stock['Name']}", 0, 1)

            self.set_font("Helvetica", "I", 10)
            self.multi_cell(0, 6, f"Thesis: {stock['AI Thesis'][:200]}...")
            self.ln(5)

    def portfolio_allocation(self, portfolio_df):
        self.add_page()
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "Portfolio Strategy", 0, 1, "L")
        self.ln(5)

        self.set_font("Helvetica", "", 11)
        self.multi_cell(
            0,
            6,
            "The following portfolio is optimized to maximize the Sharpe Ratio, balancing expected returns against volatility. "
            "Allocation is weighted towards our highest conviction ideas.",
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
            self.cell(60, 8, row["Sector"], 1, 1, "C")

        self.ln(10)

    def stock_analysis_card(self, stock_data):
        self.add_page()

        # Header Section
        self.set_font("Helvetica", "B", 20)
        self.cell(120, 10, f"{stock_data['Symbol']} - {stock_data['Name']}", 0, 0)

        # Rating Badge
        action = stock_data["Action"]
        # Strip emojis for PDF compatibility
        action_clean = (
            action.replace("🟢", "")
            .replace("🟡", "")
            .replace("🔵", "")
            .replace("⚪", "")
            .replace("🔴", "")
            .strip()
        )

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
        self.cell(
            0,
            6,
            f"Sector: {stock_data['Sector']} | Price: ${stock_data.get('Current Price', 0):.2f}",
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
                if val is None or val == "N/A":
                    return "N/A"
                return fmt.format(float(val))
            except:
                return "N/A"

        metrics = [
            (
                "Market Cap",
                f"${safe_fmt(stock_data.get('Market Cap ($B)'), '{:.1f}')}B",
            ),
            ("P/E Ratio", safe_fmt(stock_data.get("P/E"))),
            ("PEG Ratio", safe_fmt(stock_data.get("PEG"))),
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

        # Investment Thesis
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, "Investment Thesis (AI Analyst)", 0, 1)
        self.set_font("Helvetica", "", 11)
        self.multi_cell(0, 6, str(stock_data.get("AI Thesis", "No thesis generated.")))
        self.ln(10)

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
                risk_clean = risk.replace("⚠️", "").replace("❗", "").strip()
                self.cell(0, 6, f"- {risk_clean}", 0, 1)

            self.set_text_color(0, 0, 0)  # Reset
        self.ln(10)

        # Sentiment & Momentum
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 8, "Sentiment & Momentum", 0, 1)
        self.set_font("Helvetica", "", 10)

        sentiment_score = stock_data.get("News Sentiment Score", "N/A")
        earnings_surprise = stock_data.get("Earnings Surprise Avg (%)", "N/A")

        self.cell(0, 6, f"News Sentiment Score: {sentiment_score}", 0, 1)
        self.cell(0, 6, f"Avg Earnings Surprise (4Q): {earnings_surprise}%", 0, 1)


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
