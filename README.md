# India Power Stocks — Five-Year Analytics

A classroom-ready Streamlit dashboard for NTPC, Power Grid, Tata Power, Adani Power and JSW Energy.

## Features

- Five-year adjusted-price performance, CAGR, volatility, Sharpe ratio, beta and maximum drawdown
- Market-cap and risk–return peer comparisons
- Annual revenue, EBITDA, profit, cash flow and free cash flow
- Profitability, liquidity, leverage, coverage, efficiency and working-capital ratios
- Valuation multiples and an explicitly educational peer scorecard
- Excel and CSV downloads
- Responsive Mountain Path Academy design

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Upload `app.py`, `requirements.txt`, and this README to a GitHub repository. In Streamlit Community Cloud, select the repository and set the main file to `app.py`.

## Data note

The app retrieves adjusted market prices and company financial data through `yfinance`. Financial-statement field availability and reporting conventions can vary. The dashboard handles unavailable fields as missing values rather than silently substituting estimates.

This project is for education and research, not investment advice.
