from __future__ import annotations

from datetime import date, timedelta
import io
import math
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Equity Research Terminal | Quant Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

TRADING_DAYS = 252
RISK_FREE_RATE = 0.065

# Default tickers to load on first launch
DEFAULT_TICKERS = "NTPC.NS, POWERGRID.NS, TATAPOWER.NS, ADANIPOWER.NS, JSWENERGY.NS, TORNTPOWER.NS, NHPC.NS, CESC.NS"

# -----------------------------------------
# UI / THEME STYLING (BLOOMBERG / POWER BI STYLE)
# -----------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {font-family: 'Inter', sans-serif; color: #E6EDF3;}
.stApp {background: #0D1117; background-image: radial-gradient(circle at 50% 0%, #161B22 0%, #0D1117 70%);}

/* Headers */
h1, h2, h3 {color: #FFFFFF; font-weight: 700; border-bottom: 1px solid #30363D; padding-bottom: 10px; margin-top: 0px;}

/* Sidebar */
[data-testid="stSidebar"] {background: #161B22; border-right: 1px solid #30363D;}
[data-testid="stSidebar"] * {color: #C9D1D9 !important;}
[data-testid="stSidebar"] h2 {color: #58A6FF !important; border-bottom: 1px solid #30363D;}

/* Metrics */
[data-testid="stMetric"] {
    background: #161B22; border: 1px solid #30363D; padding: 15px; border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
[data-testid="stMetric"] label p {color: #8B949E !important; font-size: 0.85rem !important; font-weight: 600;}
[data-testid="stMetric"] div {color: #FFFFFF !important; font-size: 1.5rem !important; font-weight: 700; font-family: 'JetBrains Mono', monospace;}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {gap: 8px; background: transparent; border-bottom: 1px solid #30363D;}
.stTabs button[data-baseweb="tab"] {
    background: #21262D; border-radius: 6px 6px 0 0; border: 1px solid #30363D;
    color: #C9D1D9 !important; padding: 10px 20px; font-weight: 600;
}
.stTabs button[data-baseweb="tab"]:hover {background: #30363D; border-color: #58A6FF;}
.stTabs button[data-baseweb="tab"][aria-selected="true"] {
    background: #0D1117; border-bottom: 2px solid #58A6FF; color: #58A6FF !important;
    box-shadow: 0 -4px 10px rgba(88, 166, 255, 0.1);
}
.stTabs [data-baseweb="tab-highlight"] {display: none !important;}
.stTabs [data-baseweb="tab-border"] {display: none !important;}

/* Dataframes */
.stDataFrame {border: 1px solid #30363D; border-radius: 8px; overflow: hidden;}
.stDataFrame table {background: #161B22 !important;}
.stDataFrame thead th {background: #21262D !important; color: #58A6FF !important; font-weight: 700;}

/* Buttons */
.stButton button, .stDownloadButton button {
    background: #238636 !important; color: #FFFFFF !important; border: 1px solid #2EA043 !important;
    border-radius: 6px !important; font-weight: 600 !important; transition: all 0.2s !important;
}
.stButton button:hover, .stDownloadButton button:hover {
    background: #2EA043 !important; box-shadow: 0 0 10px rgba(46, 160, 67, 0.4) !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    background: #21262D !important; border: 1px solid #30363D !important; color: #58A6FF !important;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
    background: #30363D !important; border-color: #58A6FF !important; box-shadow: 0 0 8px rgba(88, 166, 255, 0.2) !important;
}

/* Inputs & Selects */
div[data-baseweb="select"] > div {background: #21262D !important; border: 1px solid #30363D !important;}
div[data-baseweb="select"] span {color: #C9D1D9 !important;}
[data-baseweb="popover"] [role="listbox"] {background: #161B22 !important; border: 1px solid #30363D !important;}
[data-baseweb="popover"] [role="option"] {color: #C9D1D9 !important;}
[data-baseweb="popover"] [role="option"]:hover, [data-baseweb="popover"] [aria-selected="true"] {background: #21262D !important;}

/* Custom Classes */
.terminal-header {font-family: 'JetBrains Mono', monospace; color: #58A6FF; letter-spacing: 1px; text-transform: uppercase; font-size: 0.9rem; margin-bottom: 5px; font-weight: 600;}
.metric_positive {color: #3FB950 !important;}
.metric_negative {color: #F85149 !important;}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------
def first_row(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.index:
            return pd.to_numeric(df.loc[name], errors="coerce")
    return pd.Series(index=df.columns, dtype=float)

def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((pd.notna(b)) & (b != 0), a / b, np.nan)

def fmt_num(v, prefix="₹", suffix=""):
    if pd.isna(v): return "—"
    av = abs(v)
    if av >= 1e12: return f"{prefix}{v/1e12:,.2f}T{suffix}"
    if av >= 1e9: return f"{prefix}{v/1e9:,.2f}B{suffix}"
    if av >= 1e7: return f"{prefix}{v/1e7:,.2f} Cr{suffix}"
    return f"{prefix}{v:,.2f}{suffix}"

def pct(v):
    return "—" if pd.isna(v) else f"{v:.2%}"

@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(ticker_map: tuple[tuple[str, str], ...], start: str, end: str):
    tickers = [t[1] for t in ticker_map]
    names = [t[0] for t in ticker_map]
    
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column", threads=True)
    
    if len(tickers) > 1:
        close = raw["Close"]
    else:
        close = raw[["Close"]]
        close.columns = tickers
        
    if isinstance(close, pd.Series): 
        close = close.to_frame(tickers[0])
        
    rename_map = {t: n for t, n in zip(tickers, names)}
    close = close.rename(columns=rename_map)
    
    return close.dropna(how="all")

@st.cache_data(ttl=21600, show_spinner=False)
def load_company(ticker: str):
    t = yf.Ticker(ticker)
    try: info = t.info or {}
    except Exception: info = {}
    try: income = t.financials.copy()
    except Exception: income = pd.DataFrame()
    try: balance = t.balance_sheet.copy()
    except Exception: balance = pd.DataFrame()
    try: cashflow = t.cashflow.copy()
    except Exception: cashflow = pd.DataFrame()
    return info, income, balance, cashflow

def annual_financials(income, balance, cashflow):
    cols = sorted(set(income.columns) | set(balance.columns) | set(cashflow.columns))
    out = pd.DataFrame(index=cols)
    out["Revenue"] = first_row(income, ["Total Revenue", "Operating Revenue"])
    out["EBITDA"] = first_row(income, ["EBITDA", "Normalized EBITDA"])
    out["EBIT"] = first_row(income, ["EBIT", "Operating Income"])
    out["Net Income"] = first_row(income, ["Net Income", "Net Income Common Stockholders"])
    out["Interest Expense"] = abs(first_row(income, ["Interest Expense", "Interest Expense Non Operating"]))
    out["Total Assets"] = first_row(balance, ["Total Assets"])
    out["Equity"] = first_row(balance, ["Stockholders Equity", "Total Equity Gross Minority Interest"])
    out["Debt"] = first_row(balance, ["Total Debt"])
    out["Current Assets"] = first_row(balance, ["Current Assets", "Total Current Assets"])
    out["Current Liabilities"] = first_row(balance, ["Current Liabilities", "Total Current Liabilities"])
    out["Cash"] = first_row(balance, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"])
    out["Inventory"] = first_row(balance, ["Inventory"])
    out["Receivables"] = first_row(balance, ["Accounts Receivable", "Receivables"])
    out["Payables"] = first_row(balance, ["Payables And Accrued Expenses", "Accounts Payable", "Payables"])
    out["Operating Cash Flow"] = first_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    out["Capital Expenditure"] = abs(first_row(cashflow, ["Capital Expenditure", "Capital Expenditures"]))
    out["Free Cash Flow"] = out["Operating Cash Flow"] - out["Capital Expenditure"]
    
    avg_assets = out["Total Assets"].rolling(2).mean()
    avg_equity = out["Equity"].rolling(2).mean()
    
    # Basic Ratios
    out["Net Margin"] = safe_div(out["Net Income"], out["Revenue"])
    out["EBITDA Margin"] = safe_div(out["EBITDA"], out["Revenue"])
    out["ROA"] = safe_div(out["Net Income"], avg_assets)
    out["ROE"] = safe_div(out["Net Income"], avg_equity)
    out["Current Ratio"] = safe_div(out["Current Assets"], out["Current Liabilities"])
    out["Quick Ratio"] = safe_div(out["Current Assets"] - out["Inventory"], out["Current Liabilities"])
    out["Debt / Equity"] = safe_div(out["Debt"], out["Equity"])
    out["Interest Coverage"] = safe_div(out["EBIT"], out["Interest Expense"])
    out["Asset Turnover"] = safe_div(out["Revenue"], avg_assets)
    
    # Advanced Feature Engineering
    out["ROCE"] = safe_div(out["EBIT"], avg_assets - out["Current Liabilities"])
    out["Debt / EBITDA"] = safe_div(out["Debt"], out["EBITDA"])
    out["FCF Margin"] = safe_div(out["Free Cash Flow"], out["Revenue"])
    out["Revenue Growth (YoY)"] = out["Revenue"].pct_change()
    out["EBITDA Growth (YoY)"] = out["EBITDA"].pct_change()
    out["PAT Growth (YoY)"] = out["Net Income"].pct_change()
    
    out["Receivable Days"] = safe_div(out["Receivables"], out["Revenue"]) * 365
    out["Inventory Days"] = safe_div(out["Inventory"], out["Revenue"]) * 365
    out["Payable Days"] = safe_div(out["Payables"], out["Revenue"]) * 365
    out["Cash Conversion Cycle"] = out["Receivable Days"] + out["Inventory Days"] - out["Payable Days"]
    
    out.index = pd.to_datetime(out.index).year
    return out.sort_index().tail(5)

def price_metrics(prices: pd.DataFrame, benchmark: pd.Series | None = None):
    ret = prices.pct_change().dropna(how="all")
    rows = []
    for c in prices:
        s, r = prices[c].dropna(), ret[c].dropna()
        if len(s) < 2: continue
        years = max((s.index[-1] - s.index[0]).days / 365.25, 1/365)
        wealth = (1 + r).cumprod()
        dd = wealth / wealth.cummax() - 1
        beta = np.nan
        if benchmark is not None:
            joined = pd.concat([r, benchmark.pct_change()], axis=1).dropna()
            if len(joined) > 2 and joined.iloc[:,1].var() != 0:
                beta = joined.cov().iloc[0,1] / joined.iloc[:,1].var()
        rows.append({"Company":c, "Last Price":s.iloc[-1], "5Y CAGR":(s.iloc[-1]/s.iloc[0])**(1/years)-1,
                     "Annual Return":r.mean()*TRADING_DAYS, "Volatility":r.std()*math.sqrt(TRADING_DAYS),
                     "Sharpe Ratio":(r.mean()*TRADING_DAYS-RISK_FREE_RATE)/(r.std()*math.sqrt(TRADING_DAYS)) if r.std() else np.nan,
                     "Max Drawdown":dd.min(), "Beta":beta})
    return pd.DataFrame(rows).set_index("Company"), ret


# -----------------------------------------
# SIDEBAR & DYNAMIC INPUT
# -----------------------------------------
with st.sidebar:
    st.markdown("## ⚡ Terminal Controls")
    
    tickers_str = st.text_area("Enter Yahoo Finance Tickers (comma-separated)", DEFAULT_TICKERS)
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    
    # Create dictionaries dynamically
    STOCKS = {t.split('.')[0]: t for t in tickers}
    COLORS = {name: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, name in enumerate(STOCKS.keys())}

    selected = st.multiselect("Universe (Active Stocks)", list(STOCKS.keys()), default=list(STOCKS.keys()))
    primary = st.selectbox("Primary Ticker (Deep Dive)", selected if selected else list(STOCKS.keys()))
    
    base = st.number_input("Investment Simulation (₹)", 10_000, 10_000_000, 100_000, 10_000)
    normalize = st.toggle("Normalize Price Chart", value=True)
    st.markdown("---")
    
    if st.button("↻ Refresh Market Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

if not selected:
    st.warning("Select at least one company from the sidebar to initialize terminal."); st.stop()

end = date.today() + timedelta(days=1)
start = end - timedelta(days=365*5 + 10)

st.markdown(f'<div class="terminal-header">Equity Research Terminal | {start.year}–{end.year} | {len(selected)} Active Tickers</div>', unsafe_allow_html=True)
st.markdown("### Quantitative Research & Portfolio Analytics")

try:
    prices = load_prices(tuple((n, STOCKS[n]) for n in selected), start.isoformat(), end.isoformat())
except Exception as e:
    st.error(f"Market data feed error. Details: {e}"); st.stop()
if prices.empty:
    st.error("No market-price data returned. Check tickers."); st.stop()

try:
    nifty = yf.download("^NSEI", start=start.isoformat(), end=end.isoformat(), auto_adjust=True, progress=False)["Close"]
    if isinstance(nifty, pd.DataFrame): nifty = nifty.iloc[:,0]
except Exception: nifty = None

pm, returns = price_metrics(prices, nifty)
infos, financials = {}, {}
with st.spinner("Loading reported financials & engineering features…"):
    for name in selected:
        info, inc, bal, cf = load_company(STOCKS[name])
        infos[name] = info
        financials[name] = annual_financials(inc, bal, cf)

latest = pm.loc[primary]
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric(f"{primary} Last Price", fmt_num(latest["Last Price"], "₹"))
c2.metric("5Y CAGR", pct(latest["5Y CAGR"]))
c3.metric("Annual Volatility", pct(latest["Volatility"]))
c4.metric("Sharpe Ratio", "—" if pd.isna(latest["Sharpe Ratio"]) else f"{latest['Sharpe Ratio']:.2f}")
c5.metric("Maximum Drawdown", pct(latest["Max Drawdown"]))

tabs = st.tabs(["Overview", "Price Action", "Financials", "Ratio Analysis", "Scoring Engine", "🧠 ML Factor Discovery", "📊 Portfolio Optimizer", "Data Export"])

# -----------------------------------------
# TAB 0: OVERVIEW
# -----------------------------------------
with tabs[0]:
    st.markdown("#### Peer Snapshot & Market Cap")
    peer=[]
    for n in selected:
        i=infos[n]; f=financials[n]; last=f.iloc[-1] if not f.empty else pd.Series(dtype=float)
        peer.append({"Company":n,"Market Cap (₹ Cr)":i.get("marketCap",np.nan)/1e7,"Price (₹)":pm.loc[n,"Last Price"],
                     "5Y CAGR":pm.loc[n,"5Y CAGR"],"ROE":i.get("returnOnEquity",last.get("ROE",np.nan)),
                     "Net Margin":i.get("profitMargins",last.get("Net Margin",np.nan)),"Debt/Equity":last.get("Debt / Equity",np.nan),
                     "P/E":i.get("trailingPE",np.nan)})
    peer_df=pd.DataFrame(peer).set_index("Company")
    st.dataframe(peer_df.style.format({"Market Cap (₹ Cr)":"{:,.0f}", "Price (₹)":"{:,.2f}", "5Y CAGR":"{:.2%}", "ROE":"{:.2%}", "Net Margin":"{:.2%}", "Debt/Equity":"{:.2f}", "P/E":"{:.2f}"}), use_container_width=True)
    
    a,b=st.columns(2)
    with a:
        fig=px.bar(peer_df.reset_index(),x="Company",y="Market Cap (₹ Cr)",color="Company",color_discrete_map=COLORS,title="Market Capitalisation (₹ Cr)")
        fig.update_layout(plot_bgcolor='#0D1117', paper_bgcolor='#0D1117', font=dict(color='#E6EDF3'))
        fig.update_layout(showlegend=False); st.plotly_chart(fig,use_container_width=True)
    with b:
        risk=pm.reset_index()
        fig=px.scatter(risk,x="Volatility",y="Annual Return",size=[max(infos[n].get("marketCap",1),1) for n in risk.Company],text="Company",color="Company",color_discrete_map=COLORS,title="Risk–Return Map")
        fig.update_traces(textposition="top center"); fig.update_xaxes(tickformat=".0%", gridcolor='#30363D');fig.update_yaxes(tickformat=".0%", gridcolor='#30363D')
        fig.update_layout(plot_bgcolor='#0D1117', paper_bgcolor='#0D1117', font=dict(color='#E6EDF3'))
        st.plotly_chart(fig,use_container_width=True)

# -----------------------------------------
# TAB 1: PRICE ACTION
# -----------------------------------------
with tabs[1]:
    plot=(prices/prices.ffill().iloc[0]*100) if normalize else prices
    label="Growth of ₹100" if normalize else "Adjusted share price (₹)"
    fig=px.line(plot,title=f"Five-Year Price Trend — {label}",color_discrete_map=COLORS)
    fig.update_layout(yaxis_title=label,xaxis_title="",legend_title="Ticker",hovermode="x unified", plot_bgcolor='#0D1117', paper_bgcolor='#0D1117', font=dict(color='#E6EDF3'))
    fig.update_xaxes(gridcolor='#30363D'); fig.update_yaxes(gridcolor='#30363D')
    st.plotly_chart(fig,use_container_width=True)
    
    invested=(prices/prices.ffill().iloc[0])*base
    fig2=px.line(invested,title=f"Growth of ₹{base:,.0f} Invested",color_discrete_map=COLORS)
    fig2.update_layout(plot_bgcolor='#0D1117', paper_bgcolor='#0D1117', font=dict(color='#E6EDF3'))
    fig2.update_xaxes(gridcolor='#30363D'); fig2.update_yaxes(gridcolor='#30363D')
    st.plotly_chart(fig2,use_container_width=True)
    
    show=pm.copy(); st.dataframe(show.style.format({"Last Price":"₹{:,.2f}","5Y CAGR":"{:.2%}","Annual Return":"{:.2%}","Volatility":"{:.2%}","Sharpe Ratio":"{:.2f}","Max Drawdown":"{:.2%}","Beta":"{:.2f}"}),use_container_width=True)
    
    corr=returns.corr(); fig=px.imshow(corr,text_auto=".2f",zmin=-1,zmax=1,color_continuous_scale="Tealgrn",title="Daily Return Correlation Matrix")
    fig.update_layout(plot_bgcolor='#0D1117', paper_bgcolor='#0D1117', font=dict(color='#E6EDF3'))
    st.plotly_chart(fig,use_container_width=True)

# -----------------------------------------
# TAB 2: FINANCIALS
# -----------------------------------------
with tabs[2]:
    f=financials[primary]
    if f.empty: st.warning("Annual financial statements unavailable for this ticker.")
    else:
        units=1e7
        long=f[["Revenue","EBITDA","Net Income","Operating Cash Flow","Free Cash Flow"]].div(units).reset_index(names="Year").melt("Year",var_name="Metric",value_name="₹ crore")
        fig=px.bar(long,x="Year",y="₹ crore",color="Metric",barmode="group",title=f"{primary}: Revenue, Profit and Cash Flow (₹ Cr)")
        fig.update_layout(plot_bgcolor='#0D1117', paper_bgcolor='#0D1117', font=dict(color='#E6EDF3'))
        fig.update_xaxes(gridcolor='#30363D'); fig.update_yaxes(gridcolor='#30363D')
        st.plotly_chart(fig,use_container_width=True)
        
        margins=f[["EBITDA Margin","Net Margin","ROA","ROE","ROCE"]].reset_index(names="Year").melt("Year",var_name="Metric",value_name="Ratio")
        fig=px.line(margins,x="Year",y="Ratio",color="Metric",markers=True,title="Profitability and Return Ratios")
        fig.update_yaxes(tickformat=".1%", gridcolor='#30363D')
        fig.update_layout(plot_bgcolor='#0D1117', paper_bgcolor='#0D1117', font=dict(color='#E6EDF3'))
        st.plotly_chart(fig,use_container_width=True)

# -----------------------------------------
# TAB 3: RATIO ANALYSIS
# -----------------------------------------
with tabs[3]:
    metric=st.selectbox("Select Ratio for Peer Comparison",["Net Margin","EBITDA Margin","ROA","ROE","ROCE","Current Ratio","Quick Ratio","Debt / Equity","Debt / EBITDA","Interest Coverage","Asset Turnover","Revenue Growth (YoY)"])
    comp=pd.DataFrame({n:financials[n][metric] for n in selected if metric in financials[n]}).sort_index()
    fig=px.line(comp,markers=True,title=f"Peer Comparison — {metric}",color_discrete_map=COLORS)
    if metric in ["Net Margin","EBITDA Margin","ROA","ROE","ROCE","Revenue Growth (YoY)"]: fig.update_yaxes(tickformat=".1%")
    fig.update_layout(plot_bgcolor='#0D1117', paper_bgcolor='#0D1117', font=dict(color='#E6EDF3'))
    fig.update_xaxes(gridcolor='#30363D'); fig.update_yaxes(gridcolor='#30363D')
    st.plotly_chart(fig,use_container_width=True)

# -----------------------------------------
# TAB 4: SCORING ENGINE (INTERACTIVE WEIGHTS)
# -----------------------------------------
with tabs[4]:
    st.markdown("#### Interactive Multi-Factor Scoring")
    st.markdown("Adjust the weights below to recalculate the composite score. Default is equal weight.")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: w_ret = st.slider("5Y CAGR", 0.0, 1.0, 0.2)
    with col2: w_risk = st.slider("Volatility (Risk)", 0.0, 1.0, 0.2)
    with col3: w_roe = st.slider("ROE", 0.0, 1.0, 0.2)
    with col4: w_debt = st.slider("Debt/Equity (Leverage)", 0.0, 1.0, 0.2)
    with col5: w_val = st.slider("P/E (Valuation)", 0.0, 1.0, 0.2)
    
    total_w = w_ret + w_risk + w_roe + w_debt + w_val
    if total_w == 0: total_w = 1 # Prevent division by zero
        
    val=[]
    for n in selected:
        i=infos[n]
        val.append({"Company":n,"P/E":i.get("trailingPE",np.nan),"Market Cap (₹ Cr)":i.get("marketCap",np.nan)/1e7})
    val_df=pd.DataFrame(val).set_index("Company")
    
    score=pd.DataFrame(index=selected)
    score["Return score"]=pm["5Y CAGR"].rank(pct=True)
    score["Risk score"]=(-pm["Volatility"]).rank(pct=True)
    score["ROE score"]=peer_df["ROE"].rank(pct=True)
    score["Leverage score"]=(-peer_df["Debt/Equity"]).rank(pct=True)
    score["Valuation score"]=(-val_df["P/E"]).rank(pct=True)
    
    score["Composite / 100"] = (
        (score["Return score"] * w_ret) +
        (score["Risk score"] * w_risk) +
        (score["ROE score"] * w_roe) +
        (score["Leverage score"] * w_debt) +
        (score["Valuation score"] * w_val)
    ) / total_w * 100
    
    score=score.sort_values("Composite / 100",ascending=False)
    fig=px.bar(score.reset_index(names="Company"),x="Company",y="Composite / 100",color="Company",color_discrete_map=COLORS,title="Weighted Peer Scorecard")
    fig.update_layout(plot_bgcolor='#0D1117', paper_bgcolor='#0D1117', font=dict(color='#E6EDF3'), showlegend=False)
    fig.update_xaxes(gridcolor='#30363D'); fig.update_yaxes(gridcolor='#30363D')
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(score.style.format("{:.1f}"),use_container_width=True)

# -----------------------------------------
# TAB 5: ML FACTOR DISCOVERY
# -----------------------------------------
with tabs[5]:
    st.markdown("#### 🧠 ML Factor Discovery & Weight Optimizer")
    st.markdown("This engine aggregates historical financials, calculates 1-Year Forward Returns, and uses a Random Forest model to identify which financial metrics actually drive stock performance.")
    
    @st.cache_data(ttl=3600, show_spinner=True)
    def build_ml_dataset(selected_stocks, start, end):
        records = []
        for name in selected_stocks:
            f = financials[name]
            if f.empty: continue
            for year in f.index:
                try:
                    p0_mask = prices.index <= pd.Timestamp(year=year, month=12, day=31)
                    p1_mask = prices.index <= pd.Timestamp(year=year+1, month=12, day=31)
                    if p0_mask.sum() > 0 and p1_mask.sum() > 0:
                        p0 = prices.loc[p0_mask, name].iloc[-1]
                        p1 = prices.loc[p1_mask, name].iloc[-1]
                        fwd_ret = (p1/p0) - 1
                        
                        row = f.loc[year].to_dict()
                        row["Company"] = name
                        row["Year"] = year
                        row["Forward_Return"] = fwd_ret
                        records.append(row)
                except:
                    continue
        return pd.DataFrame(records)

    if len(selected) < 4:
        st.warning("Please select at least 4-5 companies to run the ML Factor Discovery model effectively.")
    else:
        ml_df = build_ml_dataset(tuple(selected), start.isoformat(), end.isoformat())
        if ml_df.empty or len(ml_df) < 10:
            st.warning("Not enough historical overlap between financials and prices to train the model. Try selecting more companies.")
        else:
            st.success(f"Dataset built: {len(ml_df)} company-year observations.")
            
            features = ["ROE", "ROCE", "Debt / Equity", "Debt / EBITDA", "Net Margin", "FCF Margin", "Revenue Growth (YoY)", "EBITDA Growth (YoY)", "Interest Coverage", "Current Ratio"]
            ml_df = ml_df.dropna(subset=["Forward_Return"])
            
            X = ml_df[features].fillna(0)
            y = ml_df["Forward_Return"]
            
            if st.button("Train Random Forest & Extract Weights"):
                with st.spinner("Training ML Model..."):
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf.fit(X, y)
                    
                    importances = rf.feature_importances_
                    imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
                    imp_df = imp_df.sort_values("Importance", ascending=False)
                    
                    st.subheader("Machine-Learned Feature Importances")
                    st.markdown("These are the factors that historically drove forward 1-year returns in your selected pool.")
                    fig = px.bar(imp_df, x="Importance", y="Feature", orientation='h', title="What Actually Drives Returns?")
                    fig.update_layout(plot_bgcolor='#0D1117', paper_bgcolor='#0D1117', font=dict(color='#E6EDF3'))
                    fig.update_xaxes(gridcolor='#30363D'); fig.update_yaxes(gridcolor='#30363D')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("To apply these to the Ranking tab, map the top 5 features to the sliders manually based on these proportions.")

# -----------------------------------------
# TAB 6: PORTFOLIO OPTIMIZER
# -----------------------------------------
with tabs[6]:
    st.markdown("#### 📊 Mean-Variance Portfolio Optimizer")
    st.markdown("Uses the Sharpe Ratio maximization (Markowitz) on the top-ranked stocks to suggest optimal capital allocation.")
    
    if 'score' not in locals():
        st.warning("Please visit the 'Scoring Engine' tab first to generate scores.")
    else:
        max_n = min(10, len(selected))
        if max_n < 2:
            st.warning("Need at least 2 stocks to optimize.")
        else:
            top_n = st.slider("Select number of top-ranked stocks to optimize", 2, max_n, min(5, max_n))
            top_stocks = score.head(top_n).index.tolist()
            
            rets = returns[top_stocks].dropna()
            mean_returns = rets.mean() * TRADING_DAYS
            cov_matrix = rets.cov() * TRADING_DAYS
            
            def negative_sharpe(weights):
                ret = np.dot(weights, mean_returns)
                vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return -(ret - RISK_FREE_RATE) / vol
            
            num_assets = len(top_stocks)
            constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(num_assets))
            initial_guess = np.array([1/num_assets] * num_assets)
            
            opt_result = minimize(negative_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
            opt_weights = opt_result.x
            
            weights_df = pd.DataFrame({"Stock": top_stocks, "Optimal Weight %": opt_weights * 100})
            weights_df = weights_df.sort_values("Optimal Weight %", ascending=False)
            
            st.subheader(f"Optimal Allocation for Top {top_n} Stocks (Max Sharpe)")
            fig = px.pie(weights_df, values='Optimal Weight %', names='Stock', color_discrete_map=COLORS, title="Portfolio Allocation")
            fig.update_layout(plot_bgcolor='#0D1117', paper_bgcolor='#0D1117', font=dict(color='#E6EDF3'))
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(weights_df.style.format({"Optimal Weight %": "{:.2f}%"}), use_container_width=True)

# -----------------------------------------
# TAB 7: DATA EXPORT
# -----------------------------------------
with tabs[7]:
    st.markdown("#### Raw Statements & Data Export")
    f=financials[primary]
    statement=st.radio("View",["Income & cash flow","Balance sheet","Calculated ratios"],horizontal=True)
    mapping={"Income & cash flow":["Revenue","EBITDA","EBIT","Net Income","Interest Expense","Operating Cash Flow","Capital Expenditure","Free Cash Flow"],
             "Balance sheet":["Total Assets","Equity","Debt","Current Assets","Current Liabilities","Cash","Inventory","Receivables","Payables"],
             "Calculated ratios":["Net Margin","EBITDA Margin","ROA","ROE","ROCE","Current Ratio","Quick Ratio","Debt / Equity","Debt / EBITDA","Interest Coverage","Asset Turnover","Revenue Growth (YoY)"]}
    st.dataframe(f[mapping[statement]].T,use_container_width=True)
    
    buffer=io.BytesIO()
    with pd.ExcelWriter(buffer,engine="xlsxwriter") as writer:
        prices.to_excel(writer,sheet_name="5Y Prices")
        pm.to_excel(writer,sheet_name="Market Metrics")
        peer_df.to_excel(writer,sheet_name="Peer Snapshot")
        val_df.to_excel(writer,sheet_name="Valuation")
        score.to_excel(writer,sheet_name="Ranking")
        for n in selected: financials[n].to_excel(writer,sheet_name=n[:31])
    st.download_button("⬇ Download Complete Analysis (Excel)",buffer.getvalue(),"Equity_Research_Terminal_Export.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)