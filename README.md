# ⚡ PowerQuant — Institutional-Grade Sectoral Equity Analysis Platform

> A quantitative research platform that fuses fundamental analysis, factor-based ML, and statistical rigor to discover *what actually creates value* within a sector — not just *which stock went up*.

---

## 📌 Overview

**PowerQuant** is a Streamlit-based multi-tab equity research workspace built for the Indian power sector (extensible to any sector). It moves beyond vanilla dashboards by integrating:

- Dynamic ticker-level financial engineering
- Peer benchmarking & DuPont decomposition
- A machine-learned factor discovery engine (Lasso + Random Forest + SHAP)
- A dynamic scoring engine that auto-applies ML-derived weights
- Statistical diagnostics (VIF, multicollinearity, collinearity screening)

The core philosophy: **separate the Data Pipeline from the Presentation Layer**, so the app reads from a pre-built `power_sector_data.parquet` instead of hammering Yahoo Finance live and rate-limiting.

---

## 🧱 Architecture

```
┌──────────────────────────┐      ┌─────────────────────────────┐
│  data_builder.py (offline)│ ───▶ │  power_sector_data.parquet  │
│  • yfinance 15Y download  │      │  • Cached fundamentals      │
│  • Feature engineering    │      │  • Cached prices             │
│  • Quarterly → Annual     │      │  • Pre-computed ratios       │
└──────────────────────────┘      └────────────┬────────────────┘
                                                │
                                                ▼
                                  ┌─────────────────────────────┐
                                  │   Streamlit App (live)       │
                                  │   • Reads parquet instantly  │
                                  │   • Renders 6 tabs           │
                                  │   • ML + SHAP on demand      │
                                  └─────────────────────────────┘
```

This separation is what moves the project from a *teaching tool* to a *stable institutional-grade platform*.

---

## ✨ Features

### 📊 Tab 1 — Company Snapshot
Dynamic ticker input → instant overview of market cap, P/E, sector, beta, and key ratios pulled from cached info dicts.

### 📈 Tab 2 — Price & Returns Analysis
- 5Y / 3Y / 1Y CAGR computation
- Rolling volatility & Sharpe-style ratios
- Normalized price chart vs sector peers
- Drawdown visualization

### 🏭 Tab 3 — Peer Benchmarking
- Side-by-side fundamental heatmap (ROE, ROCE, Net Margin, FCF Margin, D/E, Interest Coverage)
- DuPont decomposition (Net Margin × Asset Turnover × Leverage)
- Growth-tier classification (High / Stable / Declining)

### 🎯 Tab 4 — Dynamic Scoring Engine
- 5-pillar composite scorecard: **Growth · Risk · Quality (ROE) · Leverage · Valuation**
- Manual slider weights *or* **🤖 Auto-Apply Machine-Learned Weights** toggle
- When ML toggle is on, weights are read from `st.session_state['ml_weights']` (SHAP-derived) and instantly re-rank the universe
- Percentile-rank based scoring (cross-sectional, robust to outliers)

### 🧠 Tab 5 — ML Factor Discovery (the crown jewel)
A full statistical-to-predictive pipeline:

| Stage | Technique | Purpose |
|-------|-----------|---------|
| 1 | **VIF (Variance Inflation Factor)** | Drop redundant, collinear features (VIF > 5 flag) |
| 2 | **LassoCV Regression** | Identify *linear* drivers of forward returns |
| 3 | **Random Forest (depth-limited)** | Capture *non-linear* relationships |
| 4 | **SHAP TreeExplainer** | Directional impact — red = high value, blue = low |
| 5 | **Auto-export weights** | SHAP-derived importance saved to session state for Tab 4 |

The SHAP summary plot explicitly answers: *does high debt push returns down? Does high ROCE push returns up?*

### 📚 Tab 6 — Sector Narrative
Contextual notes on regulation, capacity mix, and tailwinds/headwinds for the power sector.

---

## 🔬 Statistical Rigor — What Goes Beyond ML

Pure predictive accuracy is fragile. PowerQuant augments ML with **causal and statistical scaffolding** so factors survive out-of-sample:

- ✅ **Correlation matrices** — initial screening for redundant metrics
- ✅ **VIF** — multicollinearity removal before model fitting
- ✅ **LassoCV** — sparse linear feature selection
- ✅ **Random Forest** — non-linear pattern extraction
- ✅ **SHAP** — local + global explainability, not a black box

### 🚧 Currently Being Worked On (Roadmap Pointers)

- 🔲 **Ridge Regression** alongside Lasso — for stability when features are correlated but signal-bearing
- 🔲 **Gradient Boosting (XGBoost / LightGBM)** — second non-linear model for ensemble confirmation
- 🔲 **Stability testing** — re-run SHAP across rolling 5Y windows; do the *same* features stay important?
- 🔲 **Walk-forward validation** — train on Y1–Y5, predict Y6; roll forward. Eliminates look-ahead bias
- 🔲 **Purged & embargoed cross-validation** (Marcos López de Prado style) — for overlapping return observations
- 🔲 **Causal inference layer** — DoWhy / double-ML to separate *correlation* from *causation* (e.g., does lowering D/E *cause* rerating, or is it a side-effect of improving ROCE?)
- 🔲 **Feature drift monitoring** — track distribution shift in fundamentals across years
- 🔲 **Fama-MacBeth regression** — cross-sectional regressions per year, then average t-stats
- 🔲 **Information Coefficient (IC)** — Spearman rank correlation between factor scores and realized forward returns
- 🔲 **Factor turnover & decay analysis** — how long does a signal stay alive?
- 🔲 **Parquet → DuckDB migration** — for SQL-style querying of the cached fundamentals
- 🔲 **Portfolio backtester** — long-top-quintile / short-bottom-quintile simulation with transaction costs

This combination gives confidence that the discovered factors are **structural**, not artifacts of one sample period.

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| UI | Streamlit |
| Data | yfinance, pandas, parquet |
| ML | scikit-learn (RandomForest, LassoCV, StandardScaler) |
| Explainability | SHAP |
| Statistics | statsmodels (VIF) |
| Visualization | Plotly, Matplotlib |
| Storage | parquet (offline-built) |

---

## 🚀 Installation

```bash
git clone <your-repo>
cd powerquant
pip install -r requirements.txt
pip install shap statsmodels   # for ML tab
```

### Build the data cache (one-time, offline)
```bash
python data_builder.py
# → generates power_sector_data.parquet
```

### Launch the app
```bash
streamlit run app.py
```

---

## 🧭 Usage Flow

1. **Build data** → run `data_builder.py` once (or nightly via cron)
2. **Open app** → select 4+ companies in the sidebar
3. **Explore Tabs 1–3** → fundamental & price diagnostics
4. **Tab 5 (ML)** → click *Run Full ML Pipeline & SHAP Analysis*
   - Inspect VIF, Lasso coefficients, RF importance, SHAP summary
5. **Tab 4 (Scoring)** → toggle 🤖 *Auto-Apply Machine-Learned Weights*
   - Universe re-ranks instantly using SHAP-derived weights
6. **Tab 6** → read the sector narrative for qualitative context

---

## 📁 Project Structure

```
powerquant/
├── app.py                    # Streamlit entry point
├── data_builder.py           # Offline parquet builder
├── power_sector_data.parquet # Cached fundamentals + prices
├── requirements.txt
├── README.md
└── tabs/
    ├── tab1_snapshot.py
    ├── tab2_prices.py
    ├── tab3_peers.py
    ├── tab4_scoring.py
    ├── tab5_ml.py
    └── tab6_narrative.py
```

---

## ⚠️ Disclaimer

This platform is a **research and education tool**, not investment advice. Factors discovered via ML on historical Indian power-sector data may not generalize to future regimes, other sectors, or other geographies. Always validate out-of-sample before any capital deployment.

---

## 🎯 Design Principles

1. **Explainability over accuracy** — a transparent 60% R² beats a black-box 80%
2. **Statistical hygiene first** — VIF before modeling, walk-forward before trusting
3. **Separation of concerns** — data pipeline ≠ presentation layer
4. **Reproducibility** — parquet caching + fixed seeds in every ML run
5. **Causal skepticism** — "is this real, or is this noise?" is the default question

---

> Built to answer one question rigorously: **Within the power sector, which fundamental forces actually create shareholder value — and do they hold over time?**