# PEAD Forecasting (Earnings + Transcripts + Market Reactions)

This project builds a machine learning pipeline to predict **Post-Earnings Announcement Drift (PEAD)** using:
- **Earnings event features** (EPS surprise, reportTime, gap, volume shock, volatility)
- **Transcript features** from Alpha Vantage (segment sentiment aggregates; optional FinBERT embeddings)
- **Value/Glamour controls** (EP, BM, CP, Sales growth)

Outputs:
- **Probabilistic forecasts** for drift direction (5D/20D)
- Optional **return magnitude forecasts** (BHAR)
- A realistic **event-driven backtest** with probability-adjusted position sizing
- A minimal **real-time inference demo** for new earnings events

---

## Repo layout (high cohesion / low coupling)

- `configs/` — YAML configs (tickers, horizons, backtest rules)
- `data/`
  - `raw/` — cached Alpha Vantage JSON
  - `interim/` — per-ticker tables (daily, earnings, transcripts, fundamentals)
  - `processed/` — merged event-level dataset + sequences for TCN
- `src/pead/`
  - `ingest/` — Alpha Vantage client + caching
  - `transform/` — build atomic tables (daily/earnings/transcripts/fundamentals)
  - `datasets/` — merge tables + build targets + sequences + splits
  - `models/` — baselines (LogReg/XGB), multimodal fusion, TCN
  - `evaluation/` — metrics + calibration
  - `strategy/` — signals, sizing, backtest

---

## Setup

1) Create and activate an environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install:
```bash
pip install -e .
```

3) Add your API Key:
```bash
cp .env.example .env
# edit .env and set ALPHA_VANTAGE_API_KEY
```

## Typical workflow
1) Build interim + processed datasets
```bash
python -m pead.app.build_data --config configs/config.yaml
```
2) Train models (baselines + multimodel + TCN)
```bash
python -m pead.app.train_models --config configs/config.yaml
```
3) Run backtests and compare models
```bash
python -m pead.app.run_backtest --config configs/config.yaml
```
4) Real-time demo (single ticker)
```bash
python -m pead.app.realtime_demo --config configs/config.yaml --ticker NVDA
```

## Notes on timing/leakage
We use `reportTime`:
- pre-market -> tradeable start is same day open
- post-market -> tradeable start is the next day open

All features must be known by the tradeable start time.
Targets (BHAR_5d/20d) start from the tradeable start.

## Models
- Baseline: Logistic Regression (calibrated) and XGBoost
- Deep: Multimodal Fusion (tabular + transcript)
- Sequence: TCN on pre-event daily sequence (last 60 days)

## License
For academic use.
