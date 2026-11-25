# Ranked Asset Allocation – Streamlit app

App for the 2270 Financial Modeling group assignment. The user changes the assumptions of the strategy
(ETFs held, lookbacks, cash filter, transaction cost, benchmarks, years shown) and sees the performance
against the benchmark. The default settings reproduce the notebook's base case.

- App: https://jose-maria-martins-financialmodelling-app-bsmmmx.streamlit.app/
- `app.py`: the page
- `strategy.py`: configuration and calculations, copied unchanged from the notebook
- `requirements.txt`: dependencies

## Run locally

Python 3.11 or later:

```
pip install -r requirements.txt
streamlit run app.py
```

Prices come from Yahoo Finance (`yfinance`) and the T-bill rate from FRED, so an internet connection is needed.
