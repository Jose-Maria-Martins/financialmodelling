"""Ranked Asset Allocation: configuration and calculations.

Copied unchanged from the project notebook, so the app and the notebook give the same numbers.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf

# ---- Config (same values as the notebook) ----
TICKERS   = ["SPY", "EFA", "EEM", "IEF", "LQD", "TIP", "GLD", "DBC"]
START     = "2006-01-01"
END       = "2025-10-31"           # last month-end included
LOOKBACK  = (3, 6)                 # signal = average of the 3- and 6-month returns
TOP_N     = 3                      # ETFs held each month
ABS_FILTER = True                  # a selected ETF with a negative score goes to cash
COST_BPS  = 10                     # base-case cost, bps per unit of turnover
RF_SERIES = "TB3MS"                # FRED 3-month T-bill, % per year
WEIGHTS_6040 = {"SPY": 0.6, "IEF": 0.4}
PERIODS = 12                      # monthly data


def load_prices(tickers, start, end, lookback=LOOKBACK):
    """Month-end adjusted prices and a data-quality table (one row per ticker)."""
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    daily = yf.download(tickers, start=start, end=end_excl,
                        auto_adjust=True, progress=False)["Close"][tickers]
    failed = [t for t in tickers if daily[t].isna().all()]
    if failed:
        raise RuntimeError(f"Yahoo Finance returned no prices for {failed} (often a temporary "
                           "rate limit). Wait a few minutes and run all cells again.")
    quality = pd.DataFrame({
        "first valid date": daily.apply(lambda s: s.first_valid_index().date()),
        "last date":        daily.apply(lambda s: s.last_valid_index().date()),
        "% missing after first date": daily.apply(
            lambda s: s.loc[s.first_valid_index():].isna().mean() * 100).round(2),
    })
    monthly = daily.resample("ME").last()
    # A ticker may start up to `lookback` months after the first month-end; any later
    # and its signal would arrive too late, shrinking the evaluation window.
    deadline = monthly.index[0] + pd.offsets.MonthEnd(max(np.atleast_1d(lookback)))
    late = [t for t in tickers if monthly[t].first_valid_index() > deadline]
    if late:
        raise ValueError(f"No price by {deadline.date()} for {late}: no valid signal by the "
                         "evaluation start. Move START later or change the universe.")
    return monthly, quality


def load_rf(start, end, series=RF_SERIES):
    """Monthly risk-free return on month-end dates; rf[t] applies to returns in month t."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    raw = pd.read_csv(url, index_col=0, parse_dates=True)[series]
    raw = pd.to_numeric(raw, errors="coerce")                    # any '.' placeholders become NaN
    monthly_rate = raw / 100 / 12                                 # annual % -> monthly decimal
    monthly_rate.index = monthly_rate.index + pd.offsets.MonthEnd(0)   # 2006-01-01 -> 2006-01-31
    return monthly_rate.shift(1).loc[start:end].rename("rf")


def momentum_signal(prices, lookback):
    """Average trailing total return over the lookback(s), in months, at each month-end.
    lookback can be one number (e.g. 3) or several (e.g. (3, 6))."""
    lookbacks = np.atleast_1d(lookback)
    return sum(prices.pct_change(int(k), fill_method=None) for k in lookbacks) / len(lookbacks)


def evaluation_start(prices, lookback=LOOKBACK):
    """First month-end at which every ticker has a valid signal (first decision date)."""
    full = momentum_signal(prices, lookback).dropna()
    if full.empty:
        raise ValueError("No month-end where every ticker has a valid signal.")
    return full.index[0]


def target_weights(signal, top_n, abs_filter=False):
    """Weights chosen at each month-end. Adds a 'CASH' column only when abs_filter is on."""
    sig = signal.dropna()                                     # months where every ETF has a score
    chosen = sig.rank(axis=1, ascending=False, method="first") <= top_n
    if abs_filter:
        chosen = chosen & (sig > 0)
    weights = chosen.astype(float) / top_n
    if abs_filter:
        weights["CASH"] = 1 - weights.sum(axis=1)
    return weights


def simulate(prices, rf, targets, cost_bps):
    """Backtest a table of target weights (rows = decision month-ends).
    Returns monthly 'net', 'gross', 'weights' (held) and 'turnover', indexed by return month."""
    asset_ret = prices.pct_change(fill_method=None)
    if "CASH" in targets.columns:
        asset_ret["CASH"] = rf                                # cash earns the risk-free rate
    held = targets.shift(1).iloc[1:]                          # chosen at t, held during t+1
    ret = asset_ret.loc[held.index, held.columns]
    gross = (held * ret).sum(axis=1)

    drifted = held * (1 + ret)                                # weights after a month of price moves
    drifted = drifted.div(drifted.sum(axis=1), axis=0)
    rebalance = (targets.loc[held.index] - drifted).abs().sum(axis=1)   # trade at end of month t
    turnover = rebalance.shift(1).fillna(0.0)                 # stored at t+1, when its cost is paid
    net = gross - turnover * cost_bps / 10_000
    return {"net": net, "gross": gross, "weights": held, "turnover": turnover}


def run_backtest(prices, rf, lookback, top_n, cost_bps, abs_filter=ABS_FILTER, start=None):
    """Ranked asset allocation: top_n ETFs by the lookback signal, equal weight, monthly.
    start: first decision date (default: the evaluation start of the main strategy, so
    every portfolio covers the same months)."""
    start = evaluation_start(prices) if start is None else start
    targets = target_weights(momentum_signal(prices, lookback), top_n, abs_filter)
    return simulate(prices, rf, targets.loc[start:], cost_bps)


def fixed_weight_backtest(prices, rf, weights_dict, cost_bps, lookback=LOOKBACK):
    """Constant target weights, rebalanced monthly. Tickers not in weights_dict get 0."""
    dates = prices.loc[evaluation_start(prices, lookback):].index
    targets = pd.DataFrame(0.0, index=dates, columns=prices.columns)
    for ticker, w in weights_dict.items():
        targets[ticker] = w
    return simulate(prices, rf, targets, cost_bps)


def equal_weight_backtest(prices, rf, cost_bps, lookback=LOOKBACK):
    """1/N in every ETF, rebalanced monthly."""
    n = prices.shape[1]
    return fixed_weight_backtest(prices, rf, {t: 1 / n for t in prices.columns}, cost_bps, lookback)


def buy_and_hold(prices, ticker, lookback=LOOKBACK):
    """Bought at the evaluation start and never traded: net = the ticker's own return."""
    start = evaluation_start(prices, lookback)
    net = prices[ticker].pct_change(fill_method=None).loc[start:].iloc[1:]
    return {"net": net.rename(ticker)}


def max_drawdown(ret):
    wealth = (1 + ret).cumprod()
    return (wealth / wealth.cummax().clip(lower=1) - 1).min()


def performance_metrics(ret, bench, rf, periods=PERIODS):
    """All measures for one portfolio against the benchmark. ret, bench, rf: monthly decimal Series."""
    df = pd.concat({"p": ret, "b": bench, "rf": rf}, axis=1, join="inner").dropna()
    ex_p = df["p"] - df["rf"]
    ex_b = df["b"] - df["rf"]
    active = df["p"] - df["b"]
    n = len(df)

    ann_ret = df["p"].mean() * periods
    cagr = (1 + df["p"]).prod() ** (periods / n) - 1
    vol = df["p"].std() * np.sqrt(periods)
    sharpe = ex_p.mean() * periods / (ex_p.std() * np.sqrt(periods))

    # Alpha and beta: OLS on excess returns
    if np.allclose(active, 0):            # the benchmark itself
        alpha, alpha_t, beta = 0.0, np.nan, 1.0
    else:
        ols = sm.OLS(ex_p, sm.add_constant(ex_b)).fit()
        alpha = ols.params["const"] * periods
        alpha_t = ols.tvalues["const"]
        beta = ols.params.iloc[1]

    te = active.std() * np.sqrt(periods)
    ir = active.mean() * periods / te if te > 0 else np.nan
    # Appraisal ratio: alpha / annualized residual volatility (Lecture 3's second IR definition)
    appraisal = np.nan if np.allclose(active, 0) else alpha / (np.sqrt(ols.mse_resid) * np.sqrt(periods))

    return {
        "Annualized average return": ann_ret,
        "CAGR": cagr,
        "Volatility (ann.)": vol,
        "Sharpe ratio": sharpe,
        "Alpha (ann.)": alpha,
        "Alpha t-stat": alpha_t,
        "Beta": beta,
        "Tracking error (ann.)": te,
        "Information ratio": ir,
        "Appraisal ratio (alpha / residual vol.)": appraisal,
        "Maximum drawdown": max_drawdown(df["p"]),
        "Months": n,
    }


def drawdown_series(ret):
    """Drawdown from the previous peak (the start value of 1 counts as a peak); same as section 12."""
    wealth = (1 + ret).cumprod()
    return wealth / wealth.cummax().clip(lower=1) - 1


def cumulative_wealth(rets):
    """Growth of 1, with an extra starting point of 1 one month before the first return."""
    wealth = (1 + rets).cumprod()
    start = rets.index[0] - pd.offsets.MonthEnd(1)
    return pd.concat([pd.DataFrame(1.0, index=[start], columns=rets.columns), wealth])
