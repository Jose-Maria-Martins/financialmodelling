import altair as alt
import pandas as pd
import streamlit as st

import strategy as s

STRATEGY = "Strategy"
COLORS = ["#2e75b6", "#c55a11", "#38946f", "#b1439b"]      # Strategy, Equal Weight, stocks/bonds, 100% SPY
ALLOCATION_COLORS = ["#1f4e79", "#2e75b6", "#9dc3e6", "#c55a11", "#f4b183", "#ffd966", "#bf9000", "#548235", "#d9d9d9"]
PERCENT = ["Annualized average return", "CAGR", "Volatility (ann.)", "Alpha (ann.)", "Maximum drawdown"]
MEASURES = {   # shown in the table: name in the notebook -> how to read it
    "Annualized average return": "Mean monthly return × 12",
    "CAGR": "Compound growth rate per year",
    "Volatility (ann.)": "Standard deviation of monthly returns × √12",
    "Sharpe ratio": "Annualized excess return over T-bills ÷ its volatility",
    "Alpha (ann.)": "Return not explained by benchmark exposure (regression intercept × 12)",
    "Alpha t-stat": "Above about 2: unlikely to be luck alone",
    "Beta": "Sensitivity to the benchmark (not annualized)",
    "Information ratio": "Annualized active return ÷ tracking error",
    "Maximum drawdown": "Largest fall from a previous peak (month-end values)",
}


@st.cache_data(ttl="1d", show_spinner="Downloading prices and T-bill rates…")
def load_data():
    prices, _ = s.load_prices(s.TICKERS, s.START, s.END)
    return prices, s.load_rf(s.START, s.END)


def line_chart(data, colors, y_title, y_format, tip_format):
    long = data.rename_axis("Date").reset_index().melt("Date", var_name="Portfolio", value_name="Value")
    scale = alt.Scale(domain=list(data.columns), range=[colors[c] for c in data.columns])
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["Date"], empty=False)
    base = alt.Chart(long).encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y("Value:Q", title=y_title, axis=alt.Axis(format=y_format)),
        color=alt.Color("Portfolio:N", scale=scale, legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("Date:T", format="%b %Y"), "Portfolio:N",
                 alt.Tooltip("Value:Q", title=y_title, format=tip_format)])
    points = base.mark_point(size=60, filled=True).encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))).add_params(nearest)
    return (base.mark_line(strokeWidth=2) + points).properties(height=340)


def allocation_chart(held):
    # Weights are dated at the end of the month they were held, so each one is drawn back to the previous
    # month-end; the first row is repeated one month earlier so the first month is visible too.
    plot = pd.concat([held.iloc[:1].set_axis([held.index[0] - pd.offsets.MonthEnd(1)]), held])
    order = list(held.columns)
    long = plot.rename_axis("Date").reset_index().melt("Date", var_name="Asset", value_name="Weight")
    long["Order"] = long["Asset"].map(order.index)
    area = alt.Chart(long).mark_area(interpolate="step-before").encode(
        x=alt.X("Date:T", title=None),
        y=alt.Y("Weight:Q", title="Weight", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Asset:N", scale=alt.Scale(domain=order, range=ALLOCATION_COLORS[:len(order)]),
                        legend=alt.Legend(orient="top", title=None)),
        order="Order:O")
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["Date"], empty=False)
    rule = alt.Chart(held.rename_axis("Date").reset_index()).mark_rule(color="gray").encode(
        x="Date:T", opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("Date:T", format="%b %Y"), *[alt.Tooltip(f"{c}:Q", format=".0%") for c in order]],
    ).add_params(nearest)
    return (area + rule).properties(height=340)


def show(measure, value):
    if pd.isna(value):
        return "–"
    return f"{value:.2%}" if measure in PERCENT else f"{value:.2f}"


st.set_page_config(page_title="Ranked Asset Allocation", layout="wide")
st.html("<style>.stMainBlockContainer {padding-top: 3.75rem}  [data-testid='stSidebarHeader'] {margin-bottom: 0}</style>")
st.title("Ranked Asset Allocation")
st.write("Each month-end, eight ETFs are scored by the average of two past returns. The best ones are held at "
         "equal weight during the next month; with the cash filter on, a selected ETF with a negative score is "
         "replaced by T-bills. Change the assumptions on the left to see what happens.")

# ---- Assumptions (defaults = notebook base case) ----
side = st.sidebar
side.header("Assumptions")
top_n = side.slider("Number of ETFs held", 1, len(s.TICKERS), s.TOP_N)
short = side.slider("First lookback (months)", 1, 12, s.LOOKBACK[0])
long = side.slider("Second lookback (months)", 1, 12, s.LOOKBACK[1])
cash_filter = side.checkbox("Cash filter", s.ABS_FILTER,
                            help="A selected ETF with a negative score is replaced by T-bills.")
cost_bps = side.slider("Transaction cost (bps per unit of turnover)", 0, 50, s.COST_BPS, step=5)
bench_name = side.selectbox("Benchmark", ["Equal Weight", "Stocks/bonds", "100% SPY"],
                            help="Alpha, beta and the information ratio are measured against this one. Equal Weight: the 8 ETFs, "
                                 "1/8 each. Stocks/bonds: SPY and IEF. 100% SPY: only the S&P 500 ETF, bought once.")
show_all = side.checkbox("Show all benchmarks", help="Adds the other two benchmarks to the charts and the table.")
stocks = 60
if show_all or bench_name == "Stocks/bonds":
    stocks = side.slider("Stocks in the stocks/bonds benchmark (%)", 10, 90, 60, step=10,
                         help="SPY; the rest is in IEF (Treasuries), rebalanced monthly. The notebook uses 60/40.")
mix = f"{stocks}/{100 - stocks}"
if bench_name == "Stocks/bonds":
    bench_name = mix
first, last = int(s.START[:4]), int(s.END[:4])
years = side.slider("Years shown", first, last, (first, last),
                    help="The strategy rule is not refitted: this only selects which months are evaluated.")

try:
    prices, rf = load_data()
except Exception as err:
    st.error(f"The price or T-bill download failed, please reload in a few minutes. ({err})")
    st.stop()

# ---- Backtest ----
lookback = (short, long)
signal = s.momentum_signal(prices, lookback)
strat = s.run_backtest(prices, rf, lookback, top_n, cost_bps, cash_filter,
                       start=s.evaluation_start(prices, lookback))
benchmarks = {
    "Equal Weight": s.equal_weight_backtest(prices, rf, cost_bps, lookback),
    mix: s.fixed_weight_backtest(prices, rf, {"SPY": stocks / 100, "IEF": 1 - stocks / 100}, cost_bps, lookback),
    "100% SPY": s.buy_and_hold(prices, "SPY", lookback),
}
colors = dict(zip([STRATEGY, *benchmarks], COLORS))
rets = pd.DataFrame({STRATEGY: strat["net"], **{name: b["net"] for name, b in benchmarks.items()
                                                if show_all or name == bench_name}})

rets = rets.loc[str(years[0]):str(years[1])]
if len(rets) < 12:
    st.error("Choose a period of at least 12 months.")
    st.stop()
if len(rets) < 36:
    st.warning("Fewer than 36 months: alpha, beta and the ratios are very imprecise.")

base_case = (top_n == s.TOP_N and lookback == s.LOOKBACK and cash_filter == s.ABS_FILTER
             and cost_bps == s.COST_BPS and stocks == 60 and years == (first, last))
st.caption(f"{rets.index[0]:%b %Y} – {rets.index[-1]:%b %Y} ({len(rets)} months), net of costs. "
           + ("**Notebook base case.**" if base_case else "Settings differ from the notebook base case."))

# ---- Results ----
st.subheader("Cumulative performance (value of $1 invested)")
wealth = s.cumulative_wealth(rets)
for col, name in zip(st.columns(4), rets.columns):
    col.metric(name, f"${wealth[name].iloc[-1]:.2f}")
st.altair_chart(line_chart(wealth, colors, "Value of $1 invested", "$.1f", "$.2f"), width="stretch")

st.subheader(f"Performance measures (benchmark: {bench_name})")
perf = {name: s.performance_metrics(rets[name], rets[bench_name], rf) for name in rets.columns}
for m in perf.values():
    if m["Tracking error (ann.)"] < 1e-9:      # same portfolio as the benchmark: ratio not defined
        m["Information ratio"] = float("nan")
st.table(pd.DataFrame({**{name: {m: show(m, perf[name][m]) for m in MEASURES} for name in perf},
                       "How to read it": MEASURES}))

st.subheader("Drawdown from previous peak")
st.altair_chart(line_chart(rets.apply(s.drawdown_series), colors, "Drawdown", "%", ".1%"), width="stretch")

st.subheader("Strategy allocation held each month")
held = strat["weights"].loc[rets.index]
st.altair_chart(allocation_chart(held), width="stretch")

st.subheader(f"Latest allocation (decided {prices.index[-1]:%d %b %Y}, for the following month)")
weights = s.target_weights(signal, top_n, cash_filter).iloc[-1]
st.table(pd.DataFrame({"Score": signal.iloc[-1].reindex(weights.index).map(lambda v: show("CAGR", v)),
                       "Weight": weights.map("{:.1%}".format)}))

st.caption("Data: Yahoo Finance (month-end adjusted prices) and FRED TB3MS. The first purchase is free, trades are "
           "assumed at the month-end close, taxes are ignored, and 100% SPY is bought once and has no costs. "
           "Past performance does not guarantee future results. Student project, not investment advice.")
