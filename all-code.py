"""
app.py — NSE OHLC Predictor with spike-risk display and holiday-coloured calendar.

Pages:
  📈 Predict        — future / backtest / closed-day handling + risk badge
  📅 Holiday Calendar — NSE holiday calendar (dynamic)

New in this version:
  - Risk score badge (LOW / MODERATE / HIGH / EXTREME) on every prediction
  - Holiday-highlighted date picker: days that are NSE holidays shown in
    a custom HTML calendar component with amber/yellow colour
  - Backtest shows per-field error bars
"""

from __future__ import annotations

import calendar as pycal
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.utils import get_logger, set_global_seed

set_global_seed()
log = get_logger("app")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="further update on volatile",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #080c14; color: #d4dae8;
}
.kpi-card {
    background: #0e1620; border: 1px solid #1e2a3a;
    border-radius: 10px; padding: 1rem 1.25rem; text-align: center;
}
.kpi-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
    text-transform: uppercase; letter-spacing: 0.12em;
    color: #5a7393; margin-bottom: 0.3rem;
}
.kpi-value { font-family: 'IBM Plex Mono', monospace; font-size: 1.3rem; font-weight: 700; color: #e8edf5; }
.kpi-value.up   { color: #4caf82; }
.kpi-value.down { color: #e05c5c; }
.kpi-value.neutral { color: #80cbc4; }
.sec-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.14em; color: #4fc3f7; text-transform: uppercase;
    border-left: 2px solid #4fc3f7; padding-left: 0.6rem; margin: 1.4rem 0 0.8rem 0;
}
.closed-box {
    background: #1a1a0e; border: 1px solid #4a4a20;
    border-radius: 10px; padding: 1.5rem 2rem; text-align: center;
    font-family: 'IBM Plex Mono', monospace;
}
.closed-icon { font-size: 2.5rem; margin-bottom: 0.4rem; }
.closed-title { font-size: 1.1rem; font-weight: 700; color: #d4b84a; }
.closed-reason { font-size: 0.85rem; color: #8a8a6a; margin-top: 0.3rem; }
.backtest-badge {
    display: inline-block; background: #0e2040; border: 1px solid #2a4a80;
    border-radius: 5px; padding: 0.2rem 0.7rem; font-size: 0.7rem;
    font-family: 'IBM Plex Mono', monospace; color: #4fc3f7;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.8rem;
}
/* Risk badge colours */
.risk-LOW      { color: #4caf82; background: #0a2018; border: 1px solid #1a5030; }
.risk-MODERATE { color: #d4b84a; background: #1a1a08; border: 1px solid #5a5010; }
.risk-HIGH     { color: #e08050; background: #1a0e08; border: 1px solid #5a3010; }
.risk-EXTREME  { color: #e05050; background: #1a0808; border: 1px solid #5a1010; }
.risk-badge {
    display: inline-block; border-radius: 5px;
    padding: 0.3rem 0.9rem; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Assets
# ─────────────────────────────────────────────────────────────────────────────
ASSETS: dict[str, str] = {
    "Reliance Industries"  : "RELIANCE.NS",
    "Infosys"              : "INFY.NS",
    "TCS"                  : "TCS.NS",
    "HDFC Bank"            : "HDFCBANK.NS",
    "ICICI Bank"           : "ICICIBANK.NS",
    "Wipro"                : "WIPRO.NS",
    "Bajaj Finance"        : "BAJFINANCE.NS",
    "Maruti Suzuki"        : "MARUTI.NS",
    "Apple"                : "AAPL",
    "Microsoft"            : "MSFT",
    "Google (Alphabet)"    : "GOOGL",
    "NVIDIA"               : "NVDA",
    "S&P 500 ETF (SPY)"    : "SPY",
    "TATASTEEL"            : "TATASTEEL.NS",
    "NESTLE INDIA"         : "NESTLEIND.NS",
    "NIFTY 50"             : "^NSEI",
}

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 NSE OHLC Predictor")
    st.markdown("---")

    page = st.radio("Navigation", ["📈 Predict", "📅 Holiday Calendar"],
                    label_visibility="collapsed")

    st.markdown("---")
    selected_label = st.selectbox("Asset", list(ASSETS.keys()), index=0)
    ticker = ASSETS[selected_label]

    today = date.today()

    sel_date = st.date_input(
        "Date",
        value=today + timedelta(days=1),
        min_value=date(2015, 1, 1),
        max_value=today + timedelta(days=365),
        help="Past dates → Backtest. Future dates → Prediction. Holidays shown in the calendar below.",
    )

    # ── Holiday-aware mini calendar in sidebar ─────────────────────────────────

    st.markdown("---")
    run_btn = st.button("🚀 Predict / Backtest", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### 🔧 Train")
    train_ticker = st.text_input("Ticker to train", value=ticker)
    train_btn    = st.button("Train Model", use_container_width=True)
    st.caption("`python train.py --ticker TICKER`")




# ─────────────────────────────────────────────────────────────────────────────
# Training block
# ─────────────────────────────────────────────────────────────────────────────
if train_btn:
    from src.data_loader import load_ohlcv
    from src.trainer import train as run_training
    with st.spinner(f"Training {train_ticker}…"):
        try:
            df_raw = load_ohlcv(train_ticker)
            result = run_training(ticker=train_ticker, df_raw=df_raw,
                                  use_optuna=True, optuna_trials=15, deep_epochs=15)
            st.success(f"✅ Model trained for **{train_ticker}**.")
            folds = result.get("fold_metrics", [])
            if folds:
                rows = [{"Test Year": f["test_year"],
                         **{k.replace("target_", "MAE "): round(v, 4)
                            for k, v in f["mae"].items()}} for f in folds]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Training failed: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Full-page holiday calendar  (larger version for the Holiday Calendar page)
# ─────────────────────────────────────────────────────────────────────────────
def _render_full_holiday_calendar(year: int, month: int) -> None:
    from src.market_calendar import get_nse_holidays
    try:
        holidays = get_nse_holidays(year)
    except Exception:
        holidays = {}

    cal     = pycal.monthcalendar(year, month)
    mn      = date(year, month, 1).strftime("%B %Y")

    rows_html = ""
    for week in cal:
        rows_html += "<tr>"
        for i, d_num in enumerate(week):
            if d_num == 0:
                rows_html += '<td class="empty"></td>'
                continue
            d     = date(year, month, d_num)
            iso   = d.isoformat()
            is_we = i >= 5
            is_hol= iso in holidays
            is_tod= d == today
            hol_n = holidays.get(iso, "")
            cls   = "fday"
            if is_we:  cls += " fweekend"
            if is_hol: cls += " fholiday"
            if is_tod: cls += " ftoday"
            title = f'title="{hol_n}"' if hol_n else ""
            sub   = f'<span class="fhname">{hol_n}</span>' if hol_n else ""
            rows_html += f'<td class="{cls}" {title}><span class="fdnum">{d_num}</span>{sub}</td>'
        rows_html += "</tr>"

    html = f"""
<style>
.fcal {{ font-family:'IBM Plex Mono',monospace; border-collapse:collapse; width:100%; margin:0.5rem 0; }}
.fcal caption {{ font-size:1rem; font-weight:700; color:#80cbc4; padding:0.5rem 0; }}
.fcal th {{ color:#5a7393; text-align:center; padding:6px; font-size:0.8rem; font-weight:400; }}
.fcal td {{ text-align:center; padding:8px 4px; border-radius:6px; border:1px solid #0e1820;
            min-width:80px; vertical-align:top; height:52px; font-size:0.85rem; color:#c0c8d8; }}
.fcal td.empty {{ background:transparent; border:none; }}
.fcal td.fweekend {{ color:#5a6a7a; background:#0c1218; }}
.fcal td.fholiday {{ background:#2a2000 !important; color:#d4b84a !important;
                      border:1px solid #5a4a10 !important; cursor:help; }}
.fcal td.ftoday {{ outline:2px solid #80cbc4; }}
.fdnum {{ display:block; font-weight:600; }}
.fhname {{ display:block; font-size:0.6rem; color:#d4b84a; margin-top:2px;
           white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:90px; }}
.flegend {{ font-size:0.72rem; color:#5a7393; margin-top:0.5rem; }}
</style>
<table class="fcal">
  <caption>{mn}</caption>
  <thead>
    <tr>
      <th>Monday</th><th>Tuesday</th><th>Wednesday</th><th>Thursday</th>
      <th>Friday</th><th style="color:#5a6a7a">Saturday</th><th style="color:#5a6a7a">Sunday</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>
<div class="flegend">
  <span style="color:#d4b84a">■</span> NSE Holiday &nbsp;
  <span style="color:#5a6a7a">■</span> Weekend &nbsp;
  <span style="color:#80cbc4">□</span> Today
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared UI helpers
# ─────────────────────────────────────────────────────────────────────────────

def _render_ohlc_cards(pred: dict) -> None:
    st.markdown('<div class="sec-label">Predicted OHLC</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, label, key in [(c1,"Open","Open"),(c2,"High","High"),(c3,"Low","Low"),(c4,"Close","Close")]:
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">₹ {pred[key]:,.2f}</div></div>',
            unsafe_allow_html=True)


def _render_summary_metrics(pred: dict) -> None:
    st.markdown('<div class="sec-label">Summary</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    sign = "+" if pred["pct_change"] >= 0 else ""
    cls  = "up" if pred["pct_change"] >= 0 else "down"
    m1.markdown(f'<div class="kpi-card"><div class="kpi-label">Prev Close</div>'
                f'<div class="kpi-value">₹ {pred["prev_close"]:,.2f}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="kpi-card"><div class="kpi-label">Predicted % Change</div>'
                f'<div class="kpi-value {cls}">{sign}{pred["pct_change"]:.2f}%</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="kpi-card"><div class="kpi-label">Day Range (₹)</div>'
                f'<div class="kpi-value">₹ {pred["range"]:,.2f}</div></div>', unsafe_allow_html=True)
    spread = pred["range"] / pred["Open"] * 100 if pred["Open"] else 0
    m4.markdown(f'<div class="kpi-card"><div class="kpi-label">Spread %</div>'
                f'<div class="kpi-value">{spread:.2f}%</div></div>', unsafe_allow_html=True)


def _render_comparison_table(pred: dict) -> None:
    st.markdown('<div class="sec-label">Predicted vs Actual</div>', unsafe_allow_html=True)
    rows = []
    for field in ["Open", "High", "Low", "Close"]:
        p    = pred[field]
        a    = pred[f"actual_{field}"]
        e    = pred.get(f"err_{field.lower()}", abs(p - a))
        pct  = e / a * 100 if a else 0
        rows.append({"Field": field, "Predicted": f"₹{p:,.2f}", "Actual": f"₹{a:,.2f}",
                     "Abs Error": f"₹{e:,.2f}", "Error %": f"{pct:.2f}%"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_error_metrics(pred: dict) -> None:
    st.markdown('<div class="sec-label">Error Metrics</div>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    e1.markdown(f'<div class="kpi-card"><div class="kpi-label">MAE</div>'
                f'<div class="kpi-value neutral">₹ {pred["mae"]:,.4f}</div></div>', unsafe_allow_html=True)
    e2.markdown(f'<div class="kpi-card"><div class="kpi-label">RMSE</div>'
                f'<div class="kpi-value neutral">₹ {pred["rmse"]:,.4f}</div></div>', unsafe_allow_html=True)


def _render_bar_chart(pred: dict, target_date: date, ticker: str, show_actual: bool = False) -> None:
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2 if show_actual else 1,
                                 figsize=(8 if show_actual else 4, 3), facecolor="#0e1620")
        if not show_actual:
            axes = [axes]

        def _draw(ax, O, H, L, C, title_str):
            ax.set_facecolor("#0e1620")
            color = "#4caf82" if C >= O else "#e05c5c"
            ax.plot([1, 1], [L, H], color=color, linewidth=1.5)
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.6, min(O, C)), 0.8, abs(C - O) or (H - L) * 0.05,
                boxstyle="square,pad=0", linewidth=0, facecolor=color, alpha=0.9))
            for price, lbl in [(H,"H"),(L,"L"),(O,"O"),(C,"C")]:
                ax.text(2.1, price, f"{lbl}: {price:,.1f}",
                        color="#d4dae8", fontsize=7, va="center", fontfamily="monospace")
            ax.set_xlim(0, 5)
            pad = (H - L) * 0.2 or 1
            ax.set_ylim(L - pad, H + pad)
            ax.axis("off")
            ax.set_title(title_str, color="#5a7393", fontsize=8, fontfamily="monospace")

        _draw(axes[0], pred["Open"], pred["High"], pred["Low"], pred["Close"],
              f"Predicted — {target_date.strftime('%d %b %Y')}")
        if show_actual:
            _draw(axes[1], pred["actual_Open"], pred["actual_High"],
                  pred["actual_Low"], pred["actual_Close"],
                  f"Actual — {target_date.strftime('%d %b %Y')}")

        fig.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═════════════════════════════════════════════════════════════════════════════
if page == "📈 Predict":
    is_past    = sel_date < today
    mode_label = "📊 Backtest" if is_past else "🔮 Future"
    date_str   = sel_date.strftime("%A, %d %b %Y")

    st.markdown(f"## 📊 {selected_label} ({ticker})")
    st.markdown(f"**{date_str}** &nbsp;|&nbsp; {mode_label} &nbsp;|&nbsp; LSTM + Transformer + LightGBM")

    if run_btn:
        from src.predictor import predict

        with st.spinner("Running prediction…"):
            try:
                pred = predict(ticker, sel_date)
            except FileNotFoundError:
                st.warning(
                    f"⚠ No trained model for **{ticker}**.  \n"
                    f"`python train.py --ticker {ticker}`"
                )
                st.stop()
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.stop()

        # ── CLOSED ────────────────────────────────────────────────────────────
        if pred.get("market_closed"):
            reason = pred["reason"]
            if reason == "Weekend":
                st.markdown(
                    '<div class="closed-box"><div class="closed-icon">🚫</div>'
                    '<div class="closed-title">Market Closed</div>'
                    '<div class="closed-reason">Reason: Weekend</div></div>',
                    unsafe_allow_html=True)
                st.info("Select a weekday for prediction.")
            else:
                st.markdown(
                    f'<div class="closed-box"><div class="closed-icon">🚫</div>'
                    f'<div class="closed-title">Market Closed</div>'
                    f'<div class="closed-reason">Reason: {reason}</div></div>',
                    unsafe_allow_html=True)
                st.warning(f"📅 **{date_str}** is an NSE holiday: **{reason}**")
            st.stop()

        mode = pred["mode"]

        # ── RISK BADGE ─────────────────────────────────────────────────────────
        rl = pred.get("risk_label", "LOW")
        rs = pred.get("risk_score", 0.0)
        risk_icons = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🟠", "EXTREME": "🔴"}
        st.markdown(
            f'<span class="risk-badge risk-{rl}">'
            f'{risk_icons.get(rl,"⚪")} Prediction Risk: {rl} ({rs:.0%})'
            f'</span>',
            unsafe_allow_html=True,
        )
        if rl in ("HIGH", "EXTREME"):
            st.warning(
                f"⚠ **High-risk prediction day** (score: {rs:.2f}). "
                "Yesterday showed elevated range/volume vs recent baseline. "
                "Error may be larger than typical. Widen your confidence interval."
            )

        # ── BACKTEST ──────────────────────────────────────────────────────────
        if mode == "backtest":
            st.markdown('<div class="backtest-badge">📊 Backtest — Historical Date</div>',
                        unsafe_allow_html=True)
            _render_ohlc_cards(pred)
            if pred.get("data_available"):
                _render_comparison_table(pred)
                _render_error_metrics(pred)
            else:
                st.info("Actual data not available for this date.")

        # ── FUTURE ────────────────────────────────────────────────────────────
        else:
            _render_ohlc_cards(pred)
            _render_summary_metrics(pred)

        # ── Chart ──────────────────────────────────────────────────────────────
        st.markdown('<div class="sec-label">Candlestick Preview</div>', unsafe_allow_html=True)
        _render_bar_chart(pred, sel_date, ticker,
                          show_actual=mode == "backtest" and pred.get("data_available"))

        with st.expander("Raw result"):
            st.json({k: str(v) if isinstance(v, date) else v for k, v in pred.items()})

    else:
        st.info(f"👈 Click **Predict / Backtest** for {date_str}.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: HOLIDAY CALENDAR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📅 Holiday Calendar":
    from src.market_calendar import (
        get_all_holidays_for_year, get_holidays_for_month, invalidate_cache,
    )

    st.markdown("## 📅 NSE Holiday Calendar")
    st.markdown("Fetched dynamically from NSE India. No hardcoded dates.")

    cal_year  = st.selectbox("Year",  [today.year, today.year + 1], index=0)
    cal_month = st.selectbox("Month", list(range(1, 13)), index=today.month - 1,
                             format_func=lambda m: date(cal_year, m, 1).strftime("%B"))
    if st.button("🔄 Refresh from NSE"):
        invalidate_cache()
        st.success("Cache cleared.")

    month_label = date(cal_year, cal_month, 1).strftime("%B %Y")
    st.markdown(f'<div class="sec-label">Holidays in {month_label}</div>', unsafe_allow_html=True)
    with st.spinner("Fetching…"):
        mh = get_holidays_for_month(cal_year, cal_month)
    if not mh:
        st.info(f"No NSE holidays in {month_label}.")
    else:
        st.dataframe(
            pd.DataFrame([{"Date": h["date"].strftime("%d %b %Y (%A)"), "Holiday": h["name"]}
                          for h in mh]),
            use_container_width=True, hide_index=True,
        )

    st.markdown(f'<div class="sec-label">Full Year — {cal_year}</div>', unsafe_allow_html=True)
    with st.spinner("Loading…"):
        ay = get_all_holidays_for_year(cal_year)
    if not ay:
        st.warning("Could not fetch holidays. Try clicking Refresh.")
    else:
        df_h = pd.DataFrame([{"Date": h["date"].strftime("%d %b %Y"),
                               "Day": h["day"], "Holiday": h["name"]} for h in ay])
        st.dataframe(df_h, use_container_width=True, hide_index=True)
        st.caption(f"{len(df_h)} NSE trading holidays in {cal_year}")

    # Also render the holiday mini-calendar here, full-width
    st.markdown(f'<div class="sec-label">Visual Calendar — {date(cal_year, cal_month, 1).strftime("%B %Y")}</div>',
                unsafe_allow_html=True)
    _render_full_holiday_calendar(cal_year, cal_month)

    st.markdown("---")
    st.caption("Source: nseindia.com · Cache refreshes every 30 days · Hover a date for holiday name.")

"""
data_loader.py — Download and clean OHLCV data using yfinance.

Rules:
  - 15 years of daily data, auto-updated to latest date
  - Sorted ascending by date
  - Missing values handled safely
  - No leakage: this module only returns raw price data
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from src.utils import DATA_DIR, FETCH_YEARS, get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def load_ohlcv(
    ticker: str,
    use_cache: bool = True,
    cache_days: int = 1,
) -> pd.DataFrame:
    """
    Download (or load from cache) FETCH_YEARS of daily OHLCV data for *ticker*.

    Returns
    -------
    pd.DataFrame with columns [Open, High, Low, Close, Volume]
    DatetimeIndex sorted ascending, timezone-naive.

    Raises
    ------
    ValueError  – if the downloaded frame is empty or too short.
    """
    cache_file = DATA_DIR / f"{ticker.replace('/', '_').replace('^','')}_ohlcv.parquet"

    # ── Try cache ──────────────────────────────────────────────────────────────
    if use_cache and cache_file.exists():
        age = (date.today() - date.fromtimestamp(cache_file.stat().st_mtime)).days
        if age < cache_days:
            log.info("Loading %s from cache (%s).", ticker, cache_file.name)
            df = pd.read_parquet(cache_file)
            return _validate(df, ticker)

    # ── Download ───────────────────────────────────────────────────────────────
    end_date  = date.today()
    start_date = date(end_date.year - FETCH_YEARS, end_date.month, end_date.day)

    log.info("Downloading %s: %s → %s", ticker, start_date, end_date)
    raw: pd.DataFrame = yf.download(
        ticker,
        start=str(start_date),
        end=str(end_date + timedelta(days=1)),   # end is exclusive in yfinance
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"yfinance returned empty data for '{ticker}'.")

    df = _clean(raw)
    df = _validate(df, ticker)

    # ── Persist ────────────────────────────────────────────────────────────────
    df.to_parquet(cache_file)
    log.info("Saved %d rows to %s.", len(df), cache_file.name)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardise columns, handle MultiIndex, remove NaN rows."""
    # yfinance sometimes returns a MultiIndex column (ticker, field)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Keep only OHLCV
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns after download: {missing}")

    df = raw[needed].copy()

    # Ensure DatetimeIndex, timezone-naive
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"

    # Sort ascending
    df.sort_index(inplace=True)

    # Drop rows with all-NaN OHLC
    df.dropna(subset=["Open", "High", "Low", "Close"], how="all", inplace=True)

    # Forward-fill isolated NaN cells (e.g. volume gaps)
    df.ffill(inplace=True)

    # Drop any remaining NaN rows
    df.dropna(inplace=True)

    # Basic sanity: High >= Low
    bad = df["High"] < df["Low"]
    if bad.any():
        log.warning("Dropping %d rows where High < Low.", bad.sum())
        df = df[~bad]

    return df


def _validate(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    min_rows = 252 * 3   # at least 3 years
    if len(df) < min_rows:
        raise ValueError(
            f"Only {len(df)} rows for '{ticker}' — need at least {min_rows}."
        )
    log.info("%s: %d rows, %s → %s", ticker, len(df), df.index[0].date(), df.index[-1].date())
    return df
"""
deep_model.py — Multi-window Hybrid LSTM + Transformer feature extractor.

Architecture per the spec:
  Three input branches (short=30, mid=60, long=120):
    Each branch:
      LSTM  → captures temporal momentum at that window scale
      Transformer Encoder → captures regime shifts / long-range dependencies
    Each branch output: concat(LSTM_out, Transformer_out) → branch_emb

  Fusion:
    concat(branch_short_emb, branch_mid_emb, branch_long_emb)
    → Dense(embedding_dim, relu)
    → final embedding vector

  Role: feature extractor only.
        A Dense(1) head is added during training and discarded after.
        LightGBM receives the embedding vector, never raw sequences.

Transformer rules (spec §5):
  - Multi-head attention (2, 4, or 8 heads, tunable)
  - Accepts sequence embedding from LSTM output (residual path)
  - Learns regime shifts via self-attention over the full window
  - Output is sequence-level (GlobalAveragePooling1D), not token-level
  - Does NOT directly predict price
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.utils import SEQ_LONG, SEQ_MID, SEQ_SHORT, SEED, get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer encoder block
# ─────────────────────────────────────────────────────────────────────────────

def _transformer_block(
    inputs: tf.Tensor,
    num_heads: int,
    ff_dim: int,
    dropout: float,
    name_prefix: str = "tr",
) -> tf.Tensor:
    """
    Single Transformer encoder block.
    Input shape: (batch, seq_len, d_model)
    Output shape: same

    key_dim is per-head dimension.  We use d_model // num_heads so the
    total attention capacity scales with d_model regardless of head count.
    """
    d_model  = inputs.shape[-1]
    key_dim  = max(1, d_model // num_heads)

    # Multi-head self-attention
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        name=f"{name_prefix}_mha",
    )(inputs, inputs)
    attn = layers.Dropout(dropout, name=f"{name_prefix}_attn_drop")(attn)
    x    = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(inputs + attn)

    # Position-wise FFN
    ff = layers.Dense(ff_dim, activation="relu", name=f"{name_prefix}_ff1")(x)
    ff = layers.Dropout(dropout, name=f"{name_prefix}_ff_drop")(ff)
    ff = layers.Dense(d_model,  name=f"{name_prefix}_ff2")(ff)
    ff = layers.Dropout(dropout, name=f"{name_prefix}_ff2_drop")(ff)
    x  = layers.LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(x + ff)

    return x


# ─────────────────────────────────────────────────────────────────────────────
# Single-window branch: LSTM → Transformer → branch embedding
# ─────────────────────────────────────────────────────────────────────────────

def _build_branch(
    seq_input: keras.Input,
    lstm_units: int,
    lstm_layers: int,
    num_heads: int,
    tr_ff_dim: int,
    dropout: float,
    branch_name: str,
) -> tf.Tensor:
    """
    Build one LSTM+Transformer branch for a single time-window input.

    Pipeline:
      seq_input  (batch, seq_len, n_feat)
        → LSTM stack  →  lstm_out  (batch, lstm_units)
        → Transformer (on the same seq_input, projected)  →  tr_out  (batch, proj_dim)
        → Concatenate([lstm_out, tr_out])
        → Dense(branch_emb_dim, relu)

    The Transformer receives the original sequence (not LSTM output) so it
    can attend over the full window without LSTM's sequential bottleneck.
    LSTM and Transformer are thus complementary:
      LSTM  → sequential / recurrent momentum
      Transformer → global self-attention / regime awareness
    """
    n_feat   = seq_input.shape[-1]
    # Transformer projection dimension: multiple of num_heads, ≥ 16
    proj_dim = max(num_heads * 4, (n_feat // num_heads + 1) * num_heads)

    # ── LSTM branch ──────────────────────────────────────────────────────────
    x_lstm = seq_input
    for i in range(lstm_layers):
        ret_seq = (i < lstm_layers - 1)
        x_lstm  = layers.LSTM(
            lstm_units, return_sequences=ret_seq,
            name=f"{branch_name}_lstm_{i}"
        )(x_lstm)
        x_lstm  = layers.Dropout(dropout, name=f"{branch_name}_lstm_drop_{i}")(x_lstm)
    # shape: (batch, lstm_units)

    # ── Transformer branch ────────────────────────────────────────────────────
    x_tr = layers.Dense(proj_dim, name=f"{branch_name}_tr_proj")(seq_input)
    x_tr = _transformer_block(
        x_tr, num_heads=num_heads, ff_dim=tr_ff_dim,
        dropout=dropout, name_prefix=f"{branch_name}_tr",
    )
    # Sequence-level pooling (spec §5: NOT token-level prediction)
    x_tr = layers.GlobalAveragePooling1D(name=f"{branch_name}_tr_gap")(x_tr)
    # shape: (batch, proj_dim)

    # ── Merge within branch ───────────────────────────────────────────────────
    merged = layers.Concatenate(name=f"{branch_name}_concat")([x_lstm, x_tr])
    # Normalise before passing to the fusion stage
    merged = layers.LayerNormalization(epsilon=1e-6, name=f"{branch_name}_ln")(merged)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Full model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_hybrid_model(
    n_features: int,
    embedding_dim: int = 64,
    lstm_units: int = 64,
    lstm_layers: int = 1,
    num_heads: int = 4,
    transformer_ff_dim: int = 128,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
) -> tuple[keras.Model, keras.Model]:
    """
    Build the three-window hybrid LSTM+Transformer model.

    Inputs: three named inputs — "short", "mid", "long"
    Output: embedding vector of size *embedding_dim*

    Returns
    -------
    full_model : compiled with MAE loss (used during training)
    embedder   : same graph up to the embedding layer (used for feature extraction)
    """
    tf.random.set_seed(SEED)

    inp_short = keras.Input(shape=(SEQ_SHORT, n_features), name="short")
    inp_mid   = keras.Input(shape=(SEQ_MID,   n_features), name="mid")
    inp_long  = keras.Input(shape=(SEQ_LONG,  n_features), name="long")

    emb_short = _build_branch(inp_short, lstm_units, lstm_layers, num_heads,
                               transformer_ff_dim, dropout, "short")
    emb_mid   = _build_branch(inp_mid,   lstm_units, lstm_layers, num_heads,
                               transformer_ff_dim, dropout, "mid")
    emb_long  = _build_branch(inp_long,  lstm_units, lstm_layers, num_heads,
                               transformer_ff_dim, dropout, "long")

    # ── Fuse three branches ───────────────────────────────────────────────────
    fused = layers.Concatenate(name="fusion")([emb_short, emb_mid, emb_long])
    embedding = layers.Dense(embedding_dim, activation="relu", name="embedding")(fused)
    embedding = layers.Dropout(dropout, name="embed_drop")(embedding)
    # Final normalisation so embeddings are on a consistent scale for LightGBM
    embedding = layers.LayerNormalization(epsilon=1e-6, name="embed_ln")(embedding)

    # ── Training head (discarded after training) ───────────────────────────────
    output = layers.Dense(1, name="output")(embedding)

    inputs     = [inp_short, inp_mid, inp_long]
    full_model = keras.Model(inputs=inputs, outputs=output, name="hybrid_full")
    full_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mae",
    )
    embedder = keras.Model(inputs=inputs, outputs=embedding, name="embedder")

    total_params = full_model.count_params()
    log.info(
        "Built 3-window hybrid: LSTM(%d×%d) + Transformer(heads=%d) × 3 windows "
        "→ emb_dim=%d  |  params=%d",
        lstm_layers, lstm_units, num_heads, embedding_dim, total_params,
    )
    return full_model, embedder


# ─────────────────────────────────────────────────────────────────────────────
# Training helper
# ─────────────────────────────────────────────────────────────────────────────

def train_deep_model(
    model: keras.Model,
    X_seq: dict[str, np.ndarray],
    y: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    validation_split: float = 0.1,
    patience: int = 5,
) -> keras.callbacks.History:
    """
    Train *model* on multi-window inputs.

    Parameters
    ----------
    model   : compiled full_model from build_hybrid_model()
    X_seq   : {"short": arr, "mid": arr, "long": arr}
    y       : (N,) scaled target (target_open used as proxy for embedding training)
    """
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience,
        restore_best_weights=True, verbose=0,
    )
    lr_sched = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3,
        min_lr=1e-6, verbose=0,
    )
    history = model.fit(
        X_seq, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[es, lr_sched],
        verbose=0,
    )
    log.info(
        "Deep model trained: %d epochs  train_MAE=%.6f",
        len(history.epoch), history.history["loss"][-1],
    )
    return history


def extract_embeddings(embedder: keras.Model, X_seq: dict[str, np.ndarray]) -> np.ndarray:
    """
    Run multi-window X_seq through embedder.
    Returns (N, emb_dim).
    """
    return embedder.predict(X_seq, verbose=0)
"""
feature_engineering.py — Feature engineering with spike-robustness enhancements.

LEAKAGE RULE: Every feature uses only information available BEFORE the target row.

NEW in this version (spike-error diagnosis fixes):
  - Removed raw volatility feature (was noisy + redundant with range features)
  - Added ATR (Average True Range) — a proper, stable volatility proxy
  - Added range_expansion_ratio: how much yesterday's range deviated from its 10d avg
    (key signal for detecting incoming high-volatility days)
  - Added body_ratio: (|Close - Open|) / Range — measures candle body compression
  - Added upper_shadow / lower_shadow: wick ratios (regime shift signal)
  - Added volume_spike: whether yesterday's volume was anomalous
  - Added range_z_score: z-score of yesterday's range vs rolling distribution
    (most direct "are we in a spike regime?" feature available without future data)
  - Log-transformed range target: reduces regression skew on extreme days
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from src.utils import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Feature columns
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS: List[str] = [
    # Lag features
    "open_lag1", "high_lag1", "low_lag1", "close_lag1",
    "close_lag2", "close_lag3",
    # Gap features
    "gap_1", "avg_gap_5", "avg_gap_10",
    # Range features (core)
    "range_1", "avg_range_5", "avg_range_10",
    # Range regime features (NEW — spike detection)
    "range_expansion_ratio",   # range_1 / avg_range_10
    "range_z_score_10",        # (range_1 - avg_range_10) / std_range_10
    "atr_14",                  # Average True Range over 14 days (leakage-safe)
    # Candle structure (NEW — regime signals)
    "body_ratio_1",            # |Close-Open| / Range yesterday
    "upper_shadow_1",          # (High - max(Open,Close)) / Range
    "lower_shadow_1",          # (min(Open,Close) - Low) / Range
    # Return features
    "return_1", "return_3", "return_5",
    "pct_return_1", "pct_return_5",
    # Rolling statistics
    "rolling_mean_5", "rolling_mean_10",
    "rolling_std_5", "rolling_std_10",
    # Volume features
    "volume_1", "volume_change", "avg_volume_5",
    "volume_spike",            # NEW: volume_1 / avg_volume_5 (anomaly ratio)
    # Close-position features
    "close_pos_1", "avg_close_pos_5",
    # Calendar features
    "day_of_week", "week_of_year", "month", "day_of_year",
]

TARGET_COLS: List[str] = ["target_open", "target_log_range", "target_closepos"]

# Legacy name kept for compatibility with predictor inverse-transform
_TARGET_RANGE_COL = "target_log_range"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def build_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features and targets from raw OHLCV data.
    Drops NaN rows at the end. Returns clean DataFrame.
    """
    df = df_raw.copy()

    _add_lag_features(df)
    _add_gap_features(df)
    _add_range_features(df)
    _add_range_regime_features(df)
    _add_candle_structure_features(df)
    _add_return_features(df)
    _add_rolling_stats(df)
    _add_volume_features(df)
    _add_close_position_features(df)
    _add_calendar_features(df)
    _add_targets(df)

    before = len(df)
    df.dropna(subset=FEATURE_COLS + TARGET_COLS, inplace=True)
    log.info(
        "build_features: %d raw rows → %d clean rows (dropped %d).",
        before, len(df), before - len(df),
    )
    return df


def calendar_features_for_date(target_date: pd.Timestamp) -> dict:
    return {
        "day_of_week"  : int(target_date.dayofweek),
        "week_of_year" : int(target_date.isocalendar().week),
        "month"        : int(target_date.month),
        "day_of_year"  : int(target_date.day_of_year),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _add_lag_features(df: pd.DataFrame) -> None:
    df["open_lag1"]  = df["Open"].shift(1)
    df["high_lag1"]  = df["High"].shift(1)
    df["low_lag1"]   = df["Low"].shift(1)
    df["close_lag1"] = df["Close"].shift(1)
    df["close_lag2"] = df["Close"].shift(2)
    df["close_lag3"] = df["Close"].shift(3)


def _add_gap_features(df: pd.DataFrame) -> None:
    df["gap_1"]      = df["open_lag1"] - df["close_lag2"]
    daily_gap        = df["Open"].shift(1) - df["Close"].shift(2)
    df["avg_gap_5"]  = daily_gap.rolling(5).mean()
    df["avg_gap_10"] = daily_gap.rolling(10).mean()


def _add_range_features(df: pd.DataFrame) -> None:
    daily_range       = (df["High"] - df["Low"]).shift(1)
    df["range_1"]     = daily_range
    df["avg_range_5"] = daily_range.rolling(5).mean()
    df["avg_range_10"]= daily_range.rolling(10).mean()


def _add_range_regime_features(df: pd.DataFrame) -> None:
    """
    FIX #1: Proper regime / spike-detection features.

    range_expansion_ratio: how large is yesterday's range vs its 10d average?
      > 1.5 → regime expansion likely → model should widen predicted range
      This is the single most predictive leading indicator of a high-error day.

    range_z_score_10: how many SDs above/below normal is yesterday's range?
      Gives LightGBM a clear threshold split for "unusual day".

    atr_14: Average True Range (Wilder). True range accounts for gaps:
      TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
      All values shifted so only past data is used.
    """
    daily_range = (df["High"] - df["Low"]).shift(1)
    avg10       = daily_range.rolling(10).mean()
    std10       = daily_range.rolling(10).std().replace(0, np.nan)

    df["range_expansion_ratio"] = (daily_range / avg10.replace(0, np.nan)).fillna(1.0)
    df["range_z_score_10"]      = ((daily_range - avg10) / std10).fillna(0.0)

    # True range (all shifted by 1 — pure past data)
    H  = df["High"].shift(1)
    L  = df["Low"].shift(1)
    PC = df["Close"].shift(2)   # previous-previous close (one step earlier than H/L)
    tr = pd.concat([H - L, (H - PC).abs(), (L - PC).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()


def _add_candle_structure_features(df: pd.DataFrame) -> None:
    """
    FIX #2: Candle microstructure. These encode whether yesterday was a
    trend day (large body), doji (small body), or spike (long wicks).
    They are excellent LightGBM split candidates for regime detection.
    All computed on yesterday (shift(1)) — zero leakage.
    """
    H  = df["High"].shift(1)
    L  = df["Low"].shift(1)
    O  = df["Open"].shift(1)
    C  = df["Close"].shift(1)
    rng = (H - L).replace(0, np.nan)

    df["body_ratio_1"]   = ((C - O).abs() / rng).fillna(0.5)
    df["upper_shadow_1"] = ((H - pd.concat([O, C], axis=1).max(axis=1)) / rng).fillna(0.0)
    df["lower_shadow_1"] = ((pd.concat([O, C], axis=1).min(axis=1) - L) / rng).fillna(0.0)


def _add_return_features(df: pd.DataFrame) -> None:
    close = df["Close"]
    df["return_1"]     = np.log(close.shift(1) / close.shift(2))
    df["return_3"]     = np.log(close.shift(1) / close.shift(4))
    df["return_5"]     = np.log(close.shift(1) / close.shift(6))
    df["pct_return_1"] = (close.shift(1) - close.shift(2)) / close.shift(2)
    df["pct_return_5"] = (close.shift(1) - close.shift(6)) / close.shift(6)


def _add_rolling_stats(df: pd.DataFrame) -> None:
    cs = df["Close"].shift(1)
    df["rolling_mean_5"]  = cs.rolling(5).mean()
    df["rolling_mean_10"] = cs.rolling(10).mean()
    df["rolling_std_5"]   = cs.rolling(5).std()
    df["rolling_std_10"]  = cs.rolling(10).std()


def _add_volume_features(df: pd.DataFrame) -> None:
    vs = df["Volume"].shift(1)
    avg_vol5           = vs.rolling(5).mean().replace(0, np.nan)
    df["volume_1"]     = vs
    df["volume_change"]= vs / df["Volume"].shift(2) - 1.0
    df["avg_volume_5"] = avg_vol5
    # FIX #3: Volume spike ratio — a clean anomaly signal
    df["volume_spike"] = (vs / avg_vol5).fillna(1.0)


def _add_close_position_features(df: pd.DataFrame) -> None:
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    cp  = (df["Close"] - df["Low"]) / rng
    cs  = cp.shift(1)
    df["close_pos_1"]    = cs
    df["avg_close_pos_5"]= cs.rolling(5).mean()


def _add_calendar_features(df: pd.DataFrame) -> None:
    idx = df.index
    df["day_of_week"]  = idx.dayofweek.astype(np.int16)
    df["week_of_year"] = idx.isocalendar().week.values.astype(np.int16)
    df["month"]        = idx.month.astype(np.int16)
    df["day_of_year"]  = idx.day_of_year.astype(np.int16)


def _add_targets(df: pd.DataFrame) -> None:
    """
    FIX #4: Log-transform the range target.

    WHY: target_range is heavily right-skewed — extreme spike days have ranges
    5–10× the median. RobustScaler partially handles this, but LightGBM still
    optimises MAE on the raw scale, meaning spike days dominate gradients OR
    get under-weighted by quantile splits.

    Solution: predict log(range) instead of range.
    At inference we exp() it back. This compresses the tail enormously and
    makes LightGBM's splits uniform across quiet and volatile regimes.

    target_open and target_closepos are unaffected (open is in price space,
    closepos is already bounded [0,1]).
    """
    df["target_open"] = df["Open"]

    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    # log(range) — add small epsilon for numerical safety
    df["target_log_range"] = np.log(rng + 1e-6)

    df["target_closepos"] = ((df["Close"] - df["Low"]) / rng).clip(0.0, 1.0)
"""
market_calendar.py — NSE market calendar: dynamic holiday fetch + open/closed logic.

Rules (strict):
  - NO hardcoded holidays
  - Fetches from NSE website dynamically
  - Caches result to avoid repeated scraping (file cache + in-process cache)
  - is_market_open() is the single gatekeeper used by predictor and UI
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from src.utils import DATA_DIR, get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Cache file lives in data/
# ─────────────────────────────────────────────────────────────────────────────
_CACHE_FILE = DATA_DIR / "nse_holidays_cache.json"
_MAX_CACHE_AGE_DAYS = 30   # re-fetch at most once a month

# In-process cache so repeated calls within one Streamlit session are free
_MEM_CACHE: dict[int, dict[str, str]] = {}

# NSE holiday calendar URL (CM segment — equity)
_NSE_HOLIDAY_URL = (
    "https://www.nseindia.com/api/holiday-master?type=trading"
)
_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_nse_holidays(year: int) -> dict[str, str]:
    """
    Return a dict of {ISO-date-string: holiday-name} for *year*.

    e.g. {"2026-04-14": "Dr. Ambedkar Jayanti", ...}

    Fetch strategy (in order):
      1. In-process memory cache
      2. File cache (data/nse_holidays_cache.json, max 30 days old)
      3. NSE API (requests + json parse)
      4. Fallback: NSE website HTML scrape
    If all fail, returns {} with a warning — the system never crashes.
    """
    if year in _MEM_CACHE:
        return _MEM_CACHE[year]

    # Try file cache first
    cached = _load_file_cache(year)
    if cached is not None:
        _MEM_CACHE[year] = cached
        return cached

    # Try live fetch
    holidays = _fetch_from_nse_api(year)
    if not holidays:
        holidays = _fetch_from_nse_html(year)

    if holidays:
        _save_file_cache(year, holidays)
        _MEM_CACHE[year] = holidays
        log.info("NSE holidays for %d: %d holidays fetched.", year, len(holidays))
    else:
        log.warning("Could not fetch NSE holidays for %d — treating all weekdays as open.", year)
        holidays = {}
        _MEM_CACHE[year] = holidays

    return holidays


def is_market_open(target_date: date) -> tuple[bool, Optional[str]]:
    """
    Determine if NSE is open on *target_date*.

    Returns
    -------
    (True,  None)            — market is open
    (False, "Weekend")       — Saturday or Sunday
    (False, "<Holiday Name>")— NSE trading holiday
    """
    # Weekend check
    if target_date.weekday() >= 5:   # 5=Saturday, 6=Sunday
        return False, "Weekend"

    # Holiday check
    holidays = get_nse_holidays(target_date.year)
    iso = target_date.isoformat()   # "2026-04-14"
    if iso in holidays:
        return False, holidays[iso]

    return True, None


def get_holidays_for_month(year: int, month: int) -> list[dict]:
    """
    Return a list of {date, name} dicts for holidays in the given month.
    Useful for the UI holiday calendar page.
    """
    all_holidays = get_nse_holidays(year)
    result = []
    for iso_date, name in sorted(all_holidays.items()):
        try:
            d = date.fromisoformat(iso_date)
            if d.year == year and d.month == month:
                result.append({"date": d, "name": name})
        except ValueError:
            continue
    return result


def get_all_holidays_for_year(year: int) -> list[dict]:
    """Return all holidays for the year as sorted list of {date, name}."""
    all_holidays = get_nse_holidays(year)
    result = []
    for iso_date, name in sorted(all_holidays.items()):
        try:
            d = date.fromisoformat(iso_date)
            result.append({
                "date"      : d,
                "name"      : name,
                "day"       : d.strftime("%A"),
            })
        except ValueError:
            continue
    return result


# ─────────────────────────────────────────────────────────────────────────────
# NSE API fetch  (JSON endpoint)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_from_nse_api(year: int) -> dict[str, str]:
    """
    Hit the NSE holiday-master JSON API.
    Returns {iso_date: holiday_name} or {} on failure.
    """
    try:
        import requests

        session = requests.Session()
        # NSE requires a cookie from the main page first
        session.get("https://www.nseindia.com", headers=_NSE_HEADERS, timeout=10)

        resp = session.get(_NSE_HOLIDAY_URL, headers=_NSE_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # The response has a key "CM" (Capital Market) with a list of dicts
        cm_holidays = data.get("CM", [])
        if not cm_holidays:
            # Try top-level list
            cm_holidays = data if isinstance(data, list) else []

        holidays: dict[str, str] = {}
        for item in cm_holidays:
            # Fields vary: tradingDate / date / holidayDate
            raw_date = (
                item.get("tradingDate")
                or item.get("date")
                or item.get("holidayDate")
                or ""
            )
            name = (
                item.get("description")
                or item.get("holidayName")
                or item.get("name")
                or "Holiday"
            )
            iso = _parse_nse_date(raw_date)
            if iso:
                parsed_year = int(iso[:4])
                if parsed_year == year:
                    holidays[iso] = name.strip()

        return holidays

    except Exception as exc:
        log.warning("NSE API fetch failed: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# NSE HTML scrape fallback
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_from_nse_html(year: int) -> dict[str, str]:
    """
    Scrape the NSE holiday page HTML as a fallback.
    Returns {iso_date: holiday_name} or {} on failure.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        url = f"https://www.nseindia.com/market-data/holiday-calendar"
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=_NSE_HEADERS, timeout=10)
        resp = session.get(url, headers=_NSE_HEADERS, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        holidays: dict[str, str] = {}

        # Look for table rows with date patterns
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 2:
                raw_date = cells[0].get_text(strip=True)
                name     = cells[1].get_text(strip=True)
                iso = _parse_nse_date(raw_date)
                if iso:
                    parsed_year = int(iso[:4])
                    if parsed_year == year:
                        holidays[iso] = name

        return holidays

    except Exception as exc:
        log.warning("NSE HTML scrape failed: %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Date parser — handles multiple NSE date formats
# ─────────────────────────────────────────────────────────────────────────────

_DATE_FORMATS = [
    "%d-%b-%Y",   # 14-Apr-2026
    "%d-%b-%y",   # 14-Apr-26
    "%d/%m/%Y",   # 14/04/2026
    "%Y-%m-%d",   # 2026-04-14
    "%d %b %Y",   # 14 Apr 2026
    "%B %d, %Y",  # April 14, 2026
    "%d-%m-%Y",   # 14-04-2026
]


def _parse_nse_date(raw: str) -> Optional[str]:
    """Convert a raw date string to ISO format YYYY-MM-DD, or None on failure."""
    raw = raw.strip()
    if not raw:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# File cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_file_cache(year: int) -> Optional[dict[str, str]]:
    if not _CACHE_FILE.exists():
        return None
    try:
        age_days = (date.today() - date.fromtimestamp(_CACHE_FILE.stat().st_mtime)).days
        if age_days > _MAX_CACHE_AGE_DAYS:
            return None
        with _CACHE_FILE.open() as f:
            all_data = json.load(f)
        return all_data.get(str(year))
    except Exception:
        return None


def _save_file_cache(year: int, holidays: dict[str, str]) -> None:
    try:
        all_data: dict = {}
        if _CACHE_FILE.exists():
            with _CACHE_FILE.open() as f:
                all_data = json.load(f)
        all_data[str(year)] = holidays
        with _CACHE_FILE.open("w") as f:
            json.dump(all_data, f, indent=2)
    except Exception as exc:
        log.warning("Could not write holiday cache: %s", exc)


def invalidate_cache() -> None:
    """Force a fresh fetch next time (used in testing / manual override)."""
    _MEM_CACHE.clear()
    if _CACHE_FILE.exists():
        _CACHE_FILE.unlink()
    log.info("Holiday cache invalidated.")
"""
optuna_tuner.py — Optuna tuning for deep model and LightGBM.

Spec compliance:
  - 20 trials max
  - Huber loss for all LightGBM models
  - min_child_samples tuned (prevents spike-day overfitting)
  - Scalers fitted only on training data (no leakage)
  - Risk model tuned separately
"""

from __future__ import annotations

import numpy as np
import optuna
from optuna.samplers import TPESampler

from src.utils import SEED, get_logger

log = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

DEEP_HP_DEFAULTS = dict(
    lstm_units=64, lstm_layers=1, num_heads=4,
    dropout=0.2, learning_rate=1e-3,
)

LGB_HP_DEFAULTS = dict(
    max_depth=6, num_leaves=50,
    learning_rate=0.05, n_estimators=500,
    subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20,
    objective="huber",
    alpha=0.9,
)

RISK_LGB_DEFAULTS = dict(
    max_depth=4, num_leaves=31,
    learning_rate=0.05, n_estimators=200,
    subsample=0.8, colsample_bytree=0.8,
    min_child_samples=30,
    objective="binary",    # risk is a 0/1 classification target
)


# ─────────────────────────────────────────────────────────────────────────────
# Deep model tuning
# ─────────────────────────────────────────────────────────────────────────────

def tune_deep_model(
    X_seq_train: dict[str, np.ndarray],
    y_open_train: np.ndarray,
    n_trials: int = 20,
    timeout: int = 300,
    val_fraction: float = 0.15,
) -> dict:
    try:
        from src.deep_model import build_hybrid_model, train_deep_model
    except ImportError:
        log.warning("TF unavailable — using deep model defaults.")
        return DEEP_HP_DEFAULTS.copy()

    n_val = max(60, int(len(y_open_train) * val_fraction))
    X_tr  = {k: v[:-n_val] for k, v in X_seq_train.items()}
    X_vl  = {k: v[-n_val:] for k, v in X_seq_train.items()}
    y_tr  = y_open_train[:-n_val]
    y_vl  = y_open_train[-n_val:]
    n_feat = X_seq_train["short"].shape[2]

    def objective(trial: optuna.Trial) -> float:
        hp = dict(
            lstm_units    = trial.suggest_int("lstm_units", 32, 128, step=32),
            lstm_layers   = trial.suggest_int("lstm_layers", 1, 2),
            num_heads     = trial.suggest_categorical("num_heads", [2, 4, 8]),
            dropout       = trial.suggest_float("dropout", 0.1, 0.4),
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        )
        fm, _ = build_hybrid_model(n_features=n_feat, **hp)
        train_deep_model(fm, X_tr, y_tr, epochs=8, batch_size=64,
                         validation_split=0.0, patience=3)
        return float(np.mean(np.abs(fm.predict(X_vl, verbose=0).ravel() - y_vl)))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    log.info("Deep best: %s  val_MAE=%.6f", study.best_params, study.best_value)
    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM OHLC model tuning (Huber)
# ─────────────────────────────────────────────────────────────────────────────

def tune_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target_name: str,
    n_trials: int = 20,
    timeout: int = 120,
    val_fraction: float = 0.15,
) -> dict:
    try:
        import lightgbm as lgb
    except ImportError:
        log.warning("LGB unavailable for %s.", target_name)
        return LGB_HP_DEFAULTS.copy()

    n_val = max(30, int(len(y_train) * val_fraction))
    X_tr  = X_train[:-n_val];  X_vl = X_train[-n_val:]
    y_tr  = y_train[:-n_val];  y_vl = y_train[-n_val:]

    def objective(trial: optuna.Trial) -> float:
        p = dict(
            max_depth        = trial.suggest_int("max_depth", 3, 10),
            num_leaves       = trial.suggest_int("num_leaves", 20, 100),
            learning_rate    = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            n_estimators     = trial.suggest_int("n_estimators", 100, 800),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_samples= trial.suggest_int("min_child_samples", 10, 50),
            alpha            = trial.suggest_float("alpha", 0.7, 0.95),
            objective        = "huber",
            random_state     = SEED,
            verbose          = -1,
        )
        m = lgb.LGBMRegressor(**p)
        m.fit(X_tr, y_tr)
        return float(np.mean(np.abs(m.predict(X_vl) - y_vl)))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    log.info("LGB [%s] best: %s  val_MAE=%.6f", target_name, study.best_params, study.best_value)
    return study.best_params


# ─────────────────────────────────────────────────────────────────────────────
# Risk model tuning (binary classification)
# ─────────────────────────────────────────────────────────────────────────────

def tune_risk_model(
    X_train: np.ndarray,
    y_binary: np.ndarray,
    n_trials: int = 20,
    timeout: int = 60,
    val_fraction: float = 0.15,
) -> dict:
    """Tune LightGBM binary classifier for risk labelling."""
    try:
        import lightgbm as lgb
        from sklearn.metrics import log_loss
    except ImportError:
        return RISK_LGB_DEFAULTS.copy()

    n_val = max(30, int(len(y_binary) * val_fraction))
    X_tr  = X_train[:-n_val];  X_vl = X_train[-n_val:]
    y_tr  = y_binary[:-n_val]; y_vl = y_binary[-n_val:]

    def objective(trial: optuna.Trial) -> float:
        p = dict(
            max_depth        = trial.suggest_int("max_depth", 2, 8),
            num_leaves       = trial.suggest_int("num_leaves", 10, 60),
            learning_rate    = trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            n_estimators     = trial.suggest_int("n_estimators", 50, 300),
            min_child_samples= trial.suggest_int("min_child_samples", 20, 60),
            objective        = "binary",
            random_state     = SEED,
            verbose          = -1,
        )
        m = lgb.LGBMClassifier(**p)
        m.fit(X_tr, y_tr)
        probs = m.predict_proba(X_vl)[:, 1]
        return float(log_loss(y_vl, probs))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    log.info("Risk model best: %s  val_logloss=%.6f", study.best_params, study.best_value)
    return study.best_params
"""
predictor.py — Inference pipeline (upgraded).

Changes vs previous version:
  - Multi-window sequences (30/60/120) fed into embedder
  - Fused vector normalised with fused_scaler before LightGBM
  - Risk score from dedicated LightGBM classifier (spec §6)
  - Wick reconstruction (spec §4):
      High = max(Open, Close) + upper_wick_adj
      Low  = min(Open, Close) - lower_wick_adj
      where wick adjustments are derived from the predicted ATR
  - log-range inverse: range = exp(pred_log_range) - epsilon
"""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_loader import load_ohlcv
from src.feature_engineering import (
    FEATURE_COLS,
    TARGET_COLS,
    build_features,
    calendar_features_for_date,
)
from src.market_calendar import is_market_open
from src.sequence_builder import MAX_SEQ, build_inference_sequence
from src.trainer import load_artifacts
from src.utils import get_logger

log = get_logger(__name__)

CALENDAR_COLS = ["day_of_week", "week_of_year", "month", "day_of_year"]

# Risk label thresholds (spec §6)
_RISK_THRESHOLDS = [
    (0.25, "LOW"),
    (0.55, "MODERATE"),
    (0.80, "HIGH"),
    (1.01, "EXTREME"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def predict(ticker: str, target_date: date) -> dict[str, Any]:
    """
    Unified prediction. Returns one of three shapes:
      market_closed → {"market_closed": True, "reason": ..., "date": ...}
      future        → {"market_closed": False, "mode": "future", OHLC, risk, ...}
      backtest      → future + actual OHLC + error metrics
    """
    is_open, reason = is_market_open(target_date)
    if not is_open:
        return {"market_closed": True, "reason": reason, "date": target_date}

    mode = "future" if target_date >= date.today() else "backtest"

    artifacts    = load_artifacts(ticker)
    embedder     = artifacts["embedder"]
    lgbm_models  = artifacts["lgbm_models"]
    risk_model   = artifacts["risk_model"]
    feat_scaler  = artifacts["feat_scaler"]
    fused_scaler = artifacts["fused_scaler"]
    tgt_scalers  = artifacts["tgt_scalers"]

    df_raw    = load_ohlcv(ticker, use_cache=True)
    df        = build_features(df_raw)
    future_ts = pd.Timestamp(target_date)

    df_past = df[df.index < future_ts]
    if len(df_past) < MAX_SEQ:
        raise ValueError(
            f"Need ≥{MAX_SEQ} rows before {target_date}, have {len(df_past)}."
        )

    # ── Multi-window sequences ────────────────────────────────────────────────
    X_seq, X_flat = build_inference_sequence(df_past.iloc[-MAX_SEQ:], feat_scaler)

    # Override calendar features with target date's values
    cal         = calendar_features_for_date(future_ts)
    cal_indices = [FEATURE_COLS.index(c) for c in CALENDAR_COLS]
    X_flat_adj  = X_flat.copy()
    for name, idx in zip(CALENDAR_COLS, cal_indices):
        X_flat_adj[0, idx] = float(cal[name])

    # ── Deep embedding ─────────────────────────────────────────────────────────
    embedding = embedder.predict(X_seq, verbose=0)     # (1, emb_dim)
    X_fused_raw  = np.hstack([embedding, X_flat_adj])  # (1, emb_dim + n_feat)
    X_fused_norm = fused_scaler.transform(X_fused_raw) # normalised (spec §2)

    # ── Predict OHLC components ───────────────────────────────────────────────
    pred_scaled = {t: float(m.predict(X_fused_norm)[0]) for t, m in lgbm_models.items()}

    open_pred     = _inv(tgt_scalers, "target_open",      pred_scaled["target_open"])
    log_range_raw = _inv(tgt_scalers, "target_log_range", pred_scaled["target_log_range"])
    range_pred    = max(float(np.exp(log_range_raw)) - 1e-6, 0.0)
    closepos_pred = float(np.clip(
        _inv(tgt_scalers, "target_closepos", pred_scaled["target_closepos"]), 0.0, 1.0
    ))

    # ── Wick-aware OHLC reconstruction (spec §4) ──────────────────────────────
    open_pred, high_pred, low_pred, close_pred = _reconstruct_ohlc(
        open_pred, range_pred, closepos_pred, df_past
    )

    prev_close = float(df_past["Close"].iloc[-1])
    pct_change = (close_pred - prev_close) / prev_close * 100.0

    # ── Risk score from dedicated LightGBM classifier (spec §6) ───────────────
    risk_score, risk_label = _score_risk(risk_model, X_fused_norm, df_past)

    result: dict[str, Any] = {
        "market_closed": False,
        "mode"         : mode,
        "date"         : target_date,
        "Open"         : round(open_pred,  2),
        "High"         : round(high_pred,  2),
        "Low"          : round(low_pred,   2),
        "Close"        : round(close_pred, 2),
        "prev_close"   : round(prev_close, 2),
        "pct_change"   : round(pct_change, 4),
        "range"        : round(range_pred, 2),
        "risk_score"   : round(risk_score, 3),
        "risk_label"   : risk_label,
    }

    if mode == "backtest":
        result = _attach_actuals(result, df_raw, future_ts)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Wick-aware OHLC reconstruction  (spec §4)
# ─────────────────────────────────────────────────────────────────────────────

def _reconstruct_ohlc(
    open_pred: float,
    range_pred: float,
    closepos_pred: float,
    df_past: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """
    Spec §4:
      Close = Low + Range × close_position
      High  = max(Open, Close) + upper_wick_adj
      Low   = min(Open, Close) - lower_wick_adj

    Wick adjustments are estimated from the most recent ATR:
      upper_wick_adj = avg_upper_shadow_ratio × range_pred
      lower_wick_adj = avg_lower_shadow_ratio × range_pred

    This produces realistic candles instead of symmetric ones.
    """
    # Compute historical wick ratios from last 20 days
    hist  = df_past.iloc[-20:]
    rng_h = (hist["High"] - hist["Low"]).replace(0, np.nan)
    body_hi = hist[["Open", "Close"]].max(axis=1)
    body_lo = hist[["Open", "Close"]].min(axis=1)
    upper_ratios = ((hist["High"] - body_hi) / rng_h).clip(0, 0.5)
    lower_ratios = ((body_lo - hist["Low"])  / rng_h).clip(0, 0.5)

    upper_wick_adj = float(upper_ratios.median()) * range_pred
    lower_wick_adj = float(lower_ratios.median()) * range_pred

    # Base reconstruction
    low_base   = open_pred - range_pred / 2.0
    high_base  = open_pred + range_pred / 2.0
    close_pred = low_base + range_pred * closepos_pred

    # Apply wick adjustments (spec §4)
    body_top = max(open_pred, close_pred)
    body_bot = min(open_pred, close_pred)
    high_pred = body_top + upper_wick_adj
    low_pred  = body_bot - lower_wick_adj

    # Hard constraints (spec §4): High ≥ Open,Close  /  Low ≤ Open,Close
    high_pred = max(high_pred, open_pred, close_pred)
    low_pred  = min(low_pred,  open_pred, close_pred)

    return open_pred, high_pred, low_pred, close_pred


# ─────────────────────────────────────────────────────────────────────────────
# Risk scoring  (spec §6)
# ─────────────────────────────────────────────────────────────────────────────

def _score_risk(
    risk_model, X_fused_norm: np.ndarray, df_past: pd.DataFrame
) -> tuple[float, str]:
    """
    Use the dedicated LightGBM risk classifier to output a probability score.
    Falls back to the feature-based heuristic if the model is missing.
    """
    if risk_model is not None:
        try:
            if hasattr(risk_model, "predict_proba"):
                score = float(risk_model.predict_proba(X_fused_norm)[0, 1])
            else:
                score = float(np.clip(risk_model.predict(X_fused_norm)[0], 0.0, 1.0))
        except Exception as exc:
            log.warning("Risk model predict failed (%s), using heuristic.", exc)
            score = _heuristic_risk(df_past)
    else:
        score = _heuristic_risk(df_past)

    label = next(lbl for thresh, lbl in _RISK_THRESHOLDS if score < thresh)
    return score, label


def _heuristic_risk(df_past: pd.DataFrame) -> float:
    """Fallback heuristic risk score from last row features."""
    last       = df_past.iloc[-1]
    expansion  = float(last.get("range_expansion_ratio", 1.0))
    vol_spike  = float(last.get("volume_spike",          1.0))
    z          = abs(float(last.get("range_z_score_10",  0.0)))
    score = 0.45 * min((expansion - 1.0) / 2.0, 1.0) + \
            0.25 * min((vol_spike  - 1.0) / 3.0, 1.0) + \
            0.30 * min(z / 3.0, 1.0)
    return float(np.clip(score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Backtest helper
# ─────────────────────────────────────────────────────────────────────────────

def _attach_actuals(result: dict, df_raw: pd.DataFrame, ts: pd.Timestamp) -> dict:
    rows = df_raw[df_raw.index == ts]
    if rows.empty:
        result.update({
            "actual_Open": None, "actual_High": None,
            "actual_Low": None,  "actual_Close": None,
            "mae": None, "rmse": None, "data_available": False,
        })
        return result

    r  = rows.iloc[0]
    aO, aH, aL, aC = float(r["Open"]), float(r["High"]), float(r["Low"]), float(r["Close"])
    predicted = [result["Open"], result["High"], result["Low"], result["Close"]]
    actuals   = [aO, aH, aL, aC]
    result.update({
        "actual_Open"  : round(aO, 2),
        "actual_High"  : round(aH, 2),
        "actual_Low"   : round(aL, 2),
        "actual_Close" : round(aC, 2),
        "mae"          : round(mean_absolute_error(actuals, predicted), 4),
        "rmse": round(np.sqrt(mean_squared_error(actuals, predicted)), 4),
        "err_open"     : round(abs(result["Open"]  - aO), 2),
        "err_high"     : round(abs(result["High"]  - aH), 2),
        "err_low"      : round(abs(result["Low"]   - aL), 2),
        "err_close"    : round(abs(result["Close"] - aC), 2),
        "data_available": True,
    })
    return result


def _inv(tgt_scalers: dict, key: str, value: float) -> float:
    return float(tgt_scalers[key].inverse_transform([[value]])[0, 0])
"""
sequence_builder.py — Multi-window sequence builder (30 / 60 / 120 days).

Zero leakage: for each sample i, the window ending at row (i + SEQ_LONG - 1)
feeds into all three branch inputs; the target is row (i + SEQ_LONG).

Why three windows?
  SEQ_SHORT (30d) → LSTM captures recent momentum / microstructure
  SEQ_MID   (60d) → LSTM captures monthly seasonality
  SEQ_LONG (120d) → Transformer captures regime shifts / macro trends

At inference, we build each window from the latest available rows.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from src.feature_engineering import FEATURE_COLS, TARGET_COLS
from src.utils import SEQ_LONG, SEQ_MID, SEQ_SHORT, get_logger

log = get_logger(__name__)

# Alias so callers can import a single "max window" constant
MAX_SEQ = SEQ_LONG


# ─────────────────────────────────────────────────────────────────────────────
# Training-time builder
# ─────────────────────────────────────────────────────────────────────────────

def build_sequences(
    df: pd.DataFrame,
    feature_scaler: RobustScaler | None = None,
    target_scalers: dict[str, RobustScaler] | None = None,
    fit_scalers: bool = True,
) -> Tuple[
    dict[str, np.ndarray],     # X_seq: {"short": (N,30,F), "mid": (N,60,F), "long": (N,120,F)}
    np.ndarray,                # X_flat (N, F)  — last step of long window, scaled
    dict[str, np.ndarray],     # y_dict
    RobustScaler,              # feature_scaler
    dict[str, RobustScaler],   # target_scalers
    pd.DatetimeIndex,          # dates aligned with each sample
]:
    n_rows = len(df)
    if n_rows < MAX_SEQ + 1:
        raise ValueError(
            f"Only {n_rows} rows — need at least {MAX_SEQ + 1} for multi-window sequences."
        )

    feature_arr = df[FEATURE_COLS].values.astype(np.float32)

    # ── Fit / apply feature scaler ────────────────────────────────────────────
    if feature_scaler is None:
        feature_scaler = RobustScaler()
    if fit_scalers:
        fa = feature_scaler.fit_transform(feature_arr)
    else:
        fa = feature_scaler.transform(feature_arr)

    # ── Fit / apply target scalers ────────────────────────────────────────────
    if target_scalers is None:
        target_scalers = {c: RobustScaler() for c in TARGET_COLS}
    y_raw    = {c: df[c].values.astype(np.float32).reshape(-1, 1) for c in TARGET_COLS}
    y_scaled = {}
    for c in TARGET_COLS:
        if fit_scalers:
            y_scaled[c] = target_scalers[c].fit_transform(y_raw[c]).ravel()
        else:
            y_scaled[c] = target_scalers[c].transform(y_raw[c]).ravel()

    # ── Build three windows aligned on the same sample set ────────────────────
    n_samples  = n_rows - MAX_SEQ
    n_features = fa.shape[1]

    X_short = np.zeros((n_samples, SEQ_SHORT, n_features), dtype=np.float32)
    X_mid   = np.zeros((n_samples, SEQ_MID,   n_features), dtype=np.float32)
    X_long  = np.zeros((n_samples, SEQ_LONG,  n_features), dtype=np.float32)
    X_flat  = np.zeros((n_samples, n_features),            dtype=np.float32)

    for i in range(n_samples):
        end = i + MAX_SEQ          # exclusive end; row at `end` is the target
        X_long[i]  = fa[end - SEQ_LONG : end]
        X_mid[i]   = fa[end - SEQ_MID  : end]
        X_short[i] = fa[end - SEQ_SHORT: end]
        X_flat[i]  = fa[end - 1]   # latest row before the target

    X_seq = {"short": X_short, "mid": X_mid, "long": X_long}

    # Targets: row at index MAX_SEQ onwards
    y_dict = {c: y_scaled[c][MAX_SEQ:] for c in TARGET_COLS}
    dates  = df.index[MAX_SEQ:]

    log.info(
        "build_sequences: short%s mid%s long%s flat%s | %d samples",
        X_short.shape, X_mid.shape, X_long.shape, X_flat.shape, n_samples,
    )
    return X_seq, X_flat, y_dict, feature_scaler, target_scalers, dates


# ─────────────────────────────────────────────────────────────────────────────
# Inference-time builder (one sample from latest data)
# ─────────────────────────────────────────────────────────────────────────────

def build_inference_sequence(
    df: pd.DataFrame,
    feature_scaler: RobustScaler,
) -> Tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Build one multi-window sample from the last MAX_SEQ rows of *df*.

    Returns
    -------
    X_seq : {"short": (1,30,F), "mid": (1,60,F), "long": (1,120,F)}
    X_flat: (1, F)
    """
    if len(df) < MAX_SEQ:
        raise ValueError(f"Need ≥{MAX_SEQ} rows for inference, got {len(df)}.")

    fa = feature_scaler.transform(
        df[FEATURE_COLS].values[-MAX_SEQ:].astype(np.float32)
    )

    return (
        {
            "short": fa[np.newaxis, -SEQ_SHORT:, :],
            "mid"  : fa[np.newaxis, -SEQ_MID:,   :],
            "long" : fa[np.newaxis, :,            :],
        },
        fa[-1:, :],   # X_flat
    )
"""
trainer.py — Full training pipeline (upgraded).

Steps:
  1. Feature engineering
  2. Multi-window sequence building (30 / 60 / 120 days)
  3. Walk-forward validation (evaluation only — no leakage)
  4. Optuna tuning: deep model
  5. Train deep model on full data → extract embeddings
  6. Fuse embeddings + scaled tabular features
  7. Optuna tuning + train 3 LightGBM OHLC models
  8. Optuna tuning + train separate LightGBM risk model (spec §6)
  9. Save all artifacts

Spec constraints enforced:
  - No random splits (walk-forward only)
  - Scalers fitted only on train fold data (no leakage)
  - Transformer mandatory (inside build_hybrid_model)
  - Risk model is a separate LightGBM (§6)
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler

from src.deep_model import (
    build_hybrid_model,
    extract_embeddings,
    train_deep_model,
)
from src.feature_engineering import FEATURE_COLS, TARGET_COLS, build_features
from src.optuna_tuner import (
    DEEP_HP_DEFAULTS,
    LGB_HP_DEFAULTS,
    RISK_LGB_DEFAULTS,
    tune_deep_model,
    tune_lgbm,
    tune_risk_model,
)
from src.sequence_builder import MAX_SEQ, build_sequences
from src.utils import (
    SEED,
    deep_model_path,
    embedder_path,
    get_logger,
    lgbm_path,
    risk_model_path,
    scaler_path,
)
from src.walk_forward import generate_folds

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Feature columns used in the fused vector (tabular part)
# ─────────────────────────────────────────────────────────────────────────────
_RISK_FEAT_COLS = [
    "atr_14", "volume_spike", "range_expansion_ratio",
    "range_z_score_10", "range_1", "avg_range_10",
]


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def train(
    ticker: str,
    df_raw: pd.DataFrame,
    use_optuna: bool = True,
    optuna_trials: int = 20,
    deep_epochs: int = 20,
    embedding_dim: int = 64,
    run_walk_forward: bool = True,
) -> dict[str, Any]:
    from src.utils import set_global_seed
    set_global_seed(SEED)

    log.info("[%s] Step 1 — Feature engineering", ticker)
    df = build_features(df_raw)

    log.info("[%s] Step 2 — Multi-window sequence building", ticker)
    X_seq, X_flat, y_dict, feat_scaler, tgt_scalers, dates = build_sequences(
        df, fit_scalers=True
    )

    log.info("[%s] Step 3 — Walk-forward validation", ticker)
    fold_metrics: list[dict] = []
    if run_walk_forward:
        fold_metrics = _walk_forward_eval(df, feat_scaler, tgt_scalers)

    log.info("[%s] Step 4 — Tune deep model", ticker)
    best_deep_hp = DEEP_HP_DEFAULTS.copy()
    if use_optuna:
        try:
            best_deep_hp = tune_deep_model(
                X_seq, y_dict["target_open"], n_trials=optuna_trials
            )
        except Exception as exc:
            log.warning("Deep tuning failed (%s) — using defaults.", exc)

    log.info("[%s] Step 5 — Train deep model", ticker)
    full_model, embedder = build_hybrid_model(
        n_features    = X_seq["short"].shape[2],
        embedding_dim = embedding_dim,
        lstm_units    = best_deep_hp.get("lstm_units",    DEEP_HP_DEFAULTS["lstm_units"]),
        lstm_layers   = best_deep_hp.get("lstm_layers",   DEEP_HP_DEFAULTS["lstm_layers"]),
        num_heads     = best_deep_hp.get("num_heads",     DEEP_HP_DEFAULTS["num_heads"]),
        dropout       = best_deep_hp.get("dropout",       DEEP_HP_DEFAULTS["dropout"]),
        learning_rate = best_deep_hp.get("learning_rate", DEEP_HP_DEFAULTS["learning_rate"]),
    )
    train_deep_model(full_model, X_seq, y_dict["target_open"], epochs=deep_epochs)

    log.info("[%s] Step 6 — Extract embeddings + fuse features", ticker)
    embeddings = extract_embeddings(embedder, X_seq)      # (N, emb_dim)
    X_fused    = np.hstack([embeddings, X_flat])          # (N, emb_dim + n_feat)

    # Normalise fused vector (spec §2)
    fused_scaler = RobustScaler()
    X_fused_norm = fused_scaler.fit_transform(X_fused)

    log.info("[%s] Step 7 — Train LightGBM OHLC models", ticker)
    lgbm_models  = {}
    best_lgbm_hp = {}

    for target in TARGET_COLS:
        y_scaled = y_dict[target]
        lgb_hp   = LGB_HP_DEFAULTS.copy()
        if use_optuna:
            try:
                lgb_hp = tune_lgbm(
                    X_fused_norm, y_scaled, target_name=target, n_trials=optuna_trials
                )
            except Exception as exc:
                log.warning("LGB tuning failed for %s (%s).", target, exc)
        best_lgbm_hp[target] = lgb_hp
        lgbm_models[target]  = _fit_lgbm_regressor(X_fused_norm, y_scaled, lgb_hp)
        log.info("  LightGBM [%s] trained.", target)

    log.info("[%s] Step 8 — Train risk model (LightGBM §6)", ticker)
    risk_model, risk_hp = _train_risk_model(
        df, X_fused_norm, use_optuna, optuna_trials
    )

    log.info("[%s] Step 9 — Save all artifacts", ticker)
    _save_artifacts(
        ticker, full_model, embedder, lgbm_models, risk_model,
        feat_scaler, tgt_scalers, fused_scaler,
    )

    log.info("[%s] Training complete.", ticker)
    return {
        "fold_metrics"   : fold_metrics,
        "best_deep_hp"   : best_deep_hp,
        "best_lgbm_hp"   : best_lgbm_hp,
        "risk_hp"        : risk_hp,
        "embedding_dim"  : embedding_dim,
        "n_features"     : X_seq["short"].shape[2],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Risk model training (spec §6)
# ─────────────────────────────────────────────────────────────────────────────

def _train_risk_model(
    df: pd.DataFrame,
    X_fused_norm: np.ndarray,
    use_optuna: bool,
    optuna_trials: int,
):
    """
    Build and train the risk LightGBM model.

    Label: a day is "high risk" if range_expansion_ratio > 1.5 OR
           volume_spike > 2.0  (both purely backward-looking).
    The model outputs P(high_risk) — a probability in [0, 1].

    At inference this is directly the risk_score (spec §6).
    """
    # Align risk labels with the fused feature samples
    # X_fused is indexed from MAX_SEQ onwards (same as y_dict)
    risk_df = df.iloc[MAX_SEQ:].copy()

    expansion   = risk_df.get("range_expansion_ratio", pd.Series(1.0, index=risk_df.index))
    vol_spike   = risk_df.get("volume_spike",           pd.Series(1.0, index=risk_df.index))
    y_risk      = ((expansion > 1.5) | (vol_spike > 2.0)).astype(int).values

    if y_risk.sum() < 10:
        log.warning("Very few positive risk labels (%d) — using default risk params.", y_risk.sum())

    risk_hp = RISK_LGB_DEFAULTS.copy()
    if use_optuna:
        try:
            risk_hp = tune_risk_model(
                X_fused_norm, y_risk, n_trials=optuna_trials
            )
        except Exception as exc:
            log.warning("Risk model tuning failed (%s).", exc)

    try:
        import lightgbm as lgb
        p = {**RISK_LGB_DEFAULTS, **risk_hp, "random_state": SEED, "verbose": -1}
        risk_model = lgb.LGBMClassifier(**p)
    except ImportError:
        from sklearn.linear_model import LogisticRegression
        risk_model = LogisticRegression(max_iter=500)

    risk_model.fit(X_fused_norm, y_risk)
    log.info("Risk model trained. Positive class rate: %.1f%%", y_risk.mean() * 100)
    return risk_model, risk_hp


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward evaluation (leakage-free)
# ─────────────────────────────────────────────────────────────────────────────

def _walk_forward_eval(
    df: pd.DataFrame,
    feat_scaler: RobustScaler,
    tgt_scalers: dict,
) -> list[dict]:
    from src.sequence_builder import build_sequences

    folds   = generate_folds(df)
    metrics = []

    for fold in folds:
        train_df = df.iloc[fold.train_idx]
        test_df  = df.iloc[np.concatenate([fold.train_idx, fold.test_idx])]

        if len(train_df) < MAX_SEQ + 1:
            log.warning("Fold %d: insufficient train rows — skip.", fold.fold_id)
            continue

        # Fit scalers only on train data (no leakage)
        try:
            X_tr_seq, X_tr_flat, y_tr, fs, ts, _ = build_sequences(
                train_df, fit_scalers=True
            )
            X_te_seq, X_te_flat, y_te, _, _, _   = build_sequences(
                test_df, feature_scaler=fs, target_scalers=ts, fit_scalers=False
            )
        except ValueError as e:
            log.warning("Fold %d build failed: %s", fold.fold_id, e)
            continue

        # Quick deep model (minimal epochs for speed)
        n_feat = X_tr_seq["short"].shape[2]
        try:
            fm, emb = build_hybrid_model(n_features=n_feat, embedding_dim=32)
            y_dummy = np.zeros(len(X_tr_seq["short"]), dtype=np.float32)
            train_deep_model(fm, X_tr_seq, y_dummy, epochs=3, validation_split=0.0)
            emb_tr = extract_embeddings(emb, X_tr_seq)
            emb_te = extract_embeddings(emb, X_te_seq)
        except Exception:
            emb_tr = np.zeros((len(X_tr_seq["short"]), 32), np.float32)
            emb_te = np.zeros((len(X_te_seq["short"]), 32), np.float32)

        # Only test rows (not the train prefix we added for context)
        n_train_samples = len(X_tr_seq["short"])
        emb_te_only = emb_te[n_train_samples:]
        Xf_te_only  = X_te_flat[n_train_samples:]

        X_tr_f = np.hstack([emb_tr, X_tr_flat])
        X_te_f = np.hstack([emb_te_only, Xf_te_only])

        fold_mae = {}
        for tgt in TARGET_COLS:
            m  = _fit_lgbm_regressor(X_tr_f, y_tr[tgt], LGB_HP_DEFAULTS)
            pr = m.predict(X_te_f)
            # Inverse-transform for interpretable MAE
            y_te_orig = ts[tgt].inverse_transform(y_te[tgt][n_train_samples:].reshape(-1, 1)).ravel()
            pr_orig   = ts[tgt].inverse_transform(pr.reshape(-1, 1)).ravel()
            fold_mae[tgt] = float(mean_absolute_error(y_te_orig, pr_orig))

        entry = {"fold_id": fold.fold_id, "train_end": fold.train_end,
                 "test_year": fold.test_year, "mae": fold_mae}
        metrics.append(entry)
        log.info("Fold %d (test=%d): %s", fold.fold_id, fold.test_year,
                 {k: f"{v:.4f}" for k, v in fold_mae.items()})

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fit_lgbm_regressor(X: np.ndarray, y: np.ndarray, hp: dict):
    try:
        import lightgbm as lgb
        p = {**LGB_HP_DEFAULTS, **hp, "random_state": SEED, "verbose": -1}
        # Ensure regression keys only
        p.pop("objective", None)
        model = lgb.LGBMRegressor(objective="huber", **p)
    except ImportError:
        from sklearn.linear_model import Ridge
        model = Ridge()
    model.fit(X, y)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────────────────────────────

def _save_artifacts(
    ticker, full_model, embedder, lgbm_models, risk_model,
    feat_scaler, tgt_scalers, fused_scaler,
) -> None:
    try:
        full_model.save(str(deep_model_path(ticker)))
        embedder.save(str(embedder_path(ticker)))
        log.info("Deep models saved.")
    except Exception as exc:
        log.warning("Could not save Keras models: %s", exc)

    for tgt, m in lgbm_models.items():
        joblib.dump(m, lgbm_path(ticker, tgt))
    joblib.dump(risk_model, risk_model_path(ticker))
    joblib.dump(feat_scaler, scaler_path(ticker, "features"))
    joblib.dump(fused_scaler, scaler_path(ticker, "fused"))
    for tgt, sc in tgt_scalers.items():
        joblib.dump(sc, scaler_path(ticker, tgt))
    log.info("All artifacts saved for %s.", ticker)


def load_artifacts(ticker: str) -> dict:
    """
    Load all saved artifacts. Raises FileNotFoundError if not trained.
    """
    import tensorflow as tf

    ep = embedder_path(ticker)
    if not ep.exists():
        raise FileNotFoundError(f"Embedder not found: {ep}. Run training first.")

    embedder     = tf.keras.models.load_model(str(ep))
    feat_scaler  = joblib.load(scaler_path(ticker, "features"))
    fused_scaler = joblib.load(scaler_path(ticker, "fused"))
    tgt_scalers  = {t: joblib.load(scaler_path(ticker, t)) for t in TARGET_COLS}
    lgbm_models  = {t: joblib.load(lgbm_path(ticker, t))  for t in TARGET_COLS}

    rp = risk_model_path(ticker)
    risk_model = joblib.load(rp) if rp.exists() else None

    log.info("Artifacts loaded for %s.", ticker)
    return {
        "embedder"    : embedder,
        "lgbm_models" : lgbm_models,
        "risk_model"  : risk_model,
        "feat_scaler" : feat_scaler,
        "fused_scaler": fused_scaler,
        "tgt_scalers" : tgt_scalers,
    }
"""
utils.py — Shared utilities: logging, seeds, constants, path helpers.

UPGRADED: Multi-window sequence lengths (30 / 60 / 120 days),
          risk model path helper, wicks model path helper.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Project layout
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SEED        : int = 42
FETCH_YEARS : int = 15

# Multi-window sequence lengths (short / mid / long term)
SEQ_SHORT : int = 30
SEQ_MID   : int = 60
SEQ_LONG  : int = 120
SEQUENCE_LEN = SEQ_SHORT     # kept for legacy callers (inference uses longest)

# Minimum rows needed before any sample can be built
MIN_HISTORY : int = SEQ_LONG + 1

WALK_FORWARD_FOLDS = [
    {"train_end": 2018, "test_year": 2019},
    {"train_end": 2019, "test_year": 2020},
    {"train_end": 2020, "test_year": 2021},
]

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h   = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────
def _safe(ticker: str) -> str:
    return ticker.replace(".", "_").replace("^", "").replace("/", "_")

def model_path(ticker: str, suffix: str) -> Path:
    return MODELS_DIR / f"{_safe(ticker)}_{suffix}"

def scaler_path(ticker: str, name: str) -> Path:
    return model_path(ticker, f"scaler_{name}.joblib")

def lgbm_path(ticker: str, target: str) -> Path:
    return model_path(ticker, f"lgbm_{target}.joblib")

def risk_model_path(ticker: str) -> Path:
    return model_path(ticker, "lgbm_risk.joblib")

def deep_model_path(ticker: str) -> Path:
    return model_path(ticker, "deep_model.keras")

def embedder_path(ticker: str) -> Path:
    return model_path(ticker, "embedder.keras")
"""
walk_forward.py — Strict time-ordered walk-forward cross-validation.

Folds:
  Train 2010–2018 → Test 2019
  Train 2010–2019 → Test 2020
  Train 2010–2020 → Test 2021

No data leakage: test set is NEVER seen during training or scaling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

from src.utils import WALK_FORWARD_FOLDS, get_logger

log = get_logger(__name__)


@dataclass
class WalkForwardFold:
    fold_id: int
    train_end: int              # last training year (inclusive)
    test_year: int
    train_idx: np.ndarray       # integer positions in df
    test_idx: np.ndarray


def generate_folds(df: pd.DataFrame) -> list[WalkForwardFold]:
    """
    Generate WalkForwardFold objects for each entry in WALK_FORWARD_FOLDS.
    Rows outside all configured test years are silently skipped
    (we don't fail if the data doesn't reach 2021).

    Parameters
    ----------
    df : pd.DataFrame with DatetimeIndex

    Returns
    -------
    List of WalkForwardFold (may be shorter than WALK_FORWARD_FOLDS if data
    doesn't cover the test year).
    """
    folds = []

    for fold_id, spec in enumerate(WALK_FORWARD_FOLDS):
        train_end  = spec["train_end"]
        test_year  = spec["test_year"]

        train_mask = df.index.year <= train_end
        test_mask  = df.index.year == test_year

        if not train_mask.any():
            log.warning("Fold %d: no training data up to %d — skipping.", fold_id, train_end)
            continue
        if not test_mask.any():
            log.warning("Fold %d: no data for test year %d — skipping.", fold_id, test_year)
            continue

        folds.append(WalkForwardFold(
            fold_id   = fold_id,
            train_end = train_end,
            test_year = test_year,
            train_idx = np.where(train_mask)[0],
            test_idx  = np.where(test_mask)[0],
        ))
        log.info(
            "Fold %d | train: %d rows (up to %d) | test: %d rows (%d)",
            fold_id, train_mask.sum(), train_end, test_mask.sum(), test_year,
        )

    if not folds:
        raise ValueError(
            "No valid walk-forward folds could be constructed. "
            "Check that your data covers the years in WALK_FORWARD_FOLDS."
        )

    return folds
"""
train.py — CLI training script.

Usage:
    python train.py --ticker RELIANCE.NS
    python train.py --ticker RELIANCE.NS --no-optuna --epochs 10
    python train.py --ticker AAPL --trials 30 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sure src/ is importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_ohlcv
from src.trainer import train
from src.utils import MODELS_DIR, get_logger, set_global_seed

log = get_logger("train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NSE/Global stock OHLC predictor.")
    p.add_argument("--ticker",     required=True,       help="yfinance ticker, e.g. RELIANCE.NS")
    p.add_argument("--no-optuna",  action="store_true", help="Skip Optuna tuning (use defaults)")
    p.add_argument("--trials",     type=int, default=20, help="Optuna trials per model (default 20)")
    p.add_argument("--epochs",     type=int, default=20, help="Deep model max epochs (default 20)")
    p.add_argument("--emb-dim",    type=int, default=64, help="Embedding dimension (default 64)")
    p.add_argument("--no-wf",      action="store_true", help="Skip walk-forward evaluation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed()

    log.info("=== Training pipeline START === ticker=%s", args.ticker)

    # 1. Load data
    log.info("Loading OHLCV data for %s…", args.ticker)
    df_raw = load_ohlcv(args.ticker)

    # 2. Run training
    result = train(
        ticker          = args.ticker,
        df_raw          = df_raw,
        use_optuna      = not args.no_optuna,
        optuna_trials   = args.trials,
        deep_epochs     = args.epochs,
        embedding_dim   = args.emb_dim,
        run_walk_forward= not args.no_wf,
    )

    # 3. Print summary
    log.info("=== Training pipeline END ===")
    print("\n── Walk-Forward Metrics ──")
    for fold in result["fold_metrics"]:
        print(
            f"  Fold {fold['fold_id']} (test {fold['test_year']}) | "
            + " | ".join(f"{k.replace('target_','')}: {v:.4f}" for k, v in fold["mae"].items())
        )

    print("\n── Best Deep HPs ──")
    print(json.dumps(result["best_deep_hp"], indent=2))

    print("\n── Best LightGBM HPs ──")
    for tgt, hp in result["best_lgbm_hp"].items():
        print(f"  [{tgt}]  {json.dumps(hp)}")

    print(f"\nModels saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
