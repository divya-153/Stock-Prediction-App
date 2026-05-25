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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — env vars must be set by the shell

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Market V2 fix 2",
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

    # Inline HTML table with red cell highlighting for errors > ₹20
    GAP_THRESHOLD = 20.0

    rows_html = ""
    field_errors: dict[str, float] = {}
    for field in ["Open", "High", "Low", "Close"]:
        p   = pred[field]
        a   = pred[f"actual_{field}"]
        e   = pred.get(f"err_{field.lower()}", abs(p - a))
        pct = e / a * 100 if a else 0
        field_errors[field] = e

        gap_badge = "✅ Gap ≤ ₹20" if int(e) <= GAP_THRESHOLD else "❌ Gap > ₹20"
        gap_color = "#4caf82" if int(e) <= GAP_THRESHOLD else "#e05c5c"
        err_color = "#e05c5c" if int(e) > GAP_THRESHOLD else "#d4dae8"

        rows_html += f"""
        <tr>
          <td style="font-family:'IBM Plex Mono',monospace;padding:8px 12px;color:#d4dae8;border-bottom:1px solid #1e2a3a">{field}</td>
          <td style="font-family:'IBM Plex Mono',monospace;padding:8px 12px;color:{'#e05c5c' if int(e) > GAP_THRESHOLD else '#d4dae8'};border-bottom:1px solid #1e2a3a">₹{p:,.2f}</td>
          <td style="font-family:'IBM Plex Mono',monospace;padding:8px 12px;color:{'#e05c5c' if int(e) > GAP_THRESHOLD else '#d4dae8'};border-bottom:1px solid #1e2a3a">₹{a:,.2f}</td>
          <td style="font-family:'IBM Plex Mono',monospace;padding:8px 12px;color:{err_color};font-weight:{'700' if int(e) > GAP_THRESHOLD else '400'};border-bottom:1px solid #1e2a3a">₹{e:,.2f}</td>
          <td style="font-family:'IBM Plex Mono',monospace;padding:8px 12px;color:{err_color};border-bottom:1px solid #1e2a3a">{pct:.2f}%</td>
          <td style="font-family:'IBM Plex Mono',monospace;padding:8px 12px;border-bottom:1px solid #1e2a3a;color:{gap_color}">{gap_badge}</td>
        </tr>"""

    table_html = f"""
    <table style="width:100%;border-collapse:collapse;background:#0e1620;border-radius:8px;overflow:hidden;">
      <thead>
        <tr style="background:#0a1018;">
          <th style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#5a7393;text-transform:uppercase;letter-spacing:0.1em;padding:10px 12px;text-align:left;border-bottom:1px solid #1e2a3a">Field</th>
          <th style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#5a7393;text-transform:uppercase;letter-spacing:0.1em;padding:10px 12px;text-align:left;border-bottom:1px solid #1e2a3a">Predicted</th>
          <th style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#5a7393;text-transform:uppercase;letter-spacing:0.1em;padding:10px 12px;text-align:left;border-bottom:1px solid #1e2a3a">Actual</th>
          <th style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#5a7393;text-transform:uppercase;letter-spacing:0.1em;padding:10px 12px;text-align:left;border-bottom:1px solid #1e2a3a">Abs Error</th>
          <th style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#5a7393;text-transform:uppercase;letter-spacing:0.1em;padding:10px 12px;text-align:left;border-bottom:1px solid #1e2a3a">Error %</th>
          <th style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#5a7393;text-transform:uppercase;letter-spacing:0.1em;padding:10px 12px;text-align:left;border-bottom:1px solid #1e2a3a">Gap Badge</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


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
            except FileNotFoundError as fnf:
                st.warning(
                    f"⚠ No trained model found for **{ticker}**.\n\n"
                    f"Run this in your project root:\n"
                    f"```\npython train.py --ticker {ticker}\n```\n\n"
                    f"Details: `{fnf}`"
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
                "Predicted range expanded by 25–35%. Error may still be larger than typical."
            )

        # Event shock warning — shown regardless of risk level.
        if pred.get("event_shock") == 1:
            shock_reason = pred.get("event_shock_reason", "")
            if not shock_reason:
                shock_reason = "Unclassified macro / geopolitical event"
            st.error(f"🚨 **Event shock detected:** {shock_reason}")

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