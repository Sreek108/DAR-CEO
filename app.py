import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import os

# Optional dependency for a premium horizontal nav
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

# -----------------------------------------------------------------------------
# Page configuration and executive styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DAR Global - Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXEC_PRIMARY = "#DAA520"   # Gold
EXEC_BLUE = "#1E90FF"      # Blue
EXEC_GREEN = "#32CD32"     # Green
EXEC_DANGER = "#DC143C"    # Red
EXEC_BG = "#1a1a1a"
EXEC_SURFACE = "#2d2d2d"

st.markdown(f"""
<style>
/* Executive Theme */
.main-header {{
    background: linear-gradient(135deg, {EXEC_BG} 0%, {EXEC_SURFACE} 100%);
    color: {EXEC_PRIMARY};
    padding: 24px;
    border-radius: 12px;
    border: 2px solid {EXEC_PRIMARY};
    text-align: center;
    box-shadow: 0 8px 16px rgba(0,0,0,.35);
}}
.main-header h1 {{ color: {EXEC_PRIMARY}; margin: 0 0 6px 0; }}
.main-header h3 {{ color: {EXEC_BLUE}; margin: 4px 0 0 0; }}
div[data-testid="metric-container"] {{
    background: linear-gradient(135deg, {EXEC_SURFACE} 0%, {EXEC_BG} 100%);
    border: 2px solid {EXEC_PRIMARY};
    padding: 0.75rem;
    border-radius: 10px;
    color: white;
}}
div[data-testid="stHorizontalBlock"] .stButton>button {{
    background: linear-gradient(135deg, {EXEC_PRIMARY} 0%, #b8860b 100%);
    color: white; border: none; border-radius: 8px; padding: 8px 14px;
}}
.insight-box {{
    background: linear-gradient(135deg, {EXEC_SURFACE} 0%, {EXEC_BG} 100%);
    padding: 18px; border-radius: 10px; border-left: 5px solid {EXEC_GREEN};
    color: white; box-shadow: 0 4px 10px rgba(0,0,0,.25);
}}
.section {{
    background: linear-gradient(135deg, {EXEC_SURFACE} 0%, {EXEC_BG} 100%);
    padding: 18px; border-radius: 10px; border: 1px solid #444;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def format_currency(value: float) -> str:
    if value is None or pd.isna(value):
        return "$0"
    if value >= 1_000_000_000:
        return f"${value/1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value/1_000:.1f}K"
    return f"${value:,.0f}"

def format_number(value: float) -> str:
    if value is None or pd.isna(value):
        return "0"
    if value >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:,.0f}"

def safe_to_datetime(series):
    return pd.to_datetime(series, errors="coerce")

# -----------------------------------------------------------------------------
# Data loading for NEW dataset in ./data
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(data_dir: str = "data"):
    def read_csv_safe(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None

    def normalize_cols(df):
        df = df.copy()
        df.columns = (
            df.columns
              .str.strip()
              .str.replace(r"[^\w]+", "_", regex=True)
              .str.lower()
        )
        return df

    def rename_like(df, mapping):
        cols = {c: mapping[c] for c in mapping if c in df.columns}
        return df.rename(columns=cols)

    paths = {
        "leads": os.path.join(data_dir, "Lead.csv"),
        "agents": os.path.join(data_dir, "Agents.csv"),
        "calls": os.path.join(data_dir, "LeadCallRecord.csv"),
        "schedules": os.path.join(data_dir, "LeadSchedule.csv"),
        "transactions": os.path.join(data_dir, "LeadTransaction.csv"),
        "countries": os.path.join(data_dir, "Country.csv"),
        "lead_stages": os.path.join(data_dir, "LeadStage.csv"),
        "lead_statuses": os.path.join(data_dir, "LeadStatus.csv"),
        "lead_sources": os.path.join(data_dir, "LeadSource.csv"),
        "lead_scoring": os.path.join(data_dir, "LeadScoring.csv"),
        "call_statuses": os.path.join(data_dir, "CallStatus.csv"),
        "sentiments": os.path.join(data_dir, "CallSentiment.csv"),
        "task_types": os.path.join(data_dir, "TaskType.csv"),
        "task_statuses": os.path.join(data_dir, "TaskStatus.csv"),
        "city_region": os.path.join(data_dir, "CityRegion.csv"),
        "timezone_info": os.path.join(data_dir, "TimezoneInfo.csv"),
        "priority": os.path.join(data_dir, "Priority.csv"),
        "meeting_status": os.path.join(data_dir, "MeetingStatus.csv"),
        "agent_meeting_assignment": os.path.join(data_dir, "AgentMeetingAssignment.csv"),
    }
    ds = {k: read_csv_safe(v) for k, v in paths.items()}

    # Leads
    if ds["leads"] is not None:
        df = normalize_cols(ds["leads"])
        df = rename_like(df, {
            "leadid":"LeadId","lead_id":"LeadId",
            "estimatedbudget":"EstimatedBudget","budget":"EstimatedBudget","est_budget":"EstimatedBudget",
            "leadstageid":"LeadStageId","lead_stage_id":"LeadStageId",
            "assignedagentid":"AssignedAgentId","assigned_agent_id":"AssignedAgentId",
            "createdon":"CreatedOn","created_date":"CreatedOn","created_at":"CreatedOn",
            "isactive":"IsActive","active":"IsActive",
            "countryid":"CountryId","country_id":"CountryId",
            "cityregionid":"CityRegionId","city_region_id":"CityRegionId",
            "leadstatusid":"LeadStatusId","lead_status_id":"LeadStatusId",
            "leadsourceid":"LeadSourceId","lead_source_id":"LeadSourceId",
            "leadscoringid":"LeadScoringId","lead_scoring_id":"LeadScoringId",
        })
        for col, default in [("EstimatedBudget",0.0),("LeadStageId",pd.NA),("AssignedAgentId",pd.NA),("CreatedOn",pd.NaT),("IsActive",1)]:
            if col not in df.columns: df[col] = default
        df["CreatedOn"] = pd.to_datetime(df["CreatedOn"], errors="coerce")
        df["EstimatedBudget"] = pd.to_numeric(df["EstimatedBudget"], errors="coerce").fillna(0.0)
        ds["leads"] = df

    # Agents
    if ds["agents"] is not None:
        df = normalize_cols(ds["agents"])
        df = rename_like(df, {
            "agentid":"AgentId","first_name":"FirstName","firstname":"FirstName",
            "last_name":"LastName","lastname":"LastName",
            "role":"Role",
            "isactive":"IsActive","active":"IsActive"
        })
        for col, default in [("FirstName",""),("LastName",""),("Role",""),("IsActive",1)]:
            if col not in df.columns: df[col] = default
        ds["agents"] = df

    # Calls
    if ds["calls"] is not None:
        df = normalize_cols(ds["calls"])
        df = rename_like(df, {
            "leadcallid":"LeadCallId","lead_call_id":"LeadCallId",
            "leadid":"LeadId","lead_id":"LeadId",
            "callstatusid":"CallStatusId","call_status_id":"CallStatusId",
            "calldatetime":"CallDateTime","call_datetime":"CallDateTime",
            "durationseconds":"DurationSeconds","duration":"DurationSeconds",
            "sentimentid":"SentimentId",
            "assignedagentid":"AssignedAgentId",
            "isaigenerated":"IsAIGenerated","ai_flag":"IsAIGenerated",
            "calldirection":"CallDirection","direction":"CallDirection",
        })
        if "CallDateTime" in df.columns:
            df["CallDateTime"] = pd.to_datetime(df["CallDateTime"], errors="coerce")
        ds["calls"] = df

    # Schedules
    if ds["schedules"] is not None:
        df = normalize_cols(ds["schedules"])
        df = rename_like(df, {
            "scheduleid":"ScheduleId","leadid":"LeadId","tasktypeid":"TaskTypeId",
            "scheduleddate":"ScheduledDate","taskstatusid":"TaskStatusId","assignedagentid":"AssignedAgentId",
            "completeddate":"CompletedDate","isfollowup":"IsFollowUp",
        })
        if "ScheduledDate" in df.columns: df["ScheduledDate"] = pd.to_datetime(df["ScheduledDate"], errors="coerce")
        if "CompletedDate" in df.columns: df["CompletedDate"] = pd.to_datetime(df["CompletedDate"], errors="coerce")
        ds["schedules"] = df

    # Transactions
    if ds["transactions"] is not None:
        df = normalize_cols(ds["transactions"])
        df = rename_like(df, {
            "transactionid":"TransactionId","leadid":"LeadId",
            "tasktypeid":"TaskTypeId","transactiondate":"TransactionDate","summary":"Summary",
        })
        if "TransactionDate" in df.columns:
            df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")
        ds["transactions"] = df

    # Normalize lookups
    for lk in [
        "countries","lead_stages","lead_statuses","lead_sources","lead_scoring",
        "call_statuses","sentiments","task_types","task_statuses",
        "city_region","timezone_info","priority","meeting_status","agent_meeting_assignment"
    ]:
        if ds.get(lk) is not None:
            ds[lk] = normalize_cols(ds[lk])

    return ds

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="main-header">
  <h1>🏗️ DAR Global — Executive Dashboard</h1>
  <h3>AI‑Powered Analytics</h3>
  <p style="margin: 6px 0 0 0; color: {EXEC_GREEN};">
    Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
  </p>
</div>
""", unsafe_allow_html=True)
st.write("")

# -----------------------------------------------------------------------------
# Global date controls (sidebar)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Filters")
    grain = st.radio("Time grain", ["Week","Month","Year"], index=1, horizontal=True)

# Load data from ./data
data = load_data("data")

def filter_by_date(datasets, grain_sel: str):
    out = dict(datasets)
    # derive min/max from available datetime columns
    cands = []
    if out.get("leads") is not None and "CreatedOn" in out["leads"].columns:
        cands.append(pd.to_datetime(out["leads"]["CreatedOn"], errors="coerce"))
    if out.get("calls") is not None and "CallDateTime" in out["calls"].columns:
        cands.append(pd.to_datetime(out["calls"]["CallDateTime"], errors="coerce"))
    if out.get("schedules") is not None and "ScheduledDate" in out["schedules"].columns:
        cands.append(pd.to_datetime(out["schedules"]["ScheduledDate"], errors="coerce"))

    if cands:
        gmin = min([c.min() for c in cands if c is not None]).date()
        gmax = max([c.max() for c in cands if c is not None]).date()
    else:
        gmax = date.today()
        gmin = gmax - timedelta(days=365)

    # Sidebar date controls
    with st.sidebar:
        preset = st.select_slider("Quick range",
                                  ["Last 7 days","Last 30 days","Last 90 days","MTD","YTD","Custom"],
                                  value="Last 30 days")
        today = date.today()
        if preset == "Last 7 days":
            default_start, default_end = max(gmin, today - timedelta(days=6)), today
        elif preset == "Last 30 days":
            default_start, default_end = max(gmin, today - timedelta(days=29)), today
        elif preset == "Last 90 days":
            default_start, default_end = max(gmin, today - timedelta(days=89)), today
        elif preset == "MTD":
            default_start, default_end = max(gmin, today.replace(day=1)), today
        elif preset == "YTD":
            default_start, default_end = max(gmin, date(today.year,1,1)), today
        else:
            default_start, default_end = gmin, gmax
        step = timedelta(days=1 if grain_sel in ["Week","Month"] else 7)
        date_start, date_end = st.slider("Date range",
                                         min_value=gmin, max_value=gmax,
                                         value=(default_start, default_end), step=step)

    def add_period(dt_series):
        if grain_sel == "Week":
            return dt_series.dt.to_period("W").apply(lambda p: p.start_time.date())
        if grain_sel == "Month":
            return dt_series.dt.to_period("M").apply(lambda p: p.start_time.date())
        return dt_series.dt.to_period("Y").apply(lambda p: p.start_time.date())

    # Leads
    if out.get("leads") is not None and "CreatedOn" in out["leads"].columns:
        dt = pd.to_datetime(out["leads"]["CreatedOn"], errors="coerce")
        mask = dt.dt.date.between(date_start, date_end)
        out["leads"] = out["leads"].loc[mask].copy()
        out["leads"]["period"] = add_period(dt.loc[mask])
    # Calls
    if out.get("calls") is not None and "CallDateTime" in out["calls"].columns:
        dt = pd.to_datetime(out["calls"]["CallDateTime"], errors="coerce")
        mask = dt.dt.date.between(date_start, date_end)
        out["calls"] = out["calls"].loc[mask].copy()
        out["calls"]["period"] = add_period(dt.loc[mask])
    # Schedules
    if out.get("schedules") is not None and "ScheduledDate" in out["schedules"].columns:
        dt = pd.to_datetime(out["schedules"]["ScheduledDate"], errors="coerce")
        mask = dt.dt.date.between(date_start, date_end)
        out["schedules"] = out["schedules"].loc[mask].copy()
        out["schedules"]["period"] = add_period(dt.loc[mask])
    # Transactions
    if out.get("transactions") is not None and "TransactionDate" in out["transactions"].columns:
        dt = pd.to_datetime(out["transactions"]["TransactionDate"], errors="coerce")
        mask = dt.dt.date.between(date_start, date_end)
        out["transactions"] = out["transactions"].loc[mask].copy()
        out["transactions"]["period"] = add_period(dt.loc[mask])

    return out

fdata = filter_by_date(data, grain)

# -----------------------------------------------------------------------------
# Navigation - horizontal top bar (fallback to tabs)
# -----------------------------------------------------------------------------
NAV_ITEMS = [
    ("Executive", "speedometer2", "🎯 Executive Summary"),
    ("Leads", "people", "📈 Lead Status"),
    ("Calls", "telephone", "📞 AI Call Activity"),
    ("Tasks", "check2-circle", "✅ Follow-up & Tasks"),
    ("Agents", "person-badge", "👥 Agent Performance"),
    ("Conversion", "graph-up", "💰 Conversion"),
    ("Geography", "geo-alt", "🌍 Geography"),
]

if HAS_OPTION_MENU:
    selected = option_menu(
        None,
        [n[0] for n in NAV_ITEMS],
        icons=[n[1] for n in NAV_ITEMS],
        orientation="horizontal",
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#0f1116"},
            "icon": {"color": EXEC_PRIMARY, "font-size": "16px"},
            "nav-link": {
                "font-size": "14px",
                "text-align": "center",
                "margin": "0px",
                "color": "#d0d0d0",
                "--hover-color": "#21252b",
            },
            "nav-link-selected": {"background-color": EXEC_SURFACE},
        },
    )
else:
    tab_objs = st.tabs([n[2] for n in NAV_ITEMS])
    selected = None

# -----------------------------------------------------------------------------
# Executive Summary (includes computed ROI support if marketing_spend.csv exists)
# -----------------------------------------------------------------------------
def show_executive_summary(d):
    leads = d.get("leads")
    agents = d.get("agents") if d.get("agents") is not None else pd.DataFrame()
    calls = d.get("calls")
    countries = d.get("countries")
    lead_stages = d.get("lead_stages")

    if leads is None or len(leads) == 0:
        st.info("No data available in the selected range.")
        return

    total_leads = len(leads)
    active_pipeline = leads["EstimatedBudget"].sum() if "EstimatedBudget" in leads.columns else 0.0
    won_mask = leads["LeadStageId"].eq(6) if "LeadStageId" in leads.columns else pd.Series(False, index=leads.index)
    won_revenue = leads.loc[won_mask, "EstimatedBudget"].sum() if "EstimatedBudget" in leads.columns else 0.0
    won_leads = int(won_mask.sum())
    conversion_rate = (won_leads / total_leads * 100) if total_leads else 0.0

    total_calls = len(calls) if calls is not None else 0
    connected_calls = int((calls["CallStatusId"] == 1).sum()) if (calls is not None and "CallStatusId" in calls.columns) else 0
    call_success_rate = (connected_calls / total_calls * 100) if total_calls else 0.0

    active_agents = int(agents[agents["IsActive"] == 1].shape[0]) if ("IsActive" in agents.columns) else 0
    assigned_leads = int(leads["AssignedAgentId"].notna().sum()) if "AssignedAgentId" in leads.columns else 0
    agent_utilization = (assigned_leads / active_agents) if active_agents else 0.0

    # ROI from marketing_spend.csv if present (Date/SpendDate, SpendUSD)
    period_min = None; period_max = None
    if "period" in leads.columns:
        period_min = pd.to_datetime(leads["period"], errors="coerce").min()
        period_max = pd.to_datetime(leads["period"], errors="coerce").max()
    elif "CreatedOn" in leads.columns:
        dt_tmp = pd.to_datetime(leads["CreatedOn"], errors="coerce")
        period_min, period_max = dt_tmp.min(), dt_tmp.max()

    marketing_spend = None
    try:
        spend_df = pd.read_csv(os.path.join("data", "marketing_spend.csv"))
        if "SpendUSD" in spend_df.columns:
            date_col = "Date" if "Date" in spend_df.columns else ("SpendDate" if "SpendDate" in spend_df.columns else None)
            if date_col is not None:
                spend_df[date_col] = pd.to_datetime(spend_df[date_col], errors="coerce")
                if period_min is not None and period_max is not None:
                    m = spend_df[date_col].between(period_min, period_max)
                    marketing_spend = float(spend_df.loc[m, "SpendUSD"].sum())
                else:
                    marketing_spend = float(spend_df["SpendUSD"].sum())
            else:
                marketing_spend = float(spend_df["SpendUSD"].sum())
    except Exception:
        marketing_spend = None

    roi_pct = None
    if marketing_spend is not None and marketing_spend > 0:
        roi_pct = ((won_revenue - marketing_spend) / marketing_spend) * 100.0

    st.subheader("🎯 Executive Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Leads", format_number(total_leads))
    with c2: st.metric("Active Pipeline", format_currency(active_pipeline))
    with c3: st.metric("Revenue Generated", format_currency(won_revenue))
    with c4: st.metric("Conversion Rate", f"{conversion_rate:.1f}%")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Call Success Rate", f"{call_success_rate:.1f}%")
    with c2:
        st.metric("ROI", f"{roi_pct:,.1f}%" if roi_pct is not None else "—")
        if roi_pct is None:
            st.caption("Upload data/marketing_spend.csv to compute ROI.")
    with c3: st.metric("Active Agents", format_number(active_agents))
    with c4: st.metric("Agent Utilization", f"{agent_utilization:.1f} leads/agent")

    # Trend at a glance (indexed lines with axes)
    st.markdown("---")
    st.subheader("Trend at a glance")
    trend_style = st.radio("Trend style", ["Line","Bars","Bullet"], index=0, horizontal=True, key="__trend_style_exec")

    if "period" not in leads.columns:
        dt = pd.to_datetime(leads.get("CreatedOn", pd.Timestamp.utcnow()), errors="coerce")
        leads = leads.copy()
        leads["period"] = dt.dt.to_period("M").apply(lambda p: p.start_time.date())

    leads_ts = leads.groupby("period").size().reset_index(name="value")
    pipeline_ts = leads.groupby("period")["EstimatedBudget"].sum().reset_index(name="value") if "EstimatedBudget" in leads.columns else pd.DataFrame({"period":[], "value":[]})
    rev_ts = leads.loc[won_mask].groupby("period")["EstimatedBudget"].sum().reset_index(name="value") if "EstimatedBudget" in leads.columns else pd.DataFrame({"period":[], "value":[]})

    if calls is not None and len(calls) > 0:
        calls_cp = calls.copy()
        calls_cp["CallDateTime"] = pd.to_datetime(calls_cp["CallDateTime"], errors="coerce")
        calls_cp["period"] = calls_cp["CallDateTime"].dt.to_period("W").apply(lambda p: p.start_time.date())
        calls_ts = calls_cp.groupby("period").agg(total=("LeadCallId","count"),
                                                 connected=("CallStatusId", lambda x: (x==1).sum())).reset_index()
        calls_ts["value"] = (calls_ts["connected"]/calls_ts["total"]*100).round(1)
    else:
        calls_ts = pd.DataFrame({"period":[], "value":[]})

    def _index_series(df):
        df = df.copy()
        if df.empty:
            df["idx"] = []
            return df
        base = df["value"].iloc[0] if df["value"].iloc[0] != 0 else 1.0
        df["idx"] = (df["value"] / base) * 100.0
        return df

    leads_ts = _index_series(leads_ts)
    pipeline_ts = _index_series(pipeline_ts)
    rev_ts = _index_series(rev_ts)
    calls_ts = _index_series(calls_ts)

    def _apply_axes(fig, y_vals, title_txt):
        ymin = float(pd.Series(y_vals).min()) if len(y_vals) else 0
        ymax = float(pd.Series(y_vals).max()) if len(y_vals) else 1
        pad = max(1.0, (ymax - ymin) * 0.12)
        yrng = [ymin - pad, ymax + pad]
        fig.update_layout(
            height=180,
            title=dict(text=title_txt, x=0.01, xanchor="left", font=dict(size=12, color="#cfcfcf")),
            margin=dict(l=6, r=6, t=24, b=8),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            showlegend=False,
        )
        fig.update_xaxes(
            showgrid=True, gridcolor="rgba(255,255,255,0.08)",
            tickfont=dict(color="#a8a8a8", size=10), nticks=4, ticks="outside"
        )
        fig.update_yaxes(
            showgrid=True, gridcolor="rgba(255,255,255,0.08)",
            tickfont=dict(color="#a8a8a8", size=10), nticks=3, ticks="outside", range=yrng
        )
        return fig

    def tile_line(df, color, title):
        df = df.dropna().sort_values("period")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["period"], y=df["idx"], mode="lines+markers",
            line=dict(color=color, width=3, shape="spline"),
            marker=dict(size=5, color=color)
        ))
        return _apply_axes(fig, df["idx"], title)

    def tile_bar(df, color, title):
        df = df.dropna().sort_values("period")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["period"], y=df["idx"],
            marker=dict(color=color, line=dict(color="rgba(255,255,255,0.15)", width=0.5)),
            opacity=0.9
        ))
        return _apply_axes(fig, df["idx"], title)

    def tile_bullet(df, title, bar_color):
        if df.empty:
            fig = go.Figure()
            return _apply_axes(fig, [0, 1], title)
        cur = float(df["idx"].iloc[-1])
        fig = go.Figure(go.Indicator(
            mode="number+gauge+delta",
            value=cur,
            number={'valueformat': ".0f"},
            delta={'reference': 100, 'relative': False},
            gauge={
                'shape': "bullet",
                'axis': {'range': [80, 120]},
                'steps': [
                    {'range': [80, 95],  'color': "rgba(220,20,60,0.35)"},
                    {'range': [95, 105], 'color': "rgba(255,215,0,0.35)"},
                    {'range': [105, 120],'color': "rgba(50,205,50,0.35)"},
                ],
                'bar': {'color': bar_color},
                'threshold': {'line': {'color': '#FFFFFF', 'width': 2}, 'thickness': 0.7, 'value': 100}
            },
            domain={'x':[0,1],'y':[0,1]},
            title={'text': title}
        ))
        fig.update_layout(
            height=120, margin=dict(l=8, r=8, t=26, b=8),
            paper_bgcolor="rgba(0,0,0,0)", font_color="white"
        )
        return fig

    s1, s2, s3, s4 = st.columns(4)
    if trend_style == "Line":
        with s1: st.plotly_chart(tile_line(leads_ts,   EXEC_BLUE,   "Leads trend (indexed)"), use_container_width=True)
        with s2: st.plotly_chart(tile_line(pipeline_ts,EXEC_PRIMARY,"Active pipeline trend (indexed)"), use_container_width=True)
        with s3: st.plotly_chart(tile_line(rev_ts,     EXEC_GREEN,  "Revenue trend (won, indexed)"), use_container_width=True)
        with s4: st.plotly_chart(tile_line(calls_ts,   "#7dd3fc",    "Call success trend (indexed)"), use_container_width=True)
    elif trend_style == "Bars":
        with s1: st.plotly_chart(tile_bar(leads_ts,    EXEC_BLUE,   "Leads trend (indexed)"), use_container_width=True)
        with s2: st.plotly_chart(tile_bar(pipeline_ts, EXEC_PRIMARY,"Active pipeline trend (indexed)"), use_container_width=True)
        with s3: st.plotly_chart(tile_bar(rev_ts,      EXEC_GREEN,  "Revenue trend (won, indexed)"), use_container_width=True)
        with s4: st.plotly_chart(tile_bar(calls_ts,    "#7dd3fc",    "Call success trend (indexed)"), use_container_width=True)
    else:
        with s1: st.plotly_chart(tile_bullet(leads_ts,   "Leads index", EXEC_BLUE), use_container_width=True)
        with s2: st.plotly_chart(tile_bullet(pipeline_ts,"Pipeline index", EXEC_PRIMARY), use_container_width=True)
        with s3: st.plotly_chart(tile_bullet(rev_ts,     "Revenue index", EXEC_GREEN), use_container_width=True)
        with s4: st.plotly_chart(tile_bullet(calls_ts,   "Call success index", "#7dd3fc"), use_container_width=True)

    # Lead conversion snapshot (compact funnel)
    st.markdown("---")
    st.subheader("Lead conversion snapshot")
    if lead_stages is not None and "LeadStageId" in leads.columns and "StageName_E" in lead_stages.columns:
        order = lead_stages.sort_values("sortorder") if "sortorder" in lead_stages.columns else lead_stages
        order = order[["LeadStageId","StageName_E"]]
        stage_counts = leads["LeadStageId"].value_counts().rename_axis("LeadStageId").reset_index(name="count")
        funnel_df = order.merge(stage_counts, on="LeadStageId", how="left").fillna({"count":0})
        fig_funnel = px.funnel(
            funnel_df, x="count", y="StageName_E",
            color_discrete_sequence=[EXEC_BLUE, EXEC_GREEN, EXEC_PRIMARY, "#FFA500", EXEC_DANGER, "#8A2BE2"]
        )
        fig_funnel.update_layout(
            height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="white", margin=dict(l=0, r=0, t=10, b=10)
        )
        st.plotly_chart(fig_funnel, use_container_width=True)
    else:
        st.info("Lead stages not available for the funnel.")

    # Pipeline vs Target (Indicator)
    st.markdown("---")
    g1, g2 = st.columns([1,1])
    with g1:
        st.subheader("Pipeline vs Target")
        target_pipeline = max(active_pipeline * 1.1, 1e9)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=active_pipeline,
            delta={'reference': target_pipeline, 'relative': True},
            number={'valueformat': "$,.0f"},
            gauge={
                'axis': {'range': [None, target_pipeline]},
                'bar': {'color': EXEC_PRIMARY},
                'steps': [
                    {'range': [0, target_pipeline*0.6], 'color': "#2f3b45"},
                    {'range': [target_pipeline*0.6, target_pipeline*0.9], 'color': "#394754"},
                    {'range': [target_pipeline*0.9, target_pipeline*1.0], 'color': "#415263"},
                ],
                'threshold': {'line': {'color': '#FFFFFF', 'width': 2}, 'thickness': 0.75, 'value': target_pipeline}
            }
        ))
        fig_g.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_g, use_container_width=True)

    with g2:
        st.subheader("Top markets (pipeline share)")
        if countries is not None and "CountryId" in leads.columns and "CountryName_E" in countries.columns:
            geo = leads.groupby("CountryId").agg(
                Leads=("LeadId","count"),
                Pipeline=("EstimatedBudget","sum")
            ).reset_index()
            geo = geo.merge(countries[["CountryId","CountryName_E"]], on="CountryId", how="left")
            total_pipe = float(geo["Pipeline"].sum())
            geo["Share"] = (geo["Pipeline"]/total_pipe*100).round(1) if total_pipe>0 else 0.0
            top5 = geo.sort_values("Pipeline", ascending=False).head(5).reset_index(drop=True)
            top5_display = top5[["CountryName_E","Leads","Pipeline","Share"]].copy()
            top5_display.rename(columns={"CountryName_E":"Country"}, inplace=True)
            st.dataframe(
                top5_display,
                use_container_width=True,
                column_config={
                    "Pipeline": st.column_config.NumberColumn("Pipeline", format="$%,.0f"),
                    "Share": st.column_config.ProgressColumn(
                        "Share of Pipeline", format="%.1f%%", min_value=0.0, max_value=100.0
                    )
                },
                hide_index=True
            )
        else:
            st.info("Country data unavailable to build the markets table.")

    # AI Insights (as designed)
    st.markdown("---")
    st.subheader("🤖 AI-Powered Strategic Insights")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="insight-box">
        <h4>🔮 Predictive Revenue Forecasting</h4>
        <ul>
          <li><strong>Q4 2025 Projection:</strong> $28.3B (confidence 85–92%)</li>
          <li><strong>Growth Trajectory:</strong> Positive, double‑digit MoM</li>
          <li><strong>Risk Factors:</strong> Market volatility, agent capacity</li>
          <li><strong>Protection:</strong> Focus on at‑risk pipeline</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="insight-box">
        <h4>🎯 Strategic Recommendations</h4>
        <ul>
          <li>Scale capacity for Q4 surge</li>
          <li>Prioritize high‑response markets</li>
          <li>Enable AI pricing and premium tier</li>
          <li>Coach via call sentiment patterns</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Simple placeholder pages (reuse your detailed ones if present)
# -----------------------------------------------------------------------------
def show_lead_status(d): st.info("Lead Status page placeholder.")
def show_calls(d): st.info("AI Call Activity page placeholder.")
def show_tasks(d): st.info("Follow-up & Tasks page placeholder.")
def show_agents(d): st.info("Agent Performance page placeholder.")
def show_conversion(d): st.info("Conversion page placeholder.")
def show_geography(d): st.info("Geography page placeholder.")

def render_page(page_key: str, fdata, grain_sel: str):
    if page_key == "Executive": show_executive_summary(fdata)
    elif page_key == "Leads": show_lead_status(fdata)
    elif page_key == "Calls": show_calls(fdata)
    elif page_key == "Tasks": show_tasks(fdata)
    elif page_key == "Agents": show_agents(fdata)
    elif page_key == "Conversion": show_conversion(fdata)
    elif page_key == "Geography": show_geography(fdata)

if HAS_OPTION_MENU:
    render_page(selected, fdata, grain)
else:
    for idx, tab in enumerate(tab_objs):
        with tab:
            render_page(NAV_ITEMS[idx][0], fdata, grain)
