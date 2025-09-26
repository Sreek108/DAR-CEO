# app.py — DAR Global CEO Dashboard (with AI/ML Insights + Geo AI)

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import os

# Optional horizontal nav
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except Exception:
    HAS_OPTION_MENU = False

# Optional ML deps (fallback if not installed)
SKLEARN_OK = True
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score
except Exception:
    SKLEARN_OK = False

# -----------------------------------------------------------------------------
# Page config and theme
# -----------------------------------------------------------------------------
st.set_page_config(page_title="DAR Global - Executive Dashboard", page_icon="🏗️", layout="wide", initial_sidebar_state="expanded")

EXEC_PRIMARY="#DAA520"; EXEC_BLUE="#1E90FF"; EXEC_GREEN="#32CD32"; EXEC_DANGER="#DC143C"; EXEC_BG="#1a1a1a"; EXEC_SURFACE="#2d2d2d"

st.markdown(f"""
<style>
.main-header {{
    background: linear-gradient(135deg, {EXEC_BG} 0%, {EXEC_SURFACE} 100%);
    color: {EXEC_PRIMARY}; padding: 24px; border-radius: 12px; border: 2px solid {EXEC_PRIMARY};
    text-align: center; box-shadow: 0 8px 16px rgba(0,0,0,.35);
}}
.main-header h1 {{ color: {EXEC_PRIMARY}; margin: 0 0 6px 0; }}
.main-header h3 {{ color: {EXEC_BLUE}; margin: 4px 0 0 0; }}
div[data-testid="metric-container"] {{
    background: linear-gradient(135deg, {EXEC_SURFACE} 0%, {EXEC_BG} 100%);
    border: 2px solid {EXEC_PRIMARY}; padding: .75rem; border-radius: 10px; color: white;
}}
.insight-box {{
    background: linear-gradient(135deg, {EXEC_SURFACE} 0%, {EXEC_BG} 100%);
    padding: 18px; border-radius: 10px; border-left: 5px solid {EXEC_GREEN}; color: white;
}}
.section {{
    background: linear-gradient(135deg, {EXEC_SURFACE} 0%, {EXEC_BG} 100%);
    padding: 10px; border-radius: 6px; border: 1px solid #444;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def format_currency(v):
    if pd.isna(v): return "$0"
    return f"${v/1e9:.1f}B" if v>=1e9 else (f"${v/1e6:.1f}M" if v>=1e6 else f"${v:,.0f}")

def format_number(v):
    if pd.isna(v): return "0"
    return f"{v/1e6:.1f}M" if v>=1e6 else (f"{v/1e3:.1f}K" if v>=1e3 else f"{v:,.0f}")

def safe_to_datetime(series):
    return pd.to_datetime(series, errors="coerce")

# -----------------------------------------------------------------------------
# Loader for ./data (robust to plural/singular file names)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(data_dir="data"):
    def pick(*names):
        for n in names:
            p=os.path.join(data_dir,n)
            if os.path.exists(p): return p
        return None
    def read(path):
        try: return pd.read_csv(path) if path else None
        except: return None
    def norm(df):
        df=df.copy(); df.columns=(df.columns.str.strip().str.replace(r"[^\w]+","_",regex=True).str.lower()); return df
    def rename(df,map_):
        return df.rename(columns={c:map_[c] for c in map_ if c in df.columns})

    ds={}
    ds["leads"]       = read(pick("Leads.csv","Lead.csv"))
    ds["agents"]      = read(pick("Agents.csv"))
    ds["calls"]       = read(pick("LeadCallRecord.csv"))
    ds["schedules"]   = read(pick("LeadSchedule.csv"))
    ds["transactions"]= read(pick("LeadTransaction.csv"))
    ds["countries"]   = read(pick("Country.csv"))
    ds["lead_stages"] = read(pick("LeadStage.csv"))
    ds["lead_statuses"]=read(pick("LeadStatus.csv"))
    ds["lead_sources"]= read(pick("LeadSource.csv"))
    ds["lead_scoring"]= read(pick("LeadScoring.csv"))
    ds["call_statuses"]= read(pick("CallStatus.csv"))
    ds["sentiments"]  = read(pick("CallSentiment.csv"))
    ds["task_types"]  = read(pick("TaskType.csv"))
    ds["task_statuses"]=read(pick("TaskStatus.csv"))
    ds["city_region"] = read(pick("CityRegion.csv"))
    ds["timezone_info"]=read(pick("TimezoneInfo.csv"))
    ds["priority"]    = read(pick("Priority.csv"))
    ds["meeting_status"]=read(pick("MeetingStatus.csv"))
    ds["agent_meeting_assignment"]=read(pick("AgentMeetingAssignment.csv"))

    # Normalize and align to canonical names used in pages
    if ds["leads"] is not None:
        df=norm(ds["leads"])
        df=rename(df,{
            "leadid":"LeadId","lead_id":"LeadId","leadcode":"LeadCode",
            "leadstageid":"LeadStageId","leadstatusid":"LeadStatusId","leadscoringid":"LeadScoringId",
            "assignedagentid":"AssignedAgentId","createdon":"CreatedOn","isactive":"IsActive",
            "countryid":"CountryId","cityregionid":"CityRegionId",
            "estimatedbudget":"EstimatedBudget","budget":"EstimatedBudget"
        })
        for col, default in [("EstimatedBudget",0.0),("LeadStageId",pd.NA),("LeadStatusId",pd.NA),
                             ("AssignedAgentId",pd.NA),("CreatedOn",pd.NaT),("IsActive",1)]:
            if col not in df.columns: df[col]=default
        df["CreatedOn"]=pd.to_datetime(df["CreatedOn"], errors="coerce")
        df["EstimatedBudget"]=pd.to_numeric(df["EstimatedBudget"], errors="coerce").fillna(0.0)
        ds["leads"]=df

    if ds["agents"] is not None:
        df=norm(ds["agents"])
        df=rename(df,{"agentid":"AgentId","firstname":"FirstName","first_name":"FirstName","lastname":"LastName","last_name":"LastName","isactive":"IsActive"})
        for c, d in [("FirstName",""),("LastName",""),("Role",""),("IsActive",1)]: 
            if c not in df.columns: df[c]=d
        ds["agents"]=df

    if ds["calls"] is not None:
        df=norm(ds["calls"])
        df=rename(df,{
            "leadcallid":"LeadCallId","lead_id":"LeadId","leadid":"LeadId",
            "callstatusid":"CallStatusId","calldatetime":"CallDateTime","call_datetime":"CallDateTime",
            "durationseconds":"DurationSeconds","sentimentid":"SentimentId",
            "assignedagentid":"AssignedAgentId","calldirection":"CallDirection","direction":"CallDirection"
        })
        if "calldatetime" in df.columns: df["CallDateTime"]=pd.to_datetime(df["calldatetime"], errors="coerce")
        if "CallDateTime" in df.columns: df["CallDateTime"]=pd.to_datetime(df["CallDateTime"], errors="coerce")
        ds["calls"]=df

    if ds["schedules"] is not None:
        df=norm(ds["schedules"])
        df=rename(df,{"scheduleid":"ScheduleId","leadid":"LeadId","tasktypeid":"TaskTypeId","scheduleddate":"ScheduledDate","taskstatusid":"TaskStatusId","assignedagentid":"AssignedAgentId","completeddate":"CompletedDate","isfollowup":"IsFollowUp"})
        if "ScheduledDate" in df.columns: df["ScheduledDate"]=pd.to_datetime(df["ScheduledDate"], errors="coerce")
        if "CompletedDate" in df.columns: df["CompletedDate"]=pd.to_datetime(df["CompletedDate"], errors="coerce")
        ds["schedules"]=df

    if ds["transactions"] is not None:
        df=norm(ds["transactions"])
        df=rename(df,{"transactionid":"TransactionId","leadid":"LeadId","tasktypeid":"TaskTypeId","transactiondate":"TransactionDate"})
        if "TransactionDate" in df.columns: df["TransactionDate"]=pd.to_datetime(df["TransactionDate"], errors="coerce")
        ds["transactions"]=df

    for lk in ["countries","lead_stages","lead_statuses","lead_sources","lead_scoring","call_statuses","sentiments","task_types","task_statuses","city_region","timezone_info","priority","meeting_status","agent_meeting_assignment"]:
        if ds.get(lk) is not None: ds[lk]=norm(ds[lk])

    return ds

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="main-header">
  <h1> DAR Global — Executive Dashboard</h1>
  <h3>AI‑Powered Analytics</h3>
  <p style="margin: 6px 0 0 0; color: {EXEC_GREEN};">Last Updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Filters")
    grain = st.radio("Time grain", ["Week","Month","Year"], index=1, horizontal=True)

data = load_data("data")

def filter_by_date(datasets, grain_sel: str):
    out = dict(datasets)
    cands=[]
    if out.get("leads") is not None and "CreatedOn" in out["leads"].columns: cands.append(pd.to_datetime(out["leads"]["CreatedOn"], errors="coerce"))
    if out.get("calls") is not None and "CallDateTime" in out["calls"].columns: cands.append(pd.to_datetime(out["calls"]["CallDateTime"], errors="coerce"))
    if out.get("schedules") is not None and "ScheduledDate" in out["schedules"].columns: cands.append(pd.to_datetime(out["schedules"]["ScheduledDate"], errors="coerce"))

    if cands:
        gmin=min([c.min() for c in cands if c is not None]).date()
        gmax=max([c.max() for c in cands if c is not None]).date()
    else:
        gmax=date.today(); gmin=gmax-timedelta(days=365)

    with st.sidebar:
        preset = st.select_slider("Quick range", ["Last 7 days","Last 30 days","Last 90 days","MTD","YTD","Custom"], value="Last 30 days")
        today=date.today()
        if preset=="Last 7 days": default_start, default_end = max(gmin, today-timedelta(days=6)), today
        elif preset=="Last 30 days": default_start, default_end = max(gmin, today-timedelta(days=29)), today
        elif preset=="Last 90 days": default_start, default_end = max(gmin, today-timedelta(days=89)), today
        elif preset=="MTD": default_start, default_end = max(gmin, today.replace(day=1)), today
        elif preset=="YTD": default_start, default_end = max(gmin, date(today.year,1,1)), today
        else: default_start, default_end = gmin, gmax
        step = timedelta(days=1 if grain_sel in ["Week","Month"] else 7)
        date_start, date_end = st.slider("Date range", min_value=gmin, max_value=gmax, value=(default_start, default_end), step=step)

    def add_period(dt):
        if grain_sel=="Week": return dt.dt.to_period("W").apply(lambda p: p.start_time.date())
        if grain_sel=="Month": return dt.dt.to_period("M").apply(lambda p: p.start_time.date())
        return dt.dt.to_period("Y").apply(lambda p: p.start_time.date())

    # Leads
    if out.get("leads") is not None and "CreatedOn" in out["leads"].columns:
        dt=pd.to_datetime(out["leads"]["CreatedOn"], errors="coerce")
        mask=dt.dt.date.between(date_start, date_end)
        out["leads"]=out["leads"].loc[mask].copy()
        out["leads"]["period"]=add_period(dt.loc[mask])

    # Calls
    if out.get("calls") is not None and "CallDateTime" in out["calls"].columns:
        dt=pd.to_datetime(out["calls"]["CallDateTime"], errors="coerce")
        mask=dt.dt.date.between(date_start, date_end)
        out["calls"]=out["calls"].loc[mask].copy()
        out["calls"]["period"]=add_period(dt.loc[mask])

    # Schedules
    if out.get("schedules") is not None and "ScheduledDate" in out["schedules"].columns:
        dt=pd.to_datetime(out["schedules"]["ScheduledDate"], errors="coerce")
        mask=dt.dt.date.between(date_start, date_end)
        out["schedules"]=out["schedules"].loc[mask].copy()
        out["schedules"]["period"]=add_period(dt.loc[mask])

    # Transactions
    if out.get("transactions") is not None and "TransactionDate" in out["transactions"].columns:
        dt=pd.to_datetime(out["transactions"]["TransactionDate"], errors="coerce")
        mask=dt.dt.date.between(date_start, date_end)
        out["transactions"]=out["transactions"].loc[mask].copy()
        out["transactions"]["period"]=add_period(dt.loc[mask])

    # AgentMeetingAssignment — align to window via StartDateTime
    if out.get("agent_meeting_assignment") is not None:
        ama = out["agent_meeting_assignment"].copy()
        cols_lower = {c.lower(): c for c in ama.columns}
        if "startdatetime" in cols_lower:
            dtcol = cols_lower["startdatetime"]
            dt = pd.to_datetime(ama[dtcol], errors="coerce")
            mask = dt.dt.date.between(date_start, date_end)
            out["agent_meeting_assignment"] = ama.loc[mask].copy()

    return out

fdata = filter_by_date(data, grain)

# -----------------------------------------------------------------------------
# Navigation
# -----------------------------------------------------------------------------
NAV = [
    ("Executive","speedometer2","🎯 Executive Summary"),
    ("Lead Status","people","📈 Lead Status"),
    ("AI Calls","telephone","📞 AI Call Activity"),
    ("AI Insights","robot","🤖 AI Insights"),
    ("Conversion","bar-chart-line","📊 Conversion"),  # <-- comma added and icon fixed
    ("Geo AI","globe","🌍 Geo AI")
]
if HAS_OPTION_MENU:
    selected = option_menu(None, [n[0] for n in NAV], icons=[n[1] for n in NAV], orientation="horizontal", default_index=0,
                           styles={"container":{"padding":"0!important","background-color":"#0f1116"},
                                   "icon":{"color":EXEC_PRIMARY,"font-size":"16px"},
                                   "nav-link":{"font-size":"14px","color":"#d0d0d0","--hover-color":"#21252b"},
                                   "nav-link-selected":{"background-color":EXEC_SURFACE}})
else:
    tabs = st.tabs([n[2] for n in NAV])
    selected=None

# -----------------------------------------------------------------------------
# Executive Summary (Performance KPIs, trends, funnel, top markets)
# -----------------------------------------------------------------------------
def show_executive_summary(d):
    leads=d.get("leads"); agents=d.get("agents"); calls=d.get("calls")
    lead_statuses=d.get("lead_statuses"); countries=d.get("countries")

    if leads is None or len(leads)==0:
        st.info("No data available in the selected range."); return

    # 'Won' status id
    won_status_id = 9
    if lead_statuses is not None and "statusname_e" in lead_statuses.columns:
        match = lead_statuses.loc[lead_statuses["statusname_e"].str.lower()=="won"]
        if not match.empty and "leadstatusid" in match.columns:
            won_status_id = int(match.iloc[0]["leadstatusid"])

    st.subheader("Performance KPIs")
    today = pd.Timestamp.today().normalize()
    week_start = today - pd.Timedelta(days=today.weekday())
    month_start = today.replace(day=1)
    year_start = today.replace(month=1, day=1)
    date_ranges = {"Week to Date":(week_start,today),"Month to Date":(month_start,today),"Year to Date":(year_start,today)}
    cols = st.columns(3)
    meetings = d.get("agent_meeting_assignment")

    for (label, (start, end)), col in zip(date_ranges.items(), cols):
        leads_period = leads.loc[(pd.to_datetime(leads["CreatedOn"], errors="coerce")>=pd.Timestamp(start)) &
                                 (pd.to_datetime(leads["CreatedOn"], errors="coerce")<=pd.Timestamp(end))] if "CreatedOn" in leads.columns else pd.DataFrame()
        if meetings is not None and len(meetings)>0:
            m = meetings.copy(); m.columns = m.columns.str.lower()
            date_col = "startdatetime" if "startdatetime" in m.columns else None
            if date_col is not None:
                m["_dt"] = pd.to_datetime(m[date_col], errors="coerce")
                m = m[(m["_dt"]>=pd.Timestamp(start)) & (m["_dt"]<=pd.Timestamp(end))]
                if "meetingstatusid" in m.columns: m = m[m["meetingstatusid"].isin({1,6})]
                meetings_period = m
            else: meetings_period = pd.DataFrame()
        else: meetings_period = pd.DataFrame()

        total_leads_p = int(len(leads_period))
        won_leads_p = int((leads_period["LeadStatusId"]==won_status_id).sum()) if "LeadStatusId" in leads_period.columns else 0
        conv_rate_p = (won_leads_p/total_leads_p*100.0) if total_leads_p else 0.0
        meetings_scheduled = int(meetings_period["leadid"].nunique()) if "leadid" in meetings_period.columns else 0

        with col:
            st.markdown(f"#### {label}")
            st.markdown("Total Leads")
            st.markdown(f"<span style='font-size:2rem;'>{total_leads_p}</span>", unsafe_allow_html=True)
            st.markdown("Conversion Rate")
            st.markdown(f"<span style='font-size:2rem;'>{conv_rate_p:.1f}%</span>", unsafe_allow_html=True)
            st.markdown("Meetings Scheduled")
            st.markdown(f"<span style='font-size:2rem;'>{meetings_scheduled}</span>", unsafe_allow_html=True)

    # Trend at a glance
    st.markdown("---"); st.subheader("Trend at a glance")
    trend_style = st.radio("Trend style", ["Line","Bars","Bullet"], index=0, horizontal=True, key="__trend_style_exec")

    leads_local = leads.copy()
    if "period" not in leads_local.columns:
        dt=pd.to_datetime(leads_local.get("CreatedOn"), errors="coerce")
        leads_local["period"]=dt.dt.to_period("M").apply(lambda p: p.start_time.date())

    leads_ts = leads_local.groupby("period").size().reset_index(name="value")
    if "LeadStatusId" in leads_local.columns:
        per_leads = leads_local.groupby("period").size().rename("total")
        per_won = leads_local.loc[leads_local["LeadStatusId"].eq(won_status_id)].groupby("period").size().rename("won")
        conv_ts = pd.concat([per_leads, per_won], axis=1).fillna(0.0).reset_index()
        conv_ts["value"] = (conv_ts["won"]/conv_ts["total"]*100).round(1)
    else:
        conv_ts = pd.DataFrame({"period":[], "value":[]})

    meetings = d.get("agent_meeting_assignment")
    if meetings is not None and len(meetings)>0:
        m = meetings.copy(); m.columns = m.columns.str.lower()
        date_col = "startdatetime" if "startdatetime" in m.columns else None
        if date_col is not None:
            m["_period"] = pd.to_datetime(m[date_col], errors="coerce").dt.to_period("W").apply(lambda p: p.start_time.date())
            if "meetingstatusid" in m.columns: m = m[m["meetingstatusid"].isin({1,6})]
            meet_ts = m.groupby("_period").size().reset_index(name="value").rename(columns={"_period":"period"})
        else:
            meet_ts = pd.DataFrame({"period":[], "value":[]})
    else:
        meet_ts = pd.DataFrame({"period":[], "value":[]})

    def _index(df):
        df=df.copy()
        if df.empty: df["idx"]=[]; return df
        base = df["value"].iloc[0] if df["value"].iloc[0]!=0 else 1.0
        df["idx"]=(df["value"]/base)*100.0
        return df

    leads_ts = _index(leads_ts); conv_ts = _index(conv_ts); meet_ts = _index(meet_ts)

    def _apply_axes(fig, ys, title):
        ymin=float(pd.Series(ys).min()) if len(ys) else 0
        ymax=float(pd.Series(ys).max()) if len(ys) else 1
        pad=max(1.0,(ymax-ymin)*0.12); rng=[ymin-pad, ymax+pad]
        fig.update_layout(height=180, title=dict(text=title, x=0.01, font=dict(size=12, color="#cfcfcf")),
                          margin=dict(l=6,r=6,t=24,b=8), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="white", showlegend=False)
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#a8a8a8", size=10), nticks=4, ticks="outside")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", tickfont=dict(color="#a8a8a8", size=10), nticks=3, ticks="outside", range=rng)
        return fig

    def tile_line(df,color,title):
        df=df.dropna().sort_values("period"); fig=go.Figure()
        fig.add_trace(go.Scatter(x=df["period"], y=df["idx"], mode="lines+markers", line=dict(color=color, width=3, shape="spline"), marker=dict(size=5,color=color)))
        return _apply_axes(fig, df["idx"], title)
    def tile_bar(df,color,title):
        df=df.dropna().sort_values("period"); fig=go.Figure()
        fig.add_trace(go.Bar(x=df["period"], y=df["idx"], marker=dict(color=color, line=dict(color="rgba(255,255,255,0.15)", width=0.5)), opacity=0.9))
        return _apply_axes(fig, df["idx"], title)
    def tile_bullet(df,title,bar_color):
        if df.empty: fig=go.Figure(); return _apply_axes(fig, [0,1], title)
        cur=float(df["idx"].iloc[-1])
        fig=go.Figure(go.Indicator(mode="number+gauge+delta", value=cur, number={'valueformat':".0f"}, delta={'reference':100},
                                   gauge={'shape':"bullet",'axis':{'range':[80,120]},
                                          'steps':[{'range':[80,95],'color':"rgba(220,20,60,0.35)"},{'range':[95,105],'color':"rgba(255,215,0,0.35)"},
                                                   {'range':[105,120],'color':"rgba(50,205,50,0.35)"}],
                                          'bar':{'color':bar_color},'threshold':{'line':{'color':'#fff','width':2},'value':100}}))
        fig.update_layout(height=120, margin=dict(l=8,r=8,t=26,b=8), paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        return fig

    s1,s2,s3 = st.columns(3)
    if trend_style=="Line":
        with s1: st.plotly_chart(tile_line(leads_ts,EXEC_BLUE,"Leads trend (indexed)"), use_container_width=True)
        with s2: st.plotly_chart(tile_line(conv_ts,EXEC_GREEN,"Conversion rate (indexed)"), use_container_width=True)
        with s3: st.plotly_chart(tile_line(meet_ts,EXEC_PRIMARY,"Meeting scheduled (indexed)"), use_container_width=True)
    elif trend_style=="Bars":
        with s1: st.plotly_chart(tile_bar(leads_ts,EXEC_BLUE,"Leads trend (indexed)"), use_container_width=True)
        with s2: st.plotly_chart(tile_bar(conv_ts,EXEC_GREEN,"Conversion rate (indexed)"), use_container_width=True)
        with s3: st.plotly_chart(tile_bar(meet_ts,EXEC_PRIMARY,"Meeting scheduled (indexed)"), use_container_width=True)
    else:
        with s1: st.plotly_chart(tile_bullet(leads_ts,"Leads index",EXEC_BLUE), use_container_width=True)
        with s2: st.plotly_chart(tile_bullet(conv_ts,"Conversion index",EXEC_GREEN), use_container_width=True)
        with s3: st.plotly_chart(tile_bullet(meet_ts,"Meetings index",EXEC_PRIMARY), use_container_width=True)

    # Lead conversion snapshot (funnel)
    st.markdown("---"); st.subheader("Lead conversion snapshot")

    leads_df   = d.get("leads").copy()
    statuses   = d.get("lead_statuses")
    ama        = d.get("agent_meeting_assignment")

    def have(df, cols): 
        return (df is not None) and set(cols).issubset(df.columns)
    def status_ids_by_name(names):
        if statuses is None: return set()
        s = statuses.copy(); s.columns = s.columns.str.lower()
        if not {"statusname_e","leadstatusid"}.issubset(s.columns): return set()
        return set(s.loc[s["statusname_e"].str.lower().isin([n.lower() for n in names]), "leadstatusid"].astype(int).tolist())
    def status_ids_by_stage(stage_no):
        if statuses is None: return set()
        s = statuses.copy(); s.columns = s.columns.str.lower()
        if not {"leadstageid","leadstatusid"}.issubset(s.columns): return set()
        return set(s.loc[s["leadstageid"].astype("Int64")==stage_no, "leadstatusid"].astype(int).tolist())

    cohort_ids = pd.Index(leads_df["LeadId"].dropna().astype(int).unique()) if have(leads_df, ["LeadId"]) else pd.Index([])
    new_count  = int(cohort_ids.size)

    q_sid   = status_ids_by_stage(2)
    qual_ids = pd.Index(leads_df.loc[have(leads_df, ["LeadStatusId","LeadId"]) & leads_df["LeadStatusId"].isin(q_sid),"LeadId"].dropna().astype(int).unique()).intersection(cohort_ids)
    qualified_count = int(qual_ids.size)

    meet_ids = pd.Index([])
    if ama is not None:
        m = ama.copy(); m.columns = m.columns.str.lower()
        if {"leadid","meetingstatusid"}.issubset(m.columns):
            m = m[m["leadid"].isin(qual_ids)]
            m = m[m["meetingstatusid"].isin({1,6})]
            meet_ids = pd.Index(m["leadid"].dropna().astype(int).unique())
    meeting_count = int(meet_ids.size)

    neg_sid = status_ids_by_name(["On Hold","Awaiting Budget"])
    neg_ids = pd.Index(leads_df.loc[have(leads_df, ["LeadStatusId","LeadId"]) & leads_df["LeadStatusId"].isin(neg_sid),"LeadId"].dropna().astype(int).unique()).intersection(meet_ids)
    neg_count = int(neg_ids.size)

    won_sid = status_ids_by_name(["Won"])
    signed_ids = pd.Index(leads_df.loc[have(leads_df, ["LeadStatusId","LeadId"]) & leads_df["LeadStatusId"].isin(won_sid),"LeadId"].dropna().astype(int).unique()).intersection(meet_ids)
    signed_count = int(signed_ids.size)

    lost_sid = status_ids_by_name(["Lost"])
    lost_ids = pd.Index(leads_df.loc[have(leads_df, ["LeadStatusId","LeadId"]) & leads_df["LeadStatusId"].isin(lost_sid),"LeadId"].dropna().astype(int).unique()).intersection(meet_ids)
    lost_count = int(lost_ids.size)

    funnel_df = pd.DataFrame({"Stage":["New","Qualified","Meeting Scheduled","Negotiation","Contract Signed","Lost"],
                              "Count":[new_count, qualified_count, meeting_count, neg_count, signed_count, lost_count]})
    fig = px.funnel(funnel_df, x="Count", y="Stage",
                    color_discrete_sequence=[EXEC_BLUE, EXEC_GREEN, EXEC_PRIMARY, "#FFA500", "#7CFC00", EXEC_DANGER])
    fig.update_layout(height=340, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                      margin=dict(l=0, r=0, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Top markets: Country, Leads, Won/Signed
    st.markdown("---"); st.subheader("Top markets")
    if countries is not None and "CountryId" in leads_df.columns and "countryname_e" in countries.columns:
        leads_df = leads_df.copy()
        leads_df["CountryId"] = pd.to_numeric(leads_df["CountryId"], errors="coerce").astype("Int64")
        leads_df["LeadStatusId"] = pd.to_numeric(leads_df.get("LeadStatusId", pd.NA), errors="coerce").astype("Int64")
        by_country = leads_df.groupby("CountryId", dropna=True).size().reset_index(name="Leads")
        won_by_country = leads_df.loc[leads_df["LeadStatusId"]==won_status_id].groupby("CountryId", dropna=True).size().reset_index(name="Won")
        final = (by_country.merge(won_by_country, on="CountryId", how="left")
                           .merge(countries[["countryid","countryname_e"]].rename(columns={"countryid":"CountryId","countryname_e":"Country"}), on="CountryId", how="left")
                           .fillna({"Won":0})
                           .sort_values(["Leads","Won"], ascending=False)[["Country","Leads","Won"]])
        st.dataframe(final, use_container_width=True, hide_index=True,
                     column_config={"Leads": st.column_config.NumberColumn("Leads", format="%d"),
                                    "Won":   st.column_config.NumberColumn("Won/Signed", format="%d")})
    else:
        st.info("Country data unavailable to build the market list.")

# -----------------------------------------------------------------------------
# AI Insights page (Propensity + Expected Value + Actions + Forecast + Best Contact Time)
# -----------------------------------------------------------------------------
def _recent_agg(df, when_col, cutoff, days=14):
    if df is None or len(df)==0 or when_col not in df: 
        return pd.DataFrame({"LeadId":[], "n":[], "connected":[], "mean_dur":[], "last_days":[]})
    x = df.copy()
    x[when_col] = pd.to_datetime(x[when_col], errors="coerce")
    window = cutoff - pd.Timedelta(days=days)
    x = x[(x[when_col]>=window) & (x[when_col]<=cutoff)]
    g = x.groupby("LeadId").agg(
        n=("LeadId","count"),
        connected=("CallStatusId", lambda s: (s==1).mean() if "CallStatusId" in x.columns else 0.0),
        mean_dur=("DurationSeconds", "mean") if "DurationSeconds" in x.columns else ("LeadId","count")  
    ).reset_index()
    last = x.groupby("LeadId")[when_col].max().reset_index().rename(columns={when_col:"last_dt"})
    g = g.merge(last, on="LeadId", how="left")
    g["last_days"] = (cutoff - g["last_dt"]).dt.days.fillna(999)
    return g.drop(columns=["last_dt"], errors="ignore")

def _weekly_meeting_series(meets):
    if meets is None or len(meets)==0: 
        return pd.DataFrame({"week":[], "meetings":[]})
    m = meets.copy()
    dt_col = "StartDateTime" if "StartDateTime" in m.columns else ("startdatetime" if "startdatetime" in m.columns else None)
    if dt_col is None: return pd.DataFrame({"week":[], "meetings":[]})
    m["dt"] = pd.to_datetime(m[dt_col], errors="coerce")
    sid = "MeetingStatusId" if "MeetingStatusId" in m.columns else ("meetingstatusid" if "meetingstatusid" in m.columns else None)
    if sid is not None: m = m[m[sid].isin([1,6])]
    return m.groupby(pd.Grouper(key="dt", freq="W-MON")).size().reset_index(name="meetings").rename(columns={"dt":"week"})

def _weekly_wins_series(leads):
    if leads is None or len(leads)==0: 
        return pd.DataFrame({"week":[], "wins":[]})
    l = leads.copy()
    l["dt"] = pd.to_datetime(l["CreatedOn"], errors="coerce")
    l = l[l.get("LeadStatusId", pd.Series(dtype="Int64")).astype("Int64")==9]
    return l.groupby(pd.Grouper(key="dt", freq="W-MON")).size().reset_index(name="wins").rename(columns={"dt":"week"})

def _sma_forecast(vals, k=4):
    if len(vals)==0: return [0.0]*k
    s = pd.Series(vals)
    last = s.rolling(min_periods=1, window=min(4,len(s))).mean().iloc[-1]
    return [float(last)]*k

@st.cache_resource(show_spinner=False)
def _train_propensity(leads, calls, meets, statuses):
    won_id = 9
    if statuses is not None and "statusname_e" in statuses.columns:
        m = statuses.loc[statuses["statusname_e"].str.lower()=="won"]
        if not m.empty and "leadstatusid" in m.columns: won_id = int(m.iloc[0]["leadstatusid"])

    df = leads.copy()
    df["CreatedOn"] = pd.to_datetime(df.get("CreatedOn"), errors="coerce")
    df["label"] = (df.get("LeadStatusId", pd.Series(index=df.index).astype("Int64")).astype("Int64")==won_id).astype(int)
    cutoff = df["CreatedOn"].max() if "CreatedOn" in df.columns else pd.Timestamp.today()

    calls_14 = _recent_agg(calls, "CallDateTime", cutoff, 14)
    meet_norm = pd.DataFrame({"LeadId":[], "meet_n":[], "meet_connected":[], "meet_mean_dur":[], "meet_last_days":[]})
    if meets is not None and len(meets):
        m = meets.copy()
        dtc = "StartDateTime" if "StartDateTime" in m.columns else ("startdatetime" if "startdatetime" in m.columns else None)
        if dtc is not None:
            m[dtc] = pd.to_datetime(m[dtc], errors="coerce")
            sid = "MeetingStatusId" if "MeetingStatusId" in m.columns else ("meetingstatusid" if "meetingstatusid" in m.columns else None)
            if sid is not None: m = m[m[sid].isin([1,6])]
            m = m.rename(columns={dtc:"When"})
            mm = _recent_agg(m, "When", cutoff, 14)
            meet_norm = mm.rename(columns={"n":"meet_n","connected":"meet_connected","mean_dur":"meet_mean_dur","last_days":"meet_last_days"})

    X = df[["LeadId","LeadStageId","LeadStatusId","AssignedAgentId","EstimatedBudget"]].fillna(0).copy()
    X["age_days"] = (cutoff - df["CreatedOn"]).dt.days.fillna(0)
    X = X.merge(calls_14, on="LeadId", how="left").merge(meet_norm, on="LeadId", how="left").fillna(0)
    y = df["label"].values

    if SKLEARN_OK and len(df)>=20 and y.sum()>=1:
        X_fit = X.drop(columns=["LeadId"])
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X_fit, y, test_size=0.25, random_state=42, stratify=y if y.sum()>0 else None)
            base = GradientBoostingClassifier(random_state=42)
            model = CalibratedClassifierCV(base, cv=3)
            model.fit(X_tr, y_tr)
            auc = roc_auc_score(y_te, model.predict_proba(X_te)[:,1]) if len(np.unique(y_te))>1 else np.nan
            prob_all = model.predict_proba(X_fit)[:,1]
            return prob_all, auc, X
        except Exception:
            pass

    # Heuristic fallback
    prob_all = (
        0.15
        + 0.25*(X["LeadStatusId"].astype(int).isin([6,7,8]).astype(float))
        + 0.25*(X["meet_n"].clip(0,3)/3.0)
        + 0.20*(X["connected"].clip(0,1))
        + 0.15*(1.0/(1.0+X["age_days"]/30.0))
    ).clip(0,1).values
    return prob_all, np.nan, X

def show_ai_insights(d):
    st.subheader("Lead win propensity, expected value, and next‑best‑actions")
    leads = d.get("leads"); calls = d.get("calls"); meets = d.get("agent_meeting_assignment"); statuses=d.get("lead_statuses")
    if leads is None or len(leads)==0:
        st.info("No data available to train insights."); return

    win_prob, auc, X = _train_propensity(leads, calls, meets, statuses)
    if not np.isnan(auc):
        st.metric("Validation AUC", f"{auc:.3f}")
    else:
        st.info("Using heuristic scoring (ML not available or insufficient data).")

    scored = leads[["LeadId","LeadStatusId","EstimatedBudget"]].copy()
    scored["win_prob"] = win_prob
    scored["expected_value"] = (scored["win_prob"] * scored["EstimatedBudget"]).round(2)

    tmp = X.merge(scored, on="LeadId", how="left")
    def nba(r):
        if r["win_prob"]>=0.60 and r.get("meet_n",0)==0: return "Book meeting within 72h"
        if 0.30<=r["win_prob"]<0.60 and r.get("connected",0)<0.30: return "Nurture call + brochure"
        if r["win_prob"]<0.30 and r.get("n",0)>=2: return "Switch to AI Agent sequence"
        return "Maintain cadence"
    scored["next_action"] = [nba(r) for _,r in tmp.iterrows()]

    st.dataframe(
        scored.sort_values(["expected_value","win_prob"], ascending=False)
              .loc[:,["LeadId","LeadStatusId","win_prob","expected_value","next_action"]],
        use_container_width=True, hide_index=True
    )

    # Forecasts
    st.markdown("---"); st.subheader("4‑week outlook")
    wm = _weekly_meeting_series(meets)
    ww = _weekly_wins_series(leads)
    f_meet = _sma_forecast(wm["meetings"] if len(wm) else [], 4)
    f_wins = _sma_forecast(ww["wins"] if len(ww) else [], 4)
    c1,c2 = st.columns(2)
    with c1: st.metric("Forecast avg meetings / week (next 4)", f"{np.mean(f_meet):.1f}")
    with c2: st.metric("Forecast avg wins / week (next 4)", f"{np.mean(f_wins):.1f}")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(wm, x="week", y="meetings", markers=True, title="Weekly meetings")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=260)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(ww, x="week", y="wins", markers=True, title="Weekly wins")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=260)
        st.plotly_chart(fig, use_container_width=True)

    # Best contact time recommender (by connects)
    st.markdown("---"); st.subheader("Best contact time windows (connect‑rate)")
    calls_df = d.get("calls")
    if calls_df is None or len(calls_df)==0 or "CallDateTime" not in calls_df.columns:
        st.info("Not enough call data to compute contact windows.")
    else:
        c = calls_df.copy()
        c["dt"] = pd.to_datetime(c["CallDateTime"], errors="coerce")
        c["dow"] = c["dt"].dt.day_name()
        c["hour"] = c["dt"].dt.hour
        if "CallStatusId" in c.columns:
            grp = c.groupby(["dow","hour"]).agg(total=("LeadCallId","count"),
                                               connects=("CallStatusId", lambda s:(s==1).sum())).reset_index()
            grp["connect_rate"] = (grp["connects"]/grp["total"]).round(3)
            grp = grp.sort_values(["connect_rate","total"], ascending=[False,False]).head(10)
            st.dataframe(grp, use_container_width=True, hide_index=True)
        else:
            st.info("CallStatusId not present in call records.")

# -----------------------------------------------------------------------------
# Lead Status page
# -----------------------------------------------------------------------------
def show_lead_status(d):
    leads = d.get("leads"); statuses = d.get("lead_statuses")
    calls  = d.get("calls"); meets = d.get("agent_meeting_assignment")

    if leads is None or len(leads)==0:
        st.info("No lead status data in the selected range."); 
        return

    # Map status ids -> names
    name_map = {}
    if statuses is not None and {"leadstatusid","statusname_e"}.issubset(statuses.columns):
        name_map = dict(zip(statuses["leadstatusid"].astype(int), statuses["statusname_e"].astype(str)))
    leads = leads.copy()
    leads["Status"] = leads["LeadStatusId"].map(name_map).fillna(leads.get("LeadStatusId", pd.Series(dtype="Int64")).astype(str))
    leads["CreatedOn"] = pd.to_datetime(leads.get("CreatedOn"), errors="coerce")
    cutoff = leads["CreatedOn"].max() if "CreatedOn" in leads.columns else pd.Timestamp.today()
    leads["age_days"] = (cutoff - leads["CreatedOn"]).dt.days.astype("Int64")

    # Summary donut and headline KPIs (kept)
    counts = leads["Status"].value_counts().reset_index()
    counts.columns = ["Status","count"]
    c1,c2 = st.columns([2,1])
    with c1:
        fig = px.pie(counts, names="Status", values="count", hole=0.35, color_discrete_sequence=px.colors.sequential.Viridis,
                     title="Lead Status Share")
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("Total Leads", f"{len(leads):,}")
        won_id = None
        if statuses is not None and "statusname_e" in statuses.columns:
            m = statuses.loc[statuses["statusname_e"].str.lower()=="won"]
            if not m.empty: won_id = int(m.iloc[0]["leadstatusid"])
        won = int((leads.get("LeadStatusId", pd.Series(dtype="Int64")).astype("Int64")==won_id).sum()) if won_id is not None else 0
        st.metric("Won", f"{won:,}")

    # -------------- Lead Distribution Status --------------
    st.markdown("---"); st.subheader("Lead Distribution Status")

    # Horizontal bar distribution
    dist_sorted = counts.sort_values("count", ascending=True)
    fig_bar = px.bar(dist_sorted, x="count", y="Status", orientation="h",
                     title="Leads by status (sorted)", color="Status", color_discrete_sequence=px.colors.qualitative.Dark24)
    fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=360,
                          showlegend=False, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_bar, use_container_width=True)

    # Optional stacked trend by period (uses fdata period already set)
    if "period" not in leads.columns and "CreatedOn" in leads.columns:
        leads["period"] = leads["CreatedOn"].dt.to_period("M").apply(lambda p: p.start_time.date())
    if "period" in leads.columns:
        normalize = st.checkbox("Normalize to 100% per period", value=False, key="ls_norm")
        trend = leads.groupby(["period","Status"]).size().reset_index(name="count")
        if normalize:
            totals = trend.groupby("period")["count"].transform("sum").replace(0, np.nan)
            trend["count"] = (trend["count"]/totals*100).round(1)
            y_title = "Share %"
        else:
            y_title = "Count"
        fig_stack = px.bar(trend.sort_values("period"), x="period", y="count", color="Status", barmode="stack",
                           title=f"Status mix by period ({y_title})")
        fig_stack.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                                height=360, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_stack, use_container_width=True)

    # -------------- Detailed Lead Breakdown --------------
    st.markdown("---"); st.subheader("Detailed Lead Breakdown")

    # Meeting intent per status (Scheduled/Rescheduled)
    meet_rate = pd.DataFrame({"Status": pd.Series(dtype="str"), "meet_leads": pd.Series(dtype="float")})
    if meets is not None and len(meets):
        M = meets.copy(); M.columns = M.columns.str.lower()
        dtc = "startdatetime" if "startdatetime" in M.columns else None
        if dtc is not None:
            if "meetingstatusid" in M.columns:
                M = M[M["meetingstatusid"].isin({1,6})]  # Scheduled/Rescheduled
            mm = M.merge(leads[["LeadId","Status"]], left_on="leadid", right_on="LeadId", how="left")
            meet_rate = mm.groupby("Status")["leadid"].nunique().reset_index(name="meet_leads")

    # Connect rate per status
    conn_rate = pd.DataFrame({"Status": pd.Series(dtype="str"), "connect_rate": pd.Series(dtype="float")})
    if calls is not None and len(calls):
        C = calls.copy()
        C["CallDateTime"] = pd.to_datetime(C.get("CallDateTime"), errors="coerce")
        C = C.merge(leads[["LeadId","Status"]], on="LeadId", how="left")
        g = C.groupby("Status").agg(total=("LeadCallId","count"),
                                    connects=("CallStatusId", lambda s:(s==1).sum())).reset_index()
        g["connect_rate"] = (g["connects"]/g["total"]).fillna(0.0)
        conn_rate = g[["Status","connect_rate"]]

    # Aggregate breakdown
    base = leads.groupby("Status").agg(
        Leads=("LeadId","count"),
        Avg_Age_Days=("age_days","mean"),
        Pipeline=("EstimatedBudget","sum")
    ).reset_index()
    total_leads = float(base["Leads"].sum()) if len(base) else 0.0
    base["Share_%"] = (base["Leads"]/total_leads*100.0).round(1) if total_leads>0 else 0.0

    breakdown = (base.merge(meet_rate, on="Status", how="left")
                      .merge(conn_rate, on="Status", how="left"))
    breakdown["meet_leads"] = breakdown["meet_leads"].fillna(0.0)
    breakdown["Meeting_Rate_%"] = (breakdown["meet_leads"]/breakdown["Leads"]*100.0).replace([np.inf, -np.inf], 0).fillna(0.0).round(1)
    breakdown["connect_rate"] = breakdown["connect_rate"].fillna(0.0).round(2)
    breakdown["Avg_Age_Days"] = breakdown["Avg_Age_Days"].fillna(0.0).round(1)
    breakdown = breakdown.sort_values(["Leads","Pipeline"], ascending=False)

    st.dataframe(
        breakdown[["Status","Leads","Share_%","Avg_Age_Days","Meeting_Rate_%","connect_rate","Pipeline"]],
        use_container_width=True, hide_index=True,
        column_config={
            "Leads": st.column_config.NumberColumn("Leads", format="%,d"),
            "Share_%": st.column_config.ProgressColumn("Share", min_value=0.0, max_value=100.0, format="%.1f%%"),
            "Avg_Age_Days": st.column_config.NumberColumn("Avg age (days)", format="%.1f"),
            "Meeting_Rate_%": st.column_config.ProgressColumn("Meeting rate", min_value=0.0, max_value=100.0, format="%.1f%%"),
            "connect_rate": st.column_config.ProgressColumn("Connect rate", min_value=0.0, max_value=1.0, format="%.2f"),
            "Pipeline": st.column_config.NumberColumn("Pipeline", format="%.0f"),
        }
    )

# -----------------------------------------------------------------------------
# Calls page (basic)
# -----------------------------------------------------------------------------
def show_calls(d):
    st.subheader("AI Call Activity")

    calls = d.get("calls")
    if calls is None or len(calls)==0:
        st.info("No call data in the selected range.")
        return

    # Normalize types
    C = calls.copy()
    if "CallDateTime" in C.columns:
        C["CallDateTime"] = pd.to_datetime(C["CallDateTime"], errors="coerce")

    # ---------------- Daily calls and success rate ----------------
    if {"CallDateTime","LeadCallId","CallStatusId"}.issubset(C.columns):
        daily = C.groupby(C["CallDateTime"].dt.date).agg(
            Total=("LeadCallId","count"),
            Connected=("CallStatusId", lambda x: (x==1).sum())
        ).reset_index().rename(columns={"CallDateTime":"Date"})
        daily["SuccessRate"] = (daily["Connected"]/daily["Total"]*100).round(1)

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily["Date"], y=daily["Total"], mode="lines+markers",
                                     line=dict(color="#1E90FF", width=3)))
            fig.update_layout(title="Daily Calls",
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font_color="white")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily["Date"], y=daily["SuccessRate"], mode="lines+markers",
                                     line=dict(color="#32CD32", width=3)))
            fig.update_layout(title="Success Rate (%)",
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              font_color="white")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Missing fields to plot daily calls or success rate (need CallDateTime, LeadCallId, CallStatusId).")

    # ---------------- Call Status Distribution (donut + trimmed KPIs) ----------------
    st.markdown("---"); st.subheader("Call Status Distribution")

    cs = d.get("call_statuses")
    name_map = {}
    if cs is not None and {"callstatusid","statusname_e"}.issubset(cs.columns):
        name_map = dict(zip(cs["callstatusid"].astype(int), cs["statusname_e"].astype(str)))  # labels from master [file:345]

    if "CallStatusId" in C.columns:
        dist = C.copy()
        dist["Status"] = dist["CallStatusId"].map(name_map).fillna(dist["CallStatusId"].astype(str))
        donut = dist["Status"].value_counts().reset_index()
        donut.columns = ["Status","count"]

        # Bigger donut
        fig = px.pie(
            donut,
            names="Status",
            values="count",
            hole=0.35,
            color_discrete_sequence=px.colors.sequential.RdPu,
            title="Outcomes"
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
                height=500,
                margin=dict(l=10, r=10, t=60, b=80),   # <- larger bottom margin creates space
                legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
                font=dict(size=14),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="white"
        ) 
        st.plotly_chart(fig, use_container_width=True)

        # Spacer between chart and KPI row
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)# uses full width for an even bigger feel [file:324]

        # KPIs in one row
        total_calls = int(len(C))
        connected_calls = int((C["CallStatusId"]==1).sum())
        connect_rate = (connected_calls/total_calls*100.0) if total_calls else 0.0
        avg_duration = float(pd.to_numeric(C.get("DurationSeconds", pd.Series(dtype=float)), errors="coerce").dropna().mean()) if "DurationSeconds" in C.columns else 0.0  # [file:324]

        k1,k2,k3,k4 = st.columns(4)
        with k1: st.metric("Total calls", f"{total_calls:,}")                                    # [file:324]
        with k2: st.metric("Connected calls", f"{connected_calls:,}")                            # [file:324]
        with k3: st.metric("Connect rate", f"{connect_rate:.1f}%")                               # [file:324]
        with k4: st.metric("Avg duration (sec)", f"{avg_duration:.1f}")                          # [file:324]
    else:
        st.info("CallStatusId not available to render distribution and KPIs.") 
    # ---------------- Effectiveness by attempt number (kept) ----------------
    st.markdown("---"); st.subheader("Effectiveness by attempt number")
    if {"LeadId","CallDateTime","CallStatusId","LeadCallId"}.issubset(C.columns):
        A = C.copy().sort_values(["LeadId","CallDateTime"])
        A["attempt_no"] = A.groupby("LeadId").cumcount() + 1
        curve = A.groupby("attempt_no").agg(
            total=("LeadCallId","count"),
            connects=("CallStatusId", lambda s: (s==1).sum())
        ).reset_index()
        curve["connect_rate"] = (curve["connects"]/curve["total"]).fillna(0.0).round(3)
        fig = px.line(curve, x="attempt_no", y="connect_rate", markers=True,
                      title="Connect rate by attempt number")
        fig.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          font_color="white", xaxis_title="Attempt #", yaxis_title="Connect rate")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(curve, use_container_width=True, hide_index=True)
    else:
        st.info("Insufficient fields to analyze attempts (need LeadId, CallDateTime, CallStatusId, LeadCallId).")


def show_conversions(d):
    st.subheader("Conversion — Wins vs Dropped (no revenue/pipeline)")

    leads = d.get("leads")
    statuses = d.get("lead_statuses")
    meets = d.get("agent_meeting_assignment")

    if leads is None or len(leads) == 0:
        st.info("No data available in the selected range.")
        return

    # ---------- Resolve status ids (robust) ----------
    won_id, lost_id = 9, 10
    if statuses is not None and {"statusname_e", "leadstatusid"}.issubset(statuses.columns):
        s = statuses.copy()
        s["statusname_e_norm"] = s["statusname_e"].astype(str).str.strip().str.lower()
        s["leadstatusid"] = pd.to_numeric(s["leadstatusid"], errors="coerce").astype("Int64")
        w = s.loc[s["statusname_e_norm"].eq("won"), "leadstatusid"].dropna()
        l = s.loc[s["statusname_e_norm"].eq("lost"), "leadstatusid"].dropna()
        if not w.empty: won_id = int(w.iloc[0])
        if not l.empty: lost_id = int(l.iloc[0])

    # ---------- Prepare leads with period ----------
    L = leads.copy()
    L["LeadId"] = pd.to_numeric(L.get("LeadId"), errors="coerce").astype("Int64")
    L["LeadStatusId"] = pd.to_numeric(L.get("LeadStatusId"), errors="coerce").astype("Int64")
    if "period" not in L.columns:
        dt = pd.to_datetime(L.get("CreatedOn"), errors="coerce")
        L["period"] = dt.dt.to_period("M").apply(lambda p: p.start_time.date())

    # ---------- Monthly conversions vs dropped ----------
    per_total = L.groupby("period").size().rename("total")
    per_won   = L.loc[L["LeadStatusId"].eq(won_id)].groupby("period").size().rename("won")
    per_lost  = L.loc[L["LeadStatusId"].eq(lost_id)].groupby("period").size().rename("lost")
    conv = pd.concat([per_total, per_won, per_lost], axis=1).fillna(0.0).reset_index()
    conv["conv_rate"] = (conv["won"]/conv["total"]*100).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(
            conv.sort_values("period"),
            x="period", y=["won","lost"],
            barmode="group",
            color_discrete_sequence=["#32CD32","#DC143C"],
            title="Monthly conversions vs dropped"
        )
        fig.update_layout(height=340, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                          margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.line(
            conv.sort_values("period"),
            x="period", y="conv_rate", markers=True,
            title="Conversion rate trend (%)"
        )
        fig.update_layout(height=340, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                          margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ---------- KPIs (no revenue/pipeline) ----------
    total = int(L["LeadId"].nunique()) if "LeadId" in L.columns else int(len(L))
    wins = int(L["LeadStatusId"].eq(won_id).sum())
    losses = int(L["LeadStatusId"].eq(lost_id).sum())
    conv_rate_overall = (wins/total*100.0) if total else 0.0
    drop_rate_overall = (losses/total*100.0) if total else 0.0

    # YTD conversions (based on CreatedOn)
    ytd_wins = 0
    if "CreatedOn" in L.columns:
        dt = pd.to_datetime(L["CreatedOn"], errors="coerce")
        today = pd.Timestamp.today()
        ystart = pd.Timestamp(year=today.year, month=1, day=1)
        ytd_wins = int(L.loc[(dt>=ystart) & L["LeadStatusId"].eq(won_id)].shape[0])

    k1,k2,k3 = st.columns(3)
    with k1: st.metric("YTD conversions", f"{ytd_wins:,}")
    with k2: st.metric("Conversion rate", f"{conv_rate_overall:.1f}%")
    with k3: st.metric("Drop rate", f"{drop_rate_overall:.1f}%")

    # ---------- Conversion Funnel (counts only) ----------
    st.markdown("---"); st.subheader("Conversion Funnel (counts)")

    # Qualified: all statuses in stage 2; Negotiation: On Hold/Awaiting Budget
    qualified_ids, nego_ids = set(), set()
    if statuses is not None:
        s = statuses.copy()
        s["leadstageid"]  = pd.to_numeric(s.get("leadstageid"), errors="coerce").astype("Int64")
        s["leadstatusid"] = pd.to_numeric(s.get("leadstatusid"), errors="coerce").astype("Int64")
        s["statusname_e_norm"] = s.get("statusname_e", pd.Series(dtype=object)).astype(str).str.strip().str.lower()
        qualified_ids = set(s.loc[s["leadstageid"].eq(2), "leadstatusid"].dropna().astype(int).tolist())
        nego_mask = s["statusname_e_norm"].isin(["on hold","awaiting budget"])
        nego_ids = set(s.loc[nego_mask, "leadstatusid"].dropna().astype(int).tolist())

    # Initialize all stage counters to ensure no NameError
    new_count = int(L["LeadId"].nunique())
    qualified_count = int(L.loc[L["LeadStatusId"].isin(qualified_ids), "LeadId"].nunique()) if qualified_ids else 0

    meet_leads = set()
    meeting_count = 0
    if meets is not None and len(meets):
        M = meets.copy(); M.columns = M.columns.str.lower()
        if "startdatetime" in M.columns:
            if "meetingstatusid" in M.columns:
                M = M[M["meetingstatusid"].isin({1,6})]
            meet_leads = set(pd.to_numeric(M["leadid"], errors="coerce").dropna().astype(int).tolist())
            meeting_count = int(len(meet_leads))

    nego_count = int(L.loc[L["LeadStatusId"].isin(nego_ids) & L["LeadId"].isin(meet_leads), "LeadId"].nunique()) if nego_ids else 0
    won_count  = wins
    lost_count = losses

    funnel_df = pd.DataFrame({
        "Stage":["New","Qualified","Meeting Scheduled","Negotiation","Won","Lost"],
        "Count":[new_count, qualified_count, meeting_count, nego_count, won_count, lost_count]
    })

    # Optionally filter out zero-count stages to avoid awkward shapes
    show_zero_stages = False
    if not show_zero_stages:
        funnel_df = funnel_df[funnel_df["Count"]>0]

    fig = px.funnel(
        funnel_df, x="Count", y="Stage",
        color_discrete_sequence=["#1E90FF","#32CD32","#DAA520","#FFA500","#7CFC00","#DC143C"],
        title="Conversion Funnel (counts)"
    )
    fig.update_traces(textposition="inside", textinfo="value+percent initial")
    fig.update_layout(
        height=420,
        margin=dict(l=30, r=30, t=70, b=60),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Geo AI page (performance + AI recommendations)
# -----------------------------------------------------------------------------
def show_geo_ai(d):
    st.subheader("Geo AI — Market performance and opportunity")

    leads = d.get("leads")
    countries = d.get("countries")
    statuses = d.get("lead_statuses")
    calls = d.get("calls")
    meets = d.get("agent_meeting_assignment")

    if leads is None or len(leads)==0 or countries is None:
        st.info("No geo data available in the selected range.")
        return

    # Resolve Won id from master
    won_id = 9
    if statuses is not None and "statusname_e" in statuses.columns:
        m = statuses.loc[statuses["statusname_e"].str.lower()=="won"]
        if not m.empty and "leadstatusid" in m.columns:
            won_id = int(m.iloc[0]["leadstatusid"])

    L = leads.copy()
    L["CountryId"] = pd.to_numeric(L.get("CountryId"), errors="coerce").astype("Int64")
    L["LeadStatusId"] = pd.to_numeric(L.get("LeadStatusId"), errors="coerce").astype("Int64")
    L["EstimatedBudget"] = pd.to_numeric(L.get("EstimatedBudget"), errors="coerce").fillna(0.0)
    L["CreatedOn"] = pd.to_datetime(L.get("CreatedOn"), errors="coerce")

    # ---------- Section 1: Current market performance (table + map) ----------
    # Keep a base with CountryId for downstream AI merges
    perf_base = L.groupby("CountryId").agg(
        Leads=("LeadId","count"),
        Pipeline=("EstimatedBudget","sum")
    ).reset_index()
    won = L.loc[L["LeadStatusId"]==won_id].groupby("CountryId").size().reset_index(name="Won")
    perf_base = perf_base.merge(won, on="CountryId", how="left").fillna({"Won":0})
    total_pipe = float(perf_base["Pipeline"].sum())
    perf_base["Share"] = (perf_base["Pipeline"]/total_pipe*100.0).round(1) if total_pipe>0 else 0.0

    CTRY = countries.rename(columns={"countryid":"CountryId","countryname_e":"Country"})
    perf_view = perf_base.merge(CTRY[["CountryId","Country"]], on="CountryId", how="left") \
                         .sort_values(["Pipeline","Won","Leads"], ascending=False)[["Country","Leads","Won","Pipeline","Share"]]

    st.markdown("#### Current market performance")
    st.dataframe(
        perf_view, use_container_width=True, hide_index=True,
        column_config={
            "Pipeline": st.column_config.NumberColumn("Pipeline", format="%.0f"),
            "Share": st.column_config.ProgressColumn("Share", min_value=0.0, max_value=100.0, format="%.1f%%")
        }
    )

    # Performance map (choose a metric to color)
    st.markdown("##### Performance map")
    perf_metric = st.selectbox(
        "Color by",
        ["Pipeline","Leads","Won","Share"],
        index=0,
        key="geo_perf_metric"
    )

    perf_map = perf_view.copy()
    perf_map[perf_metric] = pd.to_numeric(perf_map[perf_metric], errors="coerce")

    try:
        if perf_metric == "Share":
            rng = [0, 100]
            fig_perf = px.choropleth(
                perf_map,
                locations="Country",
                locationmode="country names",
                color=perf_metric,
                hover_name="Country",
                hover_data={"Leads":":,", "Won":":,", "Pipeline":":,", "Share":".1f"},
                range_color=rng,
                color_continuous_scale="Reds",
                title=f"Current market performance — {perf_metric}"
            )
        else:
            fig_perf = px.choropleth(
                perf_map,
                locations="Country",
                locationmode="country names",
                color=perf_metric,
                hover_name="Country",
                hover_data={"Leads":":,", "Won":":,", "Pipeline":":,", "Share":".1f"},
                color_continuous_scale="Reds",
                title=f"Current market performance — {perf_metric}"
            )

        fig_perf.update_layout(
            height=420,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            margin=dict(l=0,r=0,t=30,b=0)
        )
        st.plotly_chart(fig_perf, use_container_width=True)
    except Exception:
        st.info("Map rendering skipped (requires valid country names).")

    # ---------- Section 2: Geo AI recommendations ----------
    st.markdown("---")
    st.markdown("#### Country opportunity (AI)")

    # Meeting intent (Scheduled/Rescheduled) — initialize with key to avoid KeyError
    meet_rate = pd.DataFrame({"CountryId": pd.Series(dtype="Int64"), "meet_rate": pd.Series(dtype="float")})
    if meets is not None and len(meets):
        M = meets.copy(); M.columns = M.columns.str.lower()
        dtc = "startdatetime" if "startdatetime" in M.columns else None
        if dtc is not None:
            if "meetingstatusid" in M.columns: M = M[M["meetingstatusid"].isin({1,6})]
            mm = M.merge(L[["LeadId","CountryId"]], left_on="leadid", right_on="LeadId", how="left")
            mr = mm.groupby("CountryId")["leadid"].nunique().reset_index(name="meet_leads")
            meet_rate = perf_base[["CountryId","Leads"]].merge(mr, on="CountryId", how="left").fillna({"meet_leads":0})
            meet_rate["meet_rate"] = (meet_rate["meet_leads"]/meet_rate["Leads"]).fillna(0.0)
            meet_rate = meet_rate[["CountryId","meet_rate"]]

    # Connect efficiency — initialize with key
    conn = pd.DataFrame({"CountryId": pd.Series(dtype="Int64"), "connect_rate": pd.Series(dtype="float")})
    if calls is not None and len(calls):
        C = calls.copy()
        C["CallDateTime"] = pd.to_datetime(C.get("CallDateTime"), errors="coerce")
        C = C.merge(L[["LeadId","CountryId"]], on="LeadId", how="left")
        g = C.groupby("CountryId").agg(
            total=("LeadCallId","count"),
            connects=("CallStatusId", lambda s:(s==1).sum())
        ).reset_index()
        g["connect_rate"] = (g["connects"]/g["total"]).fillna(0.0)
        conn = g[["CountryId","connect_rate"]]  # list, not set

        # ensure correct cols if set was used mistakenly
        if isinstance(conn, set) or not isinstance(conn, pd.DataFrame):
            conn = g[["CountryId","connect_rate"]]

    # Momentum (last 4 weeks vs prior 4) — initialize with key
    mom = pd.DataFrame({"CountryId": pd.Series(dtype="Int64"), "momentum": pd.Series(dtype="float")})
    if "CreatedOn" in L.columns and L["CreatedOn"].notna().any():
        W = L.copy()
        W["week"] = W["CreatedOn"].dt.to_period("W").apply(lambda p: p.start_time.date())
        wk = sorted(W["week"].dropna().unique())
        if len(wk)>=8:
            last4 = set(wk[-4:]); prev4 = set(wk[-8:-4])
            a = W[W["week"].isin(last4)].groupby("CountryId").size().reset_index(name="leads_last4")
            b = W[W["week"].isin(prev4)].groupby("CountryId").size().reset_index(name="leads_prev4")
            mom = a.merge(b, on="CountryId", how="outer").fillna(0)
            mom["momentum"] = (mom["leads_last4"] - mom["leads_prev4"]) / mom["leads_prev4"].replace(0, np.nan)
            mom["momentum"] = mom["momentum"].replace([np.inf,-np.inf], 0).fillna(0.0)
            mom = mom[["CountryId","momentum"]]
        else:
            tmp = W.groupby("CountryId").size().reset_index(name="leads_last")
            tmp["momentum"] = 0.0
            mom = tmp[["CountryId","momentum"]]

    # Combine features on CountryId (safe merges even when frames are empty)
    df = perf_base.merge(meet_rate, on="CountryId", how="left") \
                  .merge(conn, on="CountryId", how="left") \
                  .merge(mom, on="CountryId", how="left") \
                  .fillna({"meet_rate":0.0,"connect_rate":0.0,"momentum":0.0})
    df["win_rate"] = (df["Won"]/df["Leads"]).fillna(0.0)

    # Normalize + score
    def mm(s):
        s = s.astype(float)
        lo, hi = s.min(), s.max()
        if not np.isfinite(lo) or not np.isfinite(hi) or hi==lo:
            return pd.Series(0.0, index=s.index)
        return (s-lo)/(hi-lo)

    total_pipe = float(df["Pipeline"].sum())
    df["pipeline_share"] = df["Pipeline"]/total_pipe if total_pipe>0 else 0.0

    w = {"win_rate":0.35,"pipeline_share":0.30,"momentum":0.20,"connect_rate":0.10,"meet_rate":0.05}
    df["opportunity_score"] = (
        w["win_rate"]*mm(df["win_rate"]) +
        w["pipeline_share"]*mm(df["pipeline_share"]) +
        w["momentum"]*mm(df["momentum"]) +
        w["connect_rate"]*mm(df["connect_rate"]) +
        w["meet_rate"]*mm(df["meet_rate"])
    ).round(3)

    q75, q50 = df["opportunity_score"].quantile(0.75), df["opportunity_score"].quantile(0.50)
    def reco(r):
        if r["opportunity_score"]>=q75 and r["win_rate"]>=df["win_rate"].median(): return "Invest"
        if r["opportunity_score"]>=q50: return "Protect"
        if r["momentum"]>0: return "Explore"
        return "Deprioritize"
    df["recommendation"] = df.apply(reco, axis=1)

    def action(r):
        if r["recommendation"]=="Invest": return "Add senior closer capacity; accelerate meetings; allocate budget"
        if r["recommendation"]=="Protect": return "Maintain capacity; tighten SLA; defend share"
        if r["recommendation"]=="Explore": return "Low‑cost tests, partner outreach, targeted campaigns"
        return "Reduce spend; nurture via AI agent only"
    df["action"] = df.apply(action, axis=1)

    view = df.merge(CTRY[["CountryId","Country"]], on="CountryId", how="left") \
             .sort_values(["opportunity_score","Pipeline","Won"], ascending=False)

    st.dataframe(
        view[["Country","Leads","Won","win_rate","meet_rate","connect_rate","Pipeline","opportunity_score","recommendation","action"]],
        use_container_width=True, hide_index=True,
        column_config={
            "win_rate": st.column_config.NumberColumn("Win rate", format="%.1f"),
            "meet_rate": st.column_config.NumberColumn("Meet rate", format="%.1f"),
            "connect_rate": st.column_config.NumberColumn("Connect rate", format="%.1f"),
            "Pipeline": st.column_config.NumberColumn("Pipeline", format="%.0f"),
            "opportunity_score": st.column_config.ProgressColumn("Opportunity", min_value=0.0, max_value=1.0, format="%.3f"),
        }
    )

    # Opportunity map
    try:
        fig = px.choropleth(
            view, locations="Country", locationmode="country names",
            color="opportunity_score", hover_name="Country",
            color_continuous_scale="YlGnBu", title="Country opportunity map"
        )
        fig.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                          margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Map rendering skipped (requires valid country names).")

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
if HAS_OPTION_MENU:
    if selected=="Executive":          show_executive_summary(fdata)
    elif selected=="Lead Status":       show_lead_status(fdata)
    elif selected=="AI Calls":          show_calls(fdata)
    elif selected=="AI Insights":       show_ai_insights(fdata)
    elif selected=="Conversion":        show_conversions(fdata)   # Conversion page
    elif selected=="Geo AI":            show_geo_ai(fdata)
else:
    with tabs[0]: show_executive_summary(fdata)
    with tabs[1]: show_lead_status(fdata)
    with tabs[2]: show_calls(fdata)
    with tabs[3]: show_ai_insights(fdata)
    with tabs[4]: show_conversions(fdata)  # Conversion page
    with tabs[5]: show_geo_ai(fdata)
