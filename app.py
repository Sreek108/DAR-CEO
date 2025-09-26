# app.py — DAR Global CEO Dashboard (with AI/ML snapshots on every sheet)

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
.insight-box {{
    background: linear-gradient(135deg, {EXEC_SURFACE} 0%, {EXEC_BG} 100%);
    padding: 16px; border-radius: 10px; border-left: 5px solid {EXEC_GREEN}; color: white;
}}
.small-note {{ color:#b0b0b0; font-size:12px; }}
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

    for lk in ["agents","calls","schedules","transactions","countries","lead_stages","lead_statuses","lead_sources","lead_scoring","call_statuses","sentiments","task_types","task_statuses","city_region","timezone_info","priority","meeting_status","agent_meeting_assignment"]:
        if ds.get(lk) is not None:
            ds[lk]=norm(ds[lk])

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
    ("AI Insights","robot","🤖 AI Insights")
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
# Reusable AI helpers (scoring, forecast, contact time)
# -----------------------------------------------------------------------------
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

    prob_all = (
        0.15
        + 0.25*(X["LeadStatusId"].astype(int).isin([6,7,8]).astype(float))
        + 0.25*(X["meet_n"].clip(0,3)/3.0)
        + 0.20*(X["connected"].clip(0,1))
        + 0.15*(1.0/(1.0+X["age_days"]/30.0))
    ).clip(0,1).values
    return prob_all, np.nan, X

def compute_ai_snapshot(d):
    leads = d.get("leads"); calls = d.get("calls"); meets = d.get("agent_meeting_assignment"); statuses=d.get("lead_statuses")
    bullets=[]
    if leads is None or len(leads)==0:
        return ["No data available in the selected range."]
    # Propensity + EV
    win_prob, auc, X = _train_propensity(leads, calls, meets, statuses)
    df = leads.copy()
    df["win_prob"]=win_prob
    df["EV"]=(df["win_prob"]*df.get("EstimatedBudget",0)).fillna(0.0)
    top = df.sort_values(["EV","win_prob"], ascending=False).head(1)
    if len(top):
        r = top.iloc[0]
        bullets.append(f"Highest expected value lead: #{int(r['LeadId'])} with EV {format_currency(r['EV'])}; recommended action: {'Book meeting within 72h' if r['win_prob']>=0.60 else 'Nurture call + brochure'}.")
    # Forecasts
    wm = _weekly_meeting_series(meets)
    ww = _weekly_wins_series(leads)
    f_meet = _sma_forecast(wm['meetings'] if len(wm) else [], 4)
    f_wins = _sma_forecast(ww['wins'] if len(ww) else [], 4)
    if len(f_meet):
        bullets.append(f"Next 4‑weeks outlook: {np.mean(f_meet):.1f} meetings/wk and {np.mean(f_wins):.1f} wins/wk on current pace.")
    # Best contact time
    if calls is not None and len(calls)>0 and "CallDateTime" in calls.columns:
        c=calls.copy(); c["dt"]=pd.to_datetime(c["CallDateTime"], errors="coerce")
        c["dow"]=c["dt"].dt.day_name(); c["hour"]=c["dt"].dt.hour
        if "CallStatusId" in c.columns and len(c)>0:
            g=c.groupby(["dow","hour"]).agg(n=("LeadCallId","count"), conn=("CallStatusId", lambda s:(s==1).sum())).reset_index()
            g=g[g["n"]>=1]  # low-volume friendly for demo
            if len(g):
                g["rate"]=g["conn"]/g["n"]; r=g.sort_values(["rate","n"], ascending=[False,False]).iloc[0]
                hour=int(r["hour"]); nxt=(hour+1)%24
                bullets.append(f"Best connect window: {r['dow']} {hour:02d}:00–{nxt:02d}:00 based on historical connect rates.")
    return bullets

def render_ai_block(title, bullets):
    items="".join([f"<li>{b}</li>" for b in bullets])
    st.markdown(f"""
    <div class="insight-box">
      <h4>{title}</h4>
      <ul>{items}</ul>
      <div class="small-note">Insights update with current filters and date range.</div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Executive Summary (existing content trimmed for brevity)
# -----------------------------------------------------------------------------
def show_executive_summary(d):
    # ... keep your existing KPI, trends, funnel, top markets content ...
    # After your existing sections, append the AI snapshot block:
    st.markdown("---")
    render_ai_block("AI Snapshot", compute_ai_snapshot(d))

# -----------------------------------------------------------------------------
# Lead Status page (existing visuals + AI block)
# -----------------------------------------------------------------------------
def show_lead_status(d):
    leads=d.get("leads"); statuses=d.get("lead_statuses")
    if leads is None or len(leads)==0 or "LeadStatusId" not in leads.columns:
        st.info("No lead status data in the selected range."); 
    else:
        counts = leads["LeadStatusId"].value_counts().reset_index()
        counts.columns=["LeadStatusId","count"]
        lbl_map={}
        if statuses is not None and "leadstatusid" in statuses.columns:
            name_col = "statusname_e" if "statusname_e" in statuses.columns else None
            for _,r in statuses.iterrows(): lbl_map[int(r["leadstatusid"])]= str(r[name_col]) if name_col else f"Status {int(r['leadstatusid'])}"
        counts["label"] = counts["LeadStatusId"].map(lbl_map).fillna(counts["LeadStatusId"].astype(str))
        c1,c2=st.columns([2,1])
        with c1:
            fig = px.pie(counts, names="label", values="count", hole=0.35, color_discrete_sequence=px.colors.sequential.Viridis)
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.metric("Total Leads", format_number(len(leads)))
    # AI snapshot for this sheet
    st.markdown("---")
    render_ai_block("AI Snapshot", compute_ai_snapshot(d))

# -----------------------------------------------------------------------------
# Calls page (existing charts + AI block)
# -----------------------------------------------------------------------------
def show_calls(d):
    calls=d.get("calls")
    if calls is None or len(calls)==0:
        st.info("No call data in the selected range.")
    else:
        if "CallDateTime" in calls.columns:
            c=calls.copy(); c["CallDateTime"]=pd.to_datetime(c["CallDateTime"], errors="coerce")
            daily=c.groupby(c["CallDateTime"].dt.date).agg(Total=("LeadCallId","count"), Connected=("CallStatusId", lambda x:(x==1).sum())).reset_index()
            daily["SuccessRate"]=(daily["Connected"]/daily["Total"]*100).round(1)
            col1,col2=st.columns(2)
            with col1:
                fig=go.Figure(); fig.add_trace(go.Scatter(x=daily["CallDateTime"], y=daily["Total"], mode="lines+markers", line=dict(color=EXEC_BLUE,width=3)))
                fig.update_layout(title="Daily Calls", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig=go.Figure(); fig.add_trace(go.Scatter(x=daily["CallDateTime"], y=daily["SuccessRate"], mode="lines+markers", line=dict(color=EXEC_GREEN,width=3)))
                fig.update_layout(title="Success Rate (%)", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
                st.plotly_chart(fig, use_container_width=True)
        st.dataframe(calls.head(1000), use_container_width=True)
    # AI snapshot for this sheet
    st.markdown("---")
    render_ai_block("AI Snapshot", compute_ai_snapshot(d))

# -----------------------------------------------------------------------------
# Dedicated AI Insights page (optional, if already present keep it; else simple)
# -----------------------------------------------------------------------------
def show_ai_insights(d):
    st.subheader("Full AI Insights")
    render_ai_block("AI Snapshot", compute_ai_snapshot(d))
    # Optionally add deeper tables/plots here

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
if HAS_OPTION_MENU:
    if selected=="Executive": show_executive_summary(fdata)
    elif selected=="Lead Status": show_lead_status(fdata)
    elif selected=="AI Calls": show_calls(fdata)
    elif selected=="AI Insights": show_ai_insights(fdata)
else:
    with tabs[0]: show_executive_summary(fdata)
    with tabs[1]: show_lead_status(fdata)
    with tabs[2]: show_calls(fdata)
    with tabs[3]: show_ai_insights(fdata)
