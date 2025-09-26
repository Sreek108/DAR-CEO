# app.py — DAR Global CEO Dashboard (works with 2‑year synthetic dataset in ./data)

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

# Try optional ML (falls back if missing)
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
    padding: 18px; border-radius: 10px; border: 1px solid #444;
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
# Executive Summary / Lead Status / Calls — keep your existing implementations
# -----------------------------------------------------------------------------
# ... your show_executive_summary(d), show_lead_status(d), and show_calls(d) go here ...

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
    g = m.groupby(pd.Grouper(key="dt", freq="W-MON")).size().reset_index(name="meetings").rename(columns={"dt":"week"})
    return g

def _weekly_wins_series(leads):
    if leads is None or len(leads)==0: 
        return pd.DataFrame({"week":[], "wins":[]})
    l = leads.copy()
    l["dt"] = pd.to_datetime(l["CreatedOn"], errors="coerce")
    if "LeadStatusId" in l.columns:
        l = l[l["LeadStatusId"].astype("Int64")==9]
    g = l.groupby(pd.Grouper(key="dt", freq="W-MON")).size().reset_index(name="wins").rename(columns={"dt":"week"})
    return g

def _sma_forecast(vals, k=4):
    if len(vals)==0: return [0.0]*k
    s = pd.Series(vals)
    last = s.rolling(min_periods=1, window=min(4,len(s))).mean().iloc[-1]
    return [float(last)]*k

def show_ai_insights(d):
    st.subheader("Lead win propensity, expected value, and next‑best‑actions")
    leads = d.get("leads"); calls = d.get("calls"); meets = d.get("agent_meeting_assignment"); statuses = d.get("lead_statuses")
    if leads is None or len(leads)==0:
        st.info("No data available to train insights."); return

    # Label and cutoff
    won_id = 9
    if statuses is not None and "statusname_e" in statuses.columns:
        m = statuses.loc[statuses["statusname_e"].str.lower()=="won"]
        if not m.empty and "leadstatusid" in m.columns: won_id = int(m.iloc[0]["leadstatusid"])
    leads = leads.copy()
    leads["CreatedOn"] = pd.to_datetime(leads.get("CreatedOn"), errors="coerce")
    leads["label"] = (leads.get("LeadStatusId", pd.Series(index=leads.index).astype("Int64")).astype("Int64")==won_id).astype(int)
    cutoff = leads["CreatedOn"].max() if "CreatedOn" in leads.columns else pd.Timestamp.today()

    # Features (14‑day recency windows)
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

    # Assemble X
    X = leads[["LeadId","LeadStageId","LeadStatusId","AssignedAgentId","EstimatedBudget"]].fillna(0).copy()
    X["age_days"] = (cutoff - leads["CreatedOn"]).dt.days.fillna(0)
    X = X.merge(calls_14, on="LeadId", how="left").merge(meet_norm, on="LeadId", how="left").fillna(0)
    y = leads["label"].values

    # Train (or heuristic fallback)
    if SKLEARN_OK and len(leads)>=20 and y.sum()>=1:
        X_fit = X.drop(columns=["LeadId"])
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X_fit, y, test_size=0.25, random_state=42, stratify=y if y.sum()>0 else None)
            base = GradientBoostingClassifier(random_state=42)
            model = CalibratedClassifierCV(base, cv=3)
            model.fit(X_tr, y_tr)
            auc = roc_auc_score(y_te, model.predict_proba(X_te)[:,1]) if len(np.unique(y_te))>1 else np.nan
            win_prob = model.predict_proba(X_fit)[:,1]
            st.metric("Validation AUC", f"{auc:.3f}" if not np.isnan(auc) else "—")
        except Exception:
            win_prob = (
                0.15
                + 0.25*(X["LeadStatusId"].astype(int).isin([6,7,8]).astype(float))
                + 0.25*(X["meet_n"].clip(0,3)/3.0)
                + 0.20*(X["connected"].clip(0,1))
                + 0.15*(1.0/(1.0+X["age_days"]/30.0))
            ).clip(0,1).values
            st.info("Used heuristic due to small sample or split error.")
    else:
        win_prob = (
            0.15
            + 0.25*(X["LeadStatusId"].astype(int).isin([6,7,8]).astype(float))
            + 0.25*(X["meet_n"].clip(0,3)/3.0)
            + 0.20*(X["connected"].clip(0,1))
            + 0.15*(1.0/(1.0+X["age_days"]/30.0))
        ).clip(0,1).values
        st.info("Using heuristic scoring (ML not available or insufficient data).")

    # Score + Expected Value + Actions
    scored = leads[["LeadId","LeadStatusId","EstimatedBudget"]].copy()
    scored["win_prob"] = win_prob
    scored["expected_value"] = scored["win_prob"] * scored["EstimatedBudget"]

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
    if calls is None or len(calls)==0 or "CallDateTime" not in calls.columns:
        st.info("Not enough call data to compute contact windows.")
    else:
        c = calls.copy()
        c["dt"] = pd.to_datetime(c["CallDateTime"], errors="coerce")
        c["dow"] = c["dt"].dt.day_name()
        c["hour"] = c["dt"].dt.hour
        if "CallStatusId" in c.columns:
            grp = c.groupby(["dow","hour"]).agg(total=("LeadCallId","count"),
                                               connects=("CallStatusId", lambda s:(s==1).sum())).reset_index()
            grp["connect_rate"] = (grp["connects"]/grp["total"]).round(3)
            # Show top 10 windows by connect rate (min volume filter)
            grp = grp[grp["total"]>=3].sort_values(["connect_rate","total"], ascending=[False,False]).head(10)
            st.dataframe(grp, use_container_width=True, hide_index=True)
        else:
            st.info("CallStatusId not present in call records.")

# -----------------------------------------------------------------------------
# Router
# -----------------------------------------------------------------------------
if HAS_OPTION_MENU:
    if selected=="Executive": 
        # call your existing function
        # show_executive_summary(fdata)
        pass
    elif selected=="Lead Status": 
        # show_lead_status(fdata)
        pass
    elif selected=="AI Calls": 
        # show_calls(fdata)
        pass
    elif selected=="AI Insights": 
        show_ai_insights(fdata)
else:
    with tabs[0]:
        # show_executive_summary(fdata)
        pass
    with tabs[1]:
        # show_lead_status(fdata)
        pass
    with tabs[2]:
        # show_calls(fdata)
        pass
    with tabs[3]:
        show_ai_insights(fdata)
