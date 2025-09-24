# DAR Global — CEO Executive Dashboard (New Dataset)

This repo uses the new CSV suite placed under `data/` and normalizes them to the app’s canonical schema, so all pages and KPIs keep working consistently.  

## Quick start
1. Create folder and copy files:
   - Place `app.py`, `requirements.txt`, `.gitignore`, `README.md` in the repo root.  
   - Put all CSVs into `data/` (Lead.csv, LeadCallRecord.csv, Agents.csv, LeadSchedule.csv, LeadTransaction.csv, LeadStage.csv, CallStatus.csv, Country.csv, TaskStatus.csv, TaskType.csv, LeadStatus.csv, LeadSource.csv, LeadScoring.csv, CallSentiment.csv, TimezoneInfo.csv, CityRegion.csv, Priority.csv, MeetingStatus.csv, AgentMeetingAssignment.csv).  

2. Local run
pip install -r requirements.txt
streamlit run app.py
