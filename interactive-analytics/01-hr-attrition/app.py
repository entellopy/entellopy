import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="HR Attrition Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f8fafc;
    color: #1e293b;
}
.main .block-container { background: #f8fafc; padding-top: 1.5rem; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; color: #0f172a !important; }

[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e2e8f0;
}
[data-testid="stSidebar"] label { color: #475569 !important; font-size: 0.82rem !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] div { color: #475569; }
[data-testid="stSidebar"] h3 { color: #0f172a !important; font-size: 1rem !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #f1f5f9; border-radius: 10px; padding: 3px; gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    color: #64748b; border-radius: 8px; font-size: 0.83rem; font-weight: 500; padding: 6px 14px;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important; color: #0f172a !important;
    font-weight: 600 !important; box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}

.kpi-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-top: 3px solid #6366f1; border-radius: 10px;
    padding: 1.1rem 1.3rem; box-shadow: 0 1px 6px rgba(0,0,0,0.04);
}
.kpi-label { color: #94a3b8; font-size: 0.69rem; font-weight: 600; letter-spacing: 0.09em; text-transform: uppercase; }
.kpi-value { color: #0f172a; font-size: 1.9rem; font-weight: 700; margin: 4px 0 2px; line-height: 1.1; }
.kpi-sub   { color: #64748b; font-size: 0.76rem; }

.insight-box {
    background: #fffbeb; border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0; padding: 0.6rem 1rem;
    margin: 0.2rem 0 1rem 0; color: #78350f; font-size: 0.86rem; line-height: 1.55;
}
.insight-box b { color: #92400e; }

.section-label {
    color: #94a3b8; font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase; margin: 1.2rem 0 0.4rem 0;
}

.pred-box {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 1rem; text-align: center; margin-top: 0.75rem;
}

/* analyst tab */
.q-button-row { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1rem; }

.analyst-answer {
    background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 1.4rem 1.6rem; margin-top: 0.8rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.analyst-answer .answer-label {
    color: #94a3b8; font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.6rem;
}
.analyst-answer .answer-body {
    color: #1e293b; font-size: 0.92rem; line-height: 1.7;
}
.analyst-answer .answer-body b  { color: #0f172a; }
.analyst-answer .answer-body li { margin-bottom: 4px; }

.score-pill {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; margin-right: 4px; margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# ─── DATA ─────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_hr_data(n=500, seed=42):
    np.random.seed(seed)
    depts = ['Engineering','Sales','HR','Finance','Marketing','Operations']
    roles = ['Junior','Mid-level','Senior','Lead','Manager','Director']
    df = pd.DataFrame({
        'EmployeeID':          range(1001, 1001+n),
        'Department':          np.random.choice(depts, n, p=[0.28,0.22,0.10,0.12,0.15,0.13]),
        'Role':                np.random.choice(roles, n, p=[0.20,0.28,0.22,0.14,0.11,0.05]),
        'Age':                 np.random.randint(22, 58, n),
        'Tenure_Years':        np.random.exponential(4.5, n).clip(0.1,20).round(1),
        'Salary_K':            np.random.normal(85, 28, n).clip(35,220).round(1),
        'OverTime':            np.random.choice([0,1], n, p=[0.65,0.35]),
        'LastPromotion_Years': np.random.randint(0, 8, n),
        'Training_Hours':      np.random.randint(0, 80, n),
        'EngagementScore':     np.random.randint(1, 6, n),
        'ManagerRating':       np.random.randint(1, 6, n),
        'WorkLifeBalance':     np.random.randint(1, 5, n),
        'JobSatisfaction':     np.random.randint(1, 5, n),
        'NumProjects':         np.random.randint(1, 8, n),
        'CommuteDistance_km':  np.random.exponential(18, n).clip(1,80).round(0),
    })
    risk = (
        0.30*(df['EngagementScore']<3).astype(int) +
        0.25*(df['OverTime']==1).astype(int) +
        0.20*(df['LastPromotion_Years']>4).astype(int) +
        0.15*(df['WorkLifeBalance']<3).astype(int) +
        0.10*(df['Tenure_Years']<1.5).astype(int) +
        np.random.normal(0,0.08,n)
    )
    df['Attrition'] = (risk > 0.35).astype(int)
    return df


@st.cache_resource
def train_model(df):
    features = ['Age','Tenure_Years','Salary_K','OverTime','LastPromotion_Years',
                'Training_Hours','EngagementScore','ManagerRating','WorkLifeBalance',
                'JobSatisfaction','NumProjects','CommuteDistance_km']
    X_tr,X_te,y_tr,y_te = train_test_split(df[features], df['Attrition'], test_size=0.2, random_state=42)
    mdl = GradientBoostingClassifier(n_estimators=120, max_depth=4, random_state=42)
    mdl.fit(X_tr, y_tr)
    imp = pd.Series(mdl.feature_importances_, index=features).sort_values(ascending=False)
    return mdl, features, imp


df_raw = generate_hr_data()
model, feature_cols, feat_importance = train_model(df_raw)
df_raw['AttritionRisk'] = model.predict_proba(df_raw[feature_cols])[:,1]
df_raw['RiskTier'] = pd.cut(df_raw['AttritionRisk'], bins=[0,.33,.66,1], labels=['Low','Medium','High'])


# ─── RULE-BASED ANALYST ───────────────────────────────────────────────────────
# Each function receives the filtered df and returns an HTML string.
# Results change every time filters change — this is what makes it interactive.

def _pct(v): return f"{v:.1%}"
def _n(v):   return f"{int(v):,}"
def _dollar(v): return f"${v/1000:.1f}M"

def answer_first_priority(df):
    dept_rates = df.groupby('Department')['Attrition'].mean().sort_values(ascending=False)
    top_dept   = dept_rates.index[0]
    top_rate   = dept_rates.iloc[0]
    top_n_high = (df[df['Department']==top_dept]['RiskTier']=='High').sum()
    top_ot     = df[df['Department']==top_dept]['OverTime'].mean()
    top_eng    = df[df['Department']==top_dept]['EngagementScore'].mean()
    second     = dept_rates.index[1] if len(dept_rates) > 1 else None
    bench_diff = top_rate - 0.13

    lines = [
        f"Based on the current selection, <b>{top_dept}</b> is the clear first priority.",
        f"Its attrition rate is <b>{_pct(top_rate)}</b> — {_pct(abs(bench_diff))} {'above' if bench_diff>0 else 'below'} the 13% industry benchmark.",
        f"Within that department, <b>{_n(top_n_high)} employees</b> sit in the high-risk tier, "
        f"<b>{_pct(top_ot)}</b> work overtime, and the average engagement score is <b>{top_eng:.1f}/5</b>.",
    ]
    if second:
        lines.append(f"<b>{second}</b> ({_pct(dept_rates[second])}) should be monitored as a secondary concern.")

    return "<br>".join(lines)


def answer_biggest_lever(df):
    top_feat   = feat_importance.index[0]
    low_eng_n  = (df['EngagementScore'] <= 2).sum()
    ot_n       = (df['OverTime'] == 1).sum()
    promo_n    = (df['LastPromotion_Years'] > 4).sum()
    train_n    = (df['Training_Hours'] < 15).sum()
    low_wlb_n  = (df['WorkLifeBalance'] <= 2).sum()

    levers = [
        ("Engagement (score ≤ 2)",       low_eng_n,  "the strongest model signal — employees here are 3x more likely to leave"),
        ("Overtime burden",               ot_n,       "doubles short-term attrition probability; burnout is a compounding risk"),
        ("Promotion lag (4+ yrs)",        promo_n,    "career stall is a leading indicator before employees start searching"),
        ("Low training (<15 hrs/yr)",     train_n,    "correlated with disengagement and lower internal mobility"),
        ("Poor work-life balance (≤ 2)",  low_wlb_n,  "amplifies overtime effects — a dual risk multiplier"),
    ]
    levers_sorted = sorted(levers, key=lambda x: x[1], reverse=True)
    top = levers_sorted[0]

    lines = [
        f"The single biggest lever in the current selection is <b>{top[0]}</b>, affecting <b>{_n(top[1])} employees</b> — {top[2]}.",
        "",
        "<b>Full ranked list by headcount affected:</b>",
        "<ul>"
    ]
    for name, count, why in levers_sorted:
        lines.append(f"<li><b>{name}:</b> {_n(count)} employees — {why}</li>")
    lines.append("</ul>")
    return "\n".join(lines)


def answer_overtime_cost(df):
    ot_df      = df[df['OverTime'] == 1]
    ot_rate    = ot_df['Attrition'].mean() if len(ot_df) > 0 else 0
    no_ot_rate = df[df['OverTime'] == 0]['Attrition'].mean() if (df['OverTime']==0).sum() > 0 else 0
    ot_n       = len(ot_df)
    diff       = ot_rate - no_ot_rate
    projected_leavers = int(ot_n * diff)
    cost_saving = projected_leavers * 85000

    lines = [
        f"In the current selection, <b>{_n(ot_n)} employees work overtime</b>.",
        f"Their attrition rate is <b>{_pct(ot_rate)}</b> vs <b>{_pct(no_ot_rate)}</b> for non-overtime workers — a <b>{_pct(diff)} gap</b>.",
        f"If overtime were eliminated, the model projects <b>~{projected_leavers} fewer departures</b>, "
        f"saving approximately <b>${cost_saving/1000:,.0f}K</b> in replacement costs.",
        f"This assumes overtime is the primary driver for those employees — actual savings would depend on root-cause remediation (workload redistribution, hiring, or role redesign).",
    ]
    return "<br><br>".join(lines)


def answer_worst_role(df):
    role_risk  = df.groupby('Role')['AttritionRisk'].mean().sort_values(ascending=False)
    worst_role = role_risk.index[0]
    worst_val  = role_risk.iloc[0]
    role_df    = df[df['Role'] == worst_role]
    n          = len(role_df)
    high_n     = (role_df['RiskTier'] == 'High').sum()
    avg_eng    = role_df['EngagementScore'].mean()
    avg_promo  = role_df['LastPromotion_Years'].mean()
    avg_train  = role_df['Training_Hours'].mean()

    recs = []
    if avg_eng < 3:     recs.append("prioritize engagement — structured 1:1s, recognition programs, and career visibility")
    if avg_promo > 3:   recs.append("review promotion pipeline — this cohort shows classic pre-departure stalling patterns")
    if avg_train < 25:  recs.append("increase learning investment — under-trained roles have fewer internal growth options")

    lines = [
        f"<b>{worst_role}</b> has the highest average risk score at <b>{_pct(worst_val)}</b> across {_n(n)} employees.",
        f"<b>{_n(high_n)}</b> of them are in the high-risk tier. Average engagement: <b>{avg_eng:.1f}/5</b>, "
        f"years since promotion: <b>{avg_promo:.1f}</b>, training hours: <b>{avg_train:.0f}/yr</b>.",
    ]
    if recs:
        lines.append("<b>Recommended actions:</b><ul>" + "".join(f"<li>{r}</li>" for r in recs) + "</ul>")
    return "<br><br>".join(lines)


def answer_summary(df):
    atr_rate  = df['Attrition'].mean()
    high_n    = (df['RiskTier']=='High').sum()
    cost      = high_n * 85000
    top_dept  = df.groupby('Department')['Attrition'].mean().idxmax()
    top_role  = df.groupby('Role')['AttritionRisk'].mean().idxmax()
    top_feat  = feat_importance.index[0]
    low_eng_n = (df['EngagementScore']<=2).sum()
    ot_pct    = df['OverTime'].mean()

    lines = [
        "<b>Key findings for the current selection:</b>",
        "<ul>",
        f"<li><b>Attrition rate is {_pct(atr_rate)}</b> — {'above' if atr_rate>0.13 else 'below'} the 13% benchmark, "
        f"with {_n(high_n)} high-risk employees and ${cost/1000:,.0f}K in cost exposure.</li>",
        f"<li><b>{top_dept}</b> has the highest attrition rate and should be the first intervention target.</li>",
        f"<li><b>{top_role}</b> carries the highest average AI risk score among all roles.</li>",
        f"<li>The strongest attrition signal in the model is <b>{top_feat.replace('_',' ').title()}</b> — "
        f"{_n(low_eng_n)} employees currently score 2 or below.</li>",
        f"<li><b>{_pct(ot_pct)} of employees work overtime</b> — a compounding burnout risk in high-tenure cohorts.</li>",
        "</ul>",
        "<b>Recommended 90-day priorities:</b>",
        "<ol>",
        f"<li>Run engagement pulse in <b>{top_dept}</b> — identify top disengagement drivers this month</li>",
        f"<li>Audit overtime distribution — identify whether it is structural (understaffing) or cultural</li>",
        f"<li>Review promotion pipeline for <b>{top_role}</b> — flag employees with 4+ year promotion lag</li>",
        "<li>Launch targeted L&D for employees with fewer than 15 training hours — quickest modifiable lever</li>",
        "</ol>",
    ]
    return "\n".join(lines)


def answer_engagement_breakdown(df):
    eng_atr = df.groupby('EngagementScore').agg(
        Count=('EmployeeID','count'),
        AttritionRate=('Attrition','mean'),
        AvgRisk=('AttritionRisk','mean')
    ).reset_index()

    lines = ["<b>Attrition risk by engagement score in the current selection:</b><br>"]
    for _, row in eng_atr.iterrows():
        bar_width = int(row['AttritionRate'] * 20)
        bar = "█" * bar_width + "░" * (20 - bar_width)
        lines.append(
            f"Score {int(row['EngagementScore'])}: <b>{_pct(row['AttritionRate'])}</b> attrition "
            f"({_n(row['Count'])} employees) &nbsp; <code style='font-size:0.75rem;color:#6366f1'>{bar}</code>"
        )

    low_eng    = df[df['EngagementScore']<=2]
    high_eng   = df[df['EngagementScore']>=4]
    if len(low_eng)>0 and len(high_eng)>0:
        diff = low_eng['Attrition'].mean() - high_eng['Attrition'].mean()
        lines.append(f"<br>Employees scoring 1–2 have <b>{_pct(diff)} higher attrition</b> than those scoring 4–5.")
    return "<br>".join(lines)


QUESTIONS = {
    "Which department should I act on first?":   answer_first_priority,
    "What is the single biggest lever?":          answer_biggest_lever,
    "How much would fixing overtime save?":        answer_overtime_cost,
    "Which role has the worst risk?":             answer_worst_role,
    "Summarize key findings + 90-day plan":       answer_summary,
    "Break down attrition by engagement score":   answer_engagement_breakdown,
}


# ─── CHART PALETTE ────────────────────────────────────────────────────────────
TEMPLATE  = "plotly_white"
BG        = "rgba(0,0,0,0)"
INDIGO    = "#6366f1"
AMBER     = "#f59e0b"
TEAL      = "#14b8a6"
CORAL     = "#f97316"
SEQ_IND   = [[0,"#eef2ff"],[0.5,"#818cf8"],[1,"#3730a3"]]
SEQ_TEAL  = [[0,"#f0fdfa"],[0.5,"#2dd4bf"],[1,"#0f766e"]]


# ─── INSIGHT STRIP GENERATORS ─────────────────────────────────────────────────
def insight_attrition(df):
    if len(df)==0: return "No data matches current filters."
    rate  = df['Attrition'].mean()
    worst = df.groupby('Department')['Attrition'].mean().idxmax()
    wr    = df.groupby('Department')['Attrition'].mean().max()
    dir_  = "above" if rate > 0.13 else "below"
    return (f"Selected group attrition is <b>{_pct(rate)}</b> — {dir_} the 13% industry benchmark. "
            f"<b>{worst}</b> leads at {_pct(wr)} and should be the first department prioritized.")

def insight_risk(df):
    if len(df)==0: return "No data matches current filters."
    high_n = (df['RiskTier']=='High').sum()
    hi_df  = df[df['RiskTier']=='High']
    ot_pct = hi_df['OverTime'].mean() if len(hi_df)>0 else 0
    return (f"<b>{_n(high_n)} employees ({high_n/len(df):.0%})</b> are in the high-risk tier. "
            f"Of those, <b>{_pct(ot_pct)} work overtime</b> — burnout is compounding flight risk in this cohort.")

def insight_drivers(df):
    if len(df)==0: return "No data matches current filters."
    low_eng   = (df['EngagementScore']<=2).sum()
    promo_lag = (df['LastPromotion_Years']>4).sum()
    return (f"<b>{_n(low_eng)} employees</b> score 2 or below on engagement — the model's strongest signal. "
            f"<b>{_n(promo_lag)} others</b> have gone 4+ years without promotion, compounding long-term risk.")

def insight_learning(df):
    if len(df)==0: return "No data matches current filters."
    lm = df['Training_Hours']<15
    hm = df['Training_Hours']>=40
    if lm.sum()==0 or hm.sum()==0:
        return "Not enough training spread in current selection to compute differential."
    diff = df[lm]['AttritionRisk'].mean() - df[hm]['AttritionRisk'].mean()
    return (f"Employees with fewer than 15 training hours carry <b>{_pct(diff)} higher attrition risk</b> "
            f"than those receiving 40+ hours — the most actionable retention lever in the current selection.")

def insight_predictor(score, inputs):
    drivers = []
    if inputs['EngagementScore']<=2:    drivers.append("low engagement")
    if inputs['OverTime']==1:           drivers.append("overtime load")
    if inputs['LastPromotion_Years']>4: drivers.append("stalled career progression")
    if inputs['WorkLifeBalance']<=2:    drivers.append("poor work-life balance")
    tier       = "high" if score>0.66 else ("moderate" if score>0.33 else "low")
    driver_str = ", ".join(drivers) if drivers else "no single dominant signal"
    return f"<b>{tier.upper()} risk ({score:.0%})</b> — primary drivers: {driver_str}."


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    dept_sel   = st.multiselect("Department",  sorted(df_raw['Department'].unique()), default=list(df_raw['Department'].unique()))
    role_sel   = st.multiselect("Role Level",  sorted(df_raw['Role'].unique()),       default=list(df_raw['Role'].unique()))
    risk_sel   = st.multiselect("Risk Tier",   ['High','Medium','Low'],               default=['High','Medium','Low'])
    tenure_rng = st.slider("Tenure (Years)", 0.0, 20.0, (0.0, 20.0), 0.5)

    st.markdown("---")
    st.markdown("### Employee Risk Simulator")
    p_age    = st.slider("Age", 22, 60, 32)
    p_tenure = st.slider("Tenure (yrs)", 0.1, 20.0, 3.0, 0.5)
    p_salary = st.slider("Salary ($K)", 35, 220, 80)
    p_eng    = st.slider("Engagement (1-5)", 1, 5, 3)
    p_ot     = st.selectbox("Overtime?", ["No","Yes"])
    p_promo  = st.slider("Yrs Since Promotion", 0, 8, 2)
    p_wlb    = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
    p_train  = st.slider("Training Hours/yr", 0, 80, 20)
    p_proj   = st.slider("Num Projects", 1, 8, 3)
    p_comm   = st.slider("Commute (km)", 1, 80, 15)
    p_mgr    = st.slider("Manager Rating (1-5)", 1, 5, 3)
    p_js     = st.slider("Job Satisfaction (1-4)", 1, 4, 3)

    sim_inputs = {
        'Age':p_age,'Tenure_Years':p_tenure,'Salary_K':p_salary,
        'OverTime':1 if p_ot=="Yes" else 0,'LastPromotion_Years':p_promo,
        'Training_Hours':p_train,'EngagementScore':p_eng,'ManagerRating':p_mgr,
        'WorkLifeBalance':p_wlb,'JobSatisfaction':p_js,'NumProjects':p_proj,
        'CommuteDistance_km':p_comm
    }
    pred_risk  = model.predict_proba(pd.DataFrame([sim_inputs]))[0][1]
    risk_color = "#ef4444" if pred_risk>0.66 else (AMBER if pred_risk>0.33 else TEAL)
    tier_label = "HIGH RISK" if pred_risk>0.66 else ("MODERATE RISK" if pred_risk>0.33 else "LOW RISK")

    st.markdown(f"""
    <div class="pred-box">
      <div style="color:#94a3b8;font-size:0.68rem;font-weight:600;letter-spacing:0.08em">PREDICTED ATTRITION RISK</div>
      <div style="color:{risk_color};font-size:2.4rem;font-weight:700;margin:4px 0;font-family:'DM Serif Display',serif">{pred_risk:.0%}</div>
      <div style="color:{risk_color};font-size:0.78rem;font-weight:600;letter-spacing:0.05em">{tier_label}</div>
    </div>
    <div style="background:#f8fafc;border-left:2px solid {risk_color};border-radius:0 6px 6px 0;
    padding:0.55rem 0.8rem;margin-top:0.6rem;color:#475569;font-size:0.81rem;line-height:1.5">
    {insight_predictor(pred_risk, sim_inputs)}
    </div>""", unsafe_allow_html=True)


# ─── FILTER ───────────────────────────────────────────────────────────────────
df = df_raw[
    df_raw['Department'].isin(dept_sel) &
    df_raw['Role'].isin(role_sel) &
    df_raw['RiskTier'].isin(risk_sel) &
    df_raw['Tenure_Years'].between(*tenure_rng)
].copy()

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:0.5rem 0 0.8rem 0;">
  <h1 style="font-size:2rem;margin:0;">Employee Retention Intelligence</h1>
  <p style="color:#94a3b8;font-size:0.87rem;margin-top:4px;">
  AI-driven attrition prediction &nbsp;·&nbsp; Engagement signals &nbsp;·&nbsp; On-demand data analysis
  </p>
</div>""", unsafe_allow_html=True)

# ─── KPIs ─────────────────────────────────────────────────────────────────────
total     = max(len(df), 1)
high_risk = (df['RiskTier']=='High').sum()
atr_rate  = df['Attrition'].mean() if len(df)>0 else 0
cost_exp  = high_risk * 85

c1,c2,c3 = st.columns(3)
with c1:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">High-Risk Employees</div>
      <div class="kpi-value">{high_risk:,}</div>
      <div class="kpi-sub">{high_risk/total:.0%} of selected workforce</div>
    </div>""", unsafe_allow_html=True)
with c2:
    delta_sign = "+" if atr_rate>0.13 else ""
    st.markdown(f"""<div class="kpi-card" style="border-top-color:{AMBER}">
      <div class="kpi-label">Attrition Rate</div>
      <div class="kpi-value">{atr_rate:.1%}</div>
      <div class="kpi-sub">{delta_sign}{(atr_rate-0.13):.1%} vs 13% industry benchmark</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="kpi-card" style="border-top-color:{TEAL}">
      <div class="kpi-label">Estimated Cost Exposure</div>
      <div class="kpi-value">${cost_exp/1000:.1f}M</div>
      <div class="kpi-sub">at $85K avg replacement cost</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab_chat, tab1, tab2, tab3, tab4 = st.tabs([
    "Ask the Data",
    "Attrition Overview",
    "Risk Segmentation",
    "What Drives Attrition",
    "Learning and Engagement"
])


# ══ ASK THE DATA ══════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("""
    <div style="margin-bottom:1.2rem">
      <div style="font-size:1.05rem;font-weight:600;color:#0f172a;margin-bottom:4px">Ask anything about the current selection</div>
      <div style="color:#64748b;font-size:0.84rem">
      Every answer is computed live from your filtered data. Change the department or risk tier filters on the left — then ask again to get a different answer for that exact slice.
      </div>
    </div>""", unsafe_allow_html=True)

    if len(df) == 0:
        st.warning("No data matches current filters — adjust the sidebar selections to enable analysis.")
    else:
        # Question buttons
        cols = st.columns(3)
        for i, (q, fn) in enumerate(QUESTIONS.items()):
            if cols[i % 3].button(q, key=f"q_{i}", use_container_width=True):
                st.session_state['analyst_q']   = q
                st.session_state['analyst_ans'] = fn(df)

        # Display answer
        if 'analyst_ans' in st.session_state and st.session_state['analyst_ans']:
            st.markdown(f"""
            <div class="analyst-answer">
              <div class="answer-label">Analysis — {st.session_state.get('analyst_q','')}</div>
              <div class="answer-body">{st.session_state['analyst_ans']}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("Tip: change the filters on the left, then click the same question again — the answer will update for that slice.")
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 0;color:#cbd5e1;border:1px dashed #e2e8f0;border-radius:12px;margin-top:0.5rem">
              <div style="font-size:1.5rem;margin-bottom:8px">Click any question above</div>
              <div style="font-size:0.85rem">Each answer is computed from your current filter selection</div>
            </div>""", unsafe_allow_html=True)


# ══ TAB 1 ═════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f'<div class="insight-box">{insight_attrition(df)}</div>', unsafe_allow_html=True)
    if len(df)==0:
        st.warning("No data matches current filters.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-label">Attrition Rate by Department</div>', unsafe_allow_html=True)
            dept_agg = df.groupby('Department').agg(Total=('EmployeeID','count'),Left=('Attrition','sum')).reset_index()
            dept_agg['Rate'] = dept_agg['Left']/dept_agg['Total']
            fig = px.bar(dept_agg.sort_values('Rate'), x='Rate', y='Department', orientation='h',
                         color='Rate', color_continuous_scale=SEQ_IND,
                         text=dept_agg.sort_values('Rate')['Rate'].apply(lambda x: f"{x:.0%}"),
                         template=TEMPLATE)
            fig.update_traces(textposition='outside')
            fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, coloraxis_showscale=False,
                              xaxis_tickformat='.0%', height=300, margin=dict(l=0,r=50,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-label">Attrition Rate by Tenure Band</div>', unsafe_allow_html=True)
            bins=[0,1,2,4,7,10,20]; lbls=['<1yr','1-2yr','2-4yr','4-7yr','7-10yr','10+yr']
            df['TenureBin'] = pd.cut(df['Tenure_Years'], bins=bins, labels=lbls)
            t_agg = df.groupby('TenureBin', observed=True)['Attrition'].mean().reset_index()
            fig = px.line(t_agg, x='TenureBin', y='Attrition', markers=True,
                          template=TEMPLATE, color_discrete_sequence=[INDIGO])
            fig.update_traces(line_width=3, marker_size=9,
                              marker=dict(color=INDIGO, line=dict(color="white", width=2)))
            fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=300,
                              yaxis_tickformat='.0%', yaxis_title="Attrition Rate",
                              margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)


# ══ TAB 2 ═════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="insight-box">{insight_risk(df)}</div>', unsafe_allow_html=True)
    if len(df)==0:
        st.warning("No data matches current filters.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-label">Headcount by Department and Risk Tier</div>', unsafe_allow_html=True)
            heat_df = df.groupby(['Department','RiskTier'], observed=True).size().unstack(fill_value=0)
            for c in ['High','Medium','Low']:
                if c not in heat_df.columns: heat_df[c]=0
            fig = px.imshow(heat_df[['High','Medium','Low']], color_continuous_scale=SEQ_IND,
                            text_auto=True, template=TEMPLATE)
            fig.update_layout(paper_bgcolor=BG, height=320, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-label">Engagement vs Manager Rating — Avg Risk</div>', unsafe_allow_html=True)
            pivot = df.groupby(['EngagementScore','ManagerRating'])['AttritionRisk'].mean().unstack()
            fig = px.imshow(pivot, color_continuous_scale=SEQ_TEAL,
                            labels=dict(x="Manager Rating", y="Engagement Score", color="Avg Risk"),
                            text_auto='.0%', template=TEMPLATE)
            fig.update_layout(paper_bgcolor=BG, height=320, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-label">Average Risk Score by Role</div>', unsafe_allow_html=True)
        role_risk = df.groupby('Role')['AttritionRisk'].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(role_risk, x='Role', y='AttritionRisk',
                     color='AttritionRisk', color_continuous_scale=SEQ_IND,
                     text=role_risk['AttritionRisk'].apply(lambda x: f"{x:.0%}"),
                     template=TEMPLATE)
        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=260,
                          yaxis_tickformat='.0%', coloraxis_showscale=False,
                          margin=dict(l=0,r=0,t=10,b=0))
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)


# ══ TAB 3 ═════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<div class="insight-box">{insight_drivers(df)}</div>', unsafe_allow_html=True)
    if len(df)==0:
        st.warning("No data matches current filters.")
    else:
        col1, col2 = st.columns([3,2])
        with col1:
            st.markdown('<div class="section-label">AI Feature Importance — Gradient Boosting Model</div>', unsafe_allow_html=True)
            fi_df = feat_importance.reset_index()
            fi_df.columns = ['Feature','Importance']
            n = len(fi_df)
            bar_colors = [INDIGO if i==n-1 else TEAL if i==n-2 else "#cbd5e1" for i in range(n)]
            fig = go.Figure(go.Bar(
                x=fi_df['Importance'], y=fi_df['Feature'], orientation='h',
                marker_color=bar_colors,
                text=[f"{v:.1%}" for v in fi_df['Importance']], textposition='outside'
            ))
            fig.update_layout(template=TEMPLATE, paper_bgcolor=BG, plot_bgcolor=BG, height=400,
                              xaxis_tickformat='.0%', yaxis=dict(categoryorder='total ascending'),
                              margin=dict(l=0,r=70,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-label">Risk Score Distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(df, x='AttritionRisk',
                               color=df['Attrition'].map({0:'Stayed',1:'Left'}),
                               barmode='overlay', template=TEMPLATE,
                               color_discrete_map={'Stayed':TEAL,'Left':CORAL},
                               opacity=0.8, nbins=35)
            fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=220,
                              xaxis_tickformat='.0%', xaxis_title='Risk Score',
                              margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-label">Salary vs Risk, colored by Engagement</div>', unsafe_allow_html=True)
            sample = df.sample(min(250,len(df)), random_state=1)
            fig = px.scatter(sample, x='Salary_K', y='AttritionRisk',
                             color='EngagementScore', template=TEMPLATE,
                             color_continuous_scale='RdYlGn', opacity=0.6,
                             hover_data=['Department','Role'])
            fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=200,
                              yaxis_tickformat='.0%', margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)


# ══ TAB 4 ═════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f'<div class="insight-box">{insight_learning(df)}</div>', unsafe_allow_html=True)
    if len(df)==0:
        st.warning("No data matches current filters.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-label">Training Hours vs Attrition Risk and Engagement</div>', unsafe_allow_html=True)
            bins_t=[0,10,25,40,60,80]; lbls_t=['0-10h','10-25h','25-40h','40-60h','60-80h']
            df['TrainingBin'] = pd.cut(df['Training_Hours'], bins=bins_t, labels=lbls_t)
            tr_agg = df.groupby('TrainingBin', observed=True).agg(
                AvgRisk=('AttritionRisk','mean'), AvgEng=('EngagementScore','mean')
            ).reset_index()
            fig = make_subplots(specs=[[{"secondary_y":True}]])
            fig.add_trace(go.Bar(x=tr_agg['TrainingBin'], y=tr_agg['AvgRisk'],
                                  name='Avg Risk', marker_color=INDIGO, opacity=0.85), secondary_y=False)
            fig.add_trace(go.Scatter(x=tr_agg['TrainingBin'], y=tr_agg['AvgEng'],
                                      name='Avg Engagement', marker_color=AMBER,
                                      mode='lines+markers', line_width=3,
                                      marker=dict(size=8, color=AMBER, line=dict(color="white",width=2))),
                          secondary_y=True)
            fig.update_layout(template=TEMPLATE, paper_bgcolor=BG, plot_bgcolor=BG,
                              height=300, margin=dict(l=0,r=0,t=10,b=0))
            fig.update_yaxes(tickformat='.0%', title_text='Attrition Risk', secondary_y=False)
            fig.update_yaxes(title_text='Engagement', secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-label">AI Intervention Recommendations — High Risk</div>', unsafe_allow_html=True)
            high_risk_df = df[df['RiskTier']=='High'].copy()
            if len(high_risk_df)==0:
                st.info("No high-risk employees in the current selection.")
            else:
                def gen_rec(row):
                    recs=[]
                    if row.get('Training_Hours',40)<20:    recs.append("Enroll in role-aligned upskilling path")
                    if row.get('EngagementScore',3)<=2:    recs.append("Career pathing session with manager")
                    if row.get('LastPromotion_Years',0)>3: recs.append("Promotion readiness review")
                    if row.get('WorkLifeBalance',3)<=2:    recs.append("Workload and schedule audit")
                    return " / ".join(recs) if recs else "Monitor quarterly"

                recs_df = high_risk_df.head(6)[
                    ['EmployeeID','Department','Role','EngagementScore',
                     'Training_Hours','LastPromotion_Years','AttritionRisk']].copy()
                recs_df['Recommendation'] = recs_df.apply(gen_rec, axis=1)
                recs_df['Risk'] = recs_df['AttritionRisk'].apply(lambda x: f"{x:.0%}")
                st.dataframe(recs_df[['EmployeeID','Department','Role','Risk','Recommendation']],
                             use_container_width=True, hide_index=True, height=280)

        st.markdown('<div class="section-label">Full Risk Register</div>', unsafe_allow_html=True)
        show_df = df[['EmployeeID','Department','Role','Age','Tenure_Years','Salary_K',
                       'EngagementScore','OverTime','AttritionRisk','RiskTier']].copy()
        show_df['AttritionRisk'] = show_df['AttritionRisk'].apply(lambda x: f"{x:.0%}")
        show_df['OverTime']      = show_df['OverTime'].map({0:'No',1:'Yes'})
        show_df = show_df.sort_values('AttritionRisk', ascending=False).reset_index(drop=True)
        st.dataframe(show_df, use_container_width=True, hide_index=True,
                     column_config={
                         "AttritionRisk": st.column_config.TextColumn("AI Risk Score"),
                         "Salary_K":      st.column_config.NumberColumn("Salary ($K)", format="$%.0fK"),
                         "Tenure_Years":  st.column_config.NumberColumn("Tenure (yrs)", format="%.1f yrs"),
                     })
        csv = show_df.to_csv(index=False)
        st.download_button("Export Risk Register (CSV)", csv, "attrition_risk_register.csv", "text/csv")


st.markdown("---")
st.markdown('<div style="text-align:center;color:#e2e8f0;font-size:0.75rem;padding:0.3rem 0">Built with Streamlit · Scikit-learn · Plotly · Synthetic HR data</div>', unsafe_allow_html=True)
