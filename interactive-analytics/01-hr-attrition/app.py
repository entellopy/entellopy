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

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; color: #e8f0fe !important; }

[data-testid="stSidebar"] {
    background: #1a2f5e;
    border-right: 1px solid rgba(147,197,253,0.15);
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown div { color: #bfdbfe !important; }

.stTabs [data-baseweb="tab-list"] { background: #1e3a6e; border-radius: 8px; gap: 4px; }
.stTabs [data-baseweb="tab"] { color: #93c5fd; border-radius: 6px; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { background: #2563eb !important; color: white !important; }

.kpi-card {
    background: linear-gradient(135deg, #1e3a6e 0%, #1e40af 100%);
    border: 1px solid rgba(147,197,253,0.2);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
}
.kpi-label { color: #93c5fd; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; }
.kpi-value { color: #e8f0fe; font-size: 2rem; font-weight: 700; line-height: 1.1; margin: 4px 0; }
.kpi-sub   { color: #60a5fa; font-size: 0.78rem; }

.insight-box {
    background: #1e3a6e;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0 1.2rem 0;
    color: #bfdbfe;
    font-size: 0.88rem;
    line-height: 1.5;
}
.insight-box b { color: #93c5fd; }

.section-label {
    color: #60a5fa;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 1.4rem 0 0.5rem 0;
}

.pred-box {
    background: linear-gradient(135deg, #1e3a6e, #1e40af);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    margin-top: 0.75rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def generate_hr_data(n=500, seed=42):
    np.random.seed(seed)
    depts = ['Engineering', 'Sales', 'HR', 'Finance', 'Marketing', 'Operations']
    roles = ['Junior', 'Mid-level', 'Senior', 'Lead', 'Manager', 'Director']
    df = pd.DataFrame({
        'EmployeeID':          range(1001, 1001 + n),
        'Department':          np.random.choice(depts, n, p=[0.28,0.22,0.10,0.12,0.15,0.13]),
        'Role':                np.random.choice(roles, n, p=[0.20,0.28,0.22,0.14,0.11,0.05]),
        'Age':                 np.random.randint(22, 58, n),
        'Tenure_Years':        np.random.exponential(4.5, n).clip(0.1, 20).round(1),
        'Salary_K':            np.random.normal(85, 28, n).clip(35, 220).round(1),
        'OverTime':            np.random.choice([0,1], n, p=[0.65,0.35]),
        'LastPromotion_Years': np.random.randint(0, 8, n),
        'Training_Hours':      np.random.randint(0, 80, n),
        'EngagementScore':     np.random.randint(1, 6, n),
        'ManagerRating':       np.random.randint(1, 6, n),
        'WorkLifeBalance':     np.random.randint(1, 5, n),
        'JobSatisfaction':     np.random.randint(1, 5, n),
        'NumProjects':         np.random.randint(1, 8, n),
        'CommuteDistance_km':  np.random.exponential(18, n).clip(1, 80).round(0),
    })
    risk = (
        0.30 * (df['EngagementScore'] < 3).astype(int) +
        0.25 * (df['OverTime'] == 1).astype(int) +
        0.20 * (df['LastPromotion_Years'] > 4).astype(int) +
        0.15 * (df['WorkLifeBalance'] < 3).astype(int) +
        0.10 * (df['Tenure_Years'] < 1.5).astype(int) +
        np.random.normal(0, 0.08, n)
    )
    df['Attrition'] = (risk > 0.35).astype(int)
    return df


@st.cache_resource
def train_model(df):
    features = ['Age','Tenure_Years','Salary_K','OverTime','LastPromotion_Years',
                'Training_Hours','EngagementScore','ManagerRating','WorkLifeBalance',
                'JobSatisfaction','NumProjects','CommuteDistance_km']
    X_tr, X_te, y_tr, y_te = train_test_split(df[features], df['Attrition'], test_size=0.2, random_state=42)
    mdl = GradientBoostingClassifier(n_estimators=120, max_depth=4, random_state=42)
    mdl.fit(X_tr, y_tr)
    importance = pd.Series(mdl.feature_importances_, index=features).sort_values(ascending=False)
    return mdl, features, importance


df_raw = generate_hr_data()
model, feature_cols, feat_importance = train_model(df_raw)
df_raw['AttritionRisk'] = model.predict_proba(df_raw[feature_cols])[:, 1]
df_raw['RiskTier'] = pd.cut(df_raw['AttritionRisk'], bins=[0,.33,.66,1], labels=['Low','Medium','High'])


# ─── INSIGHT GENERATORS (recompute on every filter change) ────────────────────
def insight_attrition(df):
    rate  = df['Attrition'].mean()
    worst = df.groupby('Department')['Attrition'].mean().idxmax()
    worst_rate = df.groupby('Department')['Attrition'].mean().max()
    direction  = "above" if rate > 0.13 else "below"
    return (f"Attrition in the selected group is <b>{rate:.1%}</b> — {direction} the 13% industry benchmark. "
            f"<b>{worst}</b> leads at {worst_rate:.1%}, making it the priority for targeted intervention.")

def insight_risk(df):
    high_n = (df['RiskTier']=='High').sum()
    pct    = high_n / len(df)
    ot_pct = df[df['RiskTier']=='High']['OverTime'].mean()
    return (f"<b>{high_n} employees ({pct:.0%})</b> fall in the high-risk tier. "
            f"Of those, <b>{ot_pct:.0%} work overtime</b> — burnout is compounding flight risk in this cohort.")

def insight_drivers(df):
    low_eng   = (df['EngagementScore'] <= 2).sum()
    promo_lag = (df['LastPromotion_Years'] > 4).sum()
    return (f"<b>{low_eng} employees</b> score 2 or below on engagement — the model's single strongest signal. "
            f"<b>{promo_lag} others</b> have gone 4+ years without a promotion, compounding long-term risk.")

def insight_learning(df):
    low  = df[df['Training_Hours'] < 15]['AttritionRisk'].mean()
    high = df[df['Training_Hours'] >= 40]['AttritionRisk'].mean()
    diff = low - high
    return (f"Employees with fewer than 15 training hours carry <b>{diff:.0%} higher attrition risk</b> "
            f"than those receiving 40+ hours — the most actionable lever within current filters.")

def insight_predictor(score, inputs):
    drivers = []
    if inputs['EngagementScore'] <= 2:     drivers.append("low engagement")
    if inputs['OverTime'] == 1:            drivers.append("overtime load")
    if inputs['LastPromotion_Years'] > 4:  drivers.append("stalled career progression")
    if inputs['WorkLifeBalance'] <= 2:     drivers.append("poor work-life balance")
    tier = "high" if score > 0.66 else ("moderate" if score > 0.33 else "low")
    driver_str = ", ".join(drivers) if drivers else "no single dominant signal"
    return f"This profile carries <b>{tier} risk ({score:.0%})</b>. Primary drivers: {driver_str}."


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    dept_sel   = st.multiselect("Department", sorted(df_raw['Department'].unique()), default=list(df_raw['Department'].unique()))
    role_sel   = st.multiselect("Role Level",  sorted(df_raw['Role'].unique()),      default=list(df_raw['Role'].unique()))
    risk_sel   = st.multiselect("Risk Tier",   ['High','Medium','Low'],              default=['High','Medium','Low'])
    tenure_rng = st.slider("Tenure (Years)", 0.0, 20.0, (0.0, 20.0), 0.5)

    st.markdown("---")
    st.markdown("### Employee Risk Simulator")
    p_age    = st.slider("Age", 22, 60, 32)
    p_tenure = st.slider("Tenure (yrs)", 0.1, 20.0, 3.0, 0.5)
    p_salary = st.slider("Salary ($K)", 35, 220, 80)
    p_eng    = st.slider("Engagement (1-5)", 1, 5, 3)
    p_ot     = st.selectbox("Overtime?", ["No", "Yes"])
    p_promo  = st.slider("Yrs Since Promotion", 0, 8, 2)
    p_wlb    = st.slider("Work-Life Balance (1-4)", 1, 4, 3)
    p_train  = st.slider("Training Hours/yr", 0, 80, 20)
    p_proj   = st.slider("Num Projects", 1, 8, 3)
    p_comm   = st.slider("Commute (km)", 1, 80, 15)
    p_mgr    = st.slider("Manager Rating (1-5)", 1, 5, 3)
    p_js     = st.slider("Job Satisfaction (1-4)", 1, 4, 3)

    sim_inputs = {
        'Age': p_age, 'Tenure_Years': p_tenure, 'Salary_K': p_salary,
        'OverTime': 1 if p_ot == "Yes" else 0, 'LastPromotion_Years': p_promo,
        'Training_Hours': p_train, 'EngagementScore': p_eng,
        'ManagerRating': p_mgr, 'WorkLifeBalance': p_wlb,
        'JobSatisfaction': p_js, 'NumProjects': p_proj, 'CommuteDistance_km': p_comm
    }
    pred_risk  = model.predict_proba(pd.DataFrame([sim_inputs]))[0][1]
    risk_color = "#f87171" if pred_risk > 0.66 else ("#fbbf24" if pred_risk > 0.33 else "#34d399")
    tier_label = "HIGH RISK" if pred_risk > 0.66 else ("MODERATE RISK" if pred_risk > 0.33 else "LOW RISK")

    st.markdown(f"""
    <div class="pred-box">
      <div style="color:#93c5fd;font-size:0.7rem;font-weight:600;letter-spacing:0.08em">PREDICTED ATTRITION RISK</div>
      <div style="color:{risk_color};font-size:2.4rem;font-weight:700;margin:4px 0">{pred_risk:.0%}</div>
      <div style="color:{risk_color};font-size:0.8rem;font-weight:600">{tier_label}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f'<div class="insight-box" style="margin-top:0.6rem">{insight_predictor(pred_risk, sim_inputs)}</div>', unsafe_allow_html=True)


# ─── FILTER DATA ──────────────────────────────────────────────────────────────
df = df_raw[
    df_raw['Department'].isin(dept_sel) &
    df_raw['Role'].isin(role_sel) &
    df_raw['RiskTier'].isin(risk_sel) &
    df_raw['Tenure_Years'].between(*tenure_rng)
].copy()

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1.2rem 0 0.4rem 0;">
  <h1 style="font-size:2.2rem;margin:0;background:linear-gradient(90deg,#60a5fa,#818cf8);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
  Employee Retention Intelligence</h1>
  <p style="color:#4b6cb7;font-size:0.9rem;margin-top:4px;">
  AI-driven attrition prediction · Engagement signals · Personalized learning
  </p>
</div>""", unsafe_allow_html=True)

# ─── 3 KPI CARDS ──────────────────────────────────────────────────────────────
total     = len(df)
high_risk = (df['RiskTier'] == 'High').sum()
atr_rate  = df['Attrition'].mean()
cost_exp  = high_risk * 85

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">High-Risk Employees</div>
      <div class="kpi-value">{high_risk:,}</div>
      <div class="kpi-sub">{high_risk/total:.0%} of selected workforce</div>
    </div>""", unsafe_allow_html=True)
with c2:
    delta_sign = "+" if atr_rate > 0.13 else ""
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Attrition Rate</div>
      <div class="kpi-value">{atr_rate:.1%}</div>
      <div class="kpi-sub">{delta_sign}{(atr_rate-0.13):.1%} vs 13% industry benchmark</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="kpi-card">
      <div class="kpi-label">Estimated Cost Exposure</div>
      <div class="kpi-value">${cost_exp/1000:.1f}M</div>
      <div class="kpi-sub">at $85K avg replacement cost</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Attrition Overview",
    "Risk Segmentation",
    "What Drives Attrition",
    "Learning and Engagement"
])

TEMPLATE = "plotly_dark"
BG       = "rgba(0,0,0,0)"


# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(f'<div class="insight-box">{insight_attrition(df)}</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-label">Attrition Rate by Department</div>', unsafe_allow_html=True)
        dept_agg = df.groupby('Department').agg(Total=('EmployeeID','count'), Left=('Attrition','sum')).reset_index()
        dept_agg['Rate'] = dept_agg['Left'] / dept_agg['Total']
        fig = px.bar(dept_agg.sort_values('Rate'), x='Rate', y='Department', orientation='h',
                     color='Rate', color_continuous_scale='Blues',
                     text=dept_agg.sort_values('Rate')['Rate'].apply(lambda x: f"{x:.0%}"),
                     template=TEMPLATE)
        fig.update_traces(textposition='outside')
        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, coloraxis_showscale=False,
                          xaxis_tickformat='.0%', height=300, margin=dict(l=0,r=50,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-label">Attrition Rate by Tenure Band</div>', unsafe_allow_html=True)
        bins = [0,1,2,4,7,10,20]
        lbls = ['<1yr','1-2yr','2-4yr','4-7yr','7-10yr','10+yr']
        df['TenureBin'] = pd.cut(df['Tenure_Years'], bins=bins, labels=lbls)
        t_agg = df.groupby('TenureBin', observed=True)['Attrition'].mean().reset_index()
        fig = px.line(t_agg, x='TenureBin', y='Attrition', markers=True,
                      template=TEMPLATE, color_discrete_sequence=["#60a5fa"])
        fig.update_traces(line_width=3, marker_size=9)
        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=300,
                          yaxis_tickformat='.0%', yaxis_title="Attrition Rate",
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)


# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(f'<div class="insight-box">{insight_risk(df)}</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-label">Headcount by Department and Risk Tier</div>', unsafe_allow_html=True)
        heat_df = df.groupby(['Department','RiskTier'], observed=True).size().unstack(fill_value=0)
        for c in ['High','Medium','Low']:
            if c not in heat_df.columns: heat_df[c] = 0
        fig = px.imshow(heat_df[['High','Medium','Low']], color_continuous_scale='Blues',
                        text_auto=True, template=TEMPLATE)
        fig.update_layout(paper_bgcolor=BG, height=320, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-label">Engagement vs Manager Rating — Avg Risk</div>', unsafe_allow_html=True)
        pivot = df.groupby(['EngagementScore','ManagerRating'])['AttritionRisk'].mean().unstack()
        fig = px.imshow(pivot, color_continuous_scale='Blues_r',
                        labels=dict(x="Manager Rating", y="Engagement Score", color="Avg Risk"),
                        text_auto='.0%', template=TEMPLATE)
        fig.update_layout(paper_bgcolor=BG, height=320, margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-label">Average Risk Score by Role</div>', unsafe_allow_html=True)
    role_risk = df.groupby('Role')['AttritionRisk'].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(role_risk, x='Role', y='AttritionRisk',
                 color='AttritionRisk', color_continuous_scale='Blues',
                 text=role_risk['AttritionRisk'].apply(lambda x: f"{x:.0%}"),
                 template=TEMPLATE)
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=260,
                      yaxis_tickformat='.0%', coloraxis_showscale=False,
                      margin=dict(l=0,r=0,t=10,b=0))
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)


# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(f'<div class="insight-box">{insight_drivers(df)}</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="section-label">AI Feature Importance — Gradient Boosting Model</div>', unsafe_allow_html=True)
        fi_df = feat_importance.reset_index()
        fi_df.columns = ['Feature','Importance']
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Blues',
                     template=TEMPLATE, text=fi_df['Importance'].apply(lambda x: f"{x:.1%}"))
        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=400,
                          coloraxis_showscale=False, yaxis={'categoryorder':'total ascending'},
                          margin=dict(l=0,r=60,t=10,b=0))
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-label">Risk Score Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x='AttritionRisk',
                           color=df['Attrition'].map({0:'Stayed',1:'Left'}),
                           barmode='overlay', template=TEMPLATE,
                           color_discrete_map={'Stayed':'#3b82f6','Left':'#93c5fd'},
                           opacity=0.75, nbins=35)
        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=220,
                          xaxis_tickformat='.0%', xaxis_title='Risk Score',
                          margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-label">Salary vs Risk — colored by Engagement</div>', unsafe_allow_html=True)
        sample = df.sample(min(250, len(df)), random_state=1)
        fig = px.scatter(sample, x='Salary_K', y='AttritionRisk',
                         color='EngagementScore', template=TEMPLATE,
                         color_continuous_scale='Blues_r', opacity=0.65,
                         hover_data=['Department','Role'])
        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=200,
                          yaxis_tickformat='.0%', margin=dict(l=0,r=0,t=10,b=0))
        st.plotly_chart(fig, use_container_width=True)


# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown(f'<div class="insight-box">{insight_learning(df)}</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-label">Training Hours vs Attrition Risk and Engagement</div>', unsafe_allow_html=True)
        bins_t = [0,10,25,40,60,80]
        lbls_t = ['0-10h','10-25h','25-40h','40-60h','60-80h']
        df['TrainingBin'] = pd.cut(df['Training_Hours'], bins=bins_t, labels=lbls_t)
        tr_agg = df.groupby('TrainingBin', observed=True).agg(
            AvgRisk=('AttritionRisk','mean'), AvgEng=('EngagementScore','mean')
        ).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=tr_agg['TrainingBin'], y=tr_agg['AvgRisk'],
                              name='Avg Risk', marker_color='#3b82f6', opacity=0.8), secondary_y=False)
        fig.add_trace(go.Scatter(x=tr_agg['TrainingBin'], y=tr_agg['AvgEng'],
                                  name='Avg Engagement', marker_color='#93c5fd',
                                  mode='lines+markers', line_width=3), secondary_y=True)
        fig.update_layout(template=TEMPLATE, paper_bgcolor=BG, plot_bgcolor=BG,
                          height=300, margin=dict(l=0,r=0,t=10,b=0))
        fig.update_yaxes(tickformat='.0%', title_text='Attrition Risk', secondary_y=False)
        fig.update_yaxes(title_text='Engagement', secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-label">AI Intervention Recommendations — High Risk</div>', unsafe_allow_html=True)

        def gen_rec(row):
            recs = []
            if row['Training_Hours'] < 20:    recs.append("Enroll in role-aligned upskilling path")
            if row['EngagementScore'] <= 2:   recs.append("Career pathing session with manager")
            if row['LastPromotion_Years'] > 3: recs.append("Promotion readiness review")
            if row['WorkLifeBalance'] <= 2:   recs.append("Workload and schedule audit")
            return " / ".join(recs) if recs else "Monitor quarterly"

        recs_df = df[df['RiskTier']=='High'].head(6)[
            ['EmployeeID','Department','Role','EngagementScore','Training_Hours','LastPromotion_Years','AttritionRisk']
        ].copy()
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
st.markdown("""
<div style="text-align:center;color:#1e3a6e;font-size:0.78rem;padding:0.4rem 0">
Built with Streamlit · Scikit-learn · Plotly · Synthetic HR data for demonstration
</div>""", unsafe_allow_html=True)
