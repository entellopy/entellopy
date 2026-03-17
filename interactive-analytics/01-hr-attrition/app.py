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

# ─── COLOR SYSTEM ─────────────────────────────────────────────────────────────
# Framework:  deep slate/charcoal — neutral, professional
# Charts:     amber, teal, coral, sage — warm and distinct, never repeat the frame

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif !important; color: #f1f5f9 !important; }

/* ── SIDEBAR: deep charcoal, not blue ── */
[data-testid="stSidebar"] {
    background: #1c1f2e;
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] label { color: #94a3b8 !important; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown div { color: #94a3b8 !important; }

/* fix Streamlit's red "required" asterisks and widget borders */
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
    background-color: #2d3748 !important;
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] span[data-baseweb="tag"] { background: #2d3748 !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] { color: #e2e8f0 !important; }

/* tab bar */
.stTabs [data-baseweb="tab-list"] { background: #1e2535; border-radius: 8px; gap: 4px; }
.stTabs [data-baseweb="tab"] { color: #64748b; border-radius: 6px; font-size: 0.85rem; font-weight:500; }
.stTabs [aria-selected="true"] { background: #334155 !important; color: #f1f5f9 !important; }

/* KPI cards: slate, not blue */
.kpi-card {
    background: #1e2535;
    border: 1px solid rgba(255,255,255,0.08);
    border-top: 2px solid #f59e0b;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
}
.kpi-label { color: #64748b; font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; }
.kpi-value { color: #f1f5f9; font-size: 1.9rem; font-weight: 700; line-height: 1.15; margin: 4px 0 2px; }
.kpi-sub   { color: #94a3b8; font-size: 0.76rem; }

/* insight strip: amber left border, dark bg */
.insight-box {
    background: #1a1f2e;
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.65rem 1rem;
    margin: 0.3rem 0 1.1rem 0;
    color: #94a3b8;
    font-size: 0.87rem;
    line-height: 1.55;
}
.insight-box b { color: #fcd34d; }

/* sidebar insight: teal left border */
.insight-box-sim {
    background: #1a1f2e;
    border-left: 3px solid #2dd4bf;
    border-radius: 0 8px 8px 0;
    padding: 0.65rem 1rem;
    margin: 0.5rem 0 0 0;
    color: #94a3b8;
    font-size: 0.82rem;
    line-height: 1.5;
}
.insight-box-sim b { color: #5eead4; }

.section-label {
    color: #475569;
    font-size: 0.69rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 1.2rem 0 0.4rem 0;
}

/* simulator result card */
.pred-box {
    background: #1e2535;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    margin-top: 0.75rem;
}
</style>
""", unsafe_allow_html=True)


# ─── DATA ─────────────────────────────────────────────────────────────────────
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
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[features], df['Attrition'], test_size=0.2, random_state=42)
    mdl = GradientBoostingClassifier(n_estimators=120, max_depth=4, random_state=42)
    mdl.fit(X_tr, y_tr)
    importance = pd.Series(
        mdl.feature_importances_, index=features).sort_values(ascending=False)
    return mdl, features, importance


df_raw = generate_hr_data()
model, feature_cols, feat_importance = train_model(df_raw)
df_raw['AttritionRisk'] = model.predict_proba(df_raw[feature_cols])[:, 1]
df_raw['RiskTier'] = pd.cut(
    df_raw['AttritionRisk'], bins=[0, .33, .66, 1], labels=['Low', 'Medium', 'High'])


# ─── DYNAMIC INSIGHT GENERATORS ───────────────────────────────────────────────
def insight_attrition(df):
    if len(df) == 0:
        return "No data matches the current filters."
    rate  = df['Attrition'].mean()
    worst = df.groupby('Department')['Attrition'].mean().idxmax()
    wr    = df.groupby('Department')['Attrition'].mean().max()
    dir_  = "above" if rate > 0.13 else "below"
    return (f"Attrition in the selected group is <b>{rate:.1%}</b> — {dir_} the 13% industry benchmark. "
            f"<b>{worst}</b> leads at {wr:.1%}, making it the first department to prioritize.")

def insight_risk(df):
    if len(df) == 0:
        return "No data matches the current filters."
    high_n = (df['RiskTier'] == 'High').sum()
    pct    = high_n / len(df)
    hi_df  = df[df['RiskTier'] == 'High']
    ot_pct = hi_df['OverTime'].mean() if len(hi_df) > 0 else 0
    return (f"<b>{high_n} employees ({pct:.0%})</b> are in the high-risk tier. "
            f"Of those, <b>{ot_pct:.0%} work overtime</b> — burnout is compounding flight risk in this cohort.")

def insight_drivers(df):
    if len(df) == 0:
        return "No data matches the current filters."
    low_eng   = (df['EngagementScore'] <= 2).sum()
    promo_lag = (df['LastPromotion_Years'] > 4).sum()
    return (f"<b>{low_eng} employees</b> score 2 or below on engagement — the model's single strongest signal. "
            f"<b>{promo_lag} others</b> have gone 4+ years without a promotion, compounding long-term risk.")

def insight_learning(df):
    if len(df) == 0:
        return "No data matches the current filters."
    low_mask  = df['Training_Hours'] < 15
    high_mask = df['Training_Hours'] >= 40
    if low_mask.sum() == 0 or high_mask.sum() == 0:
        return "Insufficient training data spread in current selection to compute differential."
    diff = df[low_mask]['AttritionRisk'].mean() - df[high_mask]['AttritionRisk'].mean()
    return (f"Employees with fewer than 15 training hours carry <b>{diff:.0%} higher attrition risk</b> "
            f"than those receiving 40+ hours — the most actionable lever in the current selection.")

def insight_predictor(score, inputs):
    drivers = []
    if inputs['EngagementScore'] <= 2:    drivers.append("low engagement")
    if inputs['OverTime'] == 1:           drivers.append("overtime load")
    if inputs['LastPromotion_Years'] > 4: drivers.append("stalled career progression")
    if inputs['WorkLifeBalance'] <= 2:    drivers.append("poor work-life balance")
    tier       = "high" if score > 0.66 else ("moderate" if score > 0.33 else "low")
    driver_str = ", ".join(drivers) if drivers else "no single dominant signal"
    return f"This profile carries <b>{tier} risk ({score:.0%})</b>. Primary drivers: {driver_str}."


# ─── CHART PALETTE (warm/cool contrast — never the same blue as the frame) ────
TEMPLATE  = "plotly_dark"
BG        = "rgba(0,0,0,0)"
AMBER     = "#f59e0b"
TEAL      = "#2dd4bf"
CORAL     = "#fb923c"
SAGE      = "#86efac"
SLATE_LT  = "#94a3b8"
# Sequential scales that use chart colors, not UI blues
SEQ_AMBER = [[0,"#1c1f2e"],[0.5,"#92400e"],[1,"#fcd34d"]]   # dark → amber
SEQ_TEAL  = [[0,"#1c1f2e"],[0.5,"#0f766e"],[1,"#5eead4"]]   # dark → teal
CAT_COLORS = [AMBER, TEAL, CORAL, SAGE, "#a78bfa", "#f472b6"]


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filters")
    dept_sel   = st.multiselect("Department",  sorted(df_raw['Department'].unique()),
                                default=list(df_raw['Department'].unique()))
    role_sel   = st.multiselect("Role Level",  sorted(df_raw['Role'].unique()),
                                default=list(df_raw['Role'].unique()))
    risk_sel   = st.multiselect("Risk Tier",   ['High', 'Medium', 'Low'],
                                default=['High', 'Medium', 'Low'])
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
    risk_color = "#fb923c" if pred_risk > 0.66 else (AMBER if pred_risk > 0.33 else TEAL)
    tier_label = "HIGH RISK" if pred_risk > 0.66 else ("MODERATE RISK" if pred_risk > 0.33 else "LOW RISK")

    st.markdown(f"""
    <div class="pred-box">
      <div style="color:#475569;font-size:0.69rem;font-weight:600;letter-spacing:0.08em">PREDICTED ATTRITION RISK</div>
      <div style="color:{risk_color};font-size:2.4rem;font-weight:700;margin:4px 0">{pred_risk:.0%}</div>
      <div style="color:{risk_color};font-size:0.8rem;font-weight:600;letter-spacing:0.05em">{tier_label}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(
        f'<div class="insight-box-sim">{insight_predictor(pred_risk, sim_inputs)}</div>',
        unsafe_allow_html=True)


# ─── FILTER ───────────────────────────────────────────────────────────────────
df = df_raw[
    df_raw['Department'].isin(dept_sel) &
    df_raw['Role'].isin(role_sel) &
    df_raw['RiskTier'].isin(risk_sel) &
    df_raw['Tenure_Years'].between(*tenure_rng)
].copy()

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1.2rem 0 0.3rem 0;">
  <h1 style="font-size:2.1rem;margin:0;color:#f1f5f9 !important;">
  Employee Retention Intelligence</h1>
  <p style="color:#334155;font-size:0.88rem;margin-top:5px;">
  AI-driven attrition prediction &nbsp;·&nbsp; Engagement signals &nbsp;·&nbsp; Personalized learning
  </p>
</div>""", unsafe_allow_html=True)

# ─── 3 KPI CARDS ──────────────────────────────────────────────────────────────
total     = max(len(df), 1)
high_risk = (df['RiskTier'] == 'High').sum()
atr_rate  = df['Attrition'].mean() if len(df) > 0 else 0
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
      <div class="kpi-sub">{delta_sign}{(atr_rate - 0.13):.1%} vs 13% industry benchmark</div>
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


# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(f'<div class="insight-box">{insight_attrition(df)}</div>', unsafe_allow_html=True)

    if len(df) == 0:
        st.warning("No data matches the current filters. Adjust the sidebar selections.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-label">Attrition Rate by Department</div>', unsafe_allow_html=True)
            dept_agg = (df.groupby('Department')
                          .agg(Total=('EmployeeID','count'), Left=('Attrition','sum'))
                          .reset_index())
            dept_agg['Rate'] = dept_agg['Left'] / dept_agg['Total']
            fig = px.bar(dept_agg.sort_values('Rate'),
                         x='Rate', y='Department', orientation='h',
                         color='Rate',
                         color_continuous_scale=SEQ_AMBER,
                         text=dept_agg.sort_values('Rate')['Rate'].apply(lambda x: f"{x:.0%}"),
                         template=TEMPLATE)
            fig.update_traces(textposition='outside', textfont_color="#fcd34d")
            fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, coloraxis_showscale=False,
                              xaxis_tickformat='.0%', height=300,
                              margin=dict(l=0, r=50, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-label">Attrition Rate by Tenure Band</div>', unsafe_allow_html=True)
            bins = [0, 1, 2, 4, 7, 10, 20]
            lbls = ['<1yr', '1-2yr', '2-4yr', '4-7yr', '7-10yr', '10+yr']
            df['TenureBin'] = pd.cut(df['Tenure_Years'], bins=bins, labels=lbls)
            t_agg = df.groupby('TenureBin', observed=True)['Attrition'].mean().reset_index()
            fig = px.line(t_agg, x='TenureBin', y='Attrition', markers=True,
                          template=TEMPLATE, color_discrete_sequence=[TEAL])
            fig.update_traces(line_width=3, marker_size=9,
                              marker=dict(color=TEAL, line=dict(color="#1c1f2e", width=2)))
            fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=300,
                              yaxis_tickformat='.0%', yaxis_title="Attrition Rate",
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)


# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown(f'<div class="insight-box">{insight_risk(df)}</div>', unsafe_allow_html=True)

    if len(df) == 0:
        st.warning("No data matches the current filters.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-label">Headcount by Department and Risk Tier</div>', unsafe_allow_html=True)
            heat_df = (df.groupby(['Department', 'RiskTier'], observed=True)
                         .size().unstack(fill_value=0))
            for c in ['High', 'Medium', 'Low']:
                if c not in heat_df.columns:
                    heat_df[c] = 0
            fig = px.imshow(heat_df[['High', 'Medium', 'Low']],
                            color_continuous_scale=SEQ_AMBER,
                            text_auto=True, template=TEMPLATE)
            fig.update_layout(paper_bgcolor=BG, height=320,
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-label">Engagement vs Manager Rating — Avg Risk</div>', unsafe_allow_html=True)
            pivot = (df.groupby(['EngagementScore', 'ManagerRating'])['AttritionRisk']
                       .mean().unstack())
            fig = px.imshow(pivot,
                            color_continuous_scale=SEQ_TEAL,
                            labels=dict(x="Manager Rating", y="Engagement Score", color="Avg Risk"),
                            text_auto='.0%', template=TEMPLATE)
            fig.update_layout(paper_bgcolor=BG, height=320,
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-label">Average Risk Score by Role</div>', unsafe_allow_html=True)
        role_risk = (df.groupby('Role')['AttritionRisk']
                       .mean().sort_values(ascending=False).reset_index())
        fig = px.bar(role_risk, x='Role', y='AttritionRisk',
                     color='AttritionRisk',
                     color_continuous_scale=SEQ_AMBER,
                     text=role_risk['AttritionRisk'].apply(lambda x: f"{x:.0%}"),
                     template=TEMPLATE)
        fig.update_traces(textposition='outside', textfont_color="#fcd34d")
        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=260,
                          yaxis_tickformat='.0%', coloraxis_showscale=False,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)


# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown(f'<div class="insight-box">{insight_drivers(df)}</div>', unsafe_allow_html=True)

    if len(df) == 0:
        st.warning("No data matches the current filters.")
    else:
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown('<div class="section-label">AI Feature Importance — Gradient Boosting Model</div>', unsafe_allow_html=True)
            fi_df = feat_importance.reset_index()
            fi_df.columns = ['Feature', 'Importance']
            # Color bars by rank: top = amber, rest gradient to teal
            n = len(fi_df)
            bar_colors = [AMBER if i == 0 else TEAL if i == n-1 else SLATE_LT
                          for i in range(n - 1, -1, -1)]
            fig = go.Figure(go.Bar(
                x=fi_df['Importance'],
                y=fi_df['Feature'],
                orientation='h',
                marker_color=bar_colors[::-1],
                text=[f"{v:.1%}" for v in fi_df['Importance']],
                textposition='outside',
                textfont=dict(color=AMBER)
            ))
            fig.update_layout(template=TEMPLATE, paper_bgcolor=BG, plot_bgcolor=BG,
                              height=400, xaxis_tickformat='.0%',
                              yaxis=dict(categoryorder='total ascending'),
                              margin=dict(l=0, r=70, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-label">Risk Score Distribution</div>', unsafe_allow_html=True)
            fig = px.histogram(df, x='AttritionRisk',
                               color=df['Attrition'].map({0: 'Stayed', 1: 'Left'}),
                               barmode='overlay', template=TEMPLATE,
                               color_discrete_map={'Stayed': TEAL, 'Left': CORAL},
                               opacity=0.8, nbins=35)
            fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=220,
                              xaxis_tickformat='.0%', xaxis_title='Risk Score',
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-label">Salary vs Risk — colored by Engagement</div>', unsafe_allow_html=True)
            sample = df.sample(min(250, len(df)), random_state=1)
            fig = px.scatter(sample, x='Salary_K', y='AttritionRisk',
                             color='EngagementScore', template=TEMPLATE,
                             color_continuous_scale=SEQ_TEAL, opacity=0.65,
                             hover_data=['Department', 'Role'])
            fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=200,
                              yaxis_tickformat='.0%',
                              margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)


# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown(f'<div class="insight-box">{insight_learning(df)}</div>', unsafe_allow_html=True)

    if len(df) == 0:
        st.warning("No data matches the current filters.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-label">Training Hours vs Attrition Risk and Engagement</div>', unsafe_allow_html=True)
            bins_t = [0, 10, 25, 40, 60, 80]
            lbls_t = ['0-10h', '10-25h', '25-40h', '40-60h', '60-80h']
            df['TrainingBin'] = pd.cut(df['Training_Hours'], bins=bins_t, labels=lbls_t)
            tr_agg = (df.groupby('TrainingBin', observed=True)
                        .agg(AvgRisk=('AttritionRisk', 'mean'),
                             AvgEng=('EngagementScore', 'mean'))
                        .reset_index())
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=tr_agg['TrainingBin'], y=tr_agg['AvgRisk'],
                                  name='Avg Risk', marker_color=AMBER, opacity=0.85),
                          secondary_y=False)
            fig.add_trace(go.Scatter(x=tr_agg['TrainingBin'], y=tr_agg['AvgEng'],
                                      name='Avg Engagement', marker_color=TEAL,
                                      mode='lines+markers', line_width=3,
                                      marker=dict(size=8, color=TEAL,
                                                  line=dict(color="#1c1f2e", width=2))),
                          secondary_y=True)
            fig.update_layout(template=TEMPLATE, paper_bgcolor=BG, plot_bgcolor=BG,
                              height=300, margin=dict(l=0, r=0, t=10, b=0))
            fig.update_yaxes(tickformat='.0%', title_text='Attrition Risk', secondary_y=False)
            fig.update_yaxes(title_text='Engagement', secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-label">AI Intervention Recommendations — High Risk</div>', unsafe_allow_html=True)

            high_risk_df = df[df['RiskTier'] == 'High'].copy()

            if len(high_risk_df) == 0:
                st.info("No high-risk employees in the current selection.")
            else:
                def gen_rec(row):
                    recs = []
                    if row.get('Training_Hours', 40) < 20:
                        recs.append("Enroll in role-aligned upskilling path")
                    if row.get('EngagementScore', 3) <= 2:
                        recs.append("Career pathing session with manager")
                    if row.get('LastPromotion_Years', 0) > 3:
                        recs.append("Promotion readiness review")
                    if row.get('WorkLifeBalance', 3) <= 2:
                        recs.append("Workload and schedule audit")
                    return " / ".join(recs) if recs else "Monitor quarterly"

                recs_df = high_risk_df.head(6)[
                    ['EmployeeID', 'Department', 'Role',
                     'EngagementScore', 'Training_Hours',
                     'LastPromotion_Years', 'AttritionRisk']
                ].copy()
                recs_df['Recommendation'] = recs_df.apply(gen_rec, axis=1)
                recs_df['Risk'] = recs_df['AttritionRisk'].apply(lambda x: f"{x:.0%}")
                st.dataframe(
                    recs_df[['EmployeeID', 'Department', 'Role', 'Risk', 'Recommendation']],
                    use_container_width=True, hide_index=True, height=280)

        st.markdown('<div class="section-label">Full Risk Register</div>', unsafe_allow_html=True)
        show_df = df[['EmployeeID', 'Department', 'Role', 'Age', 'Tenure_Years',
                       'Salary_K', 'EngagementScore', 'OverTime',
                       'AttritionRisk', 'RiskTier']].copy()
        show_df['AttritionRisk'] = show_df['AttritionRisk'].apply(lambda x: f"{x:.0%}")
        show_df['OverTime']      = show_df['OverTime'].map({0: 'No', 1: 'Yes'})
        show_df = show_df.sort_values('AttritionRisk', ascending=False).reset_index(drop=True)
        st.dataframe(show_df, use_container_width=True, hide_index=True,
                     column_config={
                         "AttritionRisk": st.column_config.TextColumn("AI Risk Score"),
                         "Salary_K":      st.column_config.NumberColumn("Salary ($K)", format="$%.0fK"),
                         "Tenure_Years":  st.column_config.NumberColumn("Tenure (yrs)", format="%.1f yrs"),
                     })
        csv = show_df.to_csv(index=False)
        st.download_button("Export Risk Register (CSV)", csv,
                           "attrition_risk_register.csv", "text/csv")


st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#1e293b;font-size:0.76rem;padding:0.3rem 0">
Built with Streamlit · Scikit-learn · Plotly · Synthetic HR data for demonstration
</div>""", unsafe_allow_html=True)
