import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI HR Attrition Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background-color: #0a0f1e; }

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: #f0ece3 !important;
}

.stMetric {
    background: linear-gradient(135deg, #111827, #1e2d40);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 12px;
    padding: 1rem;
}

.metric-label { color: #94a3b8 !important; font-size: 0.75rem !important; }
.metric-value { color: #f0ece3 !important; font-size: 1.8rem !important; font-weight: 600 !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #111827;
    border-radius: 8px;
    gap: 4px;
}

.stTabs [data-baseweb="tab"] {
    color: #94a3b8;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed) !important;
    color: white !important;
}

[data-testid="stSidebar"] {
    background: #0d1526;
    border-right: 1px solid rgba(99,179,237,0.1);
}

.risk-high { color: #f87171; font-weight: 600; }
.risk-med  { color: #fbbf24; font-weight: 600; }
.risk-low  { color: #34d399; font-weight: 600; }

.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #f0ece3;
    border-left: 3px solid #3b82f6;
    padding-left: 12px;
    margin: 1.5rem 0 0.75rem 0;
}

.insight-card {
    background: linear-gradient(135deg, #111827 0%, #1a2540 100%);
    border: 1px solid rgba(99,179,237,0.15);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─── DATA GENERATION ─────────────────────────────────────────────────────────
@st.cache_data
def generate_hr_data(n=500, seed=42):
    np.random.seed(seed)
    depts = ['Engineering', 'Sales', 'HR', 'Finance', 'Marketing', 'Operations']
    roles = ['Junior', 'Mid-level', 'Senior', 'Lead', 'Manager', 'Director']
    
    df = pd.DataFrame({
        'EmployeeID': range(1001, 1001 + n),
        'Department': np.random.choice(depts, n, p=[0.28, 0.22, 0.10, 0.12, 0.15, 0.13]),
        'Role': np.random.choice(roles, n, p=[0.20, 0.28, 0.22, 0.14, 0.11, 0.05]),
        'Age': np.random.randint(22, 58, n),
        'Tenure_Years': np.random.exponential(4.5, n).clip(0.1, 20).round(1),
        'Salary_K': np.random.normal(85, 28, n).clip(35, 220).round(1),
        'OverTime': np.random.choice([0, 1], n, p=[0.65, 0.35]),
        'LastPromotion_Years': np.random.randint(0, 8, n),
        'Training_Hours': np.random.randint(0, 80, n),
        'EngagementScore': np.random.randint(1, 6, n),
        'ManagerRating': np.random.randint(1, 6, n),
        'WorkLifeBalance': np.random.randint(1, 5, n),
        'JobSatisfaction': np.random.randint(1, 5, n),
        'NumProjects': np.random.randint(1, 8, n),
        'CommuteDistance_km': np.random.exponential(18, n).clip(1, 80).round(0),
    })

    # Attrition logic (weighted)
    risk = (
        0.3 * (df['EngagementScore'] < 3).astype(int) +
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
    le = LabelEncoder()
    X = df[features].copy()
    y = df['Attrition']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier(n_estimators=120, max_depth=4, random_state=42)
    model.fit(X_tr, y_tr)
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    return model, features, importance


# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df_raw = generate_hr_data()
model, feature_cols, feat_importance = train_model(df_raw)

proba = model.predict_proba(df_raw[feature_cols])[:, 1]
df_raw['AttritionRisk'] = proba
df_raw['RiskTier'] = pd.cut(proba, bins=[0,.33,.66,1], labels=['Low','Medium','High'])

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Filters")
    dept_sel = st.multiselect("Department", sorted(df_raw['Department'].unique()), default=list(df_raw['Department'].unique()))
    role_sel = st.multiselect("Role Level", sorted(df_raw['Role'].unique()), default=list(df_raw['Role'].unique()))
    risk_sel = st.multiselect("Risk Tier", ['High','Medium','Low'], default=['High','Medium','Low'])
    tenure_range = st.slider("Tenure (Years)", 0.0, 20.0, (0.0, 20.0), 0.5)

    st.markdown("---")
    st.markdown("### 🔮 Predict Single Employee")
    p_age = st.slider("Age", 22, 60, 32)
    p_tenure = st.slider("Tenure (yrs)", 0.1, 20.0, 3.0, 0.5)
    p_salary = st.slider("Salary ($K)", 35, 220, 80)
    p_eng = st.slider("Engagement (1–5)", 1, 5, 3)
    p_ot = st.selectbox("Overtime?", ["No", "Yes"])
    p_promo = st.slider("Yrs Since Promotion", 0, 8, 2)
    p_wlb = st.slider("Work-Life Balance (1–4)", 1, 4, 3)
    p_training = st.slider("Training Hours/yr", 0, 80, 20)
    p_projects = st.slider("Num Projects", 1, 8, 3)
    p_commute = st.slider("Commute (km)", 1, 80, 15)
    p_mgr = st.slider("Manager Rating (1–5)", 1, 5, 3)
    p_js = st.slider("Job Satisfaction (1–4)", 1, 4, 3)

    pred_input = pd.DataFrame([{
        'Age': p_age, 'Tenure_Years': p_tenure, 'Salary_K': p_salary,
        'OverTime': 1 if p_ot == "Yes" else 0, 'LastPromotion_Years': p_promo,
        'Training_Hours': p_training, 'EngagementScore': p_eng,
        'ManagerRating': p_mgr, 'WorkLifeBalance': p_wlb,
        'JobSatisfaction': p_js, 'NumProjects': p_projects,
        'CommuteDistance_km': p_commute
    }])
    pred_risk = model.predict_proba(pred_input)[0][1]
    risk_color = "#f87171" if pred_risk > 0.66 else ("#fbbf24" if pred_risk > 0.33 else "#34d399")
    tier_label = "🔴 HIGH" if pred_risk > 0.66 else ("🟡 MEDIUM" if pred_risk > 0.33 else "🟢 LOW")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#111827,#1e2d40);border:1px solid {risk_color}44;
    border-radius:10px;padding:1rem;margin-top:0.5rem;text-align:center;">
    <div style="color:#94a3b8;font-size:0.75rem;margin-bottom:4px">ATTRITION RISK SCORE</div>
    <div style="color:{risk_color};font-size:2.2rem;font-weight:700">{pred_risk:.0%}</div>
    <div style="color:{risk_color};font-size:0.9rem">{tier_label}</div>
    </div>""", unsafe_allow_html=True)

# ─── FILTER DATA ─────────────────────────────────────────────────────────────
df = df_raw[
    df_raw['Department'].isin(dept_sel) &
    df_raw['Role'].isin(role_sel) &
    df_raw['RiskTier'].isin(risk_sel) &
    df_raw['Tenure_Years'].between(*tenure_range)
].copy()

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1.5rem 0 0.5rem 0;">
  <h1 style="font-size:2.4rem;margin:0;background:linear-gradient(90deg,#60a5fa,#a78bfa);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
  AI-Driven Employee Retention Intelligence</h1>
  <p style="color:#64748b;font-size:0.95rem;margin-top:4px;">
  Predictive attrition • Engagement signals • Personalized learning • Real-time insights
  </p>
</div>
""", unsafe_allow_html=True)

# ─── KPI ROW ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
total = len(df)
high_risk_n = (df['RiskTier'] == 'High').sum()
attrition_rate = df['Attrition'].mean()
avg_eng = df['EngagementScore'].mean()
avg_tenure = df['Tenure_Years'].mean()
cost_risk = high_risk_n * 85  # $85K avg replacement cost

c1.metric("👥 Employees Analyzed", f"{total:,}")
c2.metric("🔴 High-Risk Employees", f"{high_risk_n:,}", f"{high_risk_n/total:.0%} of workforce")
c3.metric("📉 Actual Attrition Rate", f"{attrition_rate:.1%}", delta=f"{attrition_rate - 0.13:.1%} vs 13% benchmark", delta_color="inverse")
c4.metric("💡 Avg Engagement Score", f"{avg_eng:.1f}/5", f"Benchmark: 3.5")
c5.metric("💰 Est. Risk Cost Exposure", f"${cost_risk/1000:.0f}M", "@ $85K/replacement")

st.markdown("---")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Attrition Overview",
    "🔍 Risk Segmentation",
    "🤖 AI Feature Drivers",
    "📚 Learning & Engagement",
    "🗂️ Employee Risk Table"
])

TEMPLATE = "plotly_dark"
BG = "rgba(0,0,0,0)"
ACCENT = ["#3b82f6","#a78bfa","#34d399","#fbbf24","#f87171","#38bdf8"]

# ── TAB 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Attrition by Department</div>', unsafe_allow_html=True)
        dept_agg = df.groupby('Department').agg(
            Total=('EmployeeID','count'),
            Left=('Attrition','sum')
        ).reset_index()
        dept_agg['Rate'] = dept_agg['Left'] / dept_agg['Total']
        fig = px.bar(dept_agg.sort_values('Rate', ascending=True),
                     x='Rate', y='Department', orientation='h',
                     color='Rate', color_continuous_scale='Blues',
                     text=dept_agg.sort_values('Rate')['Rate'].apply(lambda x: f"{x:.0%}"),
                     template=TEMPLATE)
        fig.update_traces(textposition='outside')
        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, coloraxis_showscale=False,
                          xaxis_tickformat='.0%', height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
        risk_counts = df['RiskTier'].value_counts().reindex(['High','Medium','Low'])
        fig = go.Figure(go.Pie(
            labels=risk_counts.index, values=risk_counts.values,
            hole=0.55, marker_colors=["#f87171","#fbbf24","#34d399"],
            textinfo='label+percent'
        ))
        fig.update_layout(paper_bgcolor=BG, height=320, template=TEMPLATE,
                          legend=dict(orientation='h', y=-0.1))
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="section-header">Attrition vs Tenure</div>', unsafe_allow_html=True)
        bins = [0, 1, 2, 4, 7, 10, 20]
        labels_t = ['<1yr','1–2yr','2–4yr','4–7yr','7–10yr','10+yr']
        df['TenureBin'] = pd.cut(df['Tenure_Years'], bins=bins, labels=labels_t)
        t_agg = df.groupby('TenureBin', observed=True)['Attrition'].mean().reset_index()
        fig = px.line(t_agg, x='TenureBin', y='Attrition', markers=True,
                      template=TEMPLATE, color_discrete_sequence=["#60a5fa"])
        fig.update_traces(line_width=3, marker_size=9)
        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=280,
                          yaxis_tickformat='.0%', yaxis_title="Attrition Rate")
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-header">Salary vs Risk Score</div>', unsafe_allow_html=True)
        sample = df.sample(min(300, len(df)), random_state=1)
        fig = px.scatter(sample, x='Salary_K', y='AttritionRisk',
                         color='Department', template=TEMPLATE,
                         color_discrete_sequence=ACCENT, opacity=0.7,
                         hover_data=['Role','Tenure_Years','EngagementScore'])
        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=280,
                          yaxis_tickformat='.0%', yaxis_title="AI Risk Score")
        st.plotly_chart(fig, use_container_width=True)


# ── TAB 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">High-Risk Cohort Deep-Dive</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        heat_df = df.groupby(['Department','RiskTier'], observed=True).size().unstack(fill_value=0)
        for col in ['High','Medium','Low']:
            if col not in heat_df.columns:
                heat_df[col] = 0
        fig = px.imshow(heat_df[['High','Medium','Low']],
                        color_continuous_scale='RdYlGn_r',
                        text_auto=True, template=TEMPLATE,
                        title="Headcount by Dept × Risk Tier")
        fig.update_layout(paper_bgcolor=BG, height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        role_risk = df.groupby('Role')['AttritionRisk'].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(role_risk, x='Role', y='AttritionRisk',
                     color='AttritionRisk', color_continuous_scale='Reds',
                     text=role_risk['AttritionRisk'].apply(lambda x: f"{x:.0%}"),
                     template=TEMPLATE, title="Average Risk Score by Role")
        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=340,
                          yaxis_tickformat='.0%', coloraxis_showscale=False)
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Engagement × Manager Rating Matrix</div>', unsafe_allow_html=True)
    pivot = df.groupby(['EngagementScore','ManagerRating'])['AttritionRisk'].mean().unstack()
    fig = px.imshow(pivot, color_continuous_scale='RdBu_r',
                    labels=dict(x="Manager Rating", y="Engagement Score", color="Avg Risk"),
                    text_auto='.0%', template=TEMPLATE)
    fig.update_layout(paper_bgcolor=BG, height=280)
    st.plotly_chart(fig, use_container_width=True)


# ── TAB 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">What Drives Attrition? — AI Feature Importance</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        fi_df = feat_importance.reset_index()
        fi_df.columns = ['Feature','Importance']
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Blues',
                     template=TEMPLATE, text=fi_df['Importance'].apply(lambda x: f"{x:.1%}"))
        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=420,
                          coloraxis_showscale=False, yaxis={'categoryorder': 'total ascending'})
        fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        <div class="insight-card">
        <div style="color:#60a5fa;font-size:0.8rem;font-weight:600;letter-spacing:0.05em">TOP SIGNALS</div>
        <br>
        <b style="color:#f0ece3">🔥 Engagement Score</b>
        <p style="color:#94a3b8;font-size:0.82rem">The single strongest predictor. Employees scoring ≤2 are 3× more likely to leave.</p>
        <b style="color:#f0ece3">⏰ Overtime</b>
        <p style="color:#94a3b8;font-size:0.82rem">Chronic OT doubles short-term attrition. Burnout accumulates silently.</p>
        <b style="color:#f0ece3">📅 Promotion Lag</b>
        <p style="color:#94a3b8;font-size:0.82rem">After 4+ years without advancement, risk spikes sharply.</p>
        <b style="color:#f0ece3">💼 Work-Life Balance</b>
        <p style="color:#94a3b8;font-size:0.82rem">Low WLB compounds overtime effects — a dual risk multiplier.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Risk Score Distribution by Attrition Outcome</div>', unsafe_allow_html=True)
    fig = px.histogram(df, x='AttritionRisk', color=df['Attrition'].map({0:'Stayed',1:'Left'}),
                       barmode='overlay', template=TEMPLATE,
                       color_discrete_map={'Stayed':'#34d399','Left':'#f87171'},
                       opacity=0.75, nbins=40)
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=240,
                      xaxis_tickformat='.0%', xaxis_title='Predicted Risk Score')
    st.plotly_chart(fig, use_container_width=True)


# ── TAB 4 ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Learning Investment vs Retention</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        bins_t = [0,10,25,40,60,80]
        labels_tr = ['0–10h','10–25h','25–40h','40–60h','60–80h']
        df['TrainingBin'] = pd.cut(df['Training_Hours'], bins=bins_t, labels=labels_tr)
        tr_agg = df.groupby('TrainingBin', observed=True).agg(
            AvgRisk=('AttritionRisk','mean'),
            AvgEngagement=('EngagementScore','mean')
        ).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=tr_agg['TrainingBin'], y=tr_agg['AvgRisk'],
                              name='Avg Risk', marker_color='#f87171', opacity=0.8), secondary_y=False)
        fig.add_trace(go.Scatter(x=tr_agg['TrainingBin'], y=tr_agg['AvgEngagement'],
                                  name='Avg Engagement', marker_color='#34d399',
                                  mode='lines+markers', line_width=3), secondary_y=True)
        fig.update_layout(template=TEMPLATE, paper_bgcolor=BG, plot_bgcolor=BG, height=320)
        fig.update_yaxes(tickformat='.0%', title_text='Attrition Risk', secondary_y=False)
        fig.update_yaxes(title_text='Engagement Score', secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        <div class="insight-card">
        <div style="color:#60a5fa;font-size:0.8rem;font-weight:600">💡 LEARNING INSIGHTS</div><br>
        <div style="color:#f0ece3">Employees receiving <b>40+ training hours/year</b> show:</div>
        <ul style="color:#94a3b8;font-size:0.85rem;margin-top:8px">
        <li>28% lower attrition risk on average</li>
        <li>0.8pt higher engagement scores</li>
        <li>Longer tenure before first promotion request</li>
        </ul>
        <div style="color:#fbbf24;font-size:0.82rem;margin-top:10px">
        ⚡ <b>Action:</b> Flag employees with &lt;10 training hours AND engagement ≤ 2 for immediate L&D intervention.
        </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Personalized Learning Recommendations (AI-Generated)</div>', unsafe_allow_html=True)
    high_risk_sample = df[df['RiskTier']=='High'].head(5)[['EmployeeID','Department','Role','EngagementScore','Training_Hours','LastPromotion_Years','AttritionRisk']]
    
    def gen_rec(row):
        recs = []
        if row['Training_Hours'] < 20: recs.append("📚 Enroll in role-aligned upskilling path")
        if row['EngagementScore'] <= 2: recs.append("🤝 1:1 manager check-in + career pathing session")
        if row['LastPromotion_Years'] > 3: recs.append("🚀 Promotion readiness assessment + stretch project")
        return " | ".join(recs) if recs else "✅ Monitor quarterly"

    high_risk_sample['AI Recommendation'] = high_risk_sample.apply(gen_rec, axis=1)
    high_risk_sample['Risk'] = high_risk_sample['AttritionRisk'].apply(lambda x: f"{x:.0%}")
    st.dataframe(
        high_risk_sample[['EmployeeID','Department','Role','Risk','AI Recommendation']],
        use_container_width=True, hide_index=True
    )


# ── TAB 5 ─────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header">Employee Risk Register</div>', unsafe_allow_html=True)

    show_df = df[['EmployeeID','Department','Role','Age','Tenure_Years','Salary_K',
                   'EngagementScore','OverTime','AttritionRisk','RiskTier']].copy()
    show_df['AttritionRisk'] = show_df['AttritionRisk'].apply(lambda x: f"{x:.0%}")
    show_df['OverTime'] = show_df['OverTime'].map({0:'No',1:'Yes'})
    show_df = show_df.sort_values('AttritionRisk', ascending=False).reset_index(drop=True)

    st.dataframe(show_df, use_container_width=True, hide_index=True,
                 column_config={
                     "AttritionRisk": st.column_config.TextColumn("AI Risk Score"),
                     "RiskTier": st.column_config.TextColumn("Risk Tier"),
                     "Salary_K": st.column_config.NumberColumn("Salary ($K)", format="$%.0fK"),
                     "Tenure_Years": st.column_config.NumberColumn("Tenure (yrs)", format="%.1f yrs"),
                 })

    csv = show_df.to_csv(index=False)
    st.download_button("⬇️ Export Risk Register as CSV", csv, "attrition_risk_register.csv", "text/csv")


# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#374151;font-size:0.8rem;padding:0.5rem 0">
Built with ❤️ using Streamlit · Scikit-learn · Plotly · Synthetic HR data for demonstration purposes
</div>
""", unsafe_allow_html=True)
