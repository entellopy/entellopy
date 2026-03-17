# AI-Driven Employee Retention Intelligence Dashboard

An interactive Streamlit dashboard using AI/ML to predict employee attrition, surface engagement signals, and generate personalized learning recommendations.

## Live Demo
[View on Streamlit Community Cloud](#) *(replace with your deployed URL)*

## What's Inside

| Tab | Description |
|-----|-------------|
| **Attrition Overview** | Department-level rates, risk distribution, tenure curves, salary vs risk scatter |
| **Risk Segmentation** | Dept × Risk heatmap, role-level risk bars, engagement × manager matrix |
| **AI Feature Drivers** | Gradient Boosting feature importance — what actually predicts attrition |
| **Learning & Engagement** | Training hours vs risk/engagement dual-axis, personalized L&D recommendations |
| **Employee Risk Table** | Sortable register with AI risk scores, exportable to CSV |

## Sidebar: Single Employee Predictor
Adjust any employee profile in the sidebar and get an **instant AI risk score** — great for HR business partner conversations.

## Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/hr-attrition-dashboard
cd hr-attrition-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `app.py` → Deploy
No secrets or API keys needed — runs entirely on synthetic data + local ML model.

## Data & Model
- **Data**: Synthetically generated (500 employees) with realistic HR distributions
- **Model**: Gradient Boosting Classifier (sklearn) trained on 12 behavioral/demographic features
- **Key drivers**: Engagement score, overtime, promotion lag, work-life balance

## Use Cases
- HR Business Partners identifying flight risks before exit interviews
- L&D teams personalizing learning paths for disengaged employees
- People Analytics teams moving from lagging → leading indicators

---
*Built with Streamlit · Plotly · Scikit-learn · Python*
