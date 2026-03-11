# ══════════════════════════════════════════════════════
# Interactive Training Hub
# Copyright (c) 2025 [Your Name]
# Licensed under CC BY-NC 4.0
# https://creativecommons.org/licenses/by-nc/4.0/
# Commercial use requires explicit written permission.
# ══════════════════════════════════════════════════════

import streamlit as st
import json
import random
import time
from datetime import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Interactive Training Hub",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=DM+Mono&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.main { background: #f0fdfc; color: #134e4a; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #ccfbf1 0%, #99f6e4 100%);
    border-right: 1px solid #5eead4;
}

/* Cards */
.card {
    background: linear-gradient(135deg, #ffffff 0%, #f0fdfa 100%);
    border: 1px solid #99f6e4;
    border-radius: 16px;
    padding: 24px;
    margin: 12px 0;
    transition: all 0.3s ease;
}
.card:hover { border-color: #0d9488; transform: translateY(-2px); box-shadow: 0 8px 24px rgba(13,148,136,0.15); }

/* XP Bar */
.xp-bar-container {
    background: #ccfbf1;
    border-radius: 20px;
    height: 18px;
    overflow: hidden;
    margin: 8px 0;
}
.xp-bar-fill {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(90deg, #0d9488, #06b6d4);
    transition: width 0.6s ease;
}

/* Badges */
.badge {
    display: inline-block;
    background: linear-gradient(135deg, #0d9488, #06b6d4);
    color: white;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78em;
    font-weight: 600;
    margin: 3px;
}
.badge-locked {
    background: #e0f2fe;
    color: #94a3b8;
}

/* Module buttons */
.stButton > button {
    background: linear-gradient(135deg, #0d9488, #06b6d4) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: scale(1.03) !important;
    box-shadow: 0 0 20px rgba(13,148,136,0.4) !important;
}

/* Metric tiles */
.metric-tile {
    background: #ffffff;
    border: 1px solid #99f6e4;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-value {
    font-size: 2em;
    font-weight: 700;
    background: linear-gradient(135deg, #0d9488, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Simulation panels */
.sim-panel {
    background: #f0fdfc;
    border: 1px solid #5eead4;
    border-radius: 12px;
    padding: 20px;
    font-family: 'DM Mono', monospace;
    font-size: 0.88em;
    color: #134e4a;
}

/* Alerts */
.alert-success {
    background: rgba(13,148,136,0.1);
    border: 1px solid #0d9488;
    border-radius: 10px;
    padding: 14px 18px;
    color: #0f766e;
}
.alert-warning {
    background: rgba(245,158,11,0.1);
    border: 1px solid #f59e0b;
    border-radius: 10px;
    padding: 14px 18px;
    color: #b45309;
}
.alert-info {
    background: rgba(6,182,212,0.1);
    border: 1px solid #06b6d4;
    border-radius: 10px;
    padding: 14px 18px;
    color: #0e7490;
}

/* Divider */
.gradient-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #0d9488, transparent);
    margin: 24px 0;
}

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: #134e4a; }
</style>
""", unsafe_allow_html=True)

# ── State initialization ──────────────────────────────────────────────────────
def init_state():
    defaults = {
        "xp": 0,
        "level": 1,
        "completed_modules": [],
        "badges": [],
        "current_module": None,
        "sim_results": {},
        "quiz_scores": {},
        "streak": 0,
        "last_activity": None,
        "username": "Marketer",
        "registered": False,
        "user_email": "",
        "joined_date": "",
        "show_profile": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ══════════════════════════════════════════════════════════════════════════════
# All secrets live in Streamlit Cloud → App Settings → Secrets
# Never hardcode passwords here — this file is public on GitHub
# ══════════════════════════════════════════════════════════════════════════════
def get_secret(key, fallback=""):
    try:
        return st.secrets[key]
    except Exception:
        return fallback

GOOGLE_FORM_URL  = get_secret("GOOGLE_FORM_URL")
FORM_ENTRY_NAME  = get_secret("FORM_ENTRY_NAME")
FORM_ENTRY_EMAIL = get_secret("FORM_ENTRY_EMAIL")
GMAIL_SENDER     = get_secret("GMAIL_SENDER")
GMAIL_APP_PASS   = get_secret("GMAIL_APP_PASS")
GOOGLE_FORM_DELETIONS_URL = get_secret("GOOGLE_FORM_DELETIONS_URL")
FORM_DEL_NAME  = get_secret("FORM_DEL_NAME")
FORM_DEL_EMAIL = get_secret("FORM_DEL_EMAIL")
FORM_DEL_DATE  = get_secret("FORM_DEL_DATE")
# ══════════════════════════════════════════════════════════════════════════════

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import urllib.request, urllib.parse

def submit_to_google_form(name: str, email: str):
    """POST signup to Google Form → auto-saves to your linked Google Sheet."""
    try:
        data = urllib.parse.urlencode({
            FORM_ENTRY_NAME: name,
            FORM_ENTRY_EMAIL: email,
        }).encode()
        req = urllib.request.Request(GOOGLE_FORM_URL, data=data)
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass  # never block the user if this fails

def submit_deletion(name: str, email: str):
    """POST deleted account info to a separate Google Form/Sheet."""
    try:
        data = urllib.parse.urlencode({
            FORM_DEL_NAME:  name,
            FORM_DEL_EMAIL: email,
            FORM_DEL_DATE:  datetime.now().strftime("%Y-%m-%d %H:%M"),
        }).encode()
        req = urllib.request.Request(GOOGLE_FORM_DELETIONS_URL, data=data)
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass

def send_welcome_email(name: str, to_email: str):
    """Send a branded welcome email via Gmail SMTP."""
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Welcome to the Interactive Training Hub, {name}! 🚀"
        msg["From"]    = GMAIL_SENDER
        msg["To"]      = to_email

        html = f"""
        <div style="font-family:sans-serif;max-width:560px;margin:auto;background:#f0fdfc;padding:32px;border-radius:16px;">
          <h1 style="color:#0d9488;">Hey {name}! 🚀</h1>
          <p style="color:#134e4a;font-size:1.05em;">
            You just unlocked access to the <strong>Interactive Training Hub</strong> — 
            the hands-on training platform built to replace boring video lectures.
          </p>
          <hr style="border:1px solid #99f6e4;margin:24px 0;">
          <p style="color:#134e4a;"><strong>What's waiting for you:</strong></p>
          <ul style="color:#0e7490;">
            <li>🎮 Real-world simulations per module</li>
            <li>📝 Quizzes that earn XP</li>
            <li>🏅 Badges to unlock as you progress</li>
            <li>⚡ Level up as you learn</li>
          </ul>
          <p style="color:#94a3b8;font-size:0.82em;margin-top:32px;">
            You received this because you signed up at the Interactive Training Hub.<br>
            No spam, ever. &nbsp;|&nbsp;
            <a href="mailto:{GMAIL_SENDER}?subject=Unsubscribe&body=Please remove me from the Interactive Training Hub mailing list.%0A%0AName: {name}%0AEmail: {to_email}"
               style="color:#0d9488;">Unsubscribe</a>
          </p>
        </div>
        """
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(GMAIL_SENDER, GMAIL_APP_PASS)
            server.sendmail(GMAIL_SENDER, to_email, msg.as_string())
    except Exception:
        pass  # silently fail — never block the user

# ── Email gate ────────────────────────────────────────────────────────────────
def show_signup_gate():
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("""
        <div style='text-align:center; padding: 40px 0 20px 0;'>
            <h1 style='color:#0d9488; font-size:2.2em; margin:10px 0;'>Interactive Training Hub</h1>
            <p style='color:#0e7490; font-size:1.1em;'>Hands-on, interactive training that replaces boring lectures.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='card' style='text-align:center;'>
            <div style='font-size:1.3em; font-weight:700; color:#134e4a; margin-bottom:8px;'>
                🎓 Get Access
            </div>
            <p style='color:#0e7490; margin-bottom:20px;'>
                Real-world simulations · XP and badges · Multiple tracks
            </p>
        </div>
        """, unsafe_allow_html=True)

        name  = st.text_input("👤 Your Name",  placeholder="e.g. María García",    key="gate_name")
        email = st.text_input("📧 Your Email", placeholder="e.g. maria@email.com", key="gate_email")
        st.markdown("")

        if st.button("Start Learning", use_container_width=True):
            if not name.strip():
                st.warning("Please enter your name!")
            elif "@" not in email or "." not in email:
                st.warning("Please enter a valid email address!")
            else:
                st.session_state.registered  = True
                st.session_state.username    = name.strip()
                st.session_state.user_email  = email.strip()
                st.session_state.joined_date = datetime.now().strftime("%B %d, %Y")
                submit_to_google_form(name.strip(), email.strip())
                send_welcome_email(name.strip(), email.strip())
                st.balloons()
                st.rerun()

        st.markdown("""
        <p style='text-align:center; color:#94a3b8; font-size:0.78em; margin-top:16px;'>
            🔒 No spam, ever. We only send course updates.
        </p>
        """, unsafe_allow_html=True)

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='display:flex; justify-content:center; gap:32px; text-align:center;'>
            <div><div style='font-size:1.6em;font-weight:700;color:#0d9488;'>Multi-track</div><div style='color:#64748b;font-size:0.85em;'>Curriculum</div></div>
            <div><div style='font-size:1.6em;font-weight:700;color:#0d9488;'>Hands-on</div><div style='color:#64748b;font-size:0.85em;'>Simulations</div></div>
            <div><div style='font-size:1.6em;font-weight:700;color:#0d9488;'>XP</div><div style='color:#64748b;font-size:0.85em;'>& Badges</div></div>
            <div><div style='font-size:1.6em;font-weight:700;color:#0d9488;'>Learn</div><div style='color:#64748b;font-size:0.85em;'>by Doing</div></div>
        </div>
        """, unsafe_allow_html=True)

# ── Gate check — runs before anything else ────────────────────────────────────
if not st.session_state.registered:
    show_signup_gate()
    st.stop()

# ── XP & Level logic ──────────────────────────────────────────────────────────
XP_PER_LEVEL = 100

def add_xp(amount, reason=""):
    st.session_state.xp += amount
    new_level = (st.session_state.xp // XP_PER_LEVEL) + 1
    if new_level > st.session_state.level:
        st.session_state.level = new_level
        st.balloons()
    check_badges()

def xp_in_current_level():
    return st.session_state.xp % XP_PER_LEVEL

def check_badges():
    badges = st.session_state.badges
    xp = st.session_state.xp
    completed = len(st.session_state.completed_modules)

    new_badges = []
    if xp >= 50 and "🌱 First Steps" not in badges:
        new_badges.append("🌱 First Steps")
    if completed >= 1 and "📘 Module Master" not in badges:
        new_badges.append("📘 Module Master")
    if completed >= 5 and "🔥 Halfway Hero" not in badges:
        new_badges.append("🔥 Halfway Hero")
    if completed >= 10 and "🏆 Digital Marketer" not in badges:
        new_badges.append("🏆 Digital Marketer")
    if xp >= 500 and "⚡ XP Grinder" not in badges:
        new_badges.append("⚡ XP Grinder")

    st.session_state.badges.extend(new_badges)

# ── Module definitions ────────────────────────────────────────────────────────
MODULES = [
    {
        "id": 1, "emoji": "🧭", "title": "Digital Marketing Foundations",
        "tagline": "What is digital marketing and why it dominates",
        "xp": 50,
        "concepts": [
            "Digital marketing is any marketing effort using the internet or electronic devices.",
            "Key channels: SEO, SEM, Social Media, Email, Content, Affiliate, Influencer.",
            "The customer journey: Awareness → Consideration → Conversion → Retention.",
            "KPIs are measurable values that show how effectively you're achieving objectives.",
        ],
        "sim_type": "budget_allocator",
        "quiz": [
            {"q": "Which stage comes AFTER Consideration in the customer journey?",
             "options": ["Awareness", "Conversion", "Retention", "Discovery"],
             "answer": "Conversion", "xp": 15},
            {"q": "CTR stands for:",
             "options": ["Click Through Rate", "Cost To Reach", "Channel Traffic Report", "Customer Trend Rate"],
             "answer": "Click Through Rate", "xp": 15},
        ]
    },
    {
        "id": 2, "emoji": "🔍", "title": "SEO Mastery",
        "tagline": "Get found on Google without paying a cent",
        "xp": 60,
        "concepts": [
            "SEO = Search Engine Optimization. Improving your organic (unpaid) rankings.",
            "On-page SEO: title tags, meta descriptions, header tags, keyword density.",
            "Off-page SEO: backlinks, domain authority, social signals.",
            "Technical SEO: site speed, mobile-friendliness, structured data, crawlability.",
        ],
        "sim_type": "keyword_battle",
        "quiz": [
            {"q": "Which factor is an ON-PAGE SEO element?",
             "options": ["Backlinks from other sites", "Page title tag", "Domain authority", "Press mentions"],
             "answer": "Page title tag", "xp": 15},
            {"q": "A 'long-tail keyword' is typically:",
             "options": ["Short and generic", "3+ words, specific, lower volume", "Only used in paid ads", "A brand name keyword"],
             "answer": "3+ words, specific, lower volume", "xp": 15},
        ]
    },
    {
        "id": 3, "emoji": "💰", "title": "Paid Advertising (SEM & PPC)",
        "tagline": "Spend smart, not hard — master Google & Meta Ads",
        "xp": 70,
        "concepts": [
            "PPC (Pay-Per-Click): you pay each time someone clicks your ad.",
            "Quality Score in Google Ads = relevance of keyword + ad + landing page.",
            "CPC (Cost Per Click) = Total Spend ÷ Total Clicks.",
            "ROAS (Return on Ad Spend) = Revenue from Ads ÷ Ad Spend.",
        ],
        "sim_type": "ad_campaign",
        "quiz": [
            {"q": "You spend $200 and get $800 in revenue from ads. Your ROAS is:",
             "options": ["2x", "4x", "0.25x", "400%"],
             "answer": "4x", "xp": 15},
            {"q": "A higher Quality Score in Google Ads typically leads to:",
             "options": ["Higher CPC", "Lower ad position", "Lower CPC and better position", "More impressions only"],
             "answer": "Lower CPC and better position", "xp": 15},
        ]
    },
    {
        "id": 4, "emoji": "📱", "title": "Social Media Marketing",
        "tagline": "Build communities that convert",
        "xp": 60,
        "concepts": [
            "Each platform has a dominant content type: TikTok=short video, LinkedIn=articles, Instagram=visuals.",
            "Engagement Rate = (Likes + Comments + Shares) ÷ Reach × 100.",
            "Algorithm factors: recency, engagement velocity, content type, user behavior.",
            "Social proof: UGC (User Generated Content) outperforms brand content in trust.",
        ],
        "sim_type": "content_calendar",
        "quiz": [
            {"q": "Which platform is BEST for B2B (business-to-business) marketing?",
             "options": ["TikTok", "Snapchat", "LinkedIn", "Pinterest"],
             "answer": "LinkedIn", "xp": 15},
            {"q": "Engagement Rate is calculated using:",
             "options": ["Followers only", "Interactions divided by reach", "Clicks divided by impressions", "Comments only"],
             "answer": "Interactions divided by reach", "xp": 15},
        ]
    },
    {
        "id": 5, "emoji": "📧", "title": "Email Marketing",
        "tagline": "The highest ROI channel — still king in 2025",
        "xp": 60,
        "concepts": [
            "Email ROI averages $36 for every $1 spent — highest of any digital channel.",
            "Open Rate = Emails Opened ÷ Emails Delivered × 100.",
            "Subject line A/B testing is the #1 lever for improving open rates.",
            "Segmentation: send the right message to the right person at the right time.",
        ],
        "sim_type": "email_ab_test",
        "quiz": [
            {"q": "You send 1,000 emails. 220 are opened. Your open rate is:",
             "options": ["0.22%", "22%", "2.2%", "220%"],
             "answer": "22%", "xp": 15},
            {"q": "Email segmentation means:",
             "options": ["Deleting inactive subscribers", "Dividing your list by behavior or traits", "Sending one email to everyone", "Using HTML templates"],
             "answer": "Dividing your list by behavior or traits", "xp": 15},
        ]
    },
    {
        "id": 6, "emoji": "✍️", "title": "Content Marketing",
        "tagline": "Be the brand people Google, not the brand that Google's you",
        "xp": 60,
        "concepts": [
            "Content marketing = creating valuable content to attract and retain an audience.",
            "Content funnel: TOFU (awareness blogs) → MOFU (case studies) → BOFU (demos, pricing).",
            "Content repurposing: 1 blog → 5 tweets → 1 video → 3 emails → 1 infographic.",
            "E-E-A-T: Experience, Expertise, Authoritativeness, Trustworthiness (Google's standard).",
        ],
        "sim_type": "content_funnel",
        "quiz": [
            {"q": "BOFU content is designed for:",
             "options": ["Brand awareness", "People ready to buy", "Casual readers", "SEO rankings only"],
             "answer": "People ready to buy", "xp": 15},
            {"q": "What does E-E-A-T stand for (Google)?",
             "options": ["Engagement, Efficiency, Authority, Traffic",
                         "Experience, Expertise, Authoritativeness, Trustworthiness",
                         "Email, Ecommerce, Analytics, Testing",
                         "Earned, Explicit, Authentic, Targeted"],
             "answer": "Experience, Expertise, Authoritativeness, Trustworthiness", "xp": 15},
        ]
    },
    {
        "id": 7, "emoji": "📊", "title": "Analytics & Data",
        "tagline": "Numbers don't lie — learn to listen",
        "xp": 70,
        "concepts": [
            "Vanity metrics (likes, followers) vs. actionable metrics (conversion rate, LTV).",
            "GA4: event-based tracking model. Every interaction is an 'event'.",
            "Conversion Rate = Conversions ÷ Total Visitors × 100.",
            "Attribution models: Last-click, First-click, Linear, Data-driven.",
        ],
        "sim_type": "analytics_dashboard",
        "quiz": [
            {"q": "1,000 visitors, 35 purchases. Conversion rate is:",
             "options": ["3.5%", "35%", "0.35%", "350%"],
             "answer": "3.5%", "xp": 15},
            {"q": "Which attribution model gives ALL credit to the last touchpoint?",
             "options": ["Linear", "First-click", "Last-click", "Data-driven"],
             "answer": "Last-click", "xp": 15},
        ]
    },
    {
        "id": 8, "emoji": "🎯", "title": "Conversion Rate Optimization (CRO)",
        "tagline": "Get more from traffic you already have",
        "xp": 70,
        "concepts": [
            "CRO = improving the % of visitors who take a desired action.",
            "A/B testing: show version A to 50% of users, version B to the other 50%.",
            "Heatmaps reveal where users click, scroll, and drop off.",
            "Landing page elements: headline, hero image, benefits, social proof, CTA.",
        ],
        "sim_type": "ab_test_simulator",
        "quiz": [
            {"q": "In an A/B test, statistical significance means:",
             "options": ["The test ran for 7 days", "Results are unlikely due to random chance", "Version B always wins", "You tested two audiences"],
             "answer": "Results are unlikely due to random chance", "xp": 15},
            {"q": "A heatmap is used to:",
             "options": ["Track email opens", "See where users interact on a page", "Measure SEO rankings", "Test ad creatives"],
             "answer": "See where users interact on a page", "xp": 15},
        ]
    },
    {
        "id": 9, "emoji": "🤖", "title": "Marketing Automation & AI",
        "tagline": "Scale yourself — automate what doesn't need you",
        "xp": 75,
        "concepts": [
            "Marketing automation = software that automates repetitive marketing tasks.",
            "Lead scoring: assign points to leads based on behavior and demographics.",
            "Drip campaigns: pre-written email sequences triggered by user actions.",
            "AI in marketing: personalization, predictive analytics, content generation, chatbots.",
        ],
        "sim_type": "lead_scoring",
        "quiz": [
            {"q": "A drip campaign is:",
             "options": ["A social media strategy", "A series of automated triggered emails", "A PPC bidding method", "A content calendar tool"],
             "answer": "A series of automated triggered emails", "xp": 15},
            {"q": "Lead scoring helps marketers:",
             "options": ["Write better ad copy", "Prioritize leads most likely to convert", "Increase website traffic", "Design landing pages"],
             "answer": "Prioritize leads most likely to convert", "xp": 15},
        ]
    },
    {
        "id": 10, "emoji": "🏆", "title": "Strategy & Campaign Planning",
        "tagline": "Bring it all together — build a full campaign",
        "xp": 100,
        "concepts": [
            "A marketing strategy defines WHO you target, WHAT you say, and WHERE you say it.",
            "SMART goals: Specific, Measurable, Achievable, Relevant, Time-bound.",
            "The RACE framework: Reach → Act → Convert → Engage.",
            "Marketing mix (4 Ps): Product, Price, Place, Promotion.",
        ],
        "sim_type": "full_campaign",
        "quiz": [
            {"q": "SMART goals require goals to be:",
             "options": ["Simple and fast", "Specific and measurable (among others)", "Social media focused", "Short-term only"],
             "answer": "Specific and measurable (among others)", "xp": 20},
            {"q": "The RACE framework's first stage 'Reach' focuses on:",
             "options": ["Customer retention", "Building brand awareness", "Conversion optimization", "Email automation"],
             "answer": "Building brand awareness", "xp": 20},
        ]
    },
]

# ── Simulations ───────────────────────────────────────────────────────────────
def sim_budget_allocator(mod_id):
    st.markdown("### 💸 Budget Allocation Simulator")
    st.markdown('<div class="alert-info">You have <strong>$1,000/month</strong> to allocate across marketing channels. Allocate wisely!</div>', unsafe_allow_html=True)
    st.markdown("")

    channels = ["SEO / Content", "Google Ads (PPC)", "Social Media Ads", "Email Marketing", "Influencer Marketing"]
    allocations = {}
    total = 0

    cols = st.columns(2)
    for i, ch in enumerate(channels):
        with cols[i % 2]:
            val = st.slider(ch, 0, 500, 100, 50, key=f"budget_{mod_id}_{i}")
            allocations[ch] = val
            total += val

    st.markdown(f"**Total Allocated: ${total} / $1,000**")
    if total > 1000:
        st.markdown('<div class="alert-warning">⚠️ Over budget! Reduce allocations.</div>', unsafe_allow_html=True)
    elif total == 1000:
        st.markdown('<div class="alert-success">✅ Perfect allocation!</div>', unsafe_allow_html=True)

    if st.button("🚀 Run Simulation", key=f"run_sim_{mod_id}"):
        if total > 1000:
            st.error("You're over budget!")
        else:
            with st.spinner("Simulating 30-day campaign..."):
                time.sleep(1.5)
            # Simulated ROI by channel
            roi_map = {"SEO / Content": 3.5, "Google Ads (PPC)": 2.8, "Social Media Ads": 2.2, "Email Marketing": 4.2, "Influencer Marketing": 1.8}
            st.markdown("#### 📈 30-Day Simulated Results")
            total_revenue = 0
            res_cols = st.columns(len([c for c in channels if allocations[c] > 0]))
            ci = 0
            for ch in channels:
                if allocations[ch] > 0:
                    revenue = round(allocations[ch] * roi_map[ch] * random.uniform(0.85, 1.15), 0)
                    total_revenue += revenue
                    with res_cols[ci]:
                        st.metric(ch, f"${int(revenue)}", f"{roi_map[ch]:.1f}x ROI")
                    ci += 1
            st.success(f"🎉 Total simulated revenue: **${int(total_revenue)}** from **${total}** spend = **{total_revenue/max(total,1):.1f}x ROAS**")
            add_xp(25, "Completed budget simulation")
            st.markdown('<div class="alert-success">+25 XP earned! 🎯</div>', unsafe_allow_html=True)


def sim_keyword_battle(mod_id):
    st.markdown("### 🔍 Keyword Strategy Battle")
    st.markdown('<div class="alert-info">Choose keywords for your blog post. Balance between search volume and competition!</div>', unsafe_allow_html=True)
    st.markdown("")

    keywords = {
        "digital marketing": {"volume": 9900, "difficulty": 85, "cpc": 4.20},
        "digital marketing tips for small business": {"volume": 480, "difficulty": 28, "cpc": 2.10},
        "how to start email marketing": {"volume": 1300, "difficulty": 42, "cpc": 3.50},
        "email marketing software": {"volume": 8100, "difficulty": 78, "cpc": 12.00},
        "best free seo tools 2025": {"volume": 720, "difficulty": 35, "cpc": 5.80},
    }

    selected = st.multiselect("Pick up to 3 keywords to target:", list(keywords.keys()), max_selections=3, key=f"kw_{mod_id}")

    if selected:
        st.markdown("#### Keyword Analysis")
        for kw in selected:
            d = keywords[kw]
            diff_color = "🟢" if d["difficulty"] < 40 else ("🟡" if d["difficulty"] < 65 else "🔴")
            st.markdown(f"""
            <div class="sim-panel">
            <strong>{kw}</strong><br>
            📊 Volume: {d['volume']:,}/mo &nbsp;|&nbsp; 
            {diff_color} Difficulty: {d['difficulty']}/100 &nbsp;|&nbsp; 
            💵 CPC: ${d['cpc']}
            </div>
            """, unsafe_allow_html=True)

        if st.button("⚔️ Evaluate Strategy", key=f"kw_eval_{mod_id}"):
            avg_diff = sum(keywords[k]["difficulty"] for k in selected) / len(selected)
            avg_vol = sum(keywords[k]["volume"] for k in selected) / len(selected)
            score = (avg_vol / 100) * (1 - avg_diff / 100)

            if avg_diff < 45:
                st.success(f"🏆 Smart targeting! Low competition = high chance of ranking. Score: {score:.1f}")
                add_xp(20, "Great keyword strategy")
                st.markdown('<div class="alert-success">+20 XP for smart keyword strategy! 🎯</div>', unsafe_allow_html=True)
            elif avg_diff < 65:
                st.warning(f"⚠️ Moderate competition. Achievable with quality content. Score: {score:.1f}")
                add_xp(10, "Decent keyword strategy")
            else:
                st.error(f"💀 Very high competition. Hard to rank without strong domain authority. Score: {score:.1f}")


def sim_ad_campaign(mod_id):
    st.markdown("### 💰 Google Ads Campaign Simulator")
    st.markdown('<div class="alert-info">Set your campaign parameters and see how they perform!</div>', unsafe_allow_html=True)
    st.markdown("")

    c1, c2 = st.columns(2)
    with c1:
        daily_budget = st.number_input("Daily Budget ($)", 5, 500, 50, key=f"budget_ad_{mod_id}")
        max_cpc = st.slider("Max CPC Bid ($)", 0.10, 10.0, 1.50, 0.10, key=f"cpc_{mod_id}")
    with c2:
        quality_score = st.slider("Quality Score (1-10)", 1, 10, 7, key=f"qs_{mod_id}")
        landing_page_cvr = st.slider("Landing Page CVR (%)", 1, 20, 5, key=f"cvr_{mod_id}")

    if st.button("▶️ Run 7-Day Campaign", key=f"run_ad_{mod_id}"):
        with st.spinner("Running campaign simulation..."):
            time.sleep(1.5)

        # Simulate results
        actual_cpc = max(0.30, max_cpc * (1 - (quality_score - 5) * 0.08))
        daily_clicks = int((daily_budget / actual_cpc) * random.uniform(0.8, 1.1))
        total_clicks = daily_clicks * 7
        total_spend = round(total_clicks * actual_cpc, 2)
        conversions = int(total_clicks * (landing_page_cvr / 100) * random.uniform(0.9, 1.1))
        revenue = conversions * random.randint(40, 120)
        roas = round(revenue / max(total_spend, 1), 2)

        st.markdown("#### 📊 7-Day Campaign Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Clicks", f"{total_clicks:,}")
        m2.metric("Actual CPC", f"${actual_cpc:.2f}", f"vs ${max_cpc:.2f} max")
        m3.metric("Conversions", f"{conversions}")
        m4.metric("ROAS", f"{roas}x")

        if roas >= 3:
            st.success(f"🚀 Profitable campaign! ROAS of {roas}x. Quality Score helps lower your CPC!")
            add_xp(30, "Profitable ad campaign")
            st.markdown('<div class="alert-success">+30 XP! Great campaign setup! 💰</div>', unsafe_allow_html=True)
        elif roas >= 1:
            st.warning(f"⚠️ Breaking even at {roas}x ROAS. Try increasing Quality Score or Landing Page CVR.")
            add_xp(15, "Ran ad simulation")
        else:
            st.error(f"📉 Losing money at {roas}x ROAS. Review your bids, quality score and landing page.")
            add_xp(10, "Ran ad simulation")


def sim_email_ab_test(mod_id):
    st.markdown("### 📧 Email Subject Line A/B Test")
    st.markdown('<div class="alert-info">Write two subject lines and see which performs better!</div>', unsafe_allow_html=True)
    st.markdown("")

    subject_a = st.text_input("Subject Line A:", "Exclusive offer just for you", key=f"sub_a_{mod_id}")
    subject_b = st.text_input("Subject Line B:", "⚡ 48-hour sale: 30% OFF ends tonight", key=f"sub_b_{mod_id}")

    list_size = st.slider("Email list size:", 500, 10000, 2000, 500, key=f"list_{mod_id}")

    if st.button("📨 Send Test Campaign", key=f"send_ab_{mod_id}"):
        with st.spinner("Sending to 50/50 split..."):
            time.sleep(1.5)

        # Score based on heuristics (emoji, urgency, personalization, length)
        def score_subject(s):
            score = 0.18  # base open rate
            if any(c in s for c in ["⚡", "🎁", "🔥", "💥", "✨"]): score += 0.06
            if any(w in s.lower() for w in ["exclusive", "you", "your"]): score += 0.03
            if any(w in s.lower() for w in ["limited", "ends", "today", "tonight", "hours"]): score += 0.05
            if "%" in s or "$" in s: score += 0.04
            if len(s) < 40: score += 0.02
            return min(score + random.uniform(-0.02, 0.02), 0.55)

        rate_a = score_subject(subject_a)
        rate_b = score_subject(subject_b)
        opens_a = int(list_size / 2 * rate_a)
        opens_b = int(list_size / 2 * rate_b)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="metric-tile">
            <div style="color:#aaa;font-size:0.85em">VERSION A</div>
            <div style="font-size:0.9em;margin:8px 0">"{subject_a}"</div>
            <div class="metric-value">{rate_a*100:.1f}%</div>
            <div style="color:#888">open rate · {opens_a:,} opens</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-tile">
            <div style="color:#aaa;font-size:0.85em">VERSION B</div>
            <div style="font-size:0.9em;margin:8px 0">"{subject_b}"</div>
            <div class="metric-value">{rate_b*100:.1f}%</div>
            <div style="color:#888">open rate · {opens_b:,} opens</div>
            </div>""", unsafe_allow_html=True)

        winner = "A" if rate_a > rate_b else "B"
        diff = abs(rate_a - rate_b) * 100
        st.success(f"🏆 Version **{winner}** wins with {diff:.1f}% higher open rate!")
        add_xp(20, "Email A/B test")
        st.markdown('<div class="alert-success">+20 XP for testing! 📧</div>', unsafe_allow_html=True)


def sim_analytics_dashboard(mod_id):
    st.markdown("### 📊 Analytics Interpretation Challenge")
    st.markdown('<div class="alert-info">Below is a month of website analytics. Identify the key insights!</div>', unsafe_allow_html=True)
    st.markdown("")

    # Fake data
    data = {
        "Sessions": [12400, 13200, 11800, 15600, 14900, 16200, 17800],
        "Bounce Rate": [68, 65, 72, 58, 61, 55, 52],
        "Conv. Rate": [2.1, 2.3, 1.9, 3.1, 2.8, 3.4, 3.7],
        "Week": ["W1", "W2", "W3", "W4", "W5", "W6", "W7"],
    }

    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Sessions/Week", "14,557", "+43% vs last month")
    c2.metric("Avg Bounce Rate", "61.6%", "-8% vs last month")
    c3.metric("Avg Conv Rate", "2.76%", "+31% vs last month")

    st.markdown("#### What's the biggest win in this data?")
    answer = st.radio("", [
        "Sessions grew the most, that's the main success",
        "Bounce rate dropped AND conversions rose = better traffic quality",
        "Conversion rate grew but it's still low",
        "Bounce rate is too high at 61%"
    ], key=f"analytics_q_{mod_id}")

    if st.button("✅ Submit Analysis", key=f"analytics_sub_{mod_id}"):
        if "Bounce rate dropped AND conversions rose" in answer:
            st.success("🎯 Correct! When bounce rate drops AND conversions rise, you're getting better quality visitors — or improving the experience. That's a double win!")
            add_xp(25, "Analytics insight")
            st.markdown('<div class="alert-success">+25 XP for sharp analysis! 📊</div>', unsafe_allow_html=True)
        else:
            st.warning("Not quite! The real insight is the combination: lower bounce + higher CVR = quality improvement, not just volume.")
            add_xp(10, "Attempted analytics")


def sim_content_funnel(mod_id):
    st.markdown("### ✍️ Content Funnel Planner")
    st.markdown('<div class="alert-info">Match the content type to the right funnel stage for your product!</div>', unsafe_allow_html=True)

    st.markdown("")
    product = st.selectbox("Your product:", ["SaaS Project Management Tool", "Online Fitness Course", "Local Restaurant", "E-commerce Sneaker Brand"], key=f"prod_{mod_id}")

    st.markdown("#### Assign content to funnel stages:")
    content_options = ["How-to blog post", "Product demo video", "Customer case study",
                       "Comparison page vs competitors", "Social media reel", "Free trial CTA", "Pricing page", "Industry report"]

    tofu = st.multiselect("🔼 TOFU (Top of Funnel — Awareness)", content_options, key=f"tofu_{mod_id}")
    mofu = st.multiselect("🟡 MOFU (Middle — Consideration)", content_options, key=f"mofu_{mod_id}")
    bofu = st.multiselect("🔽 BOFU (Bottom — Decision)", content_options, key=f"bofu_{mod_id}")

    if st.button("📋 Evaluate My Funnel", key=f"eval_funnel_{mod_id}"):
        score = 0
        feedback = []
        good_tofu = {"How-to blog post", "Social media reel", "Industry report"}
        good_mofu = {"Customer case study", "Comparison page vs competitors", "Product demo video"}
        good_bofu = {"Free trial CTA", "Pricing page"}

        score += len(set(tofu) & good_tofu) * 10
        score += len(set(mofu) & good_mofu) * 10
        score += len(set(bofu) & good_bofu) * 10

        if score >= 40:
            st.success(f"🏆 Excellent funnel strategy! Score: {score}/60")
            add_xp(25, "Content funnel")
            st.markdown('<div class="alert-success">+25 XP! You think like a content strategist! ✍️</div>', unsafe_allow_html=True)
        elif score >= 20:
            st.warning(f"👍 Good start! Score: {score}/60 — Pricing pages and CTAs belong at BOFU, how-tos at TOFU.")
            add_xp(15, "Content funnel attempt")
        else:
            st.error(f"📚 Score: {score}/60 — Review the funnel stages. TOFU = awareness, MOFU = comparison, BOFU = buying decision.")
            add_xp(10, "Attempted content funnel")


def sim_ab_test(mod_id):
    st.markdown("### 🎯 A/B Test Design Lab")
    st.markdown('<div class="alert-info">Design a landing page A/B test. Pick ONE element to test!</div>', unsafe_allow_html=True)

    element = st.selectbox("What will you test?", [
        "Headline copy", "CTA button color", "Hero image", "Page layout", "CTA text", "Social proof placement"
    ], key=f"ab_elem_{mod_id}")

    st.markdown(f"**Testing:** {element}")

    hypothesis = st.text_input("Your hypothesis:", f"Changing the {element.lower()} will increase conversions because...", key=f"hyp_{mod_id}")

    sample_size = st.slider("Test sample size per variant:", 100, 5000, 1000, 100, key=f"sample_{mod_id}")

    if st.button("🧪 Run A/B Test", key=f"run_abt_{mod_id}"):
        with st.spinner("Running test..."):
            time.sleep(1.5)

        cvr_a = random.uniform(0.025, 0.045)
        improvement = random.uniform(-0.1, 0.35)
        cvr_b = cvr_a * (1 + improvement)
        conv_a = int(sample_size * cvr_a)
        conv_b = int(sample_size * cvr_b)
        significance = random.choice([True, True, False]) if abs(improvement) > 0.1 else False

        c1, c2 = st.columns(2)
        c1.metric("Control (A)", f"{cvr_a*100:.2f}% CVR", f"{conv_a} conversions")
        c2.metric(f"Variant B ({element})", f"{cvr_b*100:.2f}% CVR", f"{conv_b} conversions", delta_color="normal")

        if significance:
            if improvement > 0:
                st.success(f"✅ Statistically significant! Variant B wins by {improvement*100:.1f}%. Ship it!")
            else:
                st.warning(f"📉 Significant result — but B lost by {abs(improvement)*100:.1f}%. Keep the control.")
        else:
            st.info("⏳ Not yet significant. Run the test longer with more traffic before deciding.")

        add_xp(20, "CRO simulation")
        st.markdown('<div class="alert-success">+20 XP for running your first A/B test! 🧪</div>', unsafe_allow_html=True)


def sim_lead_scoring(mod_id):
    st.markdown("### 🤖 Lead Scoring Engine")
    st.markdown('<div class="alert-info">Score this lead! Assign points to each behavior/attribute.</div>', unsafe_allow_html=True)

    lead = {
        "Job Title": "Marketing Manager",
        "Company Size": "50-200 employees",
        "Visited Pricing Page": "Yes (3 times)",
        "Downloaded Ebook": "Yes",
        "Opened last 3 emails": "2 of 3",
        "Requested Demo": "No",
        "Industry": "SaaS",
    }

    st.markdown("#### Lead Profile:")
    for k, v in lead.items():
        st.markdown(f"- **{k}:** {v}")

    st.markdown("#### Assign points (0-25 each):")
    c1, c2 = st.columns(2)
    scores = {}
    items = list(lead.items())
    for i, (attr, val) in enumerate(items):
        with (c1 if i < 4 else c2):
            scores[attr] = st.slider(f"{attr}", 0, 25, 10, key=f"score_{mod_id}_{i}")

    total_score = sum(scores.values())
    st.markdown(f"### Total Lead Score: **{total_score} / {len(lead)*25}**")

    # Ideal scoring for reference
    ideal = {
        "Job Title": 20, "Company Size": 15, "Visited Pricing Page": 25,
        "Downloaded Ebook": 10, "Opened last 3 emails": 15, "Requested Demo": 25, "Industry": 15
    }

    if st.button("🎯 Evaluate Scoring", key=f"eval_lead_{mod_id}"):
        ideal_total = sum(ideal.values())
        user_total = sum(scores.values())
        diff = sum(abs(scores[k] - ideal[k]) for k in scores)
        accuracy = max(0, 100 - diff)

        st.markdown("#### Expert Scoring vs Yours:")
        for attr in scores:
            delta = scores[attr] - ideal[attr]
            icon = "✅" if abs(delta) <= 5 else ("🔼" if delta > 0 else "🔽")
            st.markdown(f"- **{attr}**: You: {scores[attr]} | Expert: {ideal[attr]} {icon}")

        if accuracy >= 70:
            st.success(f"🏆 Strong scoring! Accuracy: {accuracy:.0f}% — Pricing page visits and demo requests are huge buying signals!")
            add_xp(30, "Lead scoring")
            st.markdown('<div class="alert-success">+30 XP for expert-level lead scoring! 🤖</div>', unsafe_allow_html=True)
        else:
            st.warning(f"Accuracy: {accuracy:.0f}%. Intent signals (pricing visits, demo requests) should score highest!")
            add_xp(15, "Lead scoring attempt")


def sim_content_calendar(mod_id):
    st.markdown("### 📱 Content Calendar Builder")
    st.markdown('<div class="alert-info">Plan a week of content for a brand. Pick the right format for each platform!</div>', unsafe_allow_html=True)

    platforms = {
        "Instagram": {"best": ["Carousel", "Reel", "Story"], "formats": ["Post", "Carousel", "Reel", "Story", "Live"]},
        "LinkedIn": {"best": ["Article", "Document Post"], "formats": ["Post", "Article", "Document Post", "Reel", "Poll"]},
        "TikTok": {"best": ["Short Video", "Trending Audio"], "formats": ["Short Video", "Trending Audio", "Live", "Story", "Text Post"]},
        "Twitter/X": {"best": ["Thread", "Poll"], "formats": ["Tweet", "Thread", "Poll", "Space", "Image Post"]},
    }

    niche = st.selectbox("Brand niche:", ["Tech Startup", "Fashion Brand", "Food & Beverage", "Personal Finance"], key=f"niche_{mod_id}")
    st.markdown(f"**Planning for:** {niche}")

    selections = {}
    cols = st.columns(2)
    for i, (platform, data) in enumerate(platforms.items()):
        with cols[i % 2]:
            selections[platform] = st.selectbox(f"{platform}", data["formats"], key=f"plat_{mod_id}_{i}")

    if st.button("📅 Evaluate Strategy", key=f"eval_cal_{mod_id}"):
        correct = sum(1 for p, fmt in selections.items() if fmt in platforms[p]["best"])
        if correct >= 3:
            st.success(f"🏆 {correct}/4 correct! You understand platform-native content. Reels for Instagram, Articles for LinkedIn!")
            add_xp(20, "Content calendar")
            st.markdown('<div class="alert-success">+20 XP for platform-native thinking! 📱</div>', unsafe_allow_html=True)
        else:
            st.warning(f"{correct}/4 correct. Remember: each platform has native formats that its algorithm favors.")
            add_xp(10, "Content calendar attempt")


def sim_full_campaign(mod_id):
    st.markdown("### 🏆 Full Campaign Planner — Final Challenge")
    st.markdown('<div class="alert-info">Build a complete marketing campaign strategy. This is your capstone challenge!</div>', unsafe_allow_html=True)

    st.markdown("#### Campaign Brief")
    goal = st.selectbox("Campaign Goal:", ["Generate 500 leads", "Drive $10K in sales", "Grow email list by 1,000", "Launch a new product"], key=f"goal_{mod_id}")
    budget = st.number_input("Total Budget ($):", 500, 50000, 5000, key=f"camp_budget_{mod_id}")
    timeline = st.selectbox("Timeline:", ["2 weeks", "1 month", "3 months", "6 months"], key=f"timeline_{mod_id}")

    st.markdown("#### Channel Mix")
    channels_selected = st.multiselect("Select your channels:", [
        "Google Ads", "Meta Ads", "SEO/Content", "Email Marketing", "Influencer", "Organic Social"
    ], key=f"channels_{mod_id}")

    st.markdown("#### Primary KPIs to Track")
    kpis = st.multiselect("Select 3 KPIs:", [
        "CTR", "CPC", "ROAS", "Conversion Rate", "CAC", "LTV", "Open Rate", "Impressions"
    ], max_selections=3, key=f"kpis_{mod_id}")

    if st.button("🚀 Launch Campaign Strategy", key=f"launch_{mod_id}"):
        score = 0
        feedback = []

        if len(channels_selected) >= 2: score += 25
        if len(kpis) == 3: score += 25
        if budget > 0: score += 15

        # Check KPI relevance
        relevant_kpis = {"ROAS", "Conversion Rate", "CAC", "LTV"}
        if len(set(kpis) & relevant_kpis) >= 2:
            score += 35
            feedback.append("✅ Excellent KPI selection — you're focused on business outcomes, not vanity metrics!")
        else:
            feedback.append("⚠️ Focus on outcome KPIs like ROAS, CAC, and LTV over impressions/CTR alone.")

        st.markdown("#### 🎖️ Campaign Evaluation")
        for f in feedback:
            st.markdown(f)

        if score >= 80:
            st.success(f"🏆 OUTSTANDING strategy! Score: {score}/100 — You're ready to run real campaigns!")
            add_xp(60, "Capstone campaign")
            st.markdown('<div class="alert-success">+60 XP! 🏆 Capstone complete! You are a Digital Marketer!</div>', unsafe_allow_html=True)
        elif score >= 50:
            st.warning(f"👍 Good strategy! Score: {score}/100 — Refine your KPIs to focus on ROI metrics.")
            add_xp(35, "Capstone attempt")
        else:
            st.info(f"📚 Score: {score}/100 — Review previous modules and try again!")
            add_xp(20, "Capstone attempt")


SIM_MAP = {
    "budget_allocator": sim_budget_allocator,
    "keyword_battle": sim_keyword_battle,
    "ad_campaign": sim_ad_campaign,
    "email_ab_test": sim_email_ab_test,
    "analytics_dashboard": sim_analytics_dashboard,
    "content_funnel": sim_content_funnel,
    "ab_test_simulator": sim_ab_test,
    "lead_scoring": sim_lead_scoring,
    "content_calendar": sim_content_calendar,
    "full_campaign": sim_full_campaign,
}

# ── Category / module structure ───────────────────────────────────────────────
CATEGORIES = [
    {
        "id": "dm",
        "emoji": "📣",
        "title": "Digital Marketing",
        "available": True,
        "levels": [
            {
                "id": "beginner",
                "label": "🟢 Beginner",
                "modules": [1, 2, 3],   # module IDs
            },
            {
                "id": "intermediate",
                "label": "🟡 Intermediate",
                "modules": [4, 5, 6],
            },
            {
                "id": "advanced",
                "label": "🔴 Advanced",
                "modules": [7, 8, 9, 10],
            },
        ],
    },
    {
        "id": "ai",
        "emoji": "🤖",
        "title": "Artificial Intelligence",
        "available": False,
        "levels": [],
    },
    {
        "id": "da",
        "emoji": "📊",
        "title": "Data Analytics",
        "available": False,
        "levels": [],
    },
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 Interactive Training Hub")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # ── User block (clickable → profile) ──────────────────────────────────────
    initials_sb = "".join([w[0].upper() for w in st.session_state.username.split()[:2]])
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
        <div style="background:linear-gradient(135deg,#0d9488,#06b6d4);
            width:40px;height:40px;border-radius:50%;display:flex;align-items:center;
            justify-content:center;font-weight:700;color:white;font-size:1em;flex-shrink:0;">
            {initials_sb}
        </div>
        <div>
            <div style="font-weight:700;color:#134e4a;">{st.session_state.username}</div>
            <div style="font-size:0.78em;color:#0e7490;">Level {st.session_state.level} Learner</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_p, col_l = st.columns(2)
    with col_p:
        if st.button("👤 Profile", key="sb_profile", use_container_width=True):
            st.session_state.show_profile   = True
            st.session_state.current_module = None
            st.rerun()
    with col_l:
        if st.button("🚪 Log out", key="sb_logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # ── XP bar ────────────────────────────────────────────────────────────────
    xp_pct = xp_in_current_level()
    st.markdown(f"⚡ **{st.session_state.xp} XP** · {xp_pct}/{XP_PER_LEVEL} to next level")
    st.markdown(f"""
    <div class="xp-bar-container">
        <div class="xp-bar-fill" style="width:{xp_pct}%"></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"✅ **{len(st.session_state.completed_modules)}/10** modules complete")

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # ── Category tree ─────────────────────────────────────────────────────────
    st.markdown("#### 🗂️ Training Catalog")
    for cat in CATEGORIES:
        if not cat["available"]:
            st.markdown(f"""
            <div style="padding:8px 4px;opacity:0.5;">
                {cat["emoji"]} <strong>{cat["title"]}</strong>
                <span style="font-size:0.72em;background:#e0f2fe;color:#0369a1;
                border-radius:8px;padding:2px 8px;margin-left:6px;">Coming Soon</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.expander(f"{cat['emoji']} {cat['title']}", expanded=True):
                for level in cat["levels"]:
                    with st.expander(level["label"], expanded=False):
                        for mod_id in level["modules"]:
                            mod = next((m for m in MODULES if m["id"] == mod_id), None)
                            if mod:
                                done = mod["id"] in st.session_state.completed_modules
                                icon = "✅" if done else mod["emoji"]
                                label = f"{icon} {mod['title']}"
                                if st.button(label, key=f"nav_{mod['id']}", use_container_width=True):
                                    st.session_state.current_module = mod["id"]
                                    st.session_state.show_profile   = False
                                    st.rerun()

# ── Profile page ──────────────────────────────────────────────────────────────
def show_profile():
    st.markdown("# 👤 My Profile")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        # Avatar
        initials = "".join([w[0].upper() for w in st.session_state.username.split()[:2]])
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#0d9488,#06b6d4);width:100px;height:100px;
        border-radius:50%;display:flex;align-items:center;justify-content:center;
        font-size:2.2em;font-weight:700;color:white;margin:0 auto 16px auto;text-align:center;'>
        {initials}</div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"### {st.session_state.username}")
        st.markdown(f"📧 {st.session_state.user_email}")
        joined = st.session_state.get("joined_date", "This session")
        st.markdown(f"📅 Joined: {joined}")
        st.markdown(f"⚡ Level {st.session_state.level} Learner")

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Progress stats
    st.markdown("### 📊 My Progress")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"""<div class="metric-tile"><div style="color:#aaa;font-size:.85em">XP Earned</div>
    <div class="metric-value">{st.session_state.xp}</div></div>""", unsafe_allow_html=True)
    m2.markdown(f"""<div class="metric-tile"><div style="color:#aaa;font-size:.85em">Level</div>
    <div class="metric-value">{st.session_state.level}</div></div>""", unsafe_allow_html=True)
    m3.markdown(f"""<div class="metric-tile"><div style="color:#aaa;font-size:.85em">Modules Done</div>
    <div class="metric-value">{len(st.session_state.completed_modules)}/10</div></div>""", unsafe_allow_html=True)
    m4.markdown(f"""<div class="metric-tile"><div style="color:#aaa;font-size:.85em">Badges</div>
    <div class="metric-value">{len(st.session_state.badges)}</div></div>""", unsafe_allow_html=True)

    st.markdown("")

    # Badges
    st.markdown("### 🏅 My Badges")
    all_badges = ["🌱 First Steps","📘 Module Master","🔥 Halfway Hero","🏆 Digital Marketer","⚡ XP Grinder"]
    badge_html = ""
    for b in all_badges:
        cls = "badge" if b in st.session_state.badges else "badge badge-locked"
        label = b if b in st.session_state.badges else f"{b} (locked)"
        badge_html += f'<span class="{cls}">{label}</span> '
    st.markdown(badge_html, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Completed modules list
    if st.session_state.completed_modules:
        st.markdown("### ✅ Completed Modules")
        for mid in st.session_state.completed_modules:
            mod = next((m for m in MODULES if m["id"] == mid), None)
            if mod:
                st.markdown(f"- {mod['emoji']} **Module {mod['id']}:** {mod['title']}")
    else:
        st.markdown('<div class="alert-info">You have not completed any modules yet. Go learn something! 🚀</div>', unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Delete account
    st.markdown("### ⚠️ Danger Zone")
    st.markdown('<div class="alert-warning">Deleting your account resets all progress, XP, and badges. This cannot be undone.</div>', unsafe_allow_html=True)
    st.markdown("")
    confirm = st.checkbox("I understand — delete my account and all my progress", key="delete_confirm")
    if confirm:
        if st.button("🗑️ Delete My Account", key="delete_account"):
            submit_deletion(st.session_state.username, st.session_state.user_email)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ── Main content ──────────────────────────────────────────────────────────────
if st.session_state.get("show_profile", False):
    show_profile()

elif st.session_state.get("current_category") and st.session_state.current_module is None:
    # ── Category landing page ─────────────────────────────────────────────────
    cat = next(c for c in CATEGORIES if c["id"] == st.session_state.current_category)
    if st.button("← Back to Dashboard"):
        st.session_state.current_category = None
        st.rerun()
    st.markdown(f"# {cat['emoji']} {cat['title']}")
    st.markdown("### What's ahead — choose your level and pick a module to start")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    for level in cat["levels"]:
        st.markdown(f"#### {level['label']}")
        level_modules = [m for m in MODULES if m["id"] in level["modules"]]
        cols = st.columns(len(level_modules))
        for col, m in zip(cols, level_modules):
            done = m["id"] in st.session_state.completed_modules
            badge = "✅ Completed" if done else f"🎯 +{m['xp']} XP"
            with col:
                st.markdown(f"""<div class="card">
                <div style="font-size:1.8em">{m['emoji']}</div>
                <div style="font-weight:700;font-size:0.95em">{m['title']}</div>
                <div style="color:#0e7490;font-size:0.82em;margin:6px 0">{m['tagline']}</div>
                <span class="badge">{badge}</span>
                </div>""", unsafe_allow_html=True)
                if st.button(f"Start: {m['title']}", key=f"cat_open_{m['id']}", use_container_width=True):
                    st.session_state.current_module   = m["id"]
                    st.session_state.current_category = None
                    st.session_state.show_profile     = False
                    st.rerun()
        st.markdown("")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

elif st.session_state.current_module is None:
    # ── Home / Dashboard ──────────────────────────────────────────────────────
    st.markdown("# 🚀 Interactive Training Hub")
    st.markdown("### *Hands-on training that replaces boring, passive courses*")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    total_modules = sum(len(l["modules"]) for c in CATEGORIES for l in c["levels"])
    stats = [
        ("🎯 XP Earned", st.session_state.xp),
        ("⚡ Level", st.session_state.level),
        ("📘 Modules Done", f"{len(st.session_state.completed_modules)}/{total_modules}"),
        ("🏅 Badges", len(st.session_state.badges)),
    ]
    for col, (label, val) in zip([c1, c2, c3, c4], stats):
        col.markdown(f"""<div class="metric-tile">
        <div style="color:#aaa;font-size:0.85em">{label}</div>
        <div class="metric-value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("### 📚 Your Training Catalog")
    st.markdown("Click a track to explore its modules, levels and what you will learn.")

    for cat in CATEGORIES:
        st.markdown(f"#### {cat['emoji']} {cat['title']}")
        if not cat["available"]:
            st.markdown('<div class="alert-info">🚧 This track is coming soon! Stay tuned.</div>', unsafe_allow_html=True)
            st.markdown("")
            continue
        col_info, col_btn = st.columns([3, 1])
        with col_info:
            total_cat_modules = sum(len(l["modules"]) for l in cat["levels"])
            done_cat = sum(1 for l in cat["levels"] for mid in l["modules"] if mid in st.session_state.completed_modules)
            st.markdown(f"{total_cat_modules} modules · {len(cat['levels'])} levels · {done_cat}/{total_cat_modules} completed")
        with col_btn:
            if st.button(f"Explore {cat['title']} →", key=f"home_cat_{cat['id']}", use_container_width=True):
                st.session_state.current_category = cat["id"]
                st.session_state.current_module   = None
                st.session_state.show_profile     = False
                st.rerun()
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

else:
    # ── Module View ───────────────────────────────────────────────────────────
    mod = next(m for m in MODULES if m["id"] == st.session_state.current_module)

    if st.button("← Back to Dashboard"):
        st.session_state.current_module = None
        st.session_state.show_profile   = False
        st.rerun()

    st.markdown(f"# {mod['emoji']} Module {mod['id']}: {mod['title']}")
    st.markdown(f"*{mod['tagline']}*")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📖 Learn", "🎮 Simulation", "📝 Quiz"])

    with tab1:
        st.markdown("### 💡 Key Concepts")
        for i, concept in enumerate(mod["concepts"], 1):
            st.markdown(f"""<div class="card">
            <span style="color:#7c3aed;font-weight:700;font-size:1.1em">{i}.</span> {concept}
            </div>""", unsafe_allow_html=True)

        st.markdown("")
        st.markdown('<div class="alert-info">💡 <strong>Pro tip:</strong> Don\'t just read — move to the Simulation tab and apply these concepts hands-on!</div>', unsafe_allow_html=True)

    with tab2:
        SIM_MAP[mod["sim_type"]](mod["id"])

    with tab3:
        st.markdown("### 📝 Knowledge Check")
        all_correct = True
        answers = {}

        for i, q in enumerate(mod["quiz"]):
            st.markdown(f"**Q{i+1}: {q['q']}**")
            answers[i] = st.radio("", q["options"], key=f"quiz_{mod['id']}_{i}", label_visibility="collapsed")
            st.markdown("")

        if st.button("✅ Submit Quiz", key=f"submit_quiz_{mod['id']}"):
            score = 0
            total_xp = 0
            for i, q in enumerate(mod["quiz"]):
                if answers[i] == q["answer"]:
                    st.success(f"Q{i+1}: ✅ Correct! +{q['xp']} XP")
                    score += 1
                    total_xp += q["xp"]
                else:
                    st.error(f"Q{i+1}: ❌ Incorrect. Correct answer: **{q['answer']}**")

            if score == len(mod["quiz"]):
                st.balloons()
                st.success(f"🏆 Perfect score! +{total_xp} XP earned!")
                add_xp(total_xp, f"Perfect quiz Module {mod['id']}")
                if mod["id"] not in st.session_state.completed_modules:
                    st.session_state.completed_modules.append(mod["id"])
                    add_xp(mod["xp"] // 2, "Module completion bonus")
            else:
                st.warning(f"Score: {score}/{len(mod['quiz'])} — Review the concepts and try again!")
                add_xp(total_xp, f"Quiz Module {mod['id']}")

            st.rerun()
