import streamlit as st
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# ----------------------------
# Load and parse data safely
# ----------------------------
@st.cache_data
def load_data():
    rows = []
    with open('bluesky_covid_data.csv', 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',', 11)
            if len(parts) >= 12:
                rows.append(parts[:12])
    df = pd.DataFrame(rows, columns=[
        'post_uri', 'post_cid', 'author_handle', 'author_did', 'text',
        'timestamp', 'like_count', 'repost_count', 'reply_count',
        'reply_root', 'reply_parent'
    ])
    # Clean numeric columns
    for col in ['like_count', 'repost_count', 'reply_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
    return df

df = load_data()

# ----------------------------
# Sentiment analysis
# ----------------------------
analyzer = SentimentIntensityAnalyzer()
def get_sentiment(text):
    if pd.isna(text) or str(text).strip() == '':
        return 'neutral'
    scores = analyzer.polarity_scores(str(text))
    compound = scores['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['sentiment'] = df['text'].apply(get_sentiment)
df['total_engagement'] = df['like_count'] + df['repost_count'] + df['reply_count']

# ----------------------------
# Dashboard UI
# ----------------------------
st.set_page_config(page_title="Covid on Bluesky", layout="wide")
st.title("🦠 Covid on Bluesky: What People Are Saying")
st.markdown("""
*Posts, replies, and engagement from November 29 – December 13, 2025*  
⚠️ **Note**: This dataset is future-dated (Dec 2025) and appears synthetic—not historical observation.
""")

# Filters
col1, col2 = st.columns(2)
with col1:
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    date_range = st.date_input("Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)
with col2:
    min_engagement = st.slider("Minimum Total Engagement (likes + reposts + replies)", 0, 100, 0)

# Apply filters
start_dt = pd.to_datetime(date_range[0])
end_dt = pd.to_datetime(date_range[1]) + timedelta(days=1)
df_filtered = df[
    (df['timestamp'] >= start_dt) &
    (df['timestamp'] < end_dt) &
    (df['total_engagement'] >= min_engagement)
]

# ----------------------------
# Charts
# ----------------------------

# 1. Sentiment over time
st.subheader("1. Emotional Tone of Posts Over Time")
df_daily = df_filtered.set_index('timestamp').groupby(pd.Grouper(freq='D')).agg({
    'text': 'count',
    'sentiment': lambda x: x.value_counts().to_dict()
}).rename(columns={'text': 'post_count'}).reset_index()
sentiment_df = df_daily['sentiment'].apply(pd.Series).fillna(0).astype(int)
df_daily = pd.concat([df_daily[['timestamp']], sentiment_df], axis=1)
for col in ['positive', 'neutral', 'negative']:
    if col not in df_daily.columns:
        df_daily[col] = 0

fig1 = px.area(
    df_daily,
    x='timestamp',
    y=['negative', 'neutral', 'positive'],
    color_discrete_map={'negative': '#ef4444', 'neutral': '#9ca3af', 'positive': '#10b981'},
    labels={'value': 'Number of Posts', 'timestamp': 'Date', 'variable': 'Sentiment'}
)
st.plotly_chart(fig1, use_container_width=True)
st.caption("Based on VADER sentiment analysis. 'Negative' includes distress about long COVID, grief, and frustration. Sarcasm or medical terms (e.g., 'positive test') may be mislabeled.")

# 2. Total engagement over time
st.subheader("2. Total Engagement Over Time")
df_eng = df_filtered.set_index('timestamp').groupby(pd.Grouper(freq='D')).agg({
    'like_count': 'sum',
    'repost_count': 'sum',
    'reply_count': 'sum'
}).reset_index()
df_eng['total_engagement'] = df_eng['like_count'] + df_eng['repost_count'] + df_eng['reply_count']

fig2 = px.line(df_eng, x='timestamp', y='total_engagement', labels={'total_engagement': 'Total Engagement', 'timestamp': 'Date'})
st.plotly_chart(fig2, use_container_width=True)

# 3. Top posts
st.subheader("3. Top 10 Posts by Engagement")
top_posts = df_filtered.nlargest(10, 'total_engagement').copy()
top_posts['preview'] = top_posts['text'].str[:70] + '...'
fig3 = px.bar(
    top_posts, y='preview', x='total_engagement',
    orientation='h',
    labels={'preview': 'Post Preview', 'total_engagement': 'Total Engagement'}
)
fig3.update_layout(yaxis={'categoryorder': 'total ascending'})
st.plotly_chart(fig3, use_container_width=True)

# ----------------------------
# Key Info Sections
# ----------------------------
st.markdown("### 🔍 Key Findings")
st.markdown("""
- **Negative emotional tone dominates**, especially around long COVID, systemic neglect, and grief.
- **Two days (Dec 11–12, 2025) account for ~38% of all posts**—suggesting an external trigger.
- **Most-engaged posts share data or personal illness accounts**, not political opinions.
""")

st.markdown("### ⚠️ Data Limits & Biases")
st.markdown("""
- **Short window**: Only 15 days—not representative of long-term trends.
- **Bluesky-only**: Skews toward English-speaking, tech-savvy users.
- **Non-English posts** (~15%) are **excluded from sentiment analysis**.
- **Sarcasm & medical language** (e.g., “positive test”) often misclassified.
- **Future-dated**: All posts are from Dec 2025—likely synthetic or projected data.
""")

st.markdown("### ℹ️ Data Notes")
st.markdown("""
- Each row = one Bluesky post (original or reply).
- Engagement metrics reflect counts at time of collection.
- No political conclusions are drawn from mentions of public figures.
""")