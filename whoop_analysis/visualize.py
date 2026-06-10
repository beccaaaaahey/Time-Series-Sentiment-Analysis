import os
import re
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MultiLabelBinarizer
from statsmodels.tsa.seasonal import seasonal_decompose
from plotly.subplots import make_subplots
from mlxtend.frequent_patterns import apriori, association_rules


DATA_PATH = '../bio_analysis/data/processed_sentiment.csv'
REPORT_DIR = '../bio_analysis/reports'
os.makedirs(REPORT_DIR, exist_ok=True)

ALLOWED_TAGS = [
    'Technical', 'Monetization', 'Design', 'Localization', 
    'Account/ Membership', 'Customer Service', 'Accuracy'
]

def apply_center_layout(fig, title, height=600):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor='center', font=dict(size=22)),
        template='plotly_white',
        margin=dict(l=50, r=50, t=100, b=100),
        height=height
    )
    return fig

# ==========================================
# Data processing
# ==========================================
def load_and_clean_data():
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    emotion_map = {
    'joy': 'Positive',
    'surprise': 'Positive',
    'neutral': 'Neutral',
    'anger': 'Negative',
    'disgust': 'Negative',
    'fear': 'Negative',
    'sadness': 'Negative'
    }

    df['top_emotion2'] = df['top_emotion'].str.lower().map(emotion_map)
    def clean_tags(text):
        return re.sub(r"[\[\]\(\)'\"]", "", str(text))
    
    df['tags_pure'] = df['topic_tags'].apply(clean_tags)
    df['tags_list'] = df['tags_pure'].str.split(', ').apply(lambda x: [i.strip() for i in x])
    df_exploded = df.explode('tags_list').reset_index(drop=True)
    return df, df_exploded

df, df_exploded = load_and_clean_data()

# ==========================================
# Plots
# ==========================================

def plot_sentiment_trend(df): #line (trend) chart
    df_weekly = df.resample('W', on='date')['emotion_score'].mean().reset_index()
    df_weekly['rolling_mean'] = df_weekly['emotion_score'].rolling(window=4, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_weekly['date'], y=df_weekly['emotion_score'],
        mode='lines', name='Weekly Score', line=dict(color='lightgrey', width=1)
    ))
    fig.add_trace(go.Scatter(
        x=df_weekly['date'], y=df_weekly['rolling_mean'],
        mode='lines', name='Monthly Trend', line=dict(color='royalblue', width=3)
    ))
    
    fig.update_layout(xaxis_title='Date', yaxis_title='Score (0=Neg, 1=Pos)')
    return apply_center_layout(fig, 'Whoop Sentiment Trend (Rolling Average)')

def plot_category_distribution(df_exploded): #bar chart
    neutral_order = (
    df_exploded[df_exploded['top_emotion2'] == 'Neutral'].groupby('tags_list').size().sort_values(ascending=False).index.tolist()
    )
    
    all_tags = df_exploded['tags_list'].unique()
    for tag in all_tags:
        if tag not in neutral_order:
            neutral_order.append(tag)

    fig = px.histogram(
        df_exploded, 
        x='tags_list', 
        color='top_emotion2',
        barmode='group',
        category_orders={
            "tags_list": neutral_order,
            "top_emotion2": ['Positive', 'Neutral', 'Negative']
        },
        color_discrete_map={
            'Positive': "#FEE962", 
            'Neutral': "#B4CCBE", 
            'Negative': "#E26464"
        },
        labels={
            "top_emotion2": "Top Emotion",
            "tags_list": "Product Categories"
        }
    )
    
    fig.update_xaxes(tickangle=-45)
    
    fig.update_layout(
        legend_title_text='Top Emotion'
    )
    
    return apply_center_layout(fig, 'Whoop Sentiment Distribution by Category')

def plot_category_heatmap(df): #issue heatmap
    mlb = MultiLabelBinarizer(classes=ALLOWED_TAGS)
    tag_matrix = mlb.fit_transform(df['tags_list'])
    tag_df = pd.DataFrame(tag_matrix, columns=mlb.classes_)

    co_occurrence = tag_df.T.dot(tag_df).astype(float).copy()
    fig = px.imshow(
        co_occurrence,
        text_auto=True,
        color_continuous_scale='Viridis',
        labels=dict(x="Category", y="Category", color="Overlap Count")
    )
    
    return apply_center_layout(fig, 'Whoop Category Cross-Correlation (Co-occurrence)')

def plot_ts_decomposition(df): #time series analysis
    df_ts = df.resample('W', on='date')['emotion_score'].mean().ffill()
    result = seasonal_decompose(df_ts, model='additive', period=4)

    fig = make_subplots(
        rows=4, cols=1, 
        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
        vertical_spacing=0.08
    )
    
    colors = ['#636EFA', '#EF553B', '#00CC96', "#FAB863"]
    components = [result.observed, result.trend, result.seasonal, result.resid]
    
    for i, (comp, color) in enumerate(zip(components, colors), 1):
        mode = 'lines' if i < 4 else 'markers'
        fig.add_trace(go.Scatter(x=df_ts.index, y=comp, name=comp.name, line=dict(color=color), mode=mode), row=i, col=1)

    fig.update_layout(height=900, showlegend=False)
    return apply_center_layout(fig, "Whoop Sentiment Time Series Decomposition Analysis", height=900)


def plot_whoop_issue_lift(df): #lift analysis
    issue_series = df['tags_pure'].str.get_dummies(sep=', ')
    
    frequent_itemsets = apriori(issue_series, min_support=0.01, use_colnames=True)
    if frequent_itemsets.empty:
        return None

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    rules = rules[(rules['antecedents'].apply(len) == 1) & (rules['consequents'].apply(len) == 1)].copy()
    rules['ant'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['con'] = rules['consequents'].apply(lambda x: list(x)[0])
    
    lift_pivot = rules.pivot_table(index='ant', columns='con', values='lift', aggfunc='max').fillna(1.0)

    fig = px.imshow(
        lift_pivot, 
        text_auto=".2f",
        color_continuous_scale=[[0, "#F7FBFF"], [0.25, "#FFF5F0"], [0.5, "#FD8D3C"], [1, "#800026"]],
        zmin=1.0, 
        zmax=2.0, 
        aspect="equal",
        labels=dict(x="Consequent (B)", y="Antecedent (A)", color="Lift")
    )
    
    fig.update_xaxes(side="bottom", tickangle=-45)
    return apply_center_layout(fig, "Category Association: Lift Analysis (Strength of Connection)")

# ==========================================
# Execute
# ==========================================
if __name__ == "__main__":
    plots = {
        "01_sentiment_trend.html": plot_sentiment_trend(df),
        "02_category_sentiment.html": plot_category_distribution(df_exploded),
        "03_category_heatmap.html": plot_category_heatmap(df),
        "04_time_series_decomposition.html": plot_ts_decomposition(df),
        "05_issue_lift.html": plot_whoop_issue_lift(df)
    }

    for filename, fig in plots.items():
        save_path = os.path.join(REPORT_DIR, filename)
        fig.write_html(save_path)
        print(f"Successfully generated: {save_path}")

    print("\n Completed!")