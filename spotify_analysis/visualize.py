import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mlxtend.frequent_patterns import apriori, association_rules

DATA_PATH = '../spotify_analysis/data/processed_sentiment.csv'
REPORT_DIR = '../spotify_analysis/reports'
if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

# set plot format
def apply_center_layout(fig, title, height=600):
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, xanchor='center', font=dict(size=24)),
        template='plotly_white',
        font=dict(size=14),
        margin=dict(l=80, r=80, t=100, b=100),
        height=height,
        autosize=True
    )
    return fig

# ==========================================
# Data processing
# ==========================================
def load_and_preprocess():
    df = pd.read_csv(DATA_PATH)
    df['date'] = pd.to_datetime(df['date'])
    
    df['product_list'] = df['product_type'].astype(str).str.split(', ')
    df['issue_list'] = df['issue_type'].astype(str).str.split(', ')
    
    df_exp_prod = df.explode('product_list')
    df_exp_prod['product_list'] = df_exp_prod['product_list'].str.strip()
    
    df_exp_all = df_exp_prod.explode('issue_list')
    df_exp_all['issue_list'] = df_exp_all['issue_list'].str.strip()
    
    return df, df_exp_all

df, df_exploded = load_and_preprocess()

# ==========================================
# Plots
# ==========================================

def plot_sentiment_trend(df): #line (trend) chart
    df_daily = df.groupby('date')['emotion_score'].mean()
    all_range = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
    df_plot = df_daily.reindex(all_range).reset_index().rename(columns={'index': 'date'})
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot['date'], y=df_plot['emotion_score'],
        mode='lines+markers', connectgaps=True,
        line=dict(color='lightblue', width=4),
        marker=dict(color='royalblue', size=10)
    ))
    
    fig.update_layout(
        yaxis=dict(
        range=[0.5, 0.750], 
        dtick=0.025, 
        tickformat=".3f", 
        title="Emotion Score (0.0 = Negative, 0.5 = Neutral, 1.0 = Positive)"),
        xaxis_title="Observation Date",
        annotations=[dict(x='2026-05-08', y=0.55, text="No Data", showarrow=False, font=dict(color="red"))]
    )
    return apply_center_layout(fig, 'Spotify Sentiment Trend Analysis')

def plot_product_sentiment_hist(df_exploded): #bar chart
    order = df_exploded[df_exploded['top_emotion'] == 'Neutral'].groupby('product_list').size().sort_values(ascending=False).index.tolist()
    
    fig = px.histogram(
        df_exploded, x='product_list', color='top_emotion',
        barmode='group',
        category_orders={"product_list": order, "top_emotion": ['Positive', 'Neutral', 'Negative']},
        color_discrete_map={'Positive': "#f7f97f", 'Neutral': "#9db4b5", 'Negative': "#456c6e"}
    )
    return apply_center_layout(fig, 'Sentiment Distribution by Product')

def plot_cross_heatmaps(df_exploded, mode='score'): #product issue heatmap
    if mode == 'score':
        pivot = df_exploded.pivot_table(index='product_list', columns='issue_list', values='emotion_score', aggfunc='mean')
        title = 'Pain Points: Sentiment Score by Product & Issue'
        color_scale = 'RdBu_r'
        z_range = [0.5, 0.8]
    else:
        pivot = pd.crosstab(
            df_exploded['product_list'].reset_index(drop=True), 
            df_exploded['issue_list'].reset_index(drop=True)
        )
        title = 'Issue Density: Review Count by Product & Issue'
        color_scale = 'YlOrRd'
        z_range = None
    fig = px.imshow(pivot, text_auto=".2f" if mode=='score' else True, 
                    color_continuous_scale=color_scale, range_color=z_range)
    fig.update_xaxes(side="bottom", tickangle=-45)
    return apply_center_layout(fig, title)

def plot_issue_lift(df): #lift analysis
    issue_series = df['issue_type'].str.get_dummies(sep=', ')
    frequent_itemsets = apriori(issue_series, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    
    rules = rules[(rules['antecedents'].apply(len) == 1) & (rules['consequents'].apply(len) == 1)].copy()
    rules['ant'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['con'] = rules['consequents'].apply(lambda x: list(x)[0])
    
    lift_pivot = rules.pivot_table(index='ant', columns='con', values='lift', aggfunc='max').fillna(1.0)

    fig = px.imshow(
        lift_pivot, text_auto=".2f",
        color_continuous_scale=[[0, "#F7FBFF"], [0.25, "#FFF5F0"], [0.5, "#FD8D3C"], [1, "#800026"]],
        zmin=1.0, zmax=3.0, aspect="equal"
    )
    fig.update_xaxes(side="bottom", tickangle=-45)
    return apply_center_layout(fig, "Issue Lift Analysis: Strength of Association")

# ==========================================
# Top emotional comments
# ==========================================
def generate_html_report(df):
    top_neg = df.sort_values('emotion_score').head(5)
    top_pos = df.sort_values('emotion_score', ascending=False).head(5)
    
    def make_table(data, bg_color, text_color):
        rows = "".join([f"<tr><td>{r['review']}</td><td>{r['product_type']}</td><td>{r['issue_type']}</td></tr>" for _, r in data.iterrows()])
        return f"""
        <table border="1" style="width:100%; border-collapse:collapse; margin-bottom:20px; font-family:sans-serif;">
            <tr style="background-color:{bg_color}; color:{text_color};"><th>Review</th><th>Product</th><th>Issue</th></tr>
            {rows}
        </table>"""

    content = f"<h2>Spotify Sentiment Summary</h2>" + \
              "<h3>Most Negative</h3>" + make_table(top_neg, "#456c6e", "white") + \
              "<h3>Most Positive</h3>" + make_table(top_pos, "#f7f97f", "black")
              
    with open(f'{REPORT_DIR}/sentiment_summary.html', 'w', encoding='utf-8') as f:
        f.write(content)

# ==========================================
# Execute
# ==========================================
if __name__ == "__main__":
    fig1 = plot_sentiment_trend(df)
    fig2 = plot_product_sentiment_hist(df_exploded)
    fig3 = plot_cross_heatmaps(df_exploded, mode='score')
    fig4 = plot_cross_heatmaps(df_exploded, mode='count')
    fig5 = plot_issue_lift(df)
    
    fig1.write_html(f'{REPORT_DIR}/01_trend.html')
    fig2.write_html(f'{REPORT_DIR}/02_product_dist.html')
    fig3.write_html(f'{REPORT_DIR}/03_issue_score_heatmap.html')
    fig4.write_html(f'{REPORT_DIR}/04_issue_count_heatmap.html')
    fig5.write_html(f'{REPORT_DIR}/05_lift_analysis.html')
    

    generate_html_report(df)
    
    print(f"Completed: {REPORT_DIR}")