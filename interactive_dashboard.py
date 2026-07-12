import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import        dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as    px

DB_NAME = "local_market_share.db"

def analyze_data():
    # 1. Connect and query database
    conn = sqlite3.connect(DB_NAME)

    query = """
        SELECT
            strftime('%Y-%m', c.date) as 
       year_month,
            m.standardized_machine,
            COUNT(*) as mention_count
        FROM candidates 
       c
        JOIN machine_mentions m ON c.doi = m.doi
        GROUP BY year_month, m.standardized_machine
    
       """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("No data found in the database.")
        return None

    return df

def create_dashboard(df):
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Spatial Transcriptomics Market Share Evolution (bioRxiv Mentions)"),
        dcc.Graph(id='market-share-chart'),

       dcc.Graph(id='mention-volume-chart')
    ])

    @app.callback(
        [Output('market-share-chart', 
       'figure'),
         Output('mention-volume-chart', 'figure')],
        [Input('market-share-chart', 
       'clickData')]
    )
    def update_charts(click_data):
        year_month = None
        if click_data is not None:
            year_month = click_data['points'][0]['text']
        filtered_df = df[df['year_month'] == year_month]

        fig1 = px.area(filtered_df, x='year_month', y='mention_count', 
       color='standardized_machine', title='Market Share Evolution')
        fig2 = px.bar(filtered_df, x='year_month', 
       y='mention_count', color='standardized_machine', title='Mention Volume')

        return fig1, fig2

    return app

if __name__ == "__main__":
    df = analyze_data()
    if df is not None:
        app = create_dashboard(df)
        app.run_server(debug=True)
