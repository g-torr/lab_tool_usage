import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import sqlite3
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from config import DB_NAME


def resolve_db_path():
    return DB_NAME


def analyze_data():
    conn = sqlite3.connect(resolve_db_path())

    query = """
        SELECT
            strftime('%Y-%m', c.date) AS year_month,
            COALESCE(m.resolved_machine, 'Unresolved') AS resolved_machine,
            COUNT(*) AS mention_count
        FROM candidates c
        JOIN machine_mentions m ON c.doi = m.doi
        WHERE c.date IS NOT NULL
        GROUP BY year_month, COALESCE(m.resolved_machine, 'Unresolved')
        ORDER BY year_month, mention_count DESC;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("No data found in the database.")
        return None

    print(df.head())
    return df

def create_dashboard(df):
    app = dash.Dash(__name__)

    app.layout = html.Div([
        html.H1("Spatial Transcriptomics Market Share Evolution (bioRxiv Mentions)"),
        dcc.Graph(id='market-share-chart'),
        dcc.Graph(id='mention-volume-chart')
    ])

    @app.callback(
        [Output('market-share-chart', 'figure'),
         Output('mention-volume-chart', 'figure')],
        [Input('market-share-chart', 'clickData')]
    )
    def update_charts(click_data):
        if click_data is None:
            filtered_df = df
        else:
            year_month = click_data['points'][0]['text']
            filtered_df = df[df['year_month'] == year_month]

        fig1 = px.area(filtered_df, x='year_month', y='mention_count', color='resolved_machine', title='Market Share Evolution')
        fig2 = px.bar(filtered_df, x='year_month', y='mention_count', color='resolved_machine', title='Mention Volume')

        return fig1, fig2

    return app

if __name__ == "__main__":
    df = analyze_data()
    if df is not None:
        app = create_dashboard(df)
        app.run(debug=True)
