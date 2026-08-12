import os
import logging
import psycopg2
import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.express as px

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL")


def fetch_database_data():
    """Fetches real market share data from PostgreSQL or generates fallback demo data if DB is unavailable."""
    if not DATABASE_URL:
        logger.warning("DATABASE_URL not set. Falling back to synthetic demonstration dataset.")
        return generate_demo_data()

    try:
        conn = psycopg2.connect(DATABASE_URL)
        query = """
            SELECT
                SUBSTRING(c.date FROM 1 FOR 7) AS year_month,
                COALESCE(m.resolved_machine, 'Unresolved') AS resolved_machine,
                COUNT(*) AS mention_count
            FROM candidates c
            JOIN machine_mentions m ON c.doi = m.doi
            WHERE c.date IS NOT NULL
            GROUP BY year_month, COALESCE(m.resolved_machine, 'Unresolved')
            ORDER BY year_month ASC, mention_count DESC;
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            logger.warning("Database returned 0 records. Generating synthetic fallback dataset.")
            return generate_demo_data()

        df["is_demo"] = False
        return df

    except Exception as e:
        logger.error(f"Failed to fetch data from PostgreSQL database: {e}")
        return generate_demo_data()


def generate_demo_data():
    """Generates synthetic data for previewing dashboard layout when database is offline."""
    dates = pd.date_range(start="2024-01-01", end="2026-06-01", freq="MS").strftime("%Y-%m").tolist()
    machines = [
        "NovaSeq X / 6000",
        "Chromium Controller",
        "Visium / Xenium",
        "PromethION / P2 Solo",
        "Revio / Sequel IIe",
        "Orbitrap / Q Exactive",
        "MERSCOPE",
        "Unresolved",
    ]

    records = []
    np.random.seed(42)
    for date in dates:
        for machine in machines:
            count = int(np.random.poisson(lam=np.random.randint(5, 30)))
            records.append({"year_month": date, "resolved_machine": machine, "mention_count": count})

    df = pd.DataFrame(records)
    df["is_demo"] = True
    return df


# Fetch data
df = fetch_database_data()
is_demo = df["is_demo"].iloc[0] if not df.empty else True

# Initialize Dash App
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="Spatial Transcriptomics Market Intelligence",
)

# Expose WSGI server for Gunicorn on HF Spaces
server = app.server

# Options for filter dropdowns
all_machines = sorted(df["resolved_machine"].unique().tolist()) if not df.empty else []

app.layout = html.Div(
    style={
        "fontFamily": "'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
        "backgroundColor": "#0f172a",
        "color": "#f8fafc",
        "minHeight": "100vh",
        "padding": "24px",
    },
    children=[
        # Header Banner
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "borderBottom": "1px solid #334155",
                "paddingBottom": "16px",
                "marginBottom": "24px",
            },
            children=[
                html.Div(
                    children=[
                        html.H1(
                            "Spatial Transcriptomics & Genomics Market Intelligence",
                            style={"fontSize": "24px", "fontWeight": "700", "margin": "0 0 8px 0"},
                        ),
                        html.P(
                            "Real-time equipment adoption & mention evolution extracted from bioRxiv preprints.",
                            style={"color": "#94a3b8", "margin": "0"},
                        ),
                    ]
                ),
                html.Div(
                    children=[
                        html.Span(
                            "LIVE DATABASE" if not is_demo else "DEMO MODE (Set DATABASE_URL Secret)",
                            style={
                                "backgroundColor": "#166534" if not is_demo else "#9a3412",
                                "color": "#f0fdf4" if not is_demo else "#ffedd5",
                                "padding": "6px 12px",
                                "borderRadius": "16px",
                                "fontSize": "12px",
                                "fontWeight": "600",
                            },
                        )
                    ]
                ),
            ],
        ),
        # KPI Cards
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                "gap": "16px",
                "marginBottom": "24px",
            },
            children=[
                html.Div(
                    style={
                        "backgroundColor": "#1e293b",
                        "padding": "16px",
                        "borderRadius": "8px",
                        "border": "1px solid #334155",
                    },
                    children=[
                        html.P("Total Machine Mentions", style={"color": "#94a3b8", "margin": "0", "fontSize": "13px"}),
                        html.H2(
                            f"{df['mention_count'].sum():,}" if not df.empty else "0",
                            style={"fontSize": "28px", "fontWeight": "700", "margin": "4px 0 0 0", "color": "#38bdf8"},
                        ),
                    ],
                ),
                html.Div(
                    style={
                        "backgroundColor": "#1e293b",
                        "padding": "16px",
                        "borderRadius": "8px",
                        "border": "1px solid #334155",
                    },
                    children=[
                        html.P("Unique Machine Models", style={"color": "#94a3b8", "margin": "0", "fontSize": "13px"}),
                        html.H2(
                            f"{len(all_machines):,}",
                            style={"fontSize": "28px", "fontWeight": "700", "margin": "4px 0 0 0", "color": "#4ade80"},
                        ),
                    ],
                ),
                html.Div(
                    style={
                        "backgroundColor": "#1e293b",
                        "padding": "16px",
                        "borderRadius": "8px",
                        "border": "1px solid #334155",
                    },
                    children=[
                        html.P("Time Range", style={"color": "#94a3b8", "margin": "0", "fontSize": "13px"}),
                        html.H2(
                            f"{df['year_month'].min()} → {df['year_month'].max()}" if not df.empty else "N/A",
                            style={"fontSize": "18px", "fontWeight": "700", "margin": "10px 0 0 0", "color": "#facc15"},
                        ),
                    ],
                ),
            ],
        ),
        # Controls Bar
        html.Div(
            style={
                "backgroundColor": "#1e293b",
                "padding": "16px",
                "borderRadius": "8px",
                "border": "1px solid #334155",
                "marginBottom": "24px",
            },
            children=[
                html.Label("Filter Machine Models:", style={"fontWeight": "600", "marginBottom": "8px", "display": "block"}),
                dcc.Dropdown(
                    id="machine-filter",
                    options=[{"label": m, "value": m} for m in all_machines],
                    value=all_machines,
                    multi=True,
                    style={"color": "#0f172a"},
                ),
            ],
        ),
        # Charts Grid
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr",
                "gap": "24px",
            },
            children=[
                html.Div(
                    style={
                        "backgroundColor": "#1e293b",
                        "padding": "16px",
                        "borderRadius": "8px",
                        "border": "1px solid #334155",
                    },
                    children=[dcc.Graph(id="market-share-chart")],
                ),
                html.Div(
                    style={
                        "backgroundColor": "#1e293b",
                        "padding": "16px",
                        "borderRadius": "8px",
                        "border": "1px solid #334155",
                    },
                    children=[dcc.Graph(id="mention-volume-chart")],
                ),
            ],
        ),
    ],
)


@app.callback(
    [Output("market-share-chart", "figure"), Output("mention-volume-chart", "figure")],
    [Input("machine-filter", "value")],
)
def update_charts(selected_machines):
    if not selected_machines or df.empty:
        filtered_df = df
    else:
        filtered_df = df[df["resolved_machine"].isin(selected_machines)]

    # Area Chart - Evolution
    fig_area = px.area(
        filtered_df,
        x="year_month",
        y="mention_count",
        color="resolved_machine",
        title="Market Share Trajectory Over Time (Preprint Mentions)",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_area.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        xaxis_title="Month",
        yaxis_title="Mentions",
        legend_title="Machine Model",
        margin=dict(l=20, r=20, t=50, b=20),
    )

    # Bar Chart - Cumulative Breakdown
    bar_df = filtered_df.groupby("resolved_machine", as_index=False)["mention_count"].sum().sort_values(
        "mention_count", ascending=True
    )
    fig_bar = px.bar(
        bar_df,
        x="mention_count",
        y="resolved_machine",
        orientation="h",
        title="Total Mentions by Equipment Model",
        template="plotly_dark",
        color="resolved_machine",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_bar.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        xaxis_title="Total Mention Count",
        yaxis_title="Machine Model",
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig_area, fig_bar


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)