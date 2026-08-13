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


DEMO_MACHINE_PARENTS = {
    "NovaSeq X / 6000": "Illumina",
    "Chromium Controller": "10x Genomics",
    "Visium / Xenium": "10x Genomics",
    "PromethION / P2 Solo": "Oxford Nanopore",
    "Revio / Sequel IIe": "Pacific Biosciences",
    "Orbitrap / Q Exactive": "Thermo Fisher",
    "MERSCOPE": "Vizgen",
    "Unresolved": "Unresolved",
}


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize expected dashboard columns and clean string values.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "year_month",
                "resolved_machine",
                "parent_company",
                "mention_count",
                "is_demo",
            ]
        )

    df = df.copy()

    if "resolved_machine" not in df.columns:
        df["resolved_machine"] = "Unresolved"

    if "parent_company" not in df.columns:
        df["parent_company"] = "Unresolved"

    for col in ["resolved_machine", "parent_company"]:
        df[col] = df[col].fillna("Unresolved").astype(str).str.strip()
        df.loc[df[col].isin(["", "nan", "None"]), col] = "Unresolved"

    if "mention_count" in df.columns:
        df["mention_count"] = pd.to_numeric(df["mention_count"], errors="coerce").fillna(0).astype(int)
    else:
        df["mention_count"] = 0

    return df


def fetch_database_data():
    """
    Fetch real market share data from PostgreSQL, including parent company.
    Falls back to demo data if database is unavailable.
    """
    if not DATABASE_URL:
        logger.warning("DATABASE_URL not set. Falling back to synthetic demonstration dataset.")
        return generate_demo_data()

    try:
        conn = psycopg2.connect(DATABASE_URL)

        query = """
            SELECT
                SUBSTRING(c.date FROM 1 FOR 7) AS year_month,
                COALESCE(NULLIF(TRIM(m.resolved_machine), ''), 'Unresolved') AS resolved_machine,
                COALESCE(NULLIF(TRIM(r.parent_company), ''), 'Unresolved') AS parent_company,
                COUNT(*) AS mention_count
            FROM candidates c
            JOIN machine_mentions m
                ON c.doi = m.doi
            LEFT JOIN registry_machines r
                ON TRIM(r.canonical_name) = TRIM(m.resolved_machine)
            WHERE c.date IS NOT NULL
            GROUP BY
                SUBSTRING(c.date FROM 1 FOR 7),
                COALESCE(NULLIF(TRIM(m.resolved_machine), ''), 'Unresolved'),
                COALESCE(NULLIF(TRIM(r.parent_company), ''), 'Unresolved')
            ORDER BY
                year_month ASC,
                mention_count DESC;
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            logger.warning("Database returned 0 records. Generating synthetic fallback dataset.")
            return generate_demo_data()

        df = normalize_dataframe(df)
        df["is_demo"] = False
        return df

    except Exception as e:
        logger.error(f"Failed to fetch data from PostgreSQL database: {e}")
        return generate_demo_data()


def generate_demo_data():
    """
    Generate synthetic data for previewing dashboard layout when database is offline.
    """
    dates = pd.date_range(start="2024-01-01", end="2026-06-01", freq="MS").strftime("%Y-%m").tolist()

    records = []
    np.random.seed(42)

    for date in dates:
        for machine, parent_company in DEMO_MACHINE_PARENTS.items():
            count = int(np.random.poisson(lam=np.random.randint(5, 30)))
            records.append(
                {
                    "year_month": date,
                    "resolved_machine": machine,
                    "parent_company": parent_company,
                    "mention_count": count,
                }
            )

    df = pd.DataFrame(records)
    df["is_demo"] = True
    return normalize_dataframe(df)


def kpi_card(title: str, value: str, color: str):
    return html.Div(
        style={
            "backgroundColor": "#1e293b",
            "padding": "16px",
            "borderRadius": "8px",
            "border": "1px solid #334155",
        },
        children=[
            html.P(
                title,
                style={
                    "color": "#94a3b8",
                    "margin": "0",
                    "fontSize": "13px",
                },
            ),
            html.H2(
                value,
                style={
                    "fontSize": "24px",
                    "fontWeight": "700",
                    "margin": "4px 0 0 0",
                    "color": color,
                },
            ),
        ],
    )


def chart_card(graph_id: str):
    return html.Div(
        style={
            "backgroundColor": "#1e293b",
            "padding": "16px",
            "borderRadius": "8px",
            "border": "1px solid #334155",
        },
        children=[dcc.Graph(id=graph_id)],
    )


def empty_figure(title: str):
    fig = px.bar(template="plotly_dark", title=title)
    fig.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


# Fetch data
df = fetch_database_data()

is_demo = bool(df["is_demo"].iloc[0]) if not df.empty and "is_demo" in df.columns else True

# Filter options
all_machines = sorted(df["resolved_machine"].dropna().unique().tolist()) if not df.empty else []
all_parent_companies = sorted(df["parent_company"].dropna().unique().tolist()) if not df.empty else []

# KPI values
total_mentions = f"{int(df['mention_count'].sum()):,}" if not df.empty else "0"
unique_machine_models = f"{len(all_machines):,}"
unique_parent_companies = f"{len(all_parent_companies):,}"

if not df.empty:
    time_range = f"{df['year_month'].min()} → {df['year_month'].max()}"
else:
    time_range = "N/A"


# Initialize Dash App
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="Spatial Transcriptomics Market Intelligence",
)

# Expose WSGI server for Gunicorn on HF Spaces
server = app.server


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
                            style={
                                "fontSize": "24px",
                                "fontWeight": "700",
                                "margin": "0 0 8px 0",
                            },
                        ),
                        html.P(
                            "Real-time equipment adoption & mention evolution extracted from bioRxiv preprints.",
                            style={
                                "color": "#94a3b8",
                                "margin": "0",
                            },
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
                kpi_card("Total Machine Mentions", total_mentions, "#38bdf8"),
                kpi_card("Unique Machine Models", unique_machine_models, "#4ade80"),
                kpi_card("Unique Parent Companies", unique_parent_companies, "#c084fc"),
                kpi_card("Time Range", time_range, "#facc15"),
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
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
                        "gap": "16px",
                    },
                    children=[
                        html.Div(
                            children=[
                                html.Label(
                                    "Filter Machine Models:",
                                    style={
                                        "fontWeight": "600",
                                        "marginBottom": "8px",
                                        "display": "block",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="machine-filter",
                                    options=[{"label": m, "value": m} for m in all_machines],
                                    value=all_machines,
                                    multi=True,
                                    placeholder="Select machine models",
                                    style={"color": "#0f172a"},
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Label(
                                    "Filter Parent Companies:",
                                    style={
                                        "fontWeight": "600",
                                        "marginBottom": "8px",
                                        "display": "block",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="parent-filter",
                                    options=[{"label": p, "value": p} for p in all_parent_companies],
                                    value=all_parent_companies,
                                    multi=True,
                                    placeholder="Select parent companies",
                                    style={"color": "#0f172a"},
                                ),
                            ]
                        ),
                    ],
                )
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
                chart_card("market-share-chart"),
                chart_card("mention-volume-chart"),
                chart_card("parent-company-chart"),
            ],
        ),
    ],
)


@app.callback(
    [
        Output("market-share-chart", "figure"),
        Output("mention-volume-chart", "figure"),
        Output("parent-company-chart", "figure"),
    ],
    [
        Input("machine-filter", "value"),
        Input("parent-filter", "value"),
    ],
)
def update_charts(selected_machines, selected_parent_companies):
    filtered_df = df.copy()

    if selected_machines:
        filtered_df = filtered_df[filtered_df["resolved_machine"].isin(selected_machines)]

    if selected_parent_companies:
        filtered_df = filtered_df[filtered_df["parent_company"].isin(selected_parent_companies)]

    if filtered_df.empty:
        return (
            empty_figure("Market Share Trajectory Over Time"),
            empty_figure("Total Mentions by Equipment Model"),
            empty_figure("Total Mentions by Parent Company"),
        )

    # Area Chart - Evolution over time
    fig_area = px.area(
        filtered_df.sort_values("year_month"),
        x="year_month",
        y="mention_count",
        color="resolved_machine",
        hover_data=["parent_company"],
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

    # Bar Chart - Machine-level cumulative breakdown, colored by parent company
    bar_df = (
        filtered_df.groupby(["resolved_machine", "parent_company"], as_index=False)["mention_count"]
        .sum()
        .sort_values("mention_count", ascending=True)
    )

    fig_bar = px.bar(
        bar_df,
        x="mention_count",
        y="resolved_machine",
        orientation="h",
        color="parent_company",
        hover_data=["parent_company"],
        title="Total Mentions by Equipment Model",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_bar.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        xaxis_title="Total Mention Count",
        yaxis_title="Machine Model",
        legend_title="Parent Company",
        margin=dict(l=20, r=20, t=50, b=20),
    )

    # Bar Chart - Parent company cumulative breakdown
    parent_df = (
        filtered_df.groupby("parent_company", as_index=False)["mention_count"]
        .sum()
        .sort_values("mention_count", ascending=True)
    )

    fig_parent = px.bar(
        parent_df,
        x="mention_count",
        y="parent_company",
        orientation="h",
        color="parent_company",
        title="Total Mentions by Parent Company",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_parent.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        xaxis_title="Total Mention Count",
        yaxis_title="Parent Company",
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig_area, fig_bar, fig_parent


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)