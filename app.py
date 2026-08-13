import os
import logging

import psycopg2
import pandas as pd

import dash
from dash import dcc, html, Input, Output
import plotly.express as px

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_connection():
    """
    Create PostgreSQL connection.
    Raises an error immediately if DATABASE_URL is missing.
    """
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set.")

    return psycopg2.connect(database_url)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate dataframe columns.
    """
    df = df.copy()

    required_columns = [
        "year_month",
        "resolved_machine",
        "mention_count",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise KeyError(
            f"Database query is missing required columns: {', '.join(missing_columns)}"
        )

    if "parent_company" not in df.columns:
        df["parent_company"] = "Unresolved"

    for col in ["resolved_machine", "parent_company"]:
        df[col] = df[col].fillna("Unresolved").astype(str).str.strip()
        df.loc[df[col].isin(["", "nan", "None"]), col] = "Unresolved"

    df["mention_count"] = (
        pd.to_numeric(df["mention_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    return df


def fetch_database_data():
    """
    Fetch real market share data from PostgreSQL.

    No demo fallback is used.
    If there is no data, raise an error.
    """
    conn = get_db_connection()

    try:
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

    except Exception as e:
        logger.error(f"Failed to fetch data from PostgreSQL database: {e}")
        raise RuntimeError(
            f"Failed to fetch data from PostgreSQL database: {e}"
        ) from e

    finally:
        conn.close()

    if df.empty:
        raise RuntimeError(
            "Database returned no records. "
            "The dashboard requires machine mention data to display results."
        )

    df = normalize_dataframe(df)
    df["is_demo"] = False

    return df


# Fetch data.
# If this fails, the app will raise an error instead of showing demo data.
df = fetch_database_data()

# Filter options
all_machines = sorted(df["resolved_machine"].dropna().unique().tolist())


def empty_figure(title: str):
    """
    Return an empty styled figure when filters produce no rows.
    """
    fig = px.bar(template="plotly_dark", title=title)

    fig.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


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
                            "LIVE DATABASE",
                            style={
                                "backgroundColor": "#166534",
                                "color": "#f0fdf4",
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
                        html.P(
                            "Total Machine Mentions",
                            style={
                                "color": "#94a3b8",
                                "margin": "0",
                                "fontSize": "13px",
                            },
                        ),
                        html.H2(
                            f"{int(df['mention_count'].sum()):,}",
                            style={
                                "fontSize": "28px",
                                "fontWeight": "700",
                                "margin": "4px 0 0 0",
                                "color": "#38bdf8",
                            },
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
                        html.P(
                            "Unique Machine Models",
                            style={
                                "color": "#94a3b8",
                                "margin": "0",
                                "fontSize": "13px",
                            },
                        ),
                        html.H2(
                            f"{len(all_machines):,}",
                            style={
                                "fontSize": "28px",
                                "fontWeight": "700",
                                "margin": "4px 0 0 0",
                                "color": "#4ade80",
                            },
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
                        html.P(
                            "Time Range",
                            style={
                                "color": "#94a3b8",
                                "margin": "0",
                                "fontSize": "13px",
                            },
                        ),
                        html.H2(
                            f"{df['year_month'].min()} → {df['year_month'].max()}",
                            style={
                                "fontSize": "18px",
                                "fontWeight": "700",
                                "margin": "10px 0 0 0",
                                "color": "#facc15",
                            },
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
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(320px, 1fr))",
                        "gap": "16px",
                        "alignItems": "end",
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
                                    options=[
                                        {"label": m, "value": m}
                                        for m in all_machines
                                    ],
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
                                    "Aggregation View:",
                                    style={
                                        "fontWeight": "600",
                                        "marginBottom": "8px",
                                        "display": "block",
                                    },
                                ),
                                html.Button(
                                    "Switch to Parent Company View",
                                    id="group-toggle-button",
                                    n_clicks=0,
                                    style={
                                        "backgroundColor": "#38bdf8",
                                        "color": "#0f172a",
                                        "border": "none",
                                        "padding": "10px 16px",
                                        "borderRadius": "6px",
                                        "fontWeight": "600",
                                        "cursor": "pointer",
                                        "width": "100%",
                                    },
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
    [
        Output("market-share-chart", "figure"),
        Output("mention-volume-chart", "figure"),
        Output("group-toggle-button", "children"),
    ],
    [
        Input("machine-filter", "value"),
        Input("group-toggle-button", "n_clicks"),
    ],
)
def update_charts(selected_machines, n_clicks):
    """
    Toggle charts between:
      - Tool view: resolved_machine
      - Parent Company view: parent_company
    """
    filtered_df = df.copy()

    if selected_machines:
        filtered_df = filtered_df[
            filtered_df["resolved_machine"].isin(selected_machines)
        ]

    use_parent_company = bool((n_clicks or 0) % 2)

    if use_parent_company:
        group_col = "parent_company"
        view_name = "Parent Company"
        button_label = "Switch to Tool View"
    else:
        group_col = "resolved_machine"
        view_name = "Tool"
        button_label = "Switch to Parent Company View"

    if filtered_df.empty:
        return (
            empty_figure("Market Share Trajectory Over Time"),
            empty_figure("Total Mentions"),
            button_label,
        )

    # Line / Area chart data
    line_df = (
        filtered_df.groupby(["year_month", group_col], as_index=False)["mention_count"]
        .sum()
        .sort_values(["year_month", group_col])
    )

    fig_area = px.area(
        line_df,
        x="year_month",
        y="mention_count",
        color=group_col,
        title=f"Market Share Trajectory Over Time ({view_name} View)",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    fig_area.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        xaxis_title="Month",
        yaxis_title="Mentions",
        legend_title=view_name,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    # Bar chart data
    bar_df = (
        filtered_df.groupby(group_col, as_index=False)["mention_count"]
        .sum()
        .sort_values("mention_count", ascending=True)
    )

    fig_bar = px.bar(
        bar_df,
        x="mention_count",
        y=group_col,
        orientation="h",
        color=group_col,
        title=f"Total Mentions by {view_name}",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    fig_bar.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        xaxis_title="Total Mention Count",
        yaxis_title="Parent Company" if use_parent_company else "Tool / Machine Model",
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        # Give every row enough vertical space so labels never get skipped
        height=max(500, len(bar_df) * 28),
    )

    # Force ALL category labels to be displayed (disables Plotly's auto label thinning)
    fig_bar.update_yaxes(
        tickmode="array",
        tickvals=bar_df[group_col].tolist(),
        ticktext=bar_df[group_col].tolist(),
    )
    return fig_area, fig_bar, button_label


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)