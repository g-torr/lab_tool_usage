import logging
import os

import dash
from dash import dash_table, html
import pandas as pd
import plotly.express as px
import psycopg2
from dash import Input, Output, callback_context, dcc

from src.stock_signals import build_stock_signals
from src.alpha_backtest import (
    HORIZONS, PURITY, attach_returns, build_signal_panel, fetch_prices,
    test1_panel_regression, test2_information_coefficient, to_weekly,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db_connection():
    """Create PostgreSQL connection.

    Raises an error immediately if DATABASE_URL is missing.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set.")
    return psycopg2.connect(database_url)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate dataframe columns."""
    df = df.copy()
    required_columns = ["day", "resolved_machine", "mention_count"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise KeyError(
            f"Database query is missing required columns: {', '.join(missing_columns)}"
        )

    if "parent_company" not in df.columns:
        df["parent_company"] = "Unresolved"
    if "ticker" not in df.columns:
        df["ticker"] = ""

    for col in ["resolved_machine", "parent_company", "ticker"]:
        df[col] = df[col].fillna("Unresolved").astype(str).str.strip()
        df.loc[df[col].isin(["", "nan", "None"]), col] = "Unresolved"

    df["mention_count"] = (
        pd.to_numeric(df["mention_count"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    df["day"] = pd.to_datetime(df["day"])
    return df


def fetch_database_data():
    """Fetch real market share data from PostgreSQL.

    No demo fallback is used. If there is no data, raise an error.
    """
    conn = get_db_connection()
    try:
        query = """
            SELECT 
                c.date AS day,
                COALESCE(NULLIF(TRIM(m.resolved_machine), ''), 'Unresolved') AS resolved_machine,
                COALESCE(NULLIF(TRIM(r.parent_company), ''), 'Unresolved') AS parent_company,
                COALESCE(NULLIF(TRIM(r.ticker), ''), '') AS ticker,
                COUNT(*) AS mention_count
            FROM candidates c
            JOIN machine_mentions m ON c.doi = m.doi
            LEFT JOIN registry_machines r ON TRIM(r.canonical_name) = TRIM(m.resolved_machine)
            WHERE c.date IS NOT NULL
            GROUP BY day, resolved_machine, r.parent_company, r.ticker
            ORDER BY day ASC, mention_count DESC;
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
            "Database returned no records. The dashboard requires machine mention data to display results."
        )

    df = normalize_dataframe(df)
    df["is_demo"] = False
    return df


def fetch_alpha_mention_panel():
    """Fetch daily ticker-level mention counts for the alpha analysis."""
    conn = get_db_connection()
    try:
        query = """
            SELECT c.date AS day,
                   COALESCE(NULLIF(TRIM(r.ticker), ''), 'Unresolved') AS ticker,
                   COUNT(*) AS mention_count,
                   COUNT(DISTINCT c.doi) AS doi_count
            FROM candidates c
            JOIN machine_mentions m ON c.doi = m.doi
            LEFT JOIN registry_machines r
              ON TRIM(r.canonical_name) = TRIM(m.resolved_machine)
            WHERE c.date IS NOT NULL
            GROUP BY c.date, r.ticker
            ORDER BY c.date ASC
        """
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


def run_alpha_analysis():
    """Run the registered mention-signal backtest for dashboard display."""
    mentions = fetch_alpha_mention_panel()
    if mentions.empty:
        raise RuntimeError("No ticker-level mention data is available.")
    weekly = to_weekly(build_signal_panel(mentions))
    tickers = [ticker for ticker in weekly["ticker"].dropna().unique() if ticker in PURITY]
    if not tickers:
        raise RuntimeError("No listed-company signals are available.")
    prices = fetch_prices(tickers, "2023-12-01", pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    panel = attach_returns(weekly, prices)
    rows = []
    for horizon in HORIZONS:
        beta, p_value, n_reg = test1_panel_regression(panel, horizon)
        mean_ic, icir, t_stat, n_ic = test2_information_coefficient(panel, horizon)
        rows.append({"Horizon": horizon, "Beta": beta, "p_value": p_value,
                     "Mean IC": mean_ic, "ICIR": icir, "IC t-stat": t_stat,
                     "Regression n": n_reg, "IC dates": n_ic})
    return pd.DataFrame(rows)


# Fetch data
df = fetch_database_data()
all_machines = sorted(df["resolved_machine"].dropna().unique().tolist())
all_companies = sorted(df["parent_company"].dropna().unique().tolist())


def empty_figure(title: str):
    """Return an empty styled figure when filters produce no rows."""
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
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    title="Spatial Transcriptomics Market Intelligence",
)
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
        # Header Banner Fix
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "16px",
                "marginBottom": "24px",
            },
            children=[
                html.Div(
                    style={"flex": "1"},
                    children=[
                        html.H1(
                            "Spatial Transcriptomics & Genomics Market Intelligence",
                            style={
                                "fontSize": "24px",
                                "fontWeight": 700,
                                "margin": "0 0 8px 0",
                            },
                        ),
                        html.P(
                            "Real-time equipment adoption & mention evolution extracted from bioRxiv preprints.",
                            style={"color": "#94a3b8", "margin": 0},
                        ),
                    ],
                ),
                html.A(
                    "Link to Repo",
                    href="https://github.com/g-torr/lab_tool_usage",
                    target="_blank",
                    rel="noopener noreferrer",
                    style={
                        "color": "#94a3b8",
                        "fontSize": "14px",
                        "textDecoration": "none",
                    },
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
                                "whiteSpace": "nowrap",
                            },
                        )
                    ]
                ),
            ],
        ),
        # KPI Cards Section
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
                                "color": "#38bdf8",
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
                                    "Filter Tool Models:",
                                    id="machine-filter-label",
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
                        html.Div(
                            children=[
                                html.Label(
                                    "Date Range:",
                                    style={
                                        "fontWeight": "600",
                                        "marginBottom": "8px",
                                        "display": "block",
                                    },
                                ),
                                dcc.DatePickerRange(
                                    id="date-range-picker",
                                    min_date_allowed=df["day"].min().date(),
                                    max_date_allowed=df["day"].max().date(),
                                    start_date=df["day"].min().date(),
                                    end_date=df["day"].max().date(),
                                    display_format="YYYY-MM-DD",
                                    style={"width": "100%", "color": "#000000"},
                                ),
                            ]
                        ),
                    ],
                )
            ],
        ),

        # Dashboard tabs
        dcc.Tabs(
            id="dashboard-tabs", value="market-tab",
            colors={"border": "#334155", "primary": "#38bdf8", "background": "#1e293b"},
            children=[
                dcc.Tab(label="Market Intelligence", value="market-tab", children=[
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
                html.Div(
                    style={"backgroundColor": "#1e293b", "padding": "16px", "borderRadius": "8px", "border": "1px solid #334155"},
                    children=[
                        html.H3("Public-equity research signals", style={"marginTop": 0}),
                        html.P("Evidence from equipment mentions only — not a buy recommendation. Signals exclude valuation, earnings, price, risk tolerance, and portfolio context.", style={"color": "#fbbf24", "fontSize": "13px"}),
                        dash_table.DataTable(
                            id="stock-signals-table",
                            columns=[
                                {"name": "Company", "id": "parent_company"}, {"name": "Ticker", "id": "ticker"},
                                {"name": "Mentions", "id": "mentions"}, {"name": "Share", "id": "market_share", "type": "numeric", "format": {"specifier": ".1%"}},
                                {"name": "Recent change", "id": "recent_change_pct", "type": "numeric", "format": {"specifier": "+.0f"}},
                                {"name": "Signal", "id": "signal"}, {"name": "Why", "id": "rationale"},
                            ], data=[], style_table={"overflowX": "auto"},
                            style_header={"backgroundColor": "#334155", "fontWeight": "600"},
                            style_cell={"backgroundColor": "#1e293b", "color": "#f8fafc", "padding": "10px", "textAlign": "left"},
                        ),
                    ],
                ),
            ],
        ),
                ]),
                dcc.Tab(label="Alpha Signal Analysis", value="alpha-tab", children=[
                    html.Div(style={"padding": "24px 0"}, children=[
                        html.H2("Alpha signal analysis", style={"marginTop": 0}),
                        html.P("Tests whether equipment-mention strength is associated with subsequent stock returns. Signals enter at the next close; this is research, not investment advice.", style={"color": "#94a3b8"}),
                        html.Div(style={"backgroundColor": "#172554", "border": "1px solid #1e40af", "padding": "14px 16px", "borderRadius": "6px", "marginBottom": "16px"}, children=[
                            html.H4("How to read the results", style={"margin": "0 0 8px 0"}),
                            html.P("Beta measures the change in forward log return associated with a one-unit increase in the mention signal, after controlling for prior 12-month return and ticker. Positive beta means stronger mention signals were associated with higher subsequent returns; p-value below 0.05 is a conventional, not definitive, significance threshold.", style={"margin": "6px 0", "fontSize": "13px"}),
                            html.P("Mean IC is the average daily Spearman correlation between signal strength and future returns (from -1 to +1). ICIR is Mean IC divided by its variability, while IC t-stat measures how reliably the average differs from zero. Regression n is the number of ticker observations; IC dates is the number of dates used for the correlation.", style={"margin": "6px 0", "fontSize": "13px"}),
                            html.P("Treat small samples, noisy estimates, missing prices, transaction costs, and multiple testing as important limitations. These figures show association, not proof that the signal can be traded profitably.", style={"margin": "6px 0", "fontSize": "13px", "color": "#fbbf24"}),
                        ]),
                        html.Button("Run alpha analysis", id="run-alpha-button", n_clicks=0, style={"backgroundColor": "#38bdf8", "color": "#0f172a", "border": "none", "padding": "10px 16px", "borderRadius": "6px", "fontWeight": "600", "cursor": "pointer"}),
                        html.Div(id="alpha-status", style={"margin": "16px 0", "color": "#94a3b8"}),
                        dcc.Graph(id="alpha-metrics-chart"),
                        dash_table.DataTable(
                            id="alpha-results-table", data=[],
                            columns=[{"name": c, "id": c} for c in ["Horizon", "Beta", "p_value", "Mean IC", "ICIR", "IC t-stat", "Regression n", "IC dates"]],
                            style_table={"overflowX": "auto"}, style_header={"backgroundColor": "#334155", "fontWeight": "600"},
                            style_cell={"backgroundColor": "#1e293b", "color": "#f8fafc", "padding": "10px", "textAlign": "left"},
                        ),
                    ]),
                ]),
            ],
        ),
    ],
)

@app.callback(
    [Output("alpha-metrics-chart", "figure"), Output("alpha-results-table", "data"), Output("alpha-status", "children")],
    Input("run-alpha-button", "n_clicks"),
    prevent_initial_call=True,
)
def update_alpha_analysis(n_clicks):
    try:
        results = run_alpha_analysis()
        chart_data = results[["Horizon", "Mean IC"]].dropna()
        fig = px.bar(chart_data, x="Horizon", y="Mean IC", title="Mean Information Coefficient by Horizon", template="plotly_dark", color="Horizon")
        fig.update_layout(paper_bgcolor="#1e293b", plot_bgcolor="#1e293b", margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
        return fig, results.replace({pd.NA: None}).to_dict("records"), f"Analysis complete across {len(results)} horizons."
    except Exception as exc:
        logger.exception("Alpha analysis failed")
        return empty_figure("Alpha analysis unavailable"), [], f"Alpha analysis unavailable: {exc}"

@app.callback(
    [
        Output("market-share-chart", "figure"),
        Output("mention-volume-chart", "figure"),
        Output("group-toggle-button", "children"),
        Output("stock-signals-table", "data"),
        Output("machine-filter", "options"),
        Output("machine-filter", "value"),
        Output("machine-filter-label", "children"),
    ],
    [
        Input("machine-filter", "value"),
        Input("group-toggle-button", "n_clicks"),
        Input("date-range-picker", "start_date"),
        Input("date-range-picker", "end_date"),
    ],
)
def update_charts(selected_machines, n_clicks, start_date, end_date):
    """
    Toggle charts between:
      - Tool view: resolved_machine
      - Parent Company view: parent_company
    Also filters by date range and applies 30-day rolling window.
    """
    use_parent_company = bool((n_clicks or 0) % 2)
    target_values = all_companies if use_parent_company else all_machines
    target_options = [{"label": value, "value": value} for value in target_values]
    filter_label = "Filter Parent Companies:" if use_parent_company else "Filter Tool Models:"

    # The selector changes domain with the aggregation view. Translate the
    # existing selection only when the toggle caused this callback; otherwise
    # preserve the user's current selection in the active domain.
    triggered_id = callback_context.triggered_id
    if triggered_id == "group-toggle-button":
        selected_machines = selected_machines or (all_machines if use_parent_company else all_companies)
        if use_parent_company:
            selected_machines = sorted(
                df.loc[df["resolved_machine"].isin(selected_machines), "parent_company"]
                .dropna().unique().tolist()
            )
        else:
            selected_machines = sorted(
                df.loc[df["parent_company"].isin(selected_machines), "resolved_machine"]
                .dropna().unique().tolist()
            )
    else:
        selected_machines = [value for value in (selected_machines or []) if value in target_values]

    filtered_df = df.copy()

    # Filter by the active selector domain.
    if selected_machines:
        filter_column = "parent_company" if use_parent_company else "resolved_machine"
        filtered_df = filtered_df[filtered_df[filter_column].isin(selected_machines)]

    # Filter by date range
    if start_date and end_date:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        filtered_df = filtered_df[
            (filtered_df["day"] >= start_dt) & (filtered_df["day"] <= end_dt)
        ]

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
            [],
            target_options,
            selected_machines,
            filter_label,
        )

    # Apply 30-day rolling window
    # First, ensure we have a complete date range for each group
    all_dates = pd.date_range(
        start=filtered_df["day"].min(), end=filtered_df["day"].max(), freq="D"
    )
    date_grid = pd.DataFrame({"day": all_dates})
    
    # Create a complete grid of dates x groups
    groups = filtered_df[group_col].unique()
    grid = pd.MultiIndex.from_product([all_dates, groups], names=["day", group_col]).to_frame(
        index=False
    )
    
    # Merge with actual data
    merged = pd.merge(
        grid,
        filtered_df,
        on=["day", group_col],
        how="left",
    ).fillna({"mention_count": 0})
    
    # Sort by group and date for rolling calculation
    merged = merged.sort_values([group_col, "day"])
    
    # Calculate 30-day rolling sum for each group
    merged["rolling_mention_count"] = (
        merged.groupby(group_col)["mention_count"]
        .transform(lambda x: x.rolling(window=30, min_periods=1).sum())
    )
    
    # Line / Area chart data (using rolling window)
    line_df = (
        merged[["day", group_col, "rolling_mention_count"]]
        .rename(columns={"rolling_mention_count": "mention_count"})
        .sort_values(["day", group_col])
    )

    # Use independent lines rather than a stacked area chart. Stacking can
    # make one company's boundary rise when another company's mentions fall.
    fig_area = px.line(
        line_df,
        x="day",
        y="mention_count",
        color=group_col,
        title=f"30-Day Rolling Mention Trend ({view_name} View)",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    fig_area.update_layout(
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        xaxis_title="Date",
        yaxis_title="Mentions (30-Day Rolling Sum)",
        legend_title=view_name,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    # Bar chart data (total mentions in selected period)
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
        title=f"Total Mentions by {view_name} (Selected Period)",
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
    signals = build_stock_signals(filtered_df)
    return (
        fig_area,
        fig_bar,
        button_label,
        signals.to_dict("records"),
        target_options,
        selected_machines,
        filter_label,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)