import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DB_NAME = "local_market_share.db"

def analyze_and_plot():
    # 1. Connect and query database
    conn = sqlite3.connect(DB_NAME)
    
    query = """
        SELECT 
            strftime('%Y-%m', c.date) as year_month,
            m.standardized_machine,
            COUNT(*) as mention_count
        FROM candidates c
        JOIN machine_mentions m ON c.doi = m.doi
        GROUP BY year_month, m.standardized_machine
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("No machine mentions found in the database yet to plot!")
        return

    # 2. Pivot data to get total mentions per month to compute % share
    pivot_df = df.pivot(index='year_month', columns='standardized_machine', values='mention_count').fillna(0)
    
    # Calculate market share percentage row-by-row
    market_share_df = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100

    # 3. Create the Plot
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    
    # Plot as a stacked area chart or a line chart (Line chart is cleaner for tracking specific competitors)
    for column in market_share_df.columns:
        plt.plot(market_share_df.index, market_share_df[column], marker='o', linewidth=2, label=column)
        
    plt.title("Spatial Transcriptomics Market Share Evolution (bioRxiv Mentions)", fontsize=16, fontweight='bold')
    plt.xlabel("Timeline (Year-Month)", fontsize=12)
    plt.ylabel("Estimated Market Share (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.xticks(rotation=45)
    plt.legend(title="Instrument Platform", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save chart locally
    output_image = "spatial_market_share.png"
    plt.savefig(output_image, dpi=300)
    print(f"Success! Market share trend chart generated and saved as '{output_image}'")

if __name__ == "__main__":
    analyze_and_plot()