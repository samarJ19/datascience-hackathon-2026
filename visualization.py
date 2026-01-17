import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_drift_heatmap(df, output_path="output/drift_heatmap.png"):
    """
    Generates a correlation heatmap or geospatial proxy for the PDF.
    """
    plt.figure(figsize=(10, 6))
    
    # Pivot to see Drift Ratio by District over Time (Year-Month)
    # Using 'date' string for axis
    df['period'] = df['year'].astype(str) + "-" + df['month'].astype(str).str.zfill(2)
    
    pivot = df.pivot_table(
        index="district", 
        columns="period", 
        values="drift_ratio", 
        aggfunc="mean"
    )
    
    sns.heatmap(pivot, cmap="RdYlGn_r", annot=True, fmt=".1f")
    plt.title("Identity Drift Ratio by District (High Score = High Risk)")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Generated: {output_path}")

def plot_risk_clusters(df, output_path="output/risk_clusters.png"):
    """
    Scatter plot showing the separation of Risk Profiles.
    """
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(
        data=df, 
        x="drift_ratio", 
        y="mbu_velocity", 
        hue="risk_cluster", 
        palette="viridis",
        s=100,
        alpha=0.7
    )
    
    plt.title("Risk Profiling: Identity Drift vs. MBU Velocity")
    plt.xlabel("Identity Drift (Demographic Changes / Bio Updates)")
    plt.ylabel("MBU Velocity (Child Updates / Cohort Size)")
    plt.axvline(x=1.5, color='r', linestyle='--', label='Critical Drift Threshold')
    plt.legend(title="Cluster Group")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    print(f"Generated: {output_path}")