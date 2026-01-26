import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#adding random comment to push it again 

#New comment, same job, testing webhook handler nothing else
def generate_district_comparison(indore_path, dindori_path):
    """
    Generates high-contrast visualizations to compare Urban vs. Rural performance
    across MBU, Drift, and AIHS metrics.
    """
    # 1. Load Datasets
    try:
        df_indore = pd.read_csv(indore_path)
        df_dindori = pd.read_csv(dindori_path)
    except FileNotFoundError:
        print("Error: Files not found. Please ensure input CSVs exist.")
        return

    # 2. Tag and Combine
    df_indore['District_Type'] = 'Indore (Urban)'
    df_dindori['District_Type'] = 'Dindori (Rural)'
    
    # Select only relevant columns for the "Head-to-Head"
    cols = ['District_Type', 'score_mbu', 'score_drift', 'AIHS']
    df_combined = pd.concat([df_indore[cols], df_dindori[cols]], ignore_index=True)

    # Set a professional theme
    sns.set_theme(style="whitegrid")

    # --- VISUALIZATION 1: The "Scorecard" (Grouped Bar Chart) ---
    # Best for: Comparing the raw averages side-by-side
    plt.figure(figsize=(10, 6))
    
    # Melt data for grouped bar format
    df_melted = df_combined.melt(id_vars='District_Type', var_name='Metric', value_name='Score')
    
    # Plot
    ax = sns.barplot(
        data=df_melted, 
        x='Metric', 
        y='Score', 
        hue='District_Type', 
        palette={'Indore (Urban)': '#3498db', 'Dindori (Rural)': '#e74c3c'},
        errorbar=None  # Remove error bars for cleaner look
    )
    
    # Annotate values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3, fontsize=10, fontweight='bold')  # type: ignore

    plt.title("The Welfare-Compliance Paradox: Rural vs. Urban Performance", fontsize=14, fontweight='bold')
    plt.ylim(0, 110)
    plt.ylabel("Index Score (0-100)")
    plt.xlabel("")
    plt.legend(title="Region")
    plt.tight_layout()
    plt.savefig('output/comparison_bar_chart.png')
    print("Generated: output/comparison_bar_chart.png")

    # --- VISUALIZATION 2: The "Strategy Map" (Scatter Plot) ---
    # Best for: Showing the trade-off between Integrity (Drift) and Efficiency (MBU)
    plt.figure(figsize=(10, 7))
    
    sns.scatterplot(
        data=df_combined, 
        x='score_drift', 
        y='score_mbu', 
        hue='District_Type', 
        style='District_Type',
        palette={'Indore (Urban)': '#3498db', 'Dindori (Rural)': '#e74c3c'},
        s=100, 
        alpha=0.6,
        edgecolor='w'
    )
    
    plt.title("Operational Strategy: Efficiency vs. Integrity", fontsize=14, fontweight='bold')
    plt.xlabel("Identity Integrity (Drift Score) \n <-- Low Risk | High Risk -->")
    plt.ylabel("Biometric Efficiency (MBU Score) \n <-- Lagging | Clearing Backlog -->")
    plt.axvline(x=90, color='gray', linestyle='--', alpha=0.5, label='High Integrity Zone')
    plt.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='High Efficiency Zone')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('output/comparison_scatter_strategy.png')
    print("Generated: output/comparison_scatter_strategy.png")

    # --- VISUALIZATION 3: Consistency Check (Box Plot) ---
    # Best for: Showing if the performance is consistent or varies wildly
    plt.figure(figsize=(8, 6))
    
    sns.boxplot(
        data=df_combined, 
        x='District_Type', 
        y='AIHS', 
        palette={'Indore (Urban)': '#3498db', 'Dindori (Rural)': '#e74c3c'},
        width=0.5
    )
    
    plt.title("Consistency of Service Delivery (AIHS Distribution)", fontsize=14, fontweight='bold')
    plt.ylabel("Overall Identity Health Score")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig('output/comparison_consistency_box.png')
    print("Generated: output/comparison_consistency_box.png")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    # Ensure you are pointing to the correct files
    generate_district_comparison(
        indore_path='data/output/indore/aadhaar_pulse_analysis_indore.csv', 
        dindori_path='data/output/dindori/aadhaar_pulse_analysis_dindori.csv'
    )