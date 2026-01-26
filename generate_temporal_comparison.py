import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_temporal_comparison(indore_path, dindori_path):
    """
    Generates a Time-Series comparison to visualize the 'Pulse' vs. 'Season' effect.
    """
    # 1. Load Datasets
    try:
        df_indore = pd.read_csv(indore_path)
        df_dindori = pd.read_csv(dindori_path)
    except FileNotFoundError:
        print("Error: Files not found.")
        return

    # 2. Pre-processing
    # Filter for the relevant timeline (e.g., 2025) to zoom in on the pattern
    # Assuming 'period' format is 'YYYY-MM'
    df_indore = df_indore.sort_values('period')
    df_dindori = df_dindori.sort_values('period')

    df_indore['District_Type'] = 'Indore (Urban)'
    df_dindori['District_Type'] = 'Dindori (Rural)'

    # Combine
    cols = ['period', 'AIHS', 'District_Type']
    df_combined = pd.concat([df_indore[cols], df_dindori[cols]],ignore_index=True)

    # 3. Visualization: The "Heartbeat" Chart
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    # Plot Lines
    sns.lineplot(
        data=df_combined, 
        x='period', 
        y='AIHS', 
        hue='District_Type', 
        style='District_Type',
        markers=True, 
        dashes=False,
        linewidth=2.5,
        palette={'Indore (Urban)': '#3498db', 'Dindori (Rural)': '#e74c3c'}
    )

    # 4. Formatting for Impact
    plt.title("Behavioral Fingerprinting: Urban Volatility vs. Rural Seasonality", fontsize=14, fontweight='bold')
    plt.ylabel("Identity Health Score (AIHS)")
    plt.xlabel("Timeline (2025-26)")
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.legend(title="Region")
    plt.grid(True, linestyle='--', alpha=0.5)

    # 5. Annotation (The "Proof")
    # Note: You might need to adjust the x/y coordinates based on your exact data points
    # This example assumes the dip happens around Index 3-5 (April-June)
    
    # Example Annotation for Rural Dip
    # plt.annotate('Harvest Season\n(Mass Absenteeism)', 
    #              xy=(3, 40), xytext=(3, 60),
    #              arrowprops=dict(facecolor='#e74c3c', shrink=0.05),
    #              color='#e74c3c', fontweight='bold')

    plt.tight_layout()
    plt.savefig('output/temporal_comparison.png')
    print("Generated: output/temporal_comparison.png")

if __name__ == "__main__":
    generate_temporal_comparison(
        indore_path='data/output/indore/aadhaar_pulse_analysis_indore.csv', 
        dindori_path='data/output/dindori/aadhaar_pulse_analysis_dindori.csv'
    )