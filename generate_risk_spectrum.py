import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import os

def generate_risk_spectrum_chart():
    """
    Generates a Horizontal Bar Chart visualizing the Risk Spectrum 
    from 'Digital Dark Zones' (Red) to 'Resilient Hubs' (Green).
    """
    # 1. Define the Strategic Archetypes and their File Paths
    # Note: Ensure these CSV files are present in your directory
    districts_config = [
        {"name": "Bangalore",   "file": "aadhaar_pulse_analysis_bangalore.csv"},
        {"name": "Indore",      "file": "aadhaar_pulse_analysis_indore.csv"},
        {"name": "Dindori",     "file": "aadhaar_pulse_analysis_dindori.csv"},
        {"name": "Banaskantha", "file": "aadhaar_pulse_analysis_banaskantha.csv"},
        {"name": "Sheopur",     "file": "aadhaar_pulse_analysis_sheopur.csv"}
    ]

    data = []

    # 2. Load Data and Extract Scores
    print("Loading District Data...")
    for entry in districts_config:
        file_path = entry["file"]
        district_name = entry["name"]
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # Calculate the Mean AIHS for the district
                avg_score = df['AIHS'].mean()
                data.append({"District": district_name, "AIHS": avg_score})
                print(f" - Loaded {district_name}: Score {avg_score:.1f}")
            except Exception as e:
                print(f" ! Error reading {district_name}: {e}")
        else:
            # Fallback for demonstration if file is missing (using User's approximate values)
            print(f" ! File not found: {file_path}. Using approximate value for visualization.")
            fallback_scores = {
                "Bangalore": 85.0, 
                "Indore": 77.4, 
                "Dindori": 60.0, 
                "Banaskantha": 45.0, 
                "Sheopur": 18.2
            }
            data.append({"District": district_name, "AIHS": fallback_scores.get(district_name, 0)})

    # Create DataFrame
    df_plot = pd.DataFrame(data)
    
    # 3. Sort Data for the Chart (High Score to Low Score)
    df_plot = df_plot.sort_values(by="AIHS", ascending=False)

    # 4. Create the Visualization
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    # Define a Custom Color Map (Red -> Yellow -> Green)
    # 0 = Red (Critical), 50 = Yellow (Warning), 100 = Green (Healthy)
    cmap = mcolors.LinearSegmentedColormap.from_list("risk_gradient", ["#e74c3c", "#f1c40f", "#2ecc71"])
    
    # Normalize scores to 0-1 for color mapping
    norm = mcolors.Normalize(0, 100)
    colors = [cmap(norm(val)) for val in df_plot['AIHS']]

    # Plot Horizontal Bars
    bars = plt.barh(df_plot['District'], df_plot['AIHS'], color=colors)
    
    # Invert Y-axis to have High Score (Green) at the top
    plt.gca().invert_yaxis()

    # 5. Formatting & Annotations
    plt.title("National Risk Segmentation: The Identity Vitality Spectrum", fontsize=16, fontweight='bold')
    plt.xlabel("Identity Health Score (AIHS)", fontsize=12)
    plt.ylabel("District Archetype", fontsize=12)
    plt.xlim(0, 100)
    
    # Add Score Labels to Bars
    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 1, 
            bar.get_y() + bar.get_height()/2, 
            f'{width:.1f}', 
            va='center', 
            fontweight='bold', 
            color='black'
        )

    # Add Vertical Threshold Lines
    plt.axvline(x=20, color='red', linestyle='--', alpha=0.5, label='Critical Failure (<20)')
    plt.axvline(x=50, color='orange', linestyle='--', alpha=0.5, label='Maintenance Debt (<50)')
    plt.legend(loc='lower right')

    # Add Descriptive Text for the "Red Zone"
    red_zone_district = df_plot.iloc[-1]['District']
    plt.annotate(
        f'CRITICAL RED ZONE\n(Deploy Mobile Vans)', 
        xy=(18, 4),  # Adjust coordinates based on bar position
        xytext=(25, 3.5),
        arrowprops=dict(facecolor='red', shrink=0.05),
        fontsize=10, 
        color='#c0392b', 
        fontweight='bold'
    )

    plt.tight_layout()
    output_filename = "output/risk_spectrum_chart.png"
    os.makedirs("output", exist_ok=True)
    plt.savefig(output_filename)
    print(f"\nVisualization saved to: {output_filename}")
    plt.show()

if __name__ == "__main__":
    generate_risk_spectrum_chart()