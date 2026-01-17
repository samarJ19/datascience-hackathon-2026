import pandas as pd
import os
from feature_engineering import build_features
from ml_pipeline import run_analytical_pipeline
from visualization import plot_drift_heatmap, plot_risk_clusters

# Ensure output directory exists
os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)

def main():
    print("--- Starting Aadhaar Pulse Analytical Pipeline ---")

    # 1. Load Data
    # NOTE: Ensure these paths match your actual CSV locations
    print("Loading datasets...")
    try:
        enroll = pd.read_csv("data/enrollment.csv")
        demo = pd.read_csv("data/demographic.csv")
        bio = pd.read_csv("data/biometric.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please place 'enrollment.csv', 'demographic.csv', 'biometric.csv' in the 'data/' folder.")
        return

    # 2. Feature Engineering
    print("Engineering Vitality Metrics (Drift, MBU, Dormancy)...")
    df_features = build_features(enroll, demo, bio)

    # 3. Analytical Modeling
    print("Running Risk Clustering & Health Scoring...")
    df_scored, kmeans_model = run_analytical_pipeline(df_features)

    # 4. Generate Visualizations for PDF
    print("Generating Plots...")
    plot_drift_heatmap(df_scored)
    plot_risk_clusters(df_scored)

    # 5. Output Key Findings (For your Report text)
    print("\n--- KEY INSIGHTS GENERATED ---")
    
    avg_drift = df_scored['drift_ratio'].mean()
    print(f"Average National Identity Drift Ratio: {avg_drift:.2f}")
    
    high_risk = df_scored[df_scored['drift_ratio'] > 2.0]
    print(f"Count of High-Risk Pincodes (Drift > 2.0): {len(high_risk)}")
    
    dormant = df_scored[df_scored['is_dormant'] == 1]
    print(f"Count of 'Digital Dark Zones' (High Pop, Zero Bio Updates): {len(dormant)}")
    
    top_risk_districts = high_risk['district'].value_counts().head(5)
    print("\nTop 5 Districts with Highest Identity Degradation:")
    print(top_risk_districts)
    
    # Save processed data for review
    df_scored.to_csv("output/aadhaar_pulse_analysis.csv", index=False)
    print("\nAnalysis Complete. Results saved to 'output/' directory.")

if __name__ == "__main__":
    main()