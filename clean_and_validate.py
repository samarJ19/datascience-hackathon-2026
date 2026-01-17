import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def clean_data_artifacts(input_path="output/aadhaar_pulse_analysis.csv", output_path="aadhaar_pulse_clean.csv"):
    """
    Removes aggregation artifacts (summed years/months) from the dataset.
    """
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print("Error: File not found. Make sure you ran the pipeline first.")
        return None

    initial_count = len(df)
    
    # --- CLEANING LOGIC ---
    # 1. Valid Month Check: Month must be 1-12
    # 2. Valid Year Check: Year must be reasonable (e.g., 2010-2030)
    # 3. Valid Pincode Check: Pincodes are 6 digits (< 1,000,000)
    
    # We use 'coerce' to handle non-numeric garbage if any
    df['month'] = pd.to_numeric(df['month'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['pincode'] = pd.to_numeric(df['pincode'], errors='coerce')

    df_clean = df[
        (df['month'].between(1, 12)) & 
        (df['year'].between(2010, 2030)) & 
        (df['pincode'] < 1000000)
    ].copy()

    # Re-create the 'period' column to ensure it's correct
    df_clean['period'] = df_clean['year'].astype(int).astype(str) + "-" + df_clean['month'].astype(int).astype(str).str.zfill(2)

    removed_count = initial_count - len(df_clean)
    print(f"Cleaning Complete. Removed {removed_count} corrupt rows (Aggregation Artifacts).")
    print(f"Valid rows remaining: {len(df_clean)}")
    
    # Save the clean version
    df_clean.to_csv(output_path, index=False)
    return df_clean

def plot_event_test(df, output_path="output/event_validation.png"):
    """
    Plots the Time Series to check for Seasonality/Event Dips.
    """
    print("Generating Event Test Visualization...")
    plt.figure(figsize=(12, 6))
    
    # Group by Period to get the National/District Average Trend
    # We sort by 'period' to ensure the line graph connects correctly
    temporal_trend = df.groupby('period')['AIHS'].mean().reset_index()
    temporal_trend = temporal_trend.sort_values('period')
    
    # Plotting
    sns.lineplot(data=temporal_trend, x='period', y='AIHS', marker='o', color='#2c3e50', linewidth=2.5)
    
    # Formatting for Impact
    plt.title("Event Test: Identity Vitality Trends (2025-2026)", fontsize=14, fontweight='bold')
    plt.ylabel("Average AIHS (Health Score)", fontsize=12)
    plt.xlabel("Timeline", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Highlight the "Seasonality" or "Dip"
    # Identify the lowest point automatically
    min_idx = temporal_trend['AIHS'].idxmin()
    min_period = temporal_trend.loc[min_idx, 'period']
    min_score = temporal_trend.loc[min_idx, 'AIHS']
    
    plt.annotate(f'Seasonal Dip\n({min_period})', 
                 xy=(min_idx, min_score), 
                 xytext=(min_idx, min_score + 5),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Chart saved to: {output_path}")

if __name__ == "__main__":
    # 1. Clean the Data
    df_clean = pd.read_csv("output/aadhaar_pulse_analysis.csv")
    
    # 2. Run the Event Test if data is valid
    if df_clean is not None and not df_clean.empty:
        plot_event_test(df_clean)
    else:
        print("Dataset empty after cleaning. Check input files.")