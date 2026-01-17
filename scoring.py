import numpy as np
import pandas as pd

def compute_aihs(df):
    """
    Calculates the Aadhaar Identity Health Score (AIHS) using 
    Absolute Policy Thresholds to prevent outlier skewing.
    """
    df = df.copy()
    
    # ---------------------------------------------------------
    # 1. BIOMETRIC EFFICIENCY SCORE (S_MBU)
    # ---------------------------------------------------------
    # Benchmark: Velocity >= 2.0 is perfect (100). Velocity 0 is critical (0).
    # Logic: If you update 2 children for every 1 new enrol, you are clearing backlog.
    # We multiply velocity by 50 to scale 2.0 -> 100.
    
    velocity_score = df['mbu_velocity'] * 50
    df['score_mbu'] = velocity_score.clip(lower=0, upper=100)

    # ---------------------------------------------------------
    # 2. IDENTITY INTEGRITY SCORE (S_DRIFT)
    # ---------------------------------------------------------
    # Benchmark: Drift <= 1.0 is perfect (100). Drift >= 5.0 is critical (0).
    # We use linear decay between 1.0 and 5.0.
    
    # Cap drift at 1.0 on the low end (ratios < 1 are treated as perfect 1:1)
    # Cap drift at 5.0 on the high end (anything > 5 is equally bad)
    drift_capped = df['drift_ratio'].clip(lower=1.0, upper=5.0)
    
    # Calculate penalty: Map range [1..5] to [0..100]
    # (Drift - 1) / (5 - 1) gives fraction of "badness"
    drift_penalty = ((drift_capped - 1.0) / 4.0) * 100
    
    df['score_drift'] = 100 - drift_penalty

    # ---------------------------------------------------------
    # 3. FINAL AIHS CALCULATION
    # ---------------------------------------------------------
    # Weighting: 60% Efficiency (Action), 40% Integrity (Risk)
    df['AIHS'] = (0.6 * df['score_mbu']) + (0.4 * df['score_drift'])
    
    # Round for cleaner display
    df['AIHS'] = df['AIHS'].round(2)
    
    return df