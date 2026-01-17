import numpy as np
import pandas as pd

def compute_aihs(df):
    """
    Recalibrated Scoring Logic (Stricter Benchmarks)
    to handle high-velocity backlog clearing data.
    """
    df = df.copy()
    
    # ---------------------------------------------------------
    # 1. BIOMETRIC EFFICIENCY SCORE (Stricter & Logarithmic)
    # ---------------------------------------------------------
    # Old Problem: Velocity was 17.0, Threshold was 2.0. Score saturated at 100.
    # New Logic: Use Log-Growth. 
    #   - Velocity 2.0 -> Score ~50 (Passable)
    #   - Velocity 10.0 -> Score ~85 (Good)
    #   - Velocity 20.0 -> Score ~100 (Excellent)
    
    # Formula: 30 * ln(Velocity + 1)
    # Natural Log dampens the huge numbers (like 17 vs 50).
    
    # Clip velocity to avoid negative logs (though +1 handles it)
    v = df['mbu_velocity'].clip(lower=0)
    
    # Factor 30 makes Velocity=20 hit approx 90-95 score.
    mbu_score = 30 * np.log(v + 1)
    df['score_mbu'] = mbu_score.clip(upper=100)

    # ---------------------------------------------------------
    # 2. IDENTITY INTEGRITY SCORE (Tighter Tolerance)
    # ---------------------------------------------------------
    # Old Problem: Drift 1.5 was scoring 87. Too generous.
    # New Logic: Drift > 3.0 is Critical (0).
    # Range is now [1.0, 3.0] instead of [1.0, 5.0].
    
    drift_capped = df['drift_ratio'].clip(lower=1.0, upper=3.0)
    
    # Linear penalty over the tighter range
    # If Drift = 1.0 -> Penalty 0 -> Score 100
    # If Drift = 2.0 -> Penalty 50 -> Score 50
    # If Drift = 3.0 -> Penalty 100 -> Score 0
    drift_penalty = ((drift_capped - 1.0) / 2.0) * 100
    
    df['score_drift'] = 100 - drift_penalty

    # ---------------------------------------------------------
    # 3. FINAL AIHS
    # ---------------------------------------------------------
    # Keep the 60/40 Split
    df['AIHS'] = (0.6 * df['score_mbu']) + (0.4 * df['score_drift'])
    
    # Rounding
    df['AIHS'] = df['AIHS'].round(2)
    
    return df