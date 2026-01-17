import pandas as pd
import numpy as np

def normalize_dates(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df

def aggregate_numeric(df, keys):
    numeric_cols = df.select_dtypes(include="number").columns
    return df.groupby(keys, as_index=False)[numeric_cols].sum()

def build_features(enroll, demo, bio):
    enroll = normalize_dates(enroll)
    demo = normalize_dates(demo)
    bio = normalize_dates(bio)

    keys = ["state", "district", "pincode", "year", "month"]
    
    enroll_agg = aggregate_numeric(enroll, keys)
    demo_agg = aggregate_numeric(demo, keys)
    bio_agg = aggregate_numeric(bio, keys)

    df = (
        enroll_agg
        .merge(demo_agg, on=keys, how="left")
        .merge(bio_agg, on=keys, how="left")
    ).fillna(0)

    # --- Feature Engineering ---

    # 1. Total Enrolment Proxy
    if "total_enrolment" not in df.columns:
        df["total_enrolment"] = (
            df.get("age_0_5", 0) +
            df.get("age_5_17", 0) +
            df.get("age_18_greater", 0)
        )

    # 2. Identity Drift Ratio
    # (Demographic Updates / Biometric Updates)
    # Add +1 smoothing to denominator
    demo_adult = df.get("demo_age_17_", 0)
    bio_adult = df.get("bio_age_17_", 0)
    df["drift_ratio"] = demo_adult / (bio_adult + 1)

    # 3. MBU Velocity
    # (Child Bio Updates / Child Enrolments)
    # Add +1 smoothing to denominator
    target_cohort = df.get("age_5_17", 0)
    child_updates = df.get("bio_age_5_17", 0)
    df["mbu_velocity"] = child_updates / (target_cohort + 1)

    # 4. Dormancy Flag (for Clustering/Visuals, not Scoring)
    # High Enrolment but Zero Updates
    high_pop_threshold = df["total_enrolment"].quantile(0.75)
    total_bio = df.get("bio_age_5_17", 0) + df.get("bio_age_17_", 0)
    
    df["is_dormant"] = np.where(
        (df["total_enrolment"] > high_pop_threshold) & (total_bio == 0), 
        1, 0
    )

    # Helper columns for display
    df["total_bio_updates"] = total_bio
    df["total_demo_updates"] = df.get("demo_age_5_17", 0) + df.get("demo_age_17_", 0)

    return df