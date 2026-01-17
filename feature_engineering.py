import pandas as pd
import numpy as np

def normalize_dates(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    return df

def aggregate_numeric(df, keys):
    # 1. Select all numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    
    # 2. THE FIX: Remove Year, Month, Pincode from the "To-Sum" list
    # We only want to sum 'age_0_5', 'bio_updates', etc.
    cols_to_sum = [c for c in numeric_cols if c not in keys]
    
    # 3. Group and Sum only the counts
    return df.groupby(keys, as_index=False)[cols_to_sum].sum()

def build_features(enroll, demo, bio):
    # Standardize Dates
    enroll = normalize_dates(enroll)
    demo = normalize_dates(demo)
    bio = normalize_dates(bio)

    # The Unique ID Keys
    keys = ["state", "district", "pincode", "year", "month"]
    
    # Aggregation (Now Safe)
    enroll_agg = aggregate_numeric(enroll, keys)
    demo_agg = aggregate_numeric(demo, keys)
    bio_agg = aggregate_numeric(bio, keys)

    # Merge the three datasets
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

    # 2. Identity Drift Ratio (Demo / Bio)
    demo_adult = df.get("demo_age_17_", 0)
    bio_adult = df.get("bio_age_17_", 0)
    df["drift_ratio"] = demo_adult / (bio_adult + 1)

    # 3. MBU Velocity (Child Bio / Child Enrol)
    target_cohort = df.get("age_5_17", 0)
    child_updates = df.get("bio_age_5_17", 0)
    df["mbu_velocity"] = child_updates / (target_cohort + 1)

    # 4. Dormancy Flag
    high_pop_threshold = df["total_enrolment"].quantile(0.75)
    total_bio = df.get("bio_age_5_17", 0) + df.get("bio_age_17_", 0)
    
    df["is_dormant"] = np.where(
        (df["total_enrolment"] > high_pop_threshold) & (total_bio == 0), 
        1, 0
    )

    return df