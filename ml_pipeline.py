import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scoring import compute_aihs

def run_analytical_pipeline(df):
    df = df.copy()

    # 1. Compute Scores First (Deterministic Logic)
    df = compute_aihs(df)
    
    # 2. Perform Clustering on the Risk Metrics
    # We cluster on the *Scores* now, as they are cleaner features
    features = ['score_mbu', 'score_drift', 'total_enrolment']
    X = df[features].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster 0: High Performing
    # Cluster 1: Average/Maintenance
    # Cluster 2: Critical/Dormant
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df["risk_cluster"] = clusters

    # Save models
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(kmeans, "models/kmeans.pkl")

    return df, kmeans