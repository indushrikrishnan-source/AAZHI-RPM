"""
Trains the Random Forest risk classifier on the cleaned NOAA dataset,
evaluates it (single split + 5-fold cross-validation + ROC-AUC), and
exports the global prediction grid used by the dashboard.

Input:  data/processed/noaa_cleaned.csv
Output: models/model_metrics.json
        models/full_quant_metrics.json
        models/risk_model.joblib
        docs/data/grid.json
        docs/data/samples.json
        docs/data/metrics.json
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score,
                              roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                      train_test_split)
from sklearn.neighbors import BallTree, KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/processed/noaa_cleaned.csv"
GYRE_CENTERS = np.array([[32, -145], [30, -40], [-30, -100], [-30, -15], [-35, 68]])
FEATURES = ["Latitude", "Longitude", "Year", "Month", "Ocean_enc", "Method_enc", "dist_to_gyre"]
COVERAGE_RADIUS_KM = 800


def load_and_engineer(path: str = DATA_PATH):
    df = pd.read_csv(path)
    le_ocean, le_method = LabelEncoder(), LabelEncoder()
    df["Ocean_enc"] = le_ocean.fit_transform(df["Ocean"])
    df["Method_enc"] = le_method.fit_transform(df["Sample_Method"])
    df["y"] = (df["Risk_Label"] == "High").astype(int)

    coords = df[["Latitude", "Longitude"]].values
    dists = np.min([np.sqrt(((coords - c) ** 2).sum(axis=1)) for c in GYRE_CENTERS], axis=0)
    df["dist_to_gyre"] = dists
    return df, le_ocean, le_method


def train_and_evaluate(df: pd.DataFrame):
    X, y = df[FEATURES], df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                  class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    single = {
        "accuracy": round(accuracy_score(y_test, y_pred) * 100, 1),
        "recall_high_risk": round(recall_score(y_test, y_pred) * 100, 1),
        "precision_high_risk": round(precision_score(y_test, y_pred) * 100, 1),
        "f1": round(f1_score(y_test, y_pred) * 100, 1),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importance": {k: float(v) for k, v in zip(FEATURES, clf.feature_importances_.round(4))},
        "n_train": len(X_train),
        "n_test": len(X_test),
        "coverage_by_ocean": df["Ocean"].value_counts().to_dict(),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_cv = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
    cv = {
        m: cross_val_score(clf_cv, X, y, cv=skf, scoring=m).tolist()
        for m in ["accuracy", "recall", "precision", "f1", "roc_auc"]
    }
    cv_summary = {f"{k}_mean": round(np.mean(v) * (100 if k != "roc_auc" else 1), 2 if k == "roc_auc" else 2)
                  for k, v in cv.items()}
    cv_summary.update({f"{k}_std": round(np.std(v) * (100 if k != "roc_auc" else 1), 2 if k == "roc_auc" else 2)
                        for k, v in cv.items()})

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_curve_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": single["roc_auc"]}

    full_metrics = {"single_split": single, "cross_validation_5fold": cv_summary, "roc_curve": roc_curve_data}
    return clf, single, full_metrics


def build_grid(df, clf, le_ocean, le_method):
    lat_grid = np.arange(-85, 86, 2)
    lon_grid = np.arange(-180, 181, 2)
    grid_points = np.array([[la, lo] for la in lat_grid for lo in lon_grid])

    knn_ocean = KNeighborsClassifier(n_neighbors=1).fit(df[["Latitude", "Longitude"]], df["Ocean_enc"])
    knn_method = KNeighborsClassifier(n_neighbors=1).fit(df[["Latitude", "Longitude"]], df["Method_enc"])
    grid_ocean_enc = knn_ocean.predict(grid_points)
    grid_method_enc = knn_method.predict(grid_points)
    grid_dists = np.min([np.sqrt(((grid_points - c) ** 2).sum(axis=1)) for c in GYRE_CENTERS], axis=0)

    grid_df = pd.DataFrame({
        "Latitude": grid_points[:, 0], "Longitude": grid_points[:, 1],
        "Year": 2022, "Month": 6,
        "Ocean_enc": grid_ocean_enc, "Method_enc": grid_method_enc,
        "dist_to_gyre": grid_dists,
    })
    grid_df["risk_prob"] = clf.predict_proba(grid_df[FEATURES])[:, 1].round(3)

    # Coverage mask: only keep cells within COVERAGE_RADIUS_KM of a real sample
    tree = BallTree(np.radians(df[["Latitude", "Longitude"]].values), metric="haversine")
    dist_rad, _ = tree.query(np.radians(grid_df[["Latitude", "Longitude"]].values), k=1)
    grid_df["nearest_sample_km"] = dist_rad[:, 0] * 6371.0
    grid_df = grid_df[grid_df["nearest_sample_km"] <= COVERAGE_RADIUS_KM]

    return [[round(r.Latitude, 1), round(r.Longitude, 1), round(r.risk_prob, 3)] for r in grid_df.itertuples()]


def build_samples(df, n=4000, seed=1):
    sample = df.sample(n=min(n, len(df)), random_state=seed)
    return [[round(r.Latitude, 2), round(r.Longitude, 2), r.Ocean, r.Risk_Label,
              r.Concentration_pieces_m3, int(r.Year), r.Sample_Method]
             for r in sample.itertuples()]


if __name__ == "__main__":
    df, le_ocean, le_method = load_and_engineer()
    clf, single, full_metrics = train_and_evaluate(df)

    print(f"Accuracy: {single['accuracy']}%  Recall: {single['recall_high_risk']}%  "
          f"Precision: {single['precision_high_risk']}%  ROC-AUC: {single['roc_auc']}")

    with open("models/model_metrics.json", "w") as f:
        json.dump(single, f, indent=2)
    with open("models/full_quant_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)
    joblib.dump(clf, "models/risk_model.joblib")

    threshold = float(np.quantile(df["Concentration_pieces_m3"], 0.75))

    grid = build_grid(df, clf, le_ocean, le_method)
    samples = build_samples(df)

    with open("docs/data/grid.json", "w") as f:
        json.dump(grid, f)
    with open("docs/data/samples.json", "w") as f:
        json.dump(samples, f)
    with open("docs/data/metrics.json", "w") as f:
        json.dump({"metrics": single, "threshold": round(threshold, 2)}, f, indent=2)

    print(f"\nGrid cells (coverage-masked): {len(grid)}")
    print(f"Sample points exported: {len(samples)}")
    print("Dashboard data written to docs/data/")
