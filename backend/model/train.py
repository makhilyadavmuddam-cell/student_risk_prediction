import numpy as np
import pandas as pd
import json
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score
)
from sklearn.calibration import CalibratedClassifierCV
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEATURES = [
    "attendance_rate", "attendance_trend", "avg_score", "score_trend",
    "assignments_ontime_pct", "avg_submission_delay", "library_visits_per_week",
    "failed_subjects", "sgpa", "sgpa_trend", "days_inactive", "backlogs",
    "first_gen", "is_rural", "has_scholarship"
]

FEATURE_LABELS = {
    "attendance_rate": "Attendance Rate",
    "attendance_trend": "Attendance Trend",
    "avg_score": "Average Score",
    "score_trend": "Score Trend",
    "assignments_ontime_pct": "On-Time Submissions",
    "avg_submission_delay": "Submission Delay (days)",
    "library_visits_per_week": "Library Visits/Week",
    "failed_subjects": "Failed Subjects",
    "sgpa": "SGPA",
    "sgpa_trend": "SGPA Trend",
    "days_inactive": "Days Inactive",
    "backlogs": "Backlogs",
    "first_gen": "First-Gen Student",
    "is_rural": "Rural Origin",
    "has_scholarship": "Has Scholarship",
}

def compute_shap_contributions(student_row, feature_importance, feature_means, feature_stds):
    """Compute manual SHAP-style contributions for one student."""
    protective = {
        "attendance_rate", "avg_score", "assignments_ontime_pct",
        "library_visits_per_week", "sgpa", "sgpa_trend",
        "attendance_trend", "score_trend",
    }
    contribs = []
    for feat in FEATURES:
        val = student_row.get(feat, feature_means[feat])
        mean = feature_means[feat]
        std = feature_stds[feat] if feature_stds[feat] > 0 else 1
        z = (val - mean) / std
        direction = -1 if feat in protective else 1
        impact = round(float(feature_importance.get(feat, 0)) * z * direction, 4)
        contribs.append({
            "feature": feat,
            "label": FEATURE_LABELS.get(feat, feat),
            "value": round(float(val), 2),
            "impact": impact,
            "direction": "risk" if impact > 0 else "protective",
        })
    contribs.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return contribs[:8]

def train_model():
    df = pd.read_csv("backend/data/students.csv")
    X = df[FEATURES]
    y = df["dropout"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Gradient Boosting — no xgboost needed, sklearn's GBC is excellent
    base_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    # Calibrate for proper probabilities
    model = CalibratedClassifierCV(base_model, cv=5, method='sigmoid')
    model.fit(X_train, y_train)

    # Metrics
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n=== MODEL METRICS ===")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Safe", "At-Risk"]))

    # Feature importances from underlying estimator
    fi = base_model.fit(X_train, y_train).feature_importances_
    feature_importance = dict(zip(FEATURES, fi.tolist()))

    # Save model
    os.makedirs("backend/model", exist_ok=True)
    with open("backend/model/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("backend/model/feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=2)
    with open("backend/model/feature_labels.json", "w") as f:
        json.dump(FEATURE_LABELS, f, indent=2)

    metrics = {
        "auc_roc": round(auc, 4),
        "average_precision": round(ap, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "features": FEATURES,
        "n_at_risk": int(y.sum()),
        "n_safe": int((y == 0).sum()),
    }
    with open("backend/model/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Generate 10 sample SHAP explanations ──────────────────────────────
    feature_means = df[FEATURES].mean().to_dict()
    feature_stds = df[FEATURES].std().to_dict()

    # Pick 10 diverse students: 4 high-risk, 3 medium, 3 low
    students_json_path = "backend/data/students.json"
    with open(students_json_path) as f:
        all_students = json.load(f)

    high = [s for s in all_students if s["risk_level"] == "High"][:4]
    med  = [s for s in all_students if s["risk_level"] == "Medium"][:3]
    low  = [s for s in all_students if s["risk_level"] == "Low"][:3]
    samples = high + med + low

    shap_samples = []
    for s in samples:
        feat_dict = {
            "attendance_rate": s["attendance_rate"],
            "attendance_trend": s["attendance_trend"],
            "avg_score": s["avg_score"],
            "score_trend": s["score_trend"],
            "assignments_ontime_pct": s["assignments_ontime_pct"],
            "avg_submission_delay": s["avg_submission_delay"],
            "library_visits_per_week": s["library_visits_per_week"],
            "failed_subjects": s["failed_subjects"],
            "sgpa": s["sgpa"],
            "sgpa_trend": s["sgpa_trend"],
            "days_inactive": s["days_inactive"],
            "backlogs": s["backlogs"],
            "first_gen": int(s["first_gen"]),
            "is_rural": int(s["origin"] == "Rural"),
            "has_scholarship": int(s["scholarship"] != "None"),
        }
        contribs = compute_shap_contributions(feat_dict, feature_importance, feature_means, feature_stds)
        shap_samples.append({
            "student_id": s["student_id"],
            "name": s["name"],
            "department": s["department"],
            "risk_score": s["risk_score"],
            "risk_level": s["risk_level"],
            "contributions": contribs,
        })

    with open("backend/model/shap_samples.json", "w") as f:
        json.dump(shap_samples, f, indent=2)

    print(f"\nModel saved. AUC: {auc:.4f}")
    print(f"SHAP samples saved for {len(shap_samples)} students.")
    return model, feature_importance

if __name__ == "__main__":
    train_model()
