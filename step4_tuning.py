import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ---------- 1. Load the saved training split ----------
train = pd.read_csv("data/train.csv")
X_train = train.drop(columns=["target"])
y_train = train["target"]
print("Training data:", X_train.shape)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------- 2. Define models and parameter grids ----------
# StandardScaler stays INSIDE the pipeline (no data leakage)
search_space = {
    "LogisticRegression": (
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(max_iter=5000, random_state=42))]),
        {
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__class_weight": [None, "balanced", {0: 1, 1: 3}],
        },
    ),
    "SVM": (
        Pipeline([("scaler", StandardScaler()),
                  ("clf", SVC(probability=True, random_state=42))]),
        {
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", 0.01, 0.001],
            "clf__kernel": ["rbf", "linear"],
            "clf__class_weight": [None, "balanced", {0: 1, 1: 3}],
        },
    ),
    "RandomForest": (
        Pipeline([("scaler", StandardScaler()),
                  ("clf", RandomForestClassifier(random_state=42))]),
        {
            "clf__n_estimators": [100, 300],
            "clf__max_depth": [None, 5, 10],
            "clf__class_weight": [None, "balanced", {0: 1, 1: 3}],
        },
    ),
}

# ---------- 3. Run grid search optimizing RECALL ----------
summary = []
best_models = {}

for name, (pipe, grid) in search_space.items():
    print(f"\nTuning {name} ...")
    gs = GridSearchCV(pipe, grid, scoring="recall", cv=cv, n_jobs=-1, refit=True)
    gs.fit(X_train, y_train)

    best_models[name] = gs.best_estimator_
    summary.append({
        "Model": name,
        "Best CV Recall": round(gs.best_score_, 4),
        "Best Params": gs.best_params_,
    })
    print(f"  Best CV recall: {gs.best_score_:.4f}")
    print(f"  Best params:    {gs.best_params_}")

# ---------- 4. Save every tuned model + summary ----------
for name, model in best_models.items():
    joblib.dump(model, f"models/{name}_tuned.joblib")

summary_df = pd.DataFrame(summary).sort_values("Best CV Recall", ascending=False)
summary_df.to_csv("outputs/tuning_summary.csv", index=False)

print("\n=== Tuning summary ===")
print(summary_df[["Model", "Best CV Recall"]].to_string(index=False))
print("\nSaved tuned models in 'models/' and summary in 'outputs/tuning_summary.csv'")