import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

os.makedirs("outputs", exist_ok=True)

# ---------- 1. Load data (malignant = 1) ----------
df = pd.read_csv("data/breast_cancer.csv")
X = df.drop(columns=["target", "diagnosis"])
y = df["target"]

# ---------- 2. Train/test split (test set stays untouched until Step 5) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print("Train:", X_train.shape, "| Test:", X_test.shape)
print("Malignant in train:", int(y_train.sum()), "| in test:", int(y_test.sum()))

# save split so later steps use the exact same data
X_train.assign(target=y_train).to_csv("data/train.csv", index=False)
X_test.assign(target=y_test).to_csv("data/test.csv", index=False)

# ---------- 3. Scalers and models to compare ----------
scalers = {
    "None (raw)": None,
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
}
models = {
    "LogisticRegression": LogisticRegression(max_iter=5000, random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["recall", "precision", "accuracy", "f1"]

# ---------- 4. Cross-validate every scaler x model combination ----------
rows = []
for s_name, scaler in scalers.items():
    for m_name, model in models.items():
        steps = []
        if scaler is not None:
            steps.append(("scaler", scaler))
        steps.append(("clf", model))
        pipe = Pipeline(steps)

        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring)
        rows.append({
            "Scaler": s_name,
            "Model": m_name,
            "Recall": res["test_recall"].mean(),
            "Precision": res["test_precision"].mean(),
            "Accuracy": res["test_accuracy"].mean(),
            "F1": res["test_f1"].mean(),
        })

results = pd.DataFrame(rows).round(4)
results.to_csv("outputs/scaler_comparison.csv", index=False)

print("\n=== Scaler x Model comparison (5-fold CV on training data) ===")
print(results.sort_values("Recall", ascending=False).to_string(index=False))

# ---------- 5. Plot: recall by scaler and model ----------
pivot = results.pivot(index="Model", columns="Scaler", values="Recall")
pivot = pivot[["None (raw)", "StandardScaler", "MinMaxScaler", "RobustScaler"]]
pivot.plot(kind="bar", figsize=(10, 5))
plt.title("Cross-validated Recall: effect of normalization")
plt.ylabel("Recall (malignant)")
plt.ylim(0.8, 1.0)
plt.xticks(rotation=0)
plt.legend(title="Scaler", loc="lower right")
plt.tight_layout()
plt.savefig("outputs/05_scaler_comparison.png", dpi=150)
plt.close()

print("\nSaved: outputs/scaler_comparison.csv and outputs/05_scaler_comparison.png")