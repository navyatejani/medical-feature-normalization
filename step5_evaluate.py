import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (recall_score, precision_score, fbeta_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, roc_curve, roc_auc_score)

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

TARGET_RECALL = 0.98

# ---------- 1. Load data and tuned models ----------
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
X_train, y_train = train.drop(columns=["target"]), train["target"]
X_test, y_test = test.drop(columns=["target"]), test["target"]

names = ["LogisticRegression", "SVM", "RandomForest"]
models = {n: joblib.load(f"models/{n}_tuned.joblib") for n in names}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def score_kind(name):
    return "predict_proba" if name == "RandomForest" else "decision_function"


def get_scores(model, X, kind):
    if kind == "predict_proba":
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def pick_threshold(y_true, scores, target):
    """Highest threshold that still reaches the target recall."""
    prec, rec, thr = precision_recall_curve(y_true, scores)
    ok = np.where(rec[:-1] >= target)[0]
    return thr[ok[-1]]


# ---------- 2. Choose threshold on TRAIN out-of-fold scores ----------
rows = []
info = {}
for name, model in models.items():
    kind = score_kind(name)
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method=kind)
    if oof.ndim == 2:
        oof = oof[:, 1]

    thr = pick_threshold(y_train, oof, TARGET_RECALL)
    oof_pred = (oof >= thr).astype(int)
    oof_prec = precision_score(y_train, oof_pred)
    oof_rec = recall_score(y_train, oof_pred)

    # ---------- 3. Evaluate on TEST (once) ----------
    test_scores = get_scores(model, X_test, kind)
    pred_default = model.predict(X_test)
    pred_tuned = (test_scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_tuned).ravel()

    rows.append({
        "Model": name,
        "Threshold": round(float(thr), 4),
        "Train OOF Recall": round(oof_rec, 4),
        "Train OOF Precision": round(oof_prec, 4),
        "Test Recall (default)": round(recall_score(y_test, pred_default), 4),
        "Test Recall (tuned)": round(recall_score(y_test, pred_tuned), 4),
        "Test Precision (tuned)": round(precision_score(y_test, pred_tuned), 4),
        "Test F2 (tuned)": round(fbeta_score(y_test, pred_tuned, beta=2), 4),
        "Missed cancers (FN)": int(fn),
        "False alarms (FP)": int(fp),
        "Test ROC-AUC": round(roc_auc_score(y_test, test_scores), 4),
    })
    info[name] = {"kind": kind, "thr": float(thr), "scores": test_scores,
                  "pred_default": pred_default, "pred_tuned": pred_tuned,
                  "oof_prec": oof_prec}

results = pd.DataFrame(rows)
results.to_csv("outputs/final_evaluation.csv", index=False)
print("\n=== Final evaluation (threshold chosen on train, scored on test) ===")
print(results.T.to_string(header=False))

# ---------- 4. Pick final model by TRAIN precision at target recall ----------
final_name = max(info, key=lambda n: info[n]["oof_prec"])
f = info[final_name]
print(f"\nFINAL MODEL: {final_name}  (threshold = {f['thr']:.4f}, score = {f['kind']})")

# ---------- 5. Plots for the final model ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, pred, title in [(axes[0], f["pred_default"], "Default threshold"),
                        (axes[1], f["pred_tuned"], "Recall-optimized threshold")]:
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=["Benign", "Malignant"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{final_name}\n{title}")
plt.tight_layout()
plt.savefig("outputs/06_confusion_matrices.png", dpi=150)
plt.close()

prec, rec, _ = precision_recall_curve(y_test, f["scores"])
fpr, tpr, _ = roc_curve(y_test, f["scores"])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(rec, prec)
axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
axes[0].set_title("Precision-Recall Curve")
axes[1].plot(fpr, tpr)
axes[1].plot([0, 1], [0, 1], "--", color="gray")
axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate (Recall)")
axes[1].set_title("ROC Curve")
plt.tight_layout()
plt.savefig("outputs/07_pr_roc_curves.png", dpi=150)
plt.close()

# ---------- 6. Save the final bundle for the demo ----------
joblib.dump({"model": models[final_name], "threshold": f["thr"],
             "score_kind": f["kind"], "name": final_name,
             "feature_names": list(X_train.columns)},
            "models/final_model.joblib")
print("Saved: models/final_model.joblib, outputs/final_evaluation.csv and 3 plots")