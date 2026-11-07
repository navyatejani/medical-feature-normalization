import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import confusion_matrix, recall_score, precision_score

st.set_page_config(page_title="Feature Normalization Engine", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load("models/final_model.joblib")


@st.cache_data
def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    return train, test


bundle = load_bundle()
model = bundle["model"]
saved_thr = float(bundle["threshold"])
feats = bundle["feature_names"]
train, test = load_data()
X_test, y_test = test[feats], test["target"]


def score(X):
    if bundle["score_kind"] == "predict_proba":
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


test_scores = score(X_test)
lo, hi = float(test_scores.min()), float(test_scores.max())

st.title("Medical Diagnostic Image Feature Normalization Engine")
st.caption(f"Model: {bundle['name']} with StandardScaler pipeline. "
           "Optimized for Recall (no missed malignant cases).")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Predict a patient", "Recall vs threshold", "Why normalization",
    "Model comparison", "Feature importance", "Calibration drift", "Batch upload"])

# ---------------- Tab 1: predict ----------------
with tab1:
    st.subheader("Patient prediction")
    choice = st.radio("Load a sample patient:",
                      ["Malignant example", "Benign example", "Edit manually"], horizontal=True)

    if choice == "Malignant example":
        row = test[test["target"] == 1].iloc[0]
    elif choice == "Benign example":
        row = test[test["target"] == 0].iloc[0]
    else:
        row = test.iloc[0]

    key_feats = ["mean radius", "mean texture", "mean area", "mean smoothness",
                 "mean concavity", "worst radius", "worst area", "worst concave points"]
    values = {f: float(row[f]) for f in feats}

    cols = st.columns(4)
    for i, f in enumerate(key_feats):
        values[f] = cols[i % 4].number_input(f, value=float(row[f]), format="%.4f",
                                             key=f"{choice}_{f}")
    st.caption("Other features are taken from the selected sample patient.")

    if st.button("Predict", type="primary"):
        x = pd.DataFrame([values])[feats]
        s = float(score(x)[0])
        is_malignant = s >= saved_thr
        if is_malignant:
            st.error(f"MALIGNANT suspected (score {s:.3f} >= threshold {saved_thr:.3f})")
        else:
            st.success(f"BENIGN (score {s:.3f} < threshold {saved_thr:.3f})")
        if choice != "Edit manually":
            actual = "Malignant" if row["target"] == 1 else "Benign"
            st.write(f"Actual label of this sample: **{actual}**")

# ---------------- Tab 2: threshold slider ----------------
with tab2:
    st.subheader("Recall vs precision trade-off (test set)")
    thr = st.slider("Decision threshold", lo, hi, saved_thr, step=(hi - lo) / 200)
    pred = (test_scores >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall", f"{recall_score(y_test, pred):.3f}")
    c2.metric("Precision", f"{precision_score(y_test, pred, zero_division=0):.3f}")
    c3.metric("Missed cancers (FN)", int(fn))
    c4.metric("False alarms (FP)", int(fp))

    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow([[tn, fp], [fn, tp]], cmap="Blues")
    for (i, j), v in np.ndenumerate([[tn, fp], [fn, tp]]):
        ax.text(j, i, int(v), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Benign", "Pred Malignant"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True Benign", "True Malignant"])
    st.pyplot(fig)
    st.info("Move the slider left: fewer missed cancers but more false alarms. "
            f"Saved threshold = {saved_thr:.3f}.")

# ---------------- Tab 3: normalization ----------------
with tab3:
    st.subheader("Raw vs normalized features")
    cols_show = ["mean area", "mean perimeter", "mean texture", "mean smoothness", "mean symmetry"]
    scaler = model.named_steps["scaler"]
    idx = [feats.index(c) for c in cols_show]
    scaled = (train[feats].values - scaler.mean_) / scaler.scale_
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].boxplot([train[c] for c in cols_show], tick_labels=cols_show)
    axes[0].set_title("Before: raw scales")
    axes[1].boxplot([scaled[:, i] for i in idx], tick_labels=cols_show)
    axes[1].set_title("After: StandardScaler")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    st.pyplot(fig)

# ---------------- Tab 4: model comparison ----------------
with tab4:
    st.subheader("Scaler x model experiment (5-fold CV on training data)")
    comp = pd.read_csv("outputs/scaler_comparison.csv")
    st.dataframe(comp.sort_values("Recall", ascending=False), use_container_width=True)
    st.bar_chart(comp.pivot(index="Model", columns="Scaler", values="Recall"))
    st.subheader("Final evaluation on the unseen test set")
    st.dataframe(pd.read_csv("outputs/final_evaluation.csv").set_index("Model").T,
                 use_container_width=True)
    st.image("outputs/06_confusion_matrices.png")

# ---------------- Tab 5: feature importance ----------------
with tab5:
    st.subheader("What drives the prediction?")
    clf = model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        coef = pd.Series(clf.coef_[0], index=feats)
        top = coef.reindex(coef.abs().sort_values(ascending=False).head(10).index)
        st.bar_chart(top)
        st.caption("Coefficients on standardized features. Positive pushes toward malignant, "
                   "negative toward benign. Many features are correlated, so read this as "
                   "a guide, not exact causes.")
    else:
        st.info("Coefficients are only available for linear models.")

# ---------------- Tab 6: calibration drift ----------------
with tab6:
    st.subheader("Simulating a machine with different calibration")
    st.write("Each feature is randomly rescaled by up to the chosen percentage, as if the scan "
             "came from a differently calibrated device. We compare the model as-is against "
             "normalizing the new device's batch with its own statistics.")
    drift = st.slider("Calibration drift (+/- %)", 0, 50, 25) / 100
    rng = np.random.default_rng(0)
    factors = 1 + rng.uniform(-drift, drift, size=len(feats))
    Xd = pd.DataFrame(X_test.values * factors, columns=feats)

    s_asis = model.decision_function(Xd)
    Xz = (Xd.values - Xd.values.mean(axis=0)) / Xd.values.std(axis=0)
    s_fixed = model.named_steps["clf"].decision_function(Xz)

    rows = []
    for label, sc in [("No correction (original scaler)", s_asis),
                      ("Per-device normalization", s_fixed)]:
        p = (sc >= saved_thr).astype(int)
        tn_, fp_, fn_, tp_ = confusion_matrix(y_test, p, labels=[0, 1]).ravel()
        rows.append({"Approach": label,
                     "Recall": round(recall_score(y_test, p), 3),
                     "Precision": round(precision_score(y_test, p, zero_division=0), 3),
                     "Missed cancers": int(fn_), "False alarms": int(fp_)})
    st.table(pd.DataFrame(rows).set_index("Approach"))
    st.caption("Per-device normalization assumes the batch has a reasonably representative "
               "mix of patients. Results vary with the drift level, so try several.")

# ---------------- Tab 7: batch upload ----------------
with tab7:
    st.subheader("Batch prediction from CSV")
    st.write("Upload a CSV with the 30 feature columns. To try it, upload `data/test.csv`.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        df_up = pd.read_csv(up)
        missing = [f for f in feats if f not in df_up.columns]
        if missing:
            st.error(f"Missing columns: {missing[:5]} ...")
        else:
            sc = score(df_up[feats])
            out = df_up.copy()
            out["score"] = sc
            out["prediction"] = np.where(sc >= saved_thr, "Malignant", "Benign")
            st.write(out["prediction"].value_counts().to_frame("patients"))
            st.dataframe(out[["score", "prediction"]].head(50), use_container_width=True)
            st.download_button("Download predictions", out.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")