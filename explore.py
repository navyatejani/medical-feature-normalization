import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # saves plots to files, no pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ---------- 1. Load data ----------
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df["target"] = 1 - df["target"]  # flip: Malignant = 1, Benign = 0
df["diagnosis"] = df["target"].map({1: "Malignant", 0: "Benign"})
df.to_csv("data/breast_cancer.csv", index=False)

features = list(data.feature_names)
palette = {"Malignant": "#d62728", "Benign": "#1f77b4"}

# ---------- 2. Basic checks ----------
print("Shape:", df.shape)
print("Missing values:", int(df[features].isnull().sum().sum()))
print("\nClass counts:")
print(df["diagnosis"].value_counts())
print("\nMalignant %:", round(df["target"].mean() * 100, 1))

key_cols = ["mean area", "mean perimeter", "mean texture",
            "mean smoothness", "mean compactness", "mean symmetry"]
print("\nRaw feature scales (this is the problem normalization solves):")
print(df[key_cols].describe().loc[["min", "max", "mean", "std"]].round(3))

# ---------- 3. Plot: class distribution ----------
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="diagnosis", hue="diagnosis", palette=palette, legend=False)
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig("outputs/01_class_distribution.png", dpi=150)
plt.close()

# ---------- 4. Plot: raw vs normalized scales ----------
scaled = pd.DataFrame(StandardScaler().fit_transform(df[key_cols]), columns=key_cols)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df[key_cols].plot(kind="box", ax=axes[0])
axes[0].set_title("BEFORE normalization (raw scales)")
axes[0].tick_params(axis="x", rotation=45)
scaled.plot(kind="box", ax=axes[1])
axes[1].set_title("AFTER normalization (StandardScaler)")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("outputs/02_before_after_scaling.png", dpi=150)
plt.close()

# ---------- 5. Plot: feature distributions by class ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, col in zip(axes.ravel(), ["mean area", "mean radius", "mean concavity", "mean texture"]):
    sns.histplot(data=df, x=col, hue="diagnosis", palette=palette, kde=True, ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.savefig("outputs/03_feature_distributions.png", dpi=150)
plt.close()

# ---------- 6. Plot: correlation heatmap ----------
mean_cols = features[:10]
plt.figure(figsize=(10, 8))
sns.heatmap(df[mean_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7})
plt.title("Correlation of Mean Features")
plt.tight_layout()
plt.savefig("outputs/04_correlation_heatmap.png", dpi=150)
plt.close()

# ---------- 7. Top features related to malignancy ----------
corr = df[features + ["target"]].corr()["target"].drop("target")
print("\nTop 8 features correlated with malignancy:")
print(corr.abs().sort_values(ascending=False).head(8).round(3))

print("\nDone. Plots saved in the 'outputs' folder.")