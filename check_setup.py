import numpy as np, pandas as pd, sklearn, matplotlib, seaborn, joblib, streamlit
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
print("All libraries imported OK")
print("sklearn version:", sklearn.__version__)
print("Dataset shape:", data.data.shape)
print("Classes:", dict(zip(data.target_names, np.bincount(data.target))))