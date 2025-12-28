import pandas as pd
from sklearn.datasets import make_regression, make_classification
from pathlib import Path

# Cartella data/
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# -------------------------
# REGRESSION DATASET
# -------------------------
X_reg, y_reg = make_regression(
    n_samples=500,
    n_features=6,
    noise=15.0,
    random_state=42
)

reg_df = pd.DataFrame(
    X_reg,
    columns=[f"feature_{i}" for i in range(X_reg.shape[1])]
)
reg_df["target"] = y_reg

reg_path = DATA_DIR / "regression.csv"
reg_df.to_csv(reg_path, index=False)

# -------------------------
# CLASSIFICATION DATASET
# -------------------------
X_clf, y_clf = make_classification(
    n_samples=600,
    n_features=8,
    n_informative=5,
    n_redundant=1,
    n_classes=2,
    weights=[0.7, 0.3],
    flip_y=0.02,
    random_state=42
)

clf_df = pd.DataFrame(
    X_clf,
    columns=[f"feature_{i}" for i in range(X_clf.shape[1])]
)
clf_df["target"] = y_clf

clf_path = DATA_DIR / "classification.csv"
clf_df.to_csv(clf_path, index=False)

print("Dataset creati correttamente:")
print(reg_path)
print(clf_path)
