# nba_bin_classifiers.py
#
# Fast version: no GridSearchCV, just single trained models.
# Uses:
# - Spatial features (LOC_X, LOC_Y, SHOT_DISTANCE)
# - Temporal features (PERIOD, MINUTES_REMAINING, SECONDS_REMAINING)
# - Engineered features (ANGLE, IS_CORNER, DIST_CAT, TIME_REMAINING)
# - Shooter features (PLAYER_FG_PCT, PLAYER_SHOT_ATTEMPTS, SHOOTER_TIER)

import os
import pickle
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


# -----------------------------------------------------------
# 1. Load enhanced dataset
# -----------------------------------------------------------
df = pd.read_csv("shot_charts_2025_enhanced.csv")

# -----------------------------------------------------------
# 2. Feature engineering
# -----------------------------------------------------------
df = df.copy()

# Shooting angle (radians)
df["ANGLE"] = np.arctan2(df["LOC_Y"], df["LOC_X"])

# Corner 3 flag (for 3PT shots)
df["IS_CORNER"] = (
    (df["SHOT_TYPE"] == "3PT Field Goal")
    & (df["LOC_Y"] < 92)
    & (df["LOC_X"].abs() > 220)
).astype(int)

# Total time remaining in the period
df["TIME_REMAINING"] = df["MINUTES_REMAINING"] * 60 + df["SECONDS_REMAINING"]

# Distance bucket feature
dist_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]
df["DIST_CAT"] = pd.cut(df["SHOT_DISTANCE"], bins=dist_bins, labels=False, include_lowest=True)
df["DIST_CAT"] = df["DIST_CAT"].fillna(len(dist_bins)).astype(int)

# Make sure shooter tier is a string category
df["SHOOTER_TIER"] = df["SHOOTER_TIER"].astype(str)

# -----------------------------------------------------------
# 3. Define features and target
# -----------------------------------------------------------
X = df.drop(columns=["SHOT_MADE_FLAG", "PLAYER_ID"])
y = df["SHOT_MADE_FLAG"]

numeric_features = [
    "LOC_X",
    "LOC_Y",
    "SHOT_DISTANCE",
    "PERIOD",
    "MINUTES_REMAINING",
    "SECONDS_REMAINING",
    "ANGLE",
    "TIME_REMAINING",
    "DIST_CAT",
    "PLAYER_FG_PCT",
    "PLAYER_SHOT_ATTEMPTS",
]

categorical_features = [
    "SHOT_TYPE",
    "SHOT_ZONE_BASIC",
    "IS_CORNER",
    "SHOOTER_TIER",
]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# -----------------------------------------------------------
# 4. Train/test split
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------------------------------------
# 5. Define models (single configs, no GridSearch)
# -----------------------------------------------------------
models = {
    "Logistic Regression": Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "clf",
                LogisticRegression(
                    C=2.0,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                ),
            ),
        ]
    ),
    "KNN": Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "clf",
                KNeighborsClassifier(
                    n_neighbors=11,
                    weights="distance",
                ),
            ),
        ]
    ),
    "Gaussian NB": Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", GaussianNB()),
        ]
    ),
    "Random Forest": Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}

# -----------------------------------------------------------
# 6. Training, evaluation, saving
# -----------------------------------------------------------
def evaluate_model(name, model):
    print(f"\n==== {name} ====")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    os.makedirs("Classifiers", exist_ok=True)
    trained = {}

    for name, pipe in models.items():
        print(f"\nTraining {name} ...")
        pipe.fit(X_train, y_train)
        evaluate_model(name, pipe)
        trained[name] = pipe

    # Save models
    with open(os.path.join("Classifiers", "lr.sav"), "wb") as f:
        pickle.dump(trained["Logistic Regression"], f)
    with open(os.path.join("Classifiers", "knn.sav"), "wb") as f:
        pickle.dump(trained["KNN"], f)
    with open(os.path.join("Classifiers", "nb.sav"), "wb") as f:
        pickle.dump(trained["Gaussian NB"], f)
    with open(os.path.join("Classifiers", "rf.sav"), "wb") as f:
        pickle.dump(trained["Random Forest"], f)
