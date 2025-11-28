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



df = pd.read_csv("shot_charts_2025_enhanced.csv")


df = df.copy()

df["ANGLE"] = np.arctan2(df["LOC_Y"], df["LOC_X"])

df["IS_CORNER"] = (
    (df["SHOT_TYPE"] == "3PT Field Goal")
    & (df["LOC_Y"] < 92)
    & (df["LOC_X"].abs() > 220)
).astype(int)

df["TIME_REMAINING"] = df["MINUTES_REMAINING"] * 60 + df["SECONDS_REMAINING"]

dist_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]
df["DIST_CAT"] = pd.cut(df["SHOT_DISTANCE"], bins=dist_bins, labels=False, include_lowest=True)
df["DIST_CAT"] = df["DIST_CAT"].fillna(len(dist_bins)).astype(int)

df["SHOOTER_TIER"] = df["SHOOTER_TIER"].astype(str)



X = df.drop(columns=["SHOT_MADE_FLAG", "PLAYER_ID"])
y = df["SHOT_MADE_FLAG"]


num_cols = [
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

cat_cols = [
    "SHOT_TYPE",
    "SHOT_ZONE_BASIC",
    "IS_CORNER",
    "SHOOTER_TIER",
]


pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)



models = {}

models["Logistic Regression"] = Pipeline(
    steps=[
        ("pre", pre),
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
)

models["KNN"] = Pipeline(
    steps=[
        ("pre", pre),
        (
            "clf",
            KNeighborsClassifier(
                n_neighbors=11,
                weights="distance",
            ),
        ),
    ]
)

models["Gaussian NB"] = Pipeline(
    steps=[
        ("pre", pre),
        ("clf", GaussianNB()),
    ]
)

models["Random Forest"] = Pipeline(
    steps=[
        ("pre", pre),
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
)



def eval_model(name, model, X_t, y_t):
    print("\n====", name, "====")
    y_pred = model.predict(X_t)
    print("Accuracy:", accuracy_score(y_t, y_pred))
    print(confusion_matrix(y_t, y_pred))
    print(classification_report(y_t, y_pred))



if __name__ == "__main__":

    os.makedirs("Classifiers", exist_ok=True)

    trained = {}

    for name, pipe in models.items():
        print("\nTraining", name, "...")
        pipe.fit(X_train, y_train)
        eval_model(name, pipe, X_test, y_test)
        trained[name] = pipe

    with open(os.path.join("Classifiers", "lr.sav"), "wb") as f:
        pickle.dump(trained["Logistic Regression"], f)

    with open(os.path.join("Classifiers", "knn.sav"), "wb") as f:
        pickle.dump(trained["KNN"], f)

    with open(os.path.join("Classifiers", "nb.sav"), "wb") as f:
        pickle.dump(trained["Gaussian NB"], f)

    with open(os.path.join("Classifiers", "rf.sav"), "wb") as f:
        pickle.dump(trained["Random Forest"], f)
