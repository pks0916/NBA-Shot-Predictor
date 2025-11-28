import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



df = pd.read_csv("shot_charts_2025_enhanced.csv")


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


X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)



pca_step = PCA(n_components=10, random_state=42)



models = {
    "Logistic Regression + PCA": LogisticRegression(
        C=2.0, penalty="l2", solver="lbfgs", max_iter=2000
    ),
    "KNN + PCA": KNeighborsClassifier(n_neighbors=11, weights="distance"),
    "Gaussian NB + PCA": GaussianNB(),
    "Random Forest + PCA": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    ),
}



def run_model(name, base_clf):
    pipe = Pipeline(
        steps=[
            ("pre", pre),
            ("pca", pca_step),
            ("clf", base_clf),
        ]
    )

    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)

    print("\n====", name, "====")
    print("Accuracy:", accuracy_score(y_te, y_pred))
    print(confusion_matrix(y_te, y_pred))
    print(classification_report(y_te, y_pred))



if __name__ == "__main__":
    for name, clf in models.items():
        run_model(name, clf)
