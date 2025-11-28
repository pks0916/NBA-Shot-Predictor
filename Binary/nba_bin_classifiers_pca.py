import os
import pickle
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


# -------------------------------------------------------------------
# 1. Load data
# -------------------------------------------------------------------
df = pd.read_csv("nba_bin_df.csv")

# -------------------------------------------------------------------
# 2. Feature engineering
#    (spatial + time features to give models more information)
# -------------------------------------------------------------------
df = df.copy()

# Shooting angle (radians)
df["ANGLE"] = np.arctan2(df["LOC_Y"], df["LOC_X"])

# Corner 3 flag (corner 3s behave differently)
df["IS_CORNER"] = ((df["LOC_Y"] < 92) & (np.abs(df["LOC_X"]) > 220)).astype(int)

# Total time remaining in the period (seconds)
df["TIME_REMAINING"] = df["MINUTES_REMAINING"] * 60 + df["SECONDS_REMAINING"]

# Distance bucket (short / mid / long range, etc.)
dist_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]
df["DIST_CAT"] = pd.cut(
    df["SHOT_DISTANCE"],
    bins=dist_bins,
    labels=False,
    include_lowest=True,
)
# Shots beyond 40 feet: put them into an extra bucket
df["DIST_CAT"] = df["DIST_CAT"].fillna(len(dist_bins)).astype(int)

# -------------------------------------------------------------------
# 3. Define features and target
# -------------------------------------------------------------------
X = df.drop(columns=["SHOT_MADE_FLAG"])
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
]

categorical_features = [
    "SHOT_TYPE",
    "SHOT_ZONE_BASIC",
    "IS_CORNER",
]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# -------------------------------------------------------------------
# 4. Train / test split
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# -------------------------------------------------------------------
# 5. Build and (lightly) tune models
# -------------------------------------------------------------------
models = {}

# ---------------- Logistic Regression ----------------
lr_pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=2000)),
    ]
)
lr_params = {
    "clf__C": [0.5, 1.0, 2.0, 5.0],
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs"],
}
lr_grid = GridSearchCV(
    lr_pipe,
    param_grid=lr_params,
    cv=5,
    n_jobs=-1,
    scoring="accuracy",
)
lr_grid.fit(X_train, y_train)
models["Logistic Regression"] = lr_grid

# ---------------- K-Nearest Neighbors ----------------
knn_pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", KNeighborsClassifier()),
    ]
)
knn_params = {
    "clf__n_neighbors": [5, 11, 19],
    "clf__weights": ["uniform", "distance"],
}
knn_grid = GridSearchCV(
    knn_pipe,
    param_grid=knn_params,
    cv=5,
    n_jobs=-1,
    scoring="accuracy",
)
knn_grid.fit(X_train, y_train)
models["KNN"] = knn_grid

# ---------------- Gaussian Naive Bayes ----------------
# NB doesn’t really use a big grid; just fit a clean pipeline
nb_pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", GaussianNB()),
    ]
)
nb_pipe.fit(X_train, y_train)
models["Gaussian NB"] = nb_pipe

# ---------------- Random Forest (extra model) ----------------
rf_pipe = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("clf", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ]
)
rf_params = {
    "clf__n_estimators": [150, 250],
    "clf__max_depth": [8, 12, None],
    "clf__min_samples_split": [2, 10],
    "clf__min_samples_leaf": [1, 5],
}
rf_grid = GridSearchCV(
    rf_pipe,
    param_grid=rf_params,
    cv=3,
    n_jobs=-1,
    scoring="accuracy",
)
rf_grid.fit(X_train, y_train)
models["Random Forest"] = rf_grid


# -------------------------------------------------------------------
# 6. Evaluation helper
# -------------------------------------------------------------------
def evaluate_model(name, estimator):
    """Print accuracy + confusion matrix + classification report."""
    if isinstance(estimator, GridSearchCV):
        model = estimator.best_estimator_
        print(f"\n==== {name} (best params: {estimator.best_params_}) ====")
    else:
        model = estimator
        print(f"\n==== {name} ====")

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return model


# -------------------------------------------------------------------
# 7. Run evaluation + save models
# -------------------------------------------------------------------
if __name__ == "__main__":
    final_models = {}

    for name, est in models.items():
        model = evaluate_model(name, est)
        final_models[name] = model

    # Save models to disk for later use
    os.makedirs("Classifiers", exist_ok=True)

    # Keep original filenames for backwards compatibility
    with open(os.path.join("Classifiers", "lr.sav"), "wb") as f:
        pickle.dump(final_models["Logistic Regression"], f)

    with open(os.path.join("Classifiers", "knn.sav"), "wb") as f:
        pickle.dump(final_models["KNN"], f)

    with open(os.path.join("Classifiers", "nb.sav"), "wb") as f:
        pickle.dump(final_models["Gaussian NB"], f)

    # New: save Random Forest as well
    with open(os.path.join("Classifiers", "rf.sav"), "wb") as f:
        pickle.dump(final_models["Random Forest"], f)
