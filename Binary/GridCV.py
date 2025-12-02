# nba_bin_gridsearch.py
# Cross-validated Grid Search for NBA Shot Make/Miss Models

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score


# Load Data
df = pd.read_csv("shot_charts_2025_enhanced.csv")

num_cols = [
    "LOC_X",
    "LOC_Y",
    "SHOT_DISTANCE",
    "PERIOD",
    "MINUTES_REMAINING",
    "SECONDS_REMAINING",
    "PLAYER_FG_PCT",
    "PLAYER_SHOT_ATTEMPTS",
]

cat_cols = [
    "SHOT_TYPE",
    "SHOT_ZONE_BASIC",
    "SHOOTER_TIER",
]


df["SHOT_MADE_FLAG"] = df["SHOT_MADE_FLAG"].astype(int)

df_model = df[num_cols + cat_cols + ["SHOT_MADE_FLAG"]].dropna()
X = df_model.drop("SHOT_MADE_FLAG", axis=1)
y = df_model["SHOT_MADE_FLAG"]


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)


# Define Models and Parameter Grids

# Logistic Regression
lr_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=5000))
])

lr_params = {
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__penalty": ["l2"],
    "clf__solver": ["lbfgs", "saga"]
}

# KNN
knn_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", KNeighborsClassifier())
])

knn_params = {
    "clf__n_neighbors": [3, 5, 7, 9, 11],
    "clf__weights": ["uniform", "distance"],
    "clf__metric": ["euclidean", "manhattan"]
}

# Random Forest
rf_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(random_state=42))
])

rf_params = {
    "clf__n_estimators": [50, 100, 200],
    "clf__max_depth": [None, 5, 10, 20],
    "clf__min_samples_split": [2, 5, 10],
    "clf__min_samples_leaf": [1, 2, 4]
}

# Gaussian Naive Bayes
nb_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", GaussianNB())
])

nb_params = {
    "clf__var_smoothing": [1e-9, 1e-8, 1e-7]
}


klr_params = {
    "clf__C": [1, 10, 100, 1000],
    "clf__gamma": [0.001, 0.01, 0.1, 1.0],
    "clf__max_iter": [2000]
}

# Grid Search Function
def run_grid_search(pipeline, param_grid, X_train, y_train, model_name):
    print(f"\n=== Running GridSearchCV for {model_name} ===")
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")
    return grid.best_estimator_

# Run Grid Searches
best_lr = run_grid_search(lr_pipe, lr_params, X_train, y_train, "Logistic Regression")
best_knn = run_grid_search(knn_pipe, knn_params, X_train, y_train, "KNN")
best_rf = run_grid_search(rf_pipe, rf_params, X_train, y_train, "Random Forest")
best_nb = run_grid_search(nb_pipe, nb_params, X_train, y_train, "GaussianNB")
