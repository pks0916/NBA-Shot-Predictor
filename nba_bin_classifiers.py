# nba_bin_classifiers.py
# NBA Shot Make/Miss Classification — Baseline Models (No PCA)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import rbf_kernel
import pickle

# =========================
# 1. LOAD & CLEAN DATA
# =========================

df = pd.read_csv("nba_shots.csv")   # <--- YOUR FILE NAME HERE

# Filter to only actual FG attempts
df = df[df["SHOT_ATTEMPTED_FLAG"] == 1].copy()

# Target: convert to 0/1
df["SHOT_MADE_FLAG"] = df["SHOT_MADE_FLAG"].astype(int)

# FEATURES WE WILL USE
numeric_features = [
    "LOC_X",
    "LOC_Y",
    "SHOT_DISTANCE",
    "PERIOD",
    "MINUTES_REMAINING",
    "SECONDS_REMAINING"
]

categorical_features = [
    "SHOT_TYPE",
    "SHOT_ZONE_BASIC"
]

# Select modeling columns
df_model = df[numeric_features + categorical_features + ["SHOT_MADE_FLAG"]].dropna()

X = df_model.drop("SHOT_MADE_FLAG", axis=1)
y = df_model["SHOT_MADE_FLAG"]

# =========================
# 2. TRAIN/TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =========================
# 3. PREPROCESSOR
# =========================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# =========================
# 4. MODELS
# =========================

lr = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000))
])

knn = Pipeline([
    ("prep", preprocessor),
    ("clf", KNeighborsClassifier(n_neighbors=7))
])

# GaussianNB needs dense data
def train_gaussian_nb(X_train, y_train):
    Xt = preprocessor.fit_transform(X_train)
    Xt_dense = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    nb = GaussianNB()
    nb.fit(Xt_dense, y_train)
    return nb

nb = train_gaussian_nb(X_train, y_train)

# Kernel Logistic Regression
def train_kernel_lr(X_train, y_train):
    Xt = preprocessor.fit_transform(X_train)
    Xt_dense = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    K_train = rbf_kernel(Xt_dense, Xt_dense, gamma=0.01)
    klr = LogisticRegression(C=10000, max_iter=2000)
    klr.fit(K_train, y_train)
    return klr, Xt_dense

klr, X_train_kernel = train_kernel_lr(X_train, y_train)

# Fit LR + KNN
lr.fit(X_train, y_train)
knn.fit(X_train, y_train)

# =========================
# 5. EVALUATION
# =========================

def evaluate(name, model, X_test, y_test, kernel=False):
    if kernel:
        Xt = preprocessor.transform(X_test)
        Xt_dense = Xt.toarray() if hasattr(Xt, "toarray") else Xt
        K_test = rbf_kernel(Xt_dense, X_train_kernel, gamma=0.01)
        y_pred = model.predict(K_test)
    else:
        y_pred = model.predict(X_test)

    print("\n=== {} ===".format(name))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

evaluate("Logistic Regression", lr, X_test, y_test)
evaluate("KNN (k=7)", knn, X_test, y_test)

# NB
Xt_test = preprocessor.transform(X_test)
Xt_test_dense = Xt_test.toarray() if hasattr(Xt_test, "toarray") else Xt_test
y_pred_nb = nb.predict(Xt_test_dense)
print("\n=== Gaussian Naive Bayes ===")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# Kernel LR
evaluate("Kernel Logistic Regression (RBF)", klr, X_test, y_test, kernel=True)
