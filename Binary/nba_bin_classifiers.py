# nba_bin_classifiers.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics.pairwise import rbf_kernel
import pickle

df = pd.read_csv("nba_bin_df.csv")

numeric_features = [
    "LOC_X", "LOC_Y", "SHOT_DISTANCE",
    "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING"
]

categorical_features = ["SHOT_TYPE", "SHOT_ZONE_BASIC"]

X = df[numeric_features + categorical_features]
y = df["SHOT_MADE_FLAG"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Models
lr = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000))])
knn = Pipeline([("prep", preprocessor), ("clf", KNeighborsClassifier(n_neighbors=11))])

# GaussianNB
Xt_train = preprocessor.fit_transform(X_train)
Xt_train_dense = Xt_train.toarray() if hasattr(Xt_train, "toarray") else Xt_train
nb = GaussianNB()
nb.fit(Xt_train_dense, y_train)

# Kernel LR
K_train = rbf_kernel(Xt_train_dense, Xt_train_dense, gamma=0.01)
klr = LogisticRegression(C=5000, max_iter=2000)
klr.fit(K_train, y_train)

# Evaluate
def eval_model(name, model, X_eval, y_true, kernel=False):
    if kernel:
        Xt_eval = preprocessor.transform(X_eval)
        Xt_eval_dense = Xt_eval.toarray()
        K_test = rbf_kernel(Xt_eval_dense, Xt_train_dense, gamma=0.01)
        y_pred = model.predict(K_test)
    else:
        y_pred = model.predict(X_eval)

    print("\n==== {} ====".format(name))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

eval_model("Logistic Regression", lr, X_test, y_test)
eval_model("KNN", knn, X_test, y_test)
eval_model("GaussianNB", nb, X_test, y_test, kernel=False)
eval_model("Kernel Logistic Regression", klr, X_test, y_test, kernel=True)

# Save models
import os
os.makedirs("Classifiers", exist_ok=True)
pickle.dump(lr, open("Classifiers/lr.sav", "wb"))
pickle.dump(knn, open("Classifiers/knn.sav", "wb"))
pickle.dump(nb, open("Classifiers/nb.sav", "wb"))
pickle.dump(klr, open("Classifiers/klr.sav", "wb"))
