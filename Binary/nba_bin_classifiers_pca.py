# nba_bin_classifiers_pca.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics.pairwise import rbf_kernel

df = pd.read_csv("pca_df.csv")
X = df[["PC1", "PC2"]]
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

lr = LogisticRegression(max_iter=2000)
knn = KNeighborsClassifier(n_neighbors=11)
nb = GaussianNB()

lr.fit(X_train, y_train)
knn.fit(X_train, y_train)
nb.fit(X_train, y_train)

# Kernel LR
K_train = rbf_kernel(X_train, X_train, gamma=0.01)
K_test = rbf_kernel(X_test, X_train, gamma=0.01)
klr = LogisticRegression(C=5000, max_iter=2000)
klr.fit(K_train, y_train)

def eval_model(name, model, X_eval, is_klr=False):
    y_pred = model.predict(X_eval)
    print("\n==== {} ====".format(name))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

eval_model("LR + PCA", lr, X_test)
eval_model("KNN + PCA", knn, X_test)
eval_model("NB + PCA", nb, X_test)
eval_model("Kernel LR + PCA", klr, K_test)
